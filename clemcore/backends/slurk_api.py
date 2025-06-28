import json
import logging
from typing import List, Dict, Tuple, Any

import requests
import socketio
from retry import retry

from clemcore import backends
from clemcore.backends import ModelSpec, Model
from clemcore.backends.utils import ensure_messages_format
from clemcore.clemgame.resources import load_packaged_file, load_json

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")

NAME = "slurk"


class SlurkClient:

    def __init__(self, slurk_host: str, api_key: str):
        self.slurk_host = slurk_host
        self.base_url = f"{slurk_host}/slurk/api"
        self.authorize_header = {"Authorization": f"Bearer {api_key}"}

    def create_user(self, name, token):
        response = requests.post(
            f"{self.base_url}/users", headers=self.authorize_header, json={"name": name, "token_id": token})
        self.assert_response(response, f"create user: {name}")
        return response.json()["id"]

    def create_task(self, name: str, num_users: int, layout_id: int):
        response = requests.post(f"{self.base_url}/tasks", headers=self.authorize_header,
                                 json={"name": name, "num_users": num_users, "layout_id": layout_id})
        self.assert_response(response, f"create task: {name}")
        return response.json()["id"]

    def create_permissions(self, permissions: Dict):
        response = requests.post(f"{self.base_url}/permissions", headers=self.authorize_header, json=permissions)
        self.assert_response(response, "create permissions")
        return response.json()["id"]

    def create_token(self, permissions_id: int, room_id: int, task_id: int = None):
        response = requests.post(
            f"{self.base_url}/tokens", headers=self.authorize_header,
            json={
                "permissions_id": permissions_id,
                "room_id": room_id,
                "registrations_left": 1,
                "task_id": task_id,
            },
        )
        self.assert_response(response, "create token")
        return response.json()["id"]

    def create_room(self, room_layout_id: int):
        response = requests.post(
            f"{self.base_url}/rooms", headers=self.authorize_header,
            json={"layout_id": room_layout_id},
        )
        self.assert_response(response, f"create room")
        return response.json()["id"]

    def create_room_layout(self, room_layout: Dict):
        response = requests.post(f"{self.base_url}/layouts", headers=self.authorize_header, json=room_layout)
        self.assert_response(response, f"create room layout")
        return response.json()["id"]

    def join_room(self, user_id: int, room_id: int):
        response = requests.post(f"{self.base_url}/users/{user_id}/rooms/{room_id}", headers=self.authorize_header)
        self.assert_response(response, f"user {user_id}  joins room {room_id}")

    def assert_response(self, response, description):
        if not response.ok:
            logger.error(f"`{description}` unsuccessful: {response.status_code}")
            response.raise_for_status()
        logger.debug(f"`{description}` successful.")


class Slurk(backends.RemoteBackend):

    def _make_api_client(self):
        creds = backends.load_credentials(NAME)
        return SlurkClient(creds[NAME]["slurk_host"], creds[NAME]["api_key"])  # slurk admin token

    def get_model_for(self, model_spec: ModelSpec) -> Model:
        # Note: If you want to customize the room layout for a specific game, then create a new model registry entry
        # e.g. "model_name":"slurk_taboo" with a model_spec specifying the path to the custom layout json
        if model_spec.room_layout_path == "packaged":
            task_room_layout = json.loads(load_packaged_file("resources/slurk_api/task_room_layout.json"))
        else:
            task_room_layout = load_json(model_spec.room_layout_path)
        bot_permissions = json.loads(load_packaged_file("resources/slurk_api/bot_permissions.json"))
        user_permissions = json.loads(load_packaged_file("resources/slurk_api/user_permissions.json"))

        layout_id = self.client.create_room_layout(task_room_layout)
        bot_permissions_id = self.client.create_permissions(bot_permissions)
        user_permissions_id = self.client.create_permissions(user_permissions)

        room_id = self.client.create_room(layout_id)  # each player needs its own room anyway
        bot_token = self.client.create_token(bot_permissions_id, room_id)
        # Note: You can customize the bot name for individual games as described above
        bot_id = self.client.create_user(model_spec.bot_name, bot_token)
        user_token = self.client.create_token(user_permissions_id, room_id)

        slurk_model = SlurkModel(bot_id, bot_token, room_id, model_spec).connect(self.client.slurk_host)
        self.client.join_room(bot_id, room_id)  # todo why do we need this?

        stdout_logger.info(f"SLURK_USER_TOKEN={user_token}")
        stdout_logger.info(f"{self.client.slurk_host}/login?name=player&token={user_token}")
        slurk_model.wait_for_participant()
        stdout_logger.info("Slurk user joined")
        return slurk_model


class SlurkModel(backends.Model):  # todo: make this HumanModel when HumanModel is fully integrated as a backend

    def __init__(self, user_id: int, user_token: str, room_id: int, model_spec: ModelSpec):
        super().__init__(model_spec)
        self.user_id = user_id
        self.user_token = user_token
        self.sio = socketio.Client(logger=logger)
        self.sync_event = self.sio.eio.create_event()  # the vehicle to wait until user responds

        """
            TODO: For now we can only allow a single room and player.
            What would it mean to start the benchmark once, but let multiple users accomplish it?
            For this we would need a separate approach that starts an individual benchmark run
            when a user connects.
        """
        self.room_id = room_id
        self.user_messages = list()  # we need an object to carry over the response between threads

        def store_and_unblock(data):
            if data['room'] != self.room_id:
                return
            if data["user"]["id"] == self.user_id:
                return  # ignore self
            self.user_messages.append(data["message"])  # collect user response
            self.sync_event.set()  # continue the other thread

        self.sio.on("text_message", store_and_unblock)

        def check_and_unblock(data):
            if data['room'] != self.room_id:
                return
            if data["user"]["id"] == self.user_id:
                return  # ignore self
            self.sync_event.set()  # continue the other thread

        self.sio.on("status", check_and_unblock)

    def wait_for_user_response(self, messages) -> str:
        latest_response = "Nothing has been said yet."
        if messages:
            latest_response = messages[-1]["content"]
        self.sio.emit("text", {"message": latest_response, "room": self.room_id})
        if not self.sync_event.wait(timeout=self.model_spec.response_timout):
            pass  # no user response
        self.sync_event.clear()
        user_response = self.user_messages[0]
        self.user_messages.clear()
        return user_response

    def wait_for_participant(self):
        # this works because: self.sio.on("status", check_and_unblock)
        if not self.sync_event.wait(timeout=self.model_spec.join_timout):
            raise RuntimeError("no user joined the slurk room")
        self.sync_event.clear()

    def connect(self, slurk_host):
        """Establish a connection to the remote server."""
        self.sio.connect(slurk_host,
                         headers={"Authorization": f"Bearer {self.user_token}", "user": str(self.user_id)},
                         namespaces="/")
        return self

    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """
        :param messages: for example
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        :param model: chat-gpt for chat-completion, otherwise text completion
        :return: the continuation
        """
        response_text = self.wait_for_user_response(messages)
        return messages, {"response": "slurk"}, response_text
