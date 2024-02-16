from typing import List, Tuple, Dict

import os
import logging
from time import sleep

from backends import Model, CustomResponseModel, ModelSpec
from clemgame.clemgame import GameMaster, GameBenchmark
from clemgame.slurkbot import APIChatBot
from clemgame import get_logger
from clemgame import file_utils
from games.chatgame.game import ChatGame

GAME_NAME = "chatgame"

MAX_RETRIES = 5


logger = get_logger(__name__)  # clem logging
LOG = logging.getLogger(__name__)  # slurk logging


class Chat(GameMaster, APIChatBot):
    """GameMaster that also connects to a slurk server in order to listen to
    events sent by human participants. The GameMaster is then run as a process
    that connects to the slurk server and listens to its events.

    When a new user message is registered, it is saved in the respective
    Player object.
    """
    def __init__(self,
                 experiment: Dict,
                 player_models: Tuple[Model],
                 slurk_token: str,
                 slurk_user: int,
                 slurk_task: int,
                 slurk_host: str,
                 slurk_port: str
                 ):
        GameMaster.__init__(self, GAME_NAME, experiment, player_models)
        APIChatBot.__init__(self, slurk_token, slurk_user, slurk_task, slurk_host, slurk_port)
        self.experiment = experiment
        self.player_models = player_models
        self.game = None
        # the general slurk connection logic is defined in the parent classes
        # here we just need to take care of the core interaction logic that is
        # triggered by incoming text messages
        self.sio.on("text_message", self.incoming_message())

    def incoming_message(self):
        """Handle the user message."""

        def message(data):
            """Triggered once a text message is sent (no leading /).

            Count user text messages.
            If encountering something that looks like a command
            then pass it on to be parsed as such.
            """
            LOG.debug(f"Received a message from {data['user']['name']}.")

            room_id = data["room"]
            user_id = data["user"]["id"]

            # filter irrelevant messages
            if user_id == self.user:
                return

            # if the message is part of the main discussion count it
            LOG.info(self.players_per_room)
            for usr in self.players_per_room[room_id]:
                if usr["id"] == user_id and usr["status"] == "ready":
                    usr["msg_n"] += 1

                    logger.info("Game turn: %d", self.game.current_turn)
                    # set the message as the questioner's last message
                    self.game.questioner.set_current_message(data["message"])
                    # complete the turn and pass the answerer's answer
                    self.turn()
                    self.say(self.game.answerer.current_contribution, room_id)

                    # check whether we need to listen to more user text
                    if not self.game.proceeds():
                        logger.info("DIALOG DONE.")
                        goodbye = "**We have reached the maximum number of " \
                                  "turns. The conversation will end now. " \
                                  "Goodbye.**"
                        self.say(goodbye, room_id)
                        sleep(2)
                        self.command_stop(room_id, user_id)

                elif usr["id"] == user_id and usr["status"] == "done":
                    return
                elif usr["id"] == user_id:
                    self.say("*You haven't typed /ready yet.*", room_id)
                    return

        return message

    def say(self, text: str, room: int):
        self.sio.emit(
            "text",
            {
                "message": text,
                "html": True,
                "room": room,
            },
        )

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.game = ChatGame(self.game_instance, self.player_models)

    def setup(self, **kwargs):
        self._on_setup(**kwargs)

    def play(self) -> None:
        """
        This method implements the game logic. In a slurk bot, the game logic
        is guided by events from the slurk server. We call the bot's run()
        method here and handle all game details via the slurk events.
        """
        self.run()

    def turn(self):
        """This method is called once the Questioner input was received via a
        message event. The Questioner message is appended to the dialog and
        passed to the Answerer.
        turn() is only called while we haven't reached the maximum number of
        turns.

        :returns True when there was a Questioner message that the Answerer
            replies to and False if there was no message to respond to or the
            maximum number of turns was reached.
        """
        self.game.questioner_turn()
        self.game.answerer_turn()

    def _get_recorded_turns(self, records: Dict) -> List[int]:
        raise NotImplementedError

    def _compute_turn_scores(self, records: Dict, turn_idx) -> List[Tuple[str, float]]:
        raise NotImplementedError

    def _compute_episode_scores(self, records: Dict) -> List[Tuple[str, float]]:
        raise NotImplementedError

    @classmethod
    def applies_to(cls, game_name: str) -> bool:
        return game_name == GAME_NAME


class ChatGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "A chat setting in which a user can ask questions to a bot."

    def create_game_master(self, experiment: Dict, player_models: bool) -> GameMaster:
        return Chat(experiment, player_models)


def main():
    # select one instance
    experiments = file_utils.load_json("in/instances.json", "chatgame")
    instance = experiments["experiments"][0]["game_instances"][0]

    # collect environment variables
    token = os.environ["SLURK_TOKEN"]
    user = os.environ["SLURK_USER"]
    host = "http://localhost"
    port = "5000"
    task = int(os.environ["TASK_ID"])
    waiting_room = os.environ["SLURK_WAITING_ROOM"]

    # Change "mock" to a model in order to connect to a LM,
    # e.g. OpenAI.MODEL_GPT_3
    player_backends = (CustomResponseModel(ModelSpec(model_name="_slurk_response")),
                       CustomResponseModel(ModelSpec(model_name="mock")))
    master = Chat(instance, player_backends, token, user, task, host, port)
    master.waiting_room = waiting_room
    master.setup(**instance)
    master.play()


if __name__ == '__main__':
    main()
