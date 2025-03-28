import copy
import logging
from datetime import datetime
from typing import Dict, Tuple, Any, List

from clemcore.clemgame.resources import GameResourceLocator

module_logger = logging.getLogger(__name__)


class GameRecorder(GameResourceLocator):

    def __init__(self, game_name: str, game_path: str):
        super().__init__(game_name, game_path)
        self.log_current_turn = -1
        """ Stores players and turn during the runs """
        self.interactions = {
            "players": {},
            "turns": []
        }
        """ Stores calls to the API """
        self.requests = []

    def log_next_round(self):
        """Call this method to group interactions per turn."""
        self.log_current_turn += 1
        self.interactions["turns"].append([])

    def log_key(self, key: str, value: Any):
        """Add a key and value to the internal log.
        Args:
            key: A string to identify the kind of log entry to be made.
            value: The content of the entry to be logged.
        """
        self.interactions[key] = value
        module_logger.info(f"{self.game_name}: Logged a game-specific interaction key: {key}.")

    def log_players(self, players_dic: Dict):
        """Log/record the players in this game episode.
        Args:
            players_dic: Dictionary of players in this game episode.
        """
        self.interactions["players"] = players_dic
        module_logger.info(f"{self.game_name}: Logged players metadata.")

    def log_event(self, from_: str, to: str, action: Dict, call: Tuple[Any, Any] = None):
        """Add an event to the internal log.
        It can be only an action or an action plus an API call that should have the same timestamp as the action.
        Args:
            from_: The identifier string of the Player/GM that originated the action.
            to: The identifier string of the Player/GM target of the action.
            action: The benchmark action to be logged.
            call: If given, this is a tuple whose first element is the input prompt object (after API-specific
                manipulation) as passed to the API and the second element is the raw response object as returned by the
                API.
        """
        assert self.log_current_turn >= 0, f"Call log_add_new_turn at least once " \
                                           f"(log_current_turn={self.log_current_turn})"
        timestamp = datetime.now().isoformat()
        action_obj = {
            "from": from_,
            "to": to,
            "timestamp": timestamp,
            "action": action
        }
        self.interactions["turns"][self.log_current_turn].append(copy.deepcopy(action_obj))
        module_logger.info(
            f"{self.game_name}: Logged {action['type']} action ({from_}->{to}).")
        if call:
            call_obj = {
                "timestamp": timestamp,
                "manipulated_prompt_obj": self._needs_copy(call[0]),
                "raw_response_obj": self._needs_copy(call[1])
            }
            self.requests.append(call_obj)
            module_logger.info(f"{self.game_name}: Logged a call with timestamp {timestamp}")

    @staticmethod
    def _needs_copy(call_obj):
        """Deepcopy objects that may otherwise lead to reference issues.
        Args:
            call_obj: The object to be deep-copied for safety.
        Returns:
            The deep-copy of the passed object, or the original object if it is safe to use.
        """
        if isinstance(call_obj, Dict) or isinstance(call_obj, List):
            return copy.deepcopy(call_obj)
        elif isinstance(call_obj, str):
            return call_obj[:]
        return call_obj

    def store_records(self, results_root: str, dialogue_pair_desc: str, game_record_dir: str):
        """Store benchmark records.
        Raise warnings if a mandatory element is empty or format is wrong.
        Args:
            results_root: The root path to the results directory.
            dialogue_pair_desc: A string combining the Player pair names to be used as directory name.
            game_record_dir: The game's record directory path.
        """
        if not self.interactions["players"]:
            module_logger.warning(f"Players metadada is missing!")
        else:
            for name in self.interactions["players"]:
                """The transcript builder relies on specific player identifiers."""
                try:
                    assert name == "GM" or name.startswith("Player ")
                except AssertionError:
                    module_logger.warning(f"Invalid player identifiers, html builder won't work.")
        if not self.interactions["turns"]:
            module_logger.warning(f"Interaction logs are missing!")
        if not self.requests:
            module_logger.warning(f"No calls logged!")
        self.store_results_file(self.interactions, "interactions.json",
                                dialogue_pair_desc,
                                sub_dir=game_record_dir,
                                results_dir=results_root)
        self.store_results_file(self.requests, "requests.json",
                                dialogue_pair_desc,
                                sub_dir=game_record_dir,
                                results_dir=results_root)
