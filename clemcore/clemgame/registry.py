import collections
import json
import os.path
from typing import List, Dict, Union
from types import SimpleNamespace
import logging
import nltk
import clemcore.utils.file_utils as file_utils

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")


class GameSpec(SimpleNamespace):
    """Base class for game specifications.
    Holds all necessary information to play game in clembench (see README for list of attributes)
    """

    def __init__(self, allow_underspecified: bool = False, **kwargs):
        super().__init__(**kwargs)
        # check for required fields
        if not allow_underspecified:
            if "game_name" not in self:
                raise KeyError(f"No game name specified in entry {kwargs}")
            if "game_path" not in self:
                raise KeyError(f"No game path specified in {kwargs}")
            if "players" not in self:
                raise KeyError(f"No players specified in {kwargs}")

    def __repr__(self):
        """Returns string representation of this GameSpec."""
        return f"GameSpec({str(self)})"

    def __str__(self):
        """Returns GameSpec instance attribute dict as string."""
        return str(self.__dict__)

    def __getitem__(self, item):
        """Access GameSpec instance attributes like dict items.
        Args:
            item: The string name of the instance attribute to get.
        Returns:
            The value of the GameSpec instance attribute, or if the instance does not have the attribute, the string
            passed as argument to this method.
        """
        return getattr(self, item)

    def __contains__(self, attribute):
        """Check GameSpec instance attributes like dict keys.
        Args:
            attribute: The string name of the instance attribute to check for.
        Returns:
            True if the GameSpec instance contains an attribute with the passed string name, False otherwise.
        """
        return hasattr(self, attribute)

    def to_string(self):
        return json.dumps(self.__dict__, separators=(",", ":"), indent=None)

    def to_pretty_string(self):
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_name(cls, game_name: str):
        """Create a GameSpec instance based on a game name.
        Args:
            game_name: The game name as string.
        """
        assert game_name is not None
        return cls(game_name=game_name, allow_underspecified=True)

    @classmethod
    def from_string(cls, game_spec: str):
        """Get a GameSpec instances for the passed string. This is rather intended to convert a game selector
        into a GameSpec. Hence, it must not be required to actually set a game_path or anything else.
        Takes both simple game names and (partially or fully specified) game specification data as JSON strings.
        Args:
            game_spec: A string to return a GameSpec instances for. Game name strings
                correspond to the 'game_name' key value of a game in the game registry. May also be partially or fully
                specified game specification data as JSON strings.
        Returns:
            A GameSpec instance
        """
        game_string = game_spec.replace("'", "\"")  # make this a proper json
        try:
            game_dict = json.loads(game_string)
            return GameSpec.from_dict(game_dict, allow_underspecified=True)
        except Exception as e:  # likely not a json
            return GameSpec.from_name(game_string, allow_underspecified=True)

    @classmethod
    def from_directory(cls, dir_path: str) -> List["GameSpec"]:
        file_path = os.path.join(dir_path, "clemgame.json")
        with open(file_path, encoding='utf-8') as f:
            game_spec = json.load(f)
        game_specs = []
        if isinstance(game_spec, dict):
            game_spec["game_path"] = dir_path
            game_specs.append(cls.from_dict(game_spec))
        elif isinstance(game_spec, list):
            for _spec in game_spec:
                _spec["game_path"] = dir_path
                game_specs.append(cls.from_dict(_spec))
        return game_specs

    @classmethod
    def from_dict(cls, spec: Dict, allow_underspecified: bool = False):
        """Initialize a GameSpec from a dictionary.
        Can be used to directly create a GameSpec from a game registry entry.
        Args:
            spec: A game-specifying dict.
        Returns:
            A GameSpec instance with the data specified by the passed dict.
        """
        return cls(allow_underspecified, **spec)

    def matches(self, spec: Dict):
        """Check if the game features match a given specification.
        Args:
            spec: A game-specifying dict.
        Returns:
            True if the game features match the passed specification, False otherwise.
        Raises:
            KeyError: The GameSpec instance does not contain an attribute corresponding to a key in the passed
                game-specifying dict.
        """
        for key, value in spec.items():
            if not self.__contains__(key):
                raise KeyError(f"The specified key '{key}' for selecting games is not set in the game registry "
                               f"for game '{self['game_name']}'")
            if type(self[key]) == str:
                if not self[key] == value:
                    return False
            elif type(self[key]) == list:
                if value not in self[key]:
                    return False
        return True

    def get_game_file(self):
        """Get the file path of the master.py of the game specified by this GameSpec instance.
        Main game file must be called master.py in game directory.
        Returns:
            The file path of the master.py of the game specified by this GameSpec instance as a string.
        """
        return os.path.join(self.game_path, "master.py")

    def game_file_exists(self):
        """Check if master.py can be located at the specified game_path.
        Returns:
            True if the master.py is located at the specified game_path, False otherwise.
        """
        return True if os.path.isfile(self.get_game_file()) else False

    def unify(self, other: "GameSpec") -> "GameSpec":
        """Unify two GameSpec instances.
        Args:
            other: The other GameSpec instance this instance is to be unified with.
        Returns:
            The GameSpec unification of this GameSpec instance and the passed GameSpec instance.
        Raises:
            ValueError: A ValueError exception is raised if the passed GameSpec instance does not unify with this
                GameSpec instance.
        """
        result = nltk.featstruct.unify(self.__dict__, other.__dict__)
        if result is None:
            raise ValueError(f"{self} does not unify with {other}")
        return GameSpec(**result)


class GameRegistry:

    def __init__(self, game_specs: List[GameSpec] = None):
        if game_specs is None:
            game_specs = []
        self._game_specs = game_specs

    def __len__(self):
        return len(self._game_specs)

    def __iter__(self):
        return iter(self._game_specs)

    def get_game_specs(self):
        return self._game_specs

    @classmethod
    def from_directories_and_cwd_files(cls):
        """
        Lookup game specs in the following locations:
        (1) Look for a (optional) game_registry.json in current working directory (relative to script execution)
        (2) Look for sub-directories of the current working director that contain a clemgame.json (depth=2)
        Model specs found in the (1) are listed before (2) allowing to 'favor' the ones in (1).

        Note:
        Game specs defined via (1) require the 'game_path' attribute
        while for game specs found via (2) this is set automatically.

        :return: model registry with model specs
        """
        game_registry = cls()
        try:
            game_registry_path = os.path.join(os.getcwd(), "game_registry.json")
            with open(game_registry_path, encoding='utf-8') as f:
                game_registry.register_from_list(json.load(f), game_registry_path)
        except Exception as e:
            logger.debug("File lookup failed with exception: %s", e)
        game_registry.register_from_directories(os.getcwd(), 0)
        return game_registry

    def register_from_list(self, game_specs: List[Dict], lookup_source: str = None) -> "GameRegistry":
        for game_spec_dict in game_specs:
            try:
                if lookup_source and ("lookup_source" not in game_spec_dict):
                    game_spec_dict["lookup_source"] = lookup_source
                game_spec = GameSpec.from_dict(game_spec_dict)
                self._game_specs.append(game_spec)
            except Exception as e:
                stdout_logger.warning("Game spec could not be loaded because: %s", e)
        return self

    def register_from_directories(self, current_directory: str, depth, max_depth=3):
        # deep first search to keep order of sorted directory names
        if depth > max_depth:
            return
        candidate_file_path = os.path.join(current_directory, "clemgame.json")
        try:
            if os.path.exists(candidate_file_path):
                game_specs = GameSpec.from_directory(current_directory)
                self._game_specs.extend(game_specs)
                return
            for current_file in sorted(os.listdir(current_directory)):
                file_path = os.path.join(current_directory, current_file)
                if not os.path.isdir(file_path):
                    continue
                if current_file.startswith("."):
                    continue
                if current_file in ["venv", "__pycache__", "docs",
                                    "in", "resources", "utils", "evaluation", "files"]:
                    continue
                self.register_from_directories(file_path, depth + 1, max_depth)
        except PermissionError:
            pass  # ignore
        except Exception as e:  # most likely a problem with the json file
            stdout_logger.warning("Lookup failed at '%s' with exception: %s", candidate_file_path, e)

    def get_game_specs_that_unify_with(self, game_selector: Union[str, Dict, GameSpec], verbose: bool = True) -> List[
        GameSpec]:
        """Select a list of GameSpecs from the game registry by unifying game spec dict or game name.
        Args:
            game_selector: String name of the game matching the 'game_name' value of the game registry entry to select, OR a
                GameSpec-like dict, OR a GameSpec object.
                A passed GameSpec-like dict can EITHER contain the 'benchmark' key with a list of benchmark versions value,
                in which case all games that have matching benchmark version strings in their 'benchmark' key values are
                selected, OR contain one or more other GameSpec keys, in which case all games that unify with the given key
                values are selected. If there is the 'benchmark' key, only benchmark versions are checked!
                For example: {'benchmark':['v2']} will select all games that have 'v2' in their 'benchmark' key value list.
                {'main_game': 'wordle'} will select all wordle variants, as their game registry entries have the 'main_game'
                key value 'wordle'.
        Returns:
            A list of GameSpec instances from the game registry corresponding to the passed game string, dict or GameSpec.
        Raises:
            ValueError: No game specification matching the passed game was found in the game registry.
        """
        # check if passed game is parseable JSON:
        game_is_dict = False
        try:
            game_selector = game_selector.replace("'", '"')
            game_selector = json.loads(game_selector)
            game_is_dict = True
        except Exception:
            logger.info(f"Passed game '{game_selector}' does not parse as JSON!")
            pass

        # convert passed JSON to GameSpec for unification:
        game_is_gamespec = False
        if game_is_dict:
            game_selector = GameSpec.from_dict(game_selector, allow_underspecified=True)
            game_is_gamespec = True
        elif type(game_selector) == GameSpec:
            game_is_gamespec = True

        selected_game_specs = []
        if game_selector == "all":
            selected_game_specs = self._game_specs
        elif game_is_gamespec:
            # iterate over game registry:
            for registered_game_spec in self._game_specs:

                if hasattr(game_selector, 'benchmark'):
                    # passed game spec specifies benchmark version
                    for benchmark_version in game_selector.benchmark:
                        if benchmark_version in registered_game_spec.benchmark:
                            if registered_game_spec.game_file_exists():
                                selected_game_specs.append(registered_game_spec)

                else:
                    # get unifying entries:
                    unifying_game_spec = None
                    try:
                        unifying_game_spec = game_selector.unify(registered_game_spec)
                        if unifying_game_spec.game_file_exists():
                            # print(f"Found unifying game registry entry: {unifying_game_spec}")
                            selected_game_specs.append(unifying_game_spec)
                    except ValueError:
                        continue
        else:
            # return first entry that matches game_name
            for registered_game_spec in self._game_specs:
                if registered_game_spec["game_name"] == game_selector:
                    selected_game_specs = [registered_game_spec]
                    break
        if selected_game_specs:
            if verbose:
                if game_is_gamespec:
                    stdout_logger.info(f"Found '{len(selected_game_specs)}' game matching the game_selector="
                                       f"{game_selector.to_string()}")
                else:
                    stdout_logger.info(f"Found '{len(selected_game_specs)}' game matching the game_selector="
                                       f"{json.dumps(game_selector, separators=(',', ':'), indent=None)}")
                if len(selected_game_specs) == 1:
                    stdout_logger.info(selected_game_specs[0].to_pretty_string())
                else:
                    for game_spec in selected_game_specs:
                        stdout_logger.info(game_spec.to_string())
            return selected_game_specs
        raise ValueError(f"No games found matching the game selector='{game_selector}'.")
        # extension to select subset of games
        # (postponed because it introduces more complexity
        # on things like how to specify specific episodes (which could, however be integrated into the game spec
        # and then selected through the custom game_spec for a specific run),
        # and thus can be easier done by looping over an
        # explicit list of games with a bash script (see clembench/scripts/run_benchmark.sh)

        # select relevant games from game registry
        # selected_games = []
        # properties = {}
        # is_single_game = True
        # if game_name.endswith(".json"):
        #     is_single_game = False
        #     with open(os.path.join(file_utils.project_root(), game_name)) as f:
        #         properties = json.load(f)
        #     # add default values
        #     if "lang" not in properties:
        #         properties["language"] = "en"
        #     if "image" not in properties:
        #         properties["image"] = "none"
        #     # examples:
        #     # {"benchmark" : "2.0"} # run all English textual games marked for benchmark version 2.0
        #     # {"benchmark" : "1.5", "lang": "ru"} # run all games of benchmark version 1.5 for which Russian versions exist
        #     # {"main_game": "matchit"} # to run all English textual matchit game versions
        #     # {"image": "single", "main_game": "matchit"} # to run all English multimodal matchit game versions
        #
        # if is_single_game:
        #     # return first entry that matches game_name
        #     for game in game_registry:
        #         if game["game_name"] == game_name:
        #             return game
        # else:
        #     for game in game_registry:
        #         if game.matches(properties):
        #             selected_games.append(game)
        #
        # if len(selected_games) == 0:
        #     raise ValueError(f"No games found matching the given specification '{game_name}'. "
        #                      "Make sure game name or attribute names and values match game_registry.json")
        # return selected_games
