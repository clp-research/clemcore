import abc
import csv
import importlib.resources
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Union

module_logger = logging.getLogger(__name__)


def store_file(data, file_name: str, dir_path: Union[str, Path], sub_dir: str = None, do_overwrite: bool = True) -> str:
    """Store a file.
    Base function to handle relative clembench directory paths.
    Args:
        data: Content to store in the file.
        file_name: Name of the file to store.
        dir_path: Path to the directory to store the file to.
        sub_dir: Optional subdirectories to store the file in.
        do_overwrite: Determines if existing file should be overwritten. Default: True
    Returns:
        The path to the stored file.
    """
    if sub_dir:
        dir_path = os.path.join(dir_path, sub_dir)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fp = os.path.join(dir_path, file_name)
    if not do_overwrite:
        if os.path.exists(fp):
            raise FileExistsError(fp)

    with open(fp, "w", encoding='utf-8') as f:
        if file_name.endswith(".json"):
            json.dump(data, f, ensure_ascii=False)
        else:
            f.write(data)
    return fp


def load_json(file_path: str) -> Dict:
    """
    Load a JSON file.
    Args:
        file_path: The path to the JSON file.
    Returns:
        The JSON file content as dict.
    """
    file_ending = ".json"
    if not file_path.endswith(file_ending):
        file_path = file_path + file_ending
    with open(file_path, encoding='utf8') as f:
        data = json.load(f)
    return data


def load_packaged_file(file_path: str):
    """
    Loads a file within the clemcore package.
    :param file_path: The path to the file within the clemcore package (without the 'clemcore/' prefix).
    :return: The file content
    """
    with importlib.resources.files("clemcore").joinpath(file_path).open("r") as f:
        return f.read()


def store_results_file(game_name, data, file_name: str, dialogue_pair: str,
                       sub_dir: str = None, results_dir: str = None):
    """Store a results file in your game results' directory. The top-level directory is 'results'.
    Args:
        game_name: the game name to store the results file for
        data: The data to store in the file.
        file_name: The name of the file. Can have subdirectories e.g. "sub/my_file".
        dialogue_pair: The name of the model pair directory. The directory name is retrieved from the results
            directory file structure by classes/methods that use this method.
        sub_dir: The subdirectory to store the results file in. Automatically created when given; otherwise an
            error will be thrown.
        results_dir: An (alternative) results directory structure given as a relative or absolute path.
    """
    if results_dir is None:
        results_dir = "results"  # default to a results directory in current terminal workspace
    game_results_path = os.path.join(results_dir, dialogue_pair, game_name)
    fp = store_file(data, file_name, game_results_path, sub_dir)
    module_logger.info(f"Results file stored to {fp}")


class GameResourceLocator(abc.ABC):
    """
    Provides access to game specific resources and results (based on game path and results directory)

    Note: You should access resource only via the game resource locator! The locator knows how to refer to them.
    For example use: `gm.load_json("my_file")` which is located directly at your game directory `game/my_file.json`.
    You can access subdirectories by giving `gm.load_json("sub/my_file")` in `game/sub/my_file.json`.

    Makes a distinction between game files (which live in the game path specified in `self.game_path`)
    and the results files, which live in the results directory (`clembench/results` if not set otherwise)
    under `results/dialogue_pair/self.game_name/`
    """

    def __init__(self, name: str = None, path: str = None):
        """

        Args:
            name: name of the game (optional, because not needed for GameInstanceGenerator)
            path: path to the game (optional, because not needed for GameScorer)
        """
        self.game_name = name  # for building results structure
        self.game_path = path  # for accessing game resources

    def __load_game_file(self, file_path: Union[str, Path], file_ending: str = None) -> str:
        """Load a file from a clemgame. Assumes the file to be an utf8-encoded (text) file.
        Args:
            file_path: Relative path to the file starting from game path.
            file_ending: The file type suffix of the file.
        Returns:
            The file content as returned by open->read().
        """
        if file_ending and not file_path.endswith(file_ending):
            file_path = file_path + file_ending
        fp = os.path.join(self.game_path, file_path)
        with open(fp, encoding='utf8') as f:
            data = f.read()
        return data

    def load_instances(self, instances_name):
        """Construct instances path and return json object of the instance file.
        Args:
            game_path: Path to the game directory.
            instances_name: Name of the instances JSON file.
        Returns:
            A dict containing the contents of the given instances file.
        """
        if instances_name is None:
            instances_name = "instances"
        return self.load_json(f"in/{instances_name}")

    def load_template(self, file_path: Union[str, Path]) -> str:
        """Load a .template file from the game directory.
        Args:
            file_path: Relative path to the file starting from game path.
        Returns:
            The template file content as string.
        """
        return self.__load_game_file(file_path, file_ending=".template")

    def load_json(self, file_path: Union[str, Path]) -> Dict:
        """Load a .json file from your game directory.
        Args:
            file_path: Relative path to the file starting from game path.
        Returns:
            The JSON file content as dict.
        """
        data = self.__load_game_file(file_path, file_ending=".json")
        return json.loads(data)

    def load_results_json(self, file_name: str, results_dir: str, dialogue_pair: str) -> Dict:
        """Load a .json file from the results directory for this game.
        Args:
            file_name: The name of the JSON file. Can have subdirectories e.g. "sub/my_file".
            results_dir: The string path to the results directory.
            dialogue_pair: The name of the model pair directory. The directory name is retrieved from the results
                directory file structure by classes/methods that use this method.
        Returns:
            The JSON file content as dict.
        """
        """
        todo: this is a bit delicate: we need to duplicate the code from self.load_json, 
        b.c. either we provide for all load_json cases the path from outside e.g. self.load_json(self.game_path)
        or we move this method outside GameResourceLocator, because the results is not necessarily a game file;
        considering this, we actually could also make the results live in the games (but this not yet done).
        Such a refactoring would indeed need more time and breaks things. 
        """
        file_ending = ".json"
        if not file_name.endswith(file_ending):
            file_name = file_name + file_ending
        fp = os.path.join(results_dir, dialogue_pair, self.game_name, file_name)
        with open(fp, encoding='utf8') as f:
            data = f.read()
        data = json.loads(data)
        return data

    def load_csv(self, file_name: str) -> List:
        """Load a .csv file from your game directory.
        Args:
            file_name: The name of the CSV file. Can have subdirectories e.g. "sub/my_file".
        Returns:
            A list version of the CSV file content.
        """
        # iso8859_2 was required for opening nytcrosswords.csv for clues in wordle
        rows = []
        fp = os.path.join(self.game_path, file_name)
        with open(fp, encoding='iso8859_2') as csv_file:
            data = csv.reader(csv_file, delimiter=',')
            for row in data:
                rows.append(row)
        return rows

    def load_file(self, file_name: str, file_ending: str = None) -> str:
        """Load an arbitrary file from your game directory.
        Args:
            file_name: The name of the file. Can have subdirectories e.g. "sub/my_file".
            file_ending: The file type suffix of the file. Optional: Can be part of file_name.
        Returns:
            The file content as string.
        """
        return self.__load_game_file(file_name, file_ending=file_ending)

    def store_file(self, data, file_name: str, sub_dir: str = None):
        """Store a file in your game directory.
        Args:
            data: The data to store in the file.
            file_name: The name of the file. Can have subdirectories e.g. "sub/my_file".
            sub_dir: The subdirectory to store the file in. Automatically created when given; otherwise an error will
                be thrown.
        """
        fp = store_file(data, file_name, self.game_path, sub_dir=sub_dir)
        module_logger.info("Game file stored to %s", fp)
