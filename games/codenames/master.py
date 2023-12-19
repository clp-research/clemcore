from typing import Dict, List, Tuple, Set
from string import Template
import random, string, re

from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster
from clemgame import get_logger

logger = get_logger(__name__)

GAME_NAME = "codenames"
# TODO: all words on board in CAPS?

class Guesser(Player):
    def __init__(self):
        super().__init__("programmatic")
        self.guesses = ['apple', 'banana']
        self.prefix = "GUESS: "

    def _custom_response(self, history, turn):
        # raise NotImplementedError("This should not be called, but the remote APIs.")
        # TODO: randomly guess number of words from the board
        prompt = history[-1]["content"]
        board = prompt.split('\n\n')[1].strip('{}').split(', ')
        board = [word.strip("'") for word in board]
        number_of_allowed_guesses = int(re.search(r"Please select up to ([0-9]+) words", prompt).group(1))
        self.guesses = random.sample(board, number_of_allowed_guesses)
        return self.recover_utterance()
    
    def validate_response(self, utterance, board, number_of_allowed_guesses):
        # all lines need to start with GUESS
        lines = utterance.split("\n")
        for line in lines:
            if not line.startswith(self.prefix):
                print(f"Utterance {utterance} did not start with the correct prefix. ({self.prefix})")
                return False
        guesses = [line[len(self.prefix):] for line in lines]
        # must contain one valid guess, but can only contain $number guesses max
        if not (0 < len(guesses) <= number_of_allowed_guesses):
            print(f"Number of guesses made ({len(guesses)}) is not between 0 and {number_of_allowed_guesses}")
            return False 
        # guesses must be words on the board that are not uncovered yet
        for guess in guesses:
            if not guess in board:
                print(f"Guessed word {guess} does not exist on board.")
                return False
            
        return True

    def parse_response(self, utterance):
        lines = utterance.split("\n")
        self.guesses = [line[len(self.prefix):] for line in lines]
        return self.guesses
            
    def recover_utterance(self):
        with_prefix = [self.prefix + guess for guess in self.guesses]
        return "\n".join(with_prefix)


class ClueGiver(Player):
    def __init__(self):
        super().__init__("programmatic")
        self.prefix: str = "CLUE: "
        self.clue: str = 'fruit'
        self.number_of_targets: int = 2
        self.targets: List[str] = ['banana', 'apple']

    def _custom_response(self, history, turn):
        #raise NotImplementedError("This should not be called, but the remote APIs.")
        # TODO: random clue giving? how?
        prompt = history[-1]["content"]
        team_words = re.search(r"Your team words are: \{(.*)\}", prompt).group(1)
        team_words = team_words.split(', ')
        team_words = [word.strip("'") for word in team_words]
        self.targets = random.sample(team_words, min(2, len(team_words)))
        self.number_of_targets = len(self.targets)
        self.clue = "".join(random.sample(list(string.ascii_lowercase), 6))
        return self.recover_utterance()
    
    def validate_response(self, utterance, board):
        # needs to start with correct prefix
        if not utterance.startswith(self.prefix):
            print(f"Utterance {utterance} did not start with the correct prefix. ({self.prefix})")
            return False
        utterance = utterance[len(self.prefix):]
        parts = utterance.split(', ')
        clue = parts[0]
        number_of_targets = parts[1]
        targets = parts[2:]
        
        # Clue needs to be a single word
        if not clue.isalpha() or ' ' in clue:
            print(f"Clue {clue} is not a single word.")
            return False
        # Clue needs to contain a word that is not morphologically similar to any word on the board
        # TODO: morphological relatedness!
        if clue in board:
            print(f"Clue {clue} is one of the words on the board.")
            return False
        
        # Number needs to be a valid number and in between a valid range of 1 and all words on the board (alternatively only team words?)
        if not number_of_targets.isdigit() and not (0 < int(number_of_targets) <= len(board)):
            print(f"Number {number_of_targets} is not within range of 0 and {len(board)}")
            return False
        number_of_targets = int(number_of_targets)
        # needs to contain as many target words as the number given
        if len(targets) != number_of_targets:
            print(f"Number {number_of_targets} does not match number of targets {targets}")
            return False
        # target words need to be words from the board
        for target in targets:
            if not target in board:
                print(f"Targeted word {target} does not exist on board.")
                return False
            
        return True

    def parse_response(self, utterance):
        utterance = utterance[len(self.prefix):]
        parts = utterance.split(', ')
        self.clue = parts[0]
        self.number_of_targets = int(parts[1])
        self.targets = parts[2:]
        return f"{self.clue}, {self.number_of_targets}"

    def recover_utterance(self):
        return f"{self.prefix}{self.clue}, {self.number_of_targets}, {', '.join(self.targets)}"


class CodenamesGame(DialogueGameMaster):
    """This class implements a codenames game in which player A
    is giving a clue for a set of target words on a board, 
    which player B has to guess from the given clue.
    """

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        # fetch experiment parameters
        self.experiment: str = experiment["name"]
        self.experiment_type: str = experiment["type"]
        
        self.aborted: bool = False
        self.lose: bool = False

        # save player interfaces
        self.model_a = player_backends[0]
        self.model_b = player_backends[1]
        
    def _on_setup(self, **game_instance):
        """
        Template method: must be implemented

        Use add_player() here to add the players.

        :param kwargs: of the game instance
        """
        self.game_instance = game_instance  # fetch game parameters here
        self.board: Set[str] = set(game_instance["board"])
        self.team_words: Set[str] = set(game_instance["assignments"]["team"])
        self.opponent_words: Set[str] = set(game_instance["assignments"]["opponent"])
        self.innocent_words: Set[str] = set(game_instance["assignments"]["innocent"])
        self.assassin_words: Set[str] = set(game_instance["assignments"]["assassin"])
        self.uncovered_words: Set[str] = set()

        # Create the players
        self.cluegiver = ClueGiver()#game_instance["target_name"])
        self.guesser = Guesser()#self.player_backends[0])

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.cluegiver)
        self.add_player(self.guesser)

    def _on_before_game(self):
        # Do something before the game start e.g. add the initial prompts to the message list for the players
        # Load the prepared initial prompt
        prompt_cluegiver = self.load_template("resources/initial_prompts/prompt_cluegiver")
        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words=self.team_words, opponent_words=self.opponent_words, innocent_words=self.innocent_words, assassin_words=self.assassin_words)
        self.add_user_message(self.cluegiver, instance_prompt_cluegiver)

    def _does_game_proceed(self):
        # Determine if the game should proceed. This is also called once initially.
        
        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        if len(self.uncovered_words.intersection(self.team_words)) == len(self.team_words):
            # all team words have been revealed
            return False
        elif len(self.uncovered_words.intersection(self.opponent_words)) == len(self.opponent_words):
            # all opponents words have been revealed
            return False
        elif len(self.uncovered_words.intersection(self.assassin_words)) >= 1:
            return False
        #elif self.current_turn >= 9:
            # that would only be necessary be for Codenames Duet
        #    return False
        return True

    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        """
        Hook

        Decide if a response utterance should be modified. If not simply return the utterance.

        When a modified utterance and a true value is returned, then a 'parse' event is logged.

        :param player: that produced the response
        :param utterance: to be potentially modified
        :return: the (modified) utterance and if to log the parse action (default: True)
        """
        if player == self.cluegiver:
            return player.parse_response(utterance), True
        else:
            guesses = player.parse_response(utterance)
            self.uncovered_words.update(guesses)
            return guesses, True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        """
        Hook

        Decide if an utterance should be added.

        This is also the place to check for game end conditions.

        :param player: for which the response is added as "assistant" to the history
        :param utterance: to be added
        :return: True, if the utterance is fine; False, if the response should not be added to the history
        """
        if player == self.cluegiver:
            return player.validate_response(utterance, self.board - self.uncovered_words)
        else:
            return player.validate_response(utterance, self.board - self.uncovered_words, self.cluegiver.number_of_targets)
    
    def _on_before_turn(self, current_turn):
        # add new prompts?
        prompt_cluegiver = self.load_template("resources/initial_prompts/prompt_cluegiver")
        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words=self.team_words - self.uncovered_words, 
                                                                          opponent_words=self.opponent_words - self.uncovered_words, 
                                                                          innocent_words=self.innocent_words - self.uncovered_words, 
                                                                          assassin_words=self.assassin_words - self.uncovered_words)
        self.add_user_message(self.cluegiver, instance_prompt_cluegiver)

    def _on_after_turn(self, turn_idx: int):
        pass
    
    def _after_add_player_response(self, player: Player, utterance: str):
        """
        Hook

        Add the utterance to other player's history, if necessary.
        To do this use the method add_user_message(other_player,utterance).

        :param player: that produced the response (or has been modified by the GM)
        :param utterance: that has been added
        """
        if player == self.cluegiver:
            # put message into next prompt?
            prompt_guesser = self.load_template("resources/initial_prompts/prompt_guesser")
            instance_prompt_guesser = Template(prompt_guesser).substitute(board=self.board - self.uncovered_words, clue=self.cluegiver.clue, number=self.cluegiver.number_of_targets)
            self.add_user_message(self.guesser, instance_prompt_guesser)
        else:
            # TODO: use intermittent prompt, that also includes guess from guessing player and then prompts for new clue, with updated lists of board words
            self.add_user_message(self.cluegiver, utterance)


    def compute_scores(self) -> None:
        # metrics here for episode scores
        # where to put metrics for turns?
        score = 0
        if self.success:
            score = 1
        self.log_episode_score('Accuracy', score)


class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)

    def get_description(self):
        return "Codenames game between a cluegiver and a guesser"

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return CodenamesGame(experiment, player_backends)
