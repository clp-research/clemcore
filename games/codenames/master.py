from typing import Dict, List, Tuple, Set
from string import Template
import random

from clemgame.clemgame import GameMaster, GameScorer, GameBenchmark, Player, DialogueGameMaster
from clemgame import get_logger
from clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE
from games.codenames.constants import *
from games.codenames.validation_errors import *
from .players import ClueGiver, Guesser
from .board import CodenamesBoard
from .scorer import CodenamesScorer

logger = get_logger(__name__)

class CodenamesGame(DialogueGameMaster):
    """This class implements a codenames game in which player A
    is giving a clue for a set of target words on a board, 
    which player B has to guess from the given clue.
    """

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        # fetch experiment parameters
        self.experiment = experiment
        self.opponent_difficulty: bool = experiment[OPPONENT_DIFFICULTY]

        # save player interfaces
        self.model_a: str = player_backends[0]
        self.model_b: str = player_backends[1]
        
    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.board: CodenamesBoard = CodenamesBoard(game_instance[ASSIGNMENTS][TEAM], 
                                                    game_instance[ASSIGNMENTS][OPPONENT], 
                                                    game_instance[ASSIGNMENTS][INNOCENT],
                                                    game_instance[ASSIGNMENTS][ASSASSIN],
                                                    self.experiment["flags"])
        
        self.aborted: bool = False
        self.lost: bool = False
        self.assassin_won: bool = False
        self.invalid_response: bool = False
        self.number_of_turns = 0
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0

        # Create the players
        self.cluegiver: Player = ClueGiver(self.model_a, self.experiment["flags"])
        self.guesser: Player = Guesser(self.model_b, self.experiment["flags"])

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.cluegiver)
        self.add_player(self.guesser)
    
    def _was_target(self, word: str):
        return word in self.cluegiver.targets

    def _get_cluegiver_prompt(self, initial = False) -> str:
        folder = "initial_prompts" if initial else "intermittent_prompts"
        path = f"resources/{folder}/prompt_cluegiver"
        prompt_cluegiver = self.load_template(path)

        team_words = ", ".join(self.board.get_hidden_words(TEAM))
        opponent_words = ", ".join(self.board.get_hidden_words(OPPONENT))
        innocent_words = ", ".join(self.board.get_hidden_words(INNOCENT))
        assassin_words = ", ".join(self.board.get_hidden_words(ASSASSIN))

        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words= team_words, 
                                                                          opponent_words=opponent_words, 
                                                                          innocent_words=innocent_words, 
                                                                          assassin_words=assassin_words)
        return instance_prompt_cluegiver
    
    def _get_guesser_prompt(self, initial = False) -> str:
        folder = "initial_prompts" if initial else "intermittent_prompts"
        path = f"resources/{folder}/prompt_guesser"
        prompt_guesser = self.load_template(path)
        
        board = ", ".join(self.board.get_all_hidden_words())
        instance_prompt_guesser = Template(prompt_guesser).substitute(board=board, 
                                                                      clue=self.cluegiver.clue, 
                                                                      number=self.cluegiver.number_of_targets)
        return instance_prompt_guesser
    
    def _opponent_turn(self):
        # reveal as many opponent cards as the opponent difficulty
        opponent_words = random.sample(self.board.get_hidden_words(OPPONENT), self.opponent_difficulty)
        for word in opponent_words:
            assignment = self.board.reveal_word(word, OPPONENT)
            self.log_to_self(Turn_logs.OPPONENT_REVEALED, {"assignment": assignment, "word": word})
    
    def _on_before_game(self):
        pass
        # self.add_user_message(self.cluegiver, self._get_cluegiver_prompt(initial=True))

    def _on_before_turn(self, current_turn):
        # let mock opponent reveal their cards
        if self.number_of_turns > 0:
            self._opponent_turn()

        # add new cluegiver prompt
        self.cluegiver.retries = 0
        self.guesser.retries = 0
        self.number_of_turns += 1
        initial = True if self.number_of_turns == 1 else False
        self.add_user_message(self.cluegiver, self._get_cluegiver_prompt(initial))

    def _does_game_proceed(self) -> bool:
        # Determine if the game should proceed. This is also called once initially.
        continue_game = True
        if self.invalid_response:
            self.aborted = True
            continue_game = False
        
        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        if self.board.has_team_won():
            self.lost = False
            self.assassin_won = False
            continue_game = False
        elif self.board.has_opponent_won():
            self.lost = True
            self.assassin_won = False
            continue_game = False
        elif self.board.has_team_won_through_assassin():
            self.lost = False
            self.assassin_won = True
            continue_game = False
        elif self.board.has_opponent_won_through_assassin():
            self.lost = True
            self.assassin_won = True
            continue_game = False

        if not continue_game:
            self._log_game_end()
            return False
        return True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        self.request_count += 1
        self.invalid_response = False
        if player == self.cluegiver:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words())
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                self.last_error_message = error.message
                # add response to history nonetheless... but without parsing it
                self.add_assistant_message(player, utterance)
        else:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words(), self.cluegiver.number_of_targets)
            except ValidationError as error:
                self.log_to_self(Turn_logs.VALIDATION_ERROR, error.get_dict())
                self.invalid_response = True
                self.violated_request_count += 1
                self.last_error_message = error.message
                # add response to history nonetheless... but without parsing it
                self.add_assistant_message(player, utterance)
        
        return not self.invalid_response
    
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        self.parsed_request_count += 1
        if player == self.cluegiver:
            utterance = player.parse_response(utterance)
            self.log_to_self(Turn_logs.CLUE, player.clue)
            # TODO: log target assignments here!
            self.log_to_self(Turn_logs.TARGETS, player.targets)
            return utterance, False
        else:
            parsed_utterance = player.parse_response(utterance)
            self.log_to_self(Turn_logs.GUESSES, player.guesses)
            for guess in player.guesses:
                assignment = self.board.reveal_word(guess)
                if not assignment:
                    continue
                self.log_to_self(Turn_logs.TEAM_REVEALED, {"assignment": assignment, "word": guess})
                if self._was_target(guess):
                    self.log_to_self(Turn_logs.TARGET_REVEALED, {"assignment": assignment, "word": guess})
                if not self.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break
                
            return parsed_utterance, False
        
    def _on_before_reprompt(self, player: Player):
        logger.debug("Reprompting...")
        player.retries += 1
        player.flags_engaged["REPROMPT ON ERROR"] += 1
        self.add_user_message(player, f"Your answer did not follow the requested format: {self.last_error_message}")
    
    def _should_reprompt(self, player: Player):
        # return False
        if player.flags["REPROMPT ON ERROR"]:
            if player.retries < MAX_RETRIES:
                return self.invalid_response
        return False
    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.cluegiver:
            initial = True if self.number_of_turns == 1 else False
            self.add_user_message(self.guesser, self._get_guesser_prompt(initial))

        else:
            self.add_user_message(self.cluegiver, utterance)

    def _log_game_end(self):
        # log everything that is needed for score calculation and game evaluation
        self.log_key(BOARD_STATUS, self.board.get_current_board())
        self.log_key(NUMBER_OF_TURNS, self.number_of_turns)
        self.log_key(METRIC_ABORTED, self.aborted)
        self.log_key(METRIC_LOSE, self.lost)
        self.log_key(GAME_ENDED_THROUGH_ASSASSIN, self.assassin_won)
        # METRIC_SUCCESS does not need to be logged as it is inferred from ABORTED and LOSE
        self.log_key(METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)
        self.log_key("Cluegiver engaged flags", self.cluegiver.flags_engaged)
        self.log_key("Guesser engaged flags", self.guesser.flags_engaged)

class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)
        random.seed(SEED)

    def get_description(self) -> str:
        return "Codenames game between a cluegiver and a guesser"

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return CodenamesGame(experiment, player_backends)

    def create_game_scorer(self, experiment_config, game_instance) -> GameScorer:
        return CodenamesScorer(experiment_config, game_instance)
