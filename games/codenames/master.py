from typing import Dict, List, Tuple, Set
from string import Template
import random, string, re, math

from clemgame.clemgame import GameMaster, GameScorer, GameBenchmark, Player, DialogueGameMaster
from clemgame import get_logger
from clemgame.metrics import METRIC_ABORTED, METRIC_LOSE, METRIC_REQUEST_COUNT, \
    METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, BENCH_SCORE
from games.codenames.constants import *

logger = get_logger(__name__)


class ValidationError(Exception):
    def __init__(self, message="Response does not follow the rules and is hence invalid."):
        self.message = message
        super().__init__(self.message)

# TODO: reuse players for other codename variants, e.g. Duet?
# TODO: change prompts on reprompt, let them reflect the errors
# TODO: ValidationError for "answer was too long, did not follow the specified format"
# TODO: implement mock opponent player
# TODO: change main score calculation, revealing assassin on first turn is scored too high
        # also when game is aborted in the beginning, the low number of turns gives too high overall scores

class Guesser(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.guesses: List[str] = ['guess', 'word']
        self.prefix: str = "GUESS: "
        self.retries: int = 0

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        board = prompt.split('\n\n')[1].split(', ')
        number_of_allowed_guesses = int(re.search(r"up to ([0-9]+) words", prompt).group(1))
        self.guesses = random.sample(board, number_of_allowed_guesses)
        self.guesses = [word.strip('. ') for word in self.guesses]       # was not an issue but also does not hurt
        return self.recover_utterance()
    
    def validate_response(self, utterance: str, board: List[str], number_of_allowed_guesses: int):
        # utterance needs to start with GUESS
        if not utterance.startswith(self.prefix):
            raise ValidationError(f"Utterance {utterance} did not start with the correct prefix. ({self.prefix})")
        utterance = utterance.removeprefix(self.prefix)
        guesses = utterance.split(', ')
        guesses = [word.strip('. ').lower() for word in guesses]
        # must contain one valid guess, but can only contain $number guesses max
        if not (0 < len(guesses) <= number_of_allowed_guesses):
            raise ValidationError(f"Number of guesses made ({len(guesses)}) is not between 0 and {number_of_allowed_guesses}")
        # guesses must be words on the board that are not revealed yet
        for guess in guesses:
            if not guess in board:
                raise ValidationError(f"Guessed word {guess} does not exist on board.")
            
    def parse_response(self, utterance: str) -> str:
        utterance = utterance.removeprefix(self.prefix)
        self.guesses = utterance.split(', ')
        self.guesses = [word.strip('. ').lower() for word in self.guesses]
        return f"{', '.join(self.guesses)}"
            
    def recover_utterance(self) -> str:
        return f"{self.prefix}{', '.join(self.guesses)}"

class ClueGiver(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.prefix: str = "CLUE: "
        self.clue: str = 'clue'
        self.number_of_targets: int = 2
        self.targets: List[str] = ['target', 'word']
        self.retries: int = 0

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        match = re.search(r"team words are: (.*)\.", prompt)
        if match != None:
            # Player was actually prompted (otherwise it was reprompted and the team_words stay the same)
            team_words = match.group(1)
            team_words = team_words.split(', ')
            self.targets = random.sample(team_words, min(2, len(team_words)))
        self.number_of_targets = len(self.targets)
        self.clue = "".join(random.sample(list(string.ascii_lowercase), 6))
        return self.recover_utterance(with_targets=True)
    
    def validate_response(self, utterance: str, board: List[str]):
        # needs to start with correct prefix
        if not utterance.startswith(self.prefix):
            raise ValidationError(f"Utterance {utterance} did not start with the correct prefix. ({self.prefix})")
        utterance = utterance.removeprefix(self.prefix)
        parts = utterance.split(', ')
        if len(parts) < 3:
            raise ValidationError(f"Utterance {utterance} did not contain enough parts (only {len(parts)}) of required clue, number, and targets (at least three comma-separated)")
        clue = parts[0].lower()
        number_of_targets = parts[1]
        targets = parts[2:]
        targets = [target.strip(' .').lower() for target in targets]
        
        # Clue needs to be a single word
        if not clue.isalpha() or ' ' in clue:
            raise ValidationError(f"Clue {clue} is not a single word.")
        # Clue needs to contain a word that is not morphologically similar to any word on the board
        # TODO: morphological relatedness!
        if clue in board:
            raise ValidationError(f"Clue {clue} is one of the words on the board.")
        
        # Number needs to be a valid number and in between a valid range of 1 and all words on the board (alternatively only team words?)
        if not number_of_targets.isdigit() or not (0 < int(number_of_targets) <= len(board)):
            raise ValidationError(f"Number {number_of_targets} is not within range of 0 and {len(board)}")
        number_of_targets = int(number_of_targets)
        # needs to contain as many target words as the number given
        if len(targets) != number_of_targets:
            raise ValidationError(f"Number {number_of_targets} does not match number of targets {targets}")
        # target words need to be words from the board
        for target in targets:
            if not target in board:
                raise ValidationError(f"Targeted word {target} does not exist on board.")
            
    def parse_response(self, utterance: str) -> str:
        utterance = utterance.removeprefix(self.prefix)
        parts = utterance.split(', ')
        self.clue = parts[0].lower()
        self.number_of_targets = int(parts[1])
        self.targets = parts[2:]
        self.targets = [target.strip(' .').lower() for target in self.targets]
        return f"{self.clue}, {self.number_of_targets}"

    def recover_utterance(self, with_targets = False) -> str:
        targets = ""
        if with_targets:
            targets = f", {', '.join(self.targets)}"
        return f"{self.prefix}{self.clue}, {self.number_of_targets}{targets}"

class CodenamesBoard:
    def __init__(self, team_words, opponent_words, innocent_words, assassin_words):
        self.hidden = {TEAM: team_words, INNOCENT: innocent_words, OPPONENT: opponent_words, ASSASSIN: assassin_words}
        self.revealed = {TEAM: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []},
                         OPPONENT: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []}}

    def get_current_board(self) -> Dict:
        return {HIDDEN: self.hidden, 
                REVEALED: self.revealed}

    def get_all_hidden_words(self) -> List:
        hidden_words = []
        for assignment in self.hidden:
            hidden_words.extend(self.hidden[assignment])
        return hidden_words

    def get_hidden_words(self, with_assignment: str) -> List:
        return self.hidden[with_assignment]

    def reveal_word(self, word: str, by: str = TEAM):
        for assignment in self.hidden:
            if word in self.hidden[assignment]:
                self.revealed[by][assignment].append(word)
                self.hidden[assignment].remove(word)
                return assignment

        raise ValueError(f"Word {word} was not found amongst the hidden words on the board.")
    
    def should_continue_after_revealing(self, word: str, by: str = TEAM):
        return word in self.revealed[by][by]
    
    def has_team_won(self) -> bool:
        return len(self.hidden[TEAM]) == 0
    
    def has_team_won_through_assassin(self) -> bool:
        return len(self.revealed[OPPONENT][ASSASSIN]) >= 1

    def has_opponent_won(self) -> bool:
        return len(self.hidden[OPPONENT]) == 0

    def has_opponent_won_through_assassin(self) -> bool:
        return len(self.revealed[TEAM][ASSASSIN]) >= 1
    
class CodenamesGame(DialogueGameMaster):
    """This class implements a codenames game in which player A
    is giving a clue for a set of target words on a board, 
    which player B has to guess from the given clue.
    """

    def __init__(self, experiment: Dict, player_backends: List[str]):
        super().__init__(GAME_NAME, experiment, player_backends)
        # fetch experiment parameters
        self.experiment: str = experiment[NAME]
        self.experiment_type: str = experiment[TYPE]
        self.opponent_difficulty: bool = experiment[OPPONENT_DIFFICULTY]

        # save player interfaces
        self.model_a: str = player_backends[0]
        self.model_b: str = player_backends[1]
        
    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.board: CodenamesBoard = CodenamesBoard(game_instance[ASSIGNMENTS][TEAM], 
                                                    game_instance[ASSIGNMENTS][OPPONENT], 
                                                    game_instance[ASSIGNMENTS][INNOCENT],
                                                    game_instance[ASSIGNMENTS][ASSASSIN])
        
        self.aborted: bool = False
        self.lost: bool = False
        self.invalid_response: bool = False
        self.number_of_turns = 0
        self.request_count = 0
        self.parsed_request_count = 0
        self.violated_request_count = 0

        # Create the players
        self.cluegiver: Player = ClueGiver(self.player_backends[0])
        self.guesser: Player = Guesser(self.player_backends[1])

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
    
    def _on_before_game(self):
        pass
        # self.add_user_message(self.cluegiver, self._get_cluegiver_prompt(initial=True))

    def _on_before_turn(self, current_turn):
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
            self.log_key(GAME_END, Game_ends.ABORTED)
            self.aborted = True
            continue_game = False
        
        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        if self.board.has_team_won():
            self.log_key(GAME_END, Game_ends.TEAM_WON)
            self.lost = False
            continue_game = False
        elif self.board.has_opponent_won():
            self.log_key(GAME_END, Game_ends.OPPONENT_WON)
            self.lost = True
            continue_game = False
        elif self.board.has_team_won_through_assassin():
            self.log_key(GAME_END, Game_ends.TEAM_WON_THROUGH_ASSASSIN)
            self.lost = False
            continue_game = False
        elif self.board.has_opponent_won_through_assassin():
            self.log_key(GAME_END, Game_ends.OPPONENT_WON_THROUGH_ASSASSIN)
            self.lost = True
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
                print(error.message)
                self.log_to_self(Turn_logs.CLUEGIVER_INVALID_FORMAT, error.message)
                self.invalid_response = True
                self.violated_request_count += 1
        else:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words(), self.cluegiver.number_of_targets)
            except ValidationError as error:
                self.log_to_self(Turn_logs.GUESSER_INVALID_FORMAT, error.message)
                self.invalid_response = True
                self.violated_request_count += 1
        
        return not self.invalid_response
    
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        self.parsed_request_count += 1
        if player == self.cluegiver:
            utterance = player.parse_response(utterance)
            self.log_to_self(Turn_logs.TARGETS, self.cluegiver.targets)
            return utterance, False
        else:
            parsed_utterance = player.parse_response(utterance)
            for guess in player.guesses:
                assignment = self.board.reveal_word(guess)
                self.log_to_self(Turn_logs.TEAM_REVEALED, assignment)
                if self._was_target(guess):
                    self.log_to_self("target revealed", guess)
                if not self.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break
                
            return parsed_utterance, False
        
    def _on_before_reprompt(self, player: Player):
        logger.debug("Reprompting...")
        player.retries += 1
        self.request_count += 1
        self.add_user_message(player, "Your answer did not follow the requested format, please give a new answer that follows the format.")
    
    def _should_reprompt(self, player: Player):
        return False
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
        # METRIC_SUCCESS does not need to be logged as it is inferred from ABORTED and LOSE
        self.log_key(METRIC_REQUEST_COUNT, self.request_count)
        self.log_key(METRIC_REQUEST_COUNT_PARSED, self.parsed_request_count)
        self.log_key(METRIC_REQUEST_COUNT_VIOLATED, self.violated_request_count)            


class CodenamesScorer(GameScorer):
    def __init__(self):
        super().__init__(GAME_NAME)

    def score_turns(self):
        for turn_idx, turn in enumerate(self.episode_interactions["turns"]):
            # Metrics per turn:
            # target-precision, target-recall, target-f1
            # team-precision
            # invalid formats per player
            turn_score = {Turn_logs.TARGETS: [], Turn_logs.CLUEGIVER_INVALID_FORMAT: 0, Turn_logs.GUESSER_INVALID_FORMAT: 0, 
                          REVEALED: {TARGET: 0, TEAM: 0, OPPONENT: 0, INNOCENT: 0, ASSASSIN: 0, "total": 0}}
            for event in turn:
                action = event["action"]
                match action["type"]:
                    case Turn_logs.CLUEGIVER_INVALID_FORMAT:
                        turn_score[Turn_logs.CLUEGIVER_INVALID_FORMAT] += 1
                    case Turn_logs.GUESSER_INVALID_FORMAT:
                        turn_score[Turn_logs.GUESSER_INVALID_FORMAT] += 1
                    case Turn_logs.TARGETS:
                        turn_score[Turn_logs.TARGETS] = action["content"]
                    case Turn_logs.TEAM_REVEALED:
                        turn_score[REVEALED][action["content"]] += 1
                        turn_score[REVEALED]["total"] += 1

            sum_revealed_words = turn_score[REVEALED][TEAM] + turn_score[REVEALED][OPPONENT] + turn_score[REVEALED][INNOCENT] + turn_score[REVEALED][ASSASSIN]
            target_precision = 0
            target_recall = 0
            target_f1 = 0
            team_precision = 0
            if sum_revealed_words:
                target_precision = turn_score[REVEALED][TARGET] / sum_revealed_words
                team_precision = turn_score[REVEALED][TEAM] / sum_revealed_words
            
            if len(turn_score[Turn_logs.TARGETS]) > 0:
                target_recall = turn_score[REVEALED][TARGET] / len(turn_score[Turn_logs.TARGETS])
            if target_precision + target_recall > 0:
                target_f1 = 2 * target_precision * target_recall / (target_precision + target_recall)
            
            self.log_turn_score(turn_idx, "turn", turn_score)
            self.log_turn_score(turn_idx, Turn_logs.CLUEGIVER_INVALID_FORMAT, turn_score[Turn_logs.CLUEGIVER_INVALID_FORMAT])
            self.log_turn_score(turn_idx, Turn_logs.GUESSER_INVALID_FORMAT, turn_score[Turn_logs.GUESSER_INVALID_FORMAT])
            self.log_turn_score(turn_idx, "target precision", target_precision)
            self.log_turn_score(turn_idx, "target recall", target_recall)
            self.log_turn_score(turn_idx, "target f1", target_f1)
            self.log_turn_score(turn_idx, "team precision", team_precision)

    def score_game(self):
        # game-specific scores
        number_of_turns = self.episode_interactions[NUMBER_OF_TURNS]
        self.log_episode_score(NUMBER_OF_TURNS, number_of_turns)

        # plus all required game scores
        super().score_game()
    
    def score_game_end(self):
        super().score_game_end()
        # plus game specific things
        # won or lost through assassin or through revealing all words of one team

        game_end = self.episode_interactions[GAME_END]
        end_score = 5       # assume that game was aborted
        match game_end:
            case Game_ends.TEAM_WON:
                end_score = 1
            case Game_ends.TEAM_WON_THROUGH_ASSASSIN:
                end_score = 2
            case Game_ends.OPPONENT_WON:
                end_score = 3
            case Game_ends.TEAM_WON_THROUGH_ASSASSIN:
                end_score = 4
        self.log_episode_score(GAME_END, end_score)

        self.board_at_end = self.episode_interactions[BOARD_STATUS]
        # self.log_episode_score(BOARD_STATUS, board_at_end)

        self.log_episode_score("team words revealed/all team words", len(self.board_at_end[REVEALED][TEAM][TEAM]) / 9)
        self.log_episode_score("other words revealed/all words revealed", 1 - (len(self.board_at_end[REVEALED][TEAM][ASSASSIN]) + len(self.board_at_end[REVEALED][TEAM][OPPONENT]) + len(self.board_at_end[REVEALED][TEAM][INNOCENT])) / 16)
       
    def log_main_score(self):
        # all logged scores are available via self.scores["episode scores"][score_name]
        # or self.scores["turn scores"][turn_idx][score_name]

        # Main Score: log19(1 + (#team - #opponent - assassin_true*#hidden_opponent + 9 offset) / #turns) * 100
        main_score = len(self.board_at_end[REVEALED][TEAM][TEAM]) - len(self.board_at_end[REVEALED][TEAM][OPPONENT]) - len(self.board_at_end[REVEALED][TEAM][ASSASSIN]) * len(self.board_at_end[HIDDEN][OPPONENT]) + 9 # offset
        main_score = (main_score / self.scores["episode scores"][NUMBER_OF_TURNS]) + 1
        main_score = math.log(main_score, 19) * 100
        assert 0 <= main_score <= 100, f"Main Score of {main_score} is not between 0 and 100"
        self.log_episode_score(BENCH_SCORE, main_score)


class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)
        random.seed(SEED)

    def get_description(self) -> str:
        return "Codenames game between a cluegiver and a guesser"

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return CodenamesGame(experiment, player_backends)

    def create_game_scorer(self) -> GameScorer:
        return CodenamesScorer()
