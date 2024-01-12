from typing import Dict, List, Tuple, Set
from string import Template
import random, string, re, math

from clemgame.clemgame import GameMaster, GameBenchmark, Player, DialogueGameMaster
from clemgame import get_logger

logger = get_logger(__name__)

GAME_NAME = "codenames"
SEED = 418
MAX_RETRIES = 2

class ValidationError(Exception):
    def __init__(self, message="Response does not follow the rules and is hence invalid."):
        self.message = message
        super().__init__(self.message)

# TODO: all words on board in CAPS?
# TODO: reuse players for other codename variants, e.g. Duet?
# TODO: intermittent prompts e.g. "The next clue is:<CLUE>. The list of words now is: <List of words>"
# TODO: change prompts on reprompt, let them reflect the errors
# TODO: ValidationError for "answer was too long, did not follow the specified format"
# TODO: implement mock opponent player

class Guesser(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.guesses: List[str] = ['apple', 'banana']
        self.prefix: str = "GUESS: "
        self.retries: int = 0

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        board = prompt.split('\n\n')[1].split(', ')
        number_of_allowed_guesses = int(re.search(r"Please select up to ([0-9]+) words", prompt).group(1))
        self.guesses = random.sample(board, number_of_allowed_guesses)
        return self.recover_utterance()
    
    def validate_response(self, utterance: str, board: List[str], number_of_allowed_guesses: int):
        # all lines need to start with GUESS
        lines = utterance.split("\n")
        for line in lines:
            if not line.startswith(self.prefix):
                raise ValidationError(f"Utterance {utterance} did not start with the correct prefix. ({self.prefix})")
        guesses = [line[len(self.prefix):] for line in lines]
        # must contain one valid guess, but can only contain $number guesses max
        if not (0 < len(guesses) <= number_of_allowed_guesses):
            raise ValidationError(f"Number of guesses made ({len(guesses)}) is not between 0 and {number_of_allowed_guesses}")
        # guesses must be words on the board that are not revealed yet
        for guess in guesses:
            if not guess in board:
                raise ValidationError(f"Guessed word {guess} does not exist on board.")
            
    def parse_response(self, utterance: str) -> str:
        lines = utterance.split("\n")
        self.guesses = [line[len(self.prefix):] for line in lines]
        return self.guesses
            
    def recover_utterance(self) -> str:
        with_prefix = [self.prefix + guess for guess in self.guesses]
        return "\n".join(with_prefix)


class ClueGiver(Player):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.prefix: str = "CLUE: "
        self.clue: str = 'fruit'
        self.number_of_targets: int = 2
        self.targets: List[str] = ['banana', 'apple']
        self.retries: int = 0

    def _custom_response(self, history, turn) -> str:
        prompt = history[-1]["content"]
        match = re.search(r"Your team words are: (.*)", prompt)
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
        utterance = utterance[len(self.prefix):]
        parts = utterance.split(', ')
        if len(parts) < 3:
            raise ValidationError(f"Utterance {utterance} did not contain enough parts (only {len(parts)}) of required clue, number, and targets (at least three comma-separated)")
        clue = parts[0]
        number_of_targets = parts[1]
        targets = parts[2:]
        
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
        utterance = utterance[len(self.prefix):]
        parts = utterance.split(', ')
        self.clue = parts[0]
        self.number_of_targets = int(parts[1])
        self.targets = parts[2:]
        return f"{self.clue}, {self.number_of_targets}"

    def recover_utterance(self, with_targets = False) -> str:
        targets = ""
        if with_targets:
            targets = f", {', '.join(self.targets)}"
        return f"{self.prefix}{self.clue}, {self.number_of_targets}{targets}"

class CodenamesBoard:
    def __init__(self, team_words, opponent_words, innocent_words, assassin_words):
        self.hidden = {"team": team_words, "innocent": innocent_words, "opponent": opponent_words, "assassin": assassin_words}
        self.revealed = {"team": {"team": [], "innocent": [], "opponent": [], "assassin": []},
                         "opponent": {"team": [], "innocent": [], "opponent": [], "assassin": []}}

    def get_current_board(self) -> Dict:
        return {"hidden": self.hidden, 
                "revealed": self.revealed}

    def get_all_hidden_words(self) -> List:
        hidden_words = []
        for assignment in self.hidden:
            hidden_words.extend(self.hidden[assignment])
        return hidden_words

    def get_hidden_words(self, with_assignment: str) -> List:
        return self.hidden[with_assignment]

    def reveal_word(self, word: str, by: str = "team"):
        for assignment in self.hidden:
            if word in self.hidden[assignment]:
                self.revealed[by][assignment].append(word)
                self.hidden[assignment].remove(word)
                return assignment

        raise ValueError(f"Word {word} was not found amongst the hidden words on the board.")
    
    def should_continue_after_revealing(self, word: str, by: str = "team"):
        return word in self.revealed[by][by]
    
    def has_team_won(self) -> bool:
        return len(self.hidden["team"]) == 0

    def has_opponent_won(self) -> bool:
        return len(self.hidden["opponent"]) == 0

    def has_assassin_won(self) -> bool:
        return len(self.revealed["team"]["assassin"]) >= 1 or len(self.revealed["opponent"]["assassin"]) >= 1


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
        
        #self.aborted: bool = False
        #self.lose: bool = False
        self.invalid_response: bool = False

        # save player interfaces
        self.model_a: str = player_backends[0]
        self.model_b: str = player_backends[1]
        
    def _on_setup(self, **game_instance):
        self.game_instance = game_instance  # fetch game parameters here
        self.board: CodenamesBoard = CodenamesBoard(game_instance["assignments"]["team"], 
                                                    game_instance["assignments"]["opponent"], 
                                                    game_instance["assignments"]["innocent"],
                                                    game_instance["assignments"]["assassin"])

        # Create the players
        self.cluegiver: Player = ClueGiver(self.player_backends[0])
        self.guesser: Player = Guesser(self.player_backends[1])

        # Add the players: these will be logged to the records interactions.json
        # Note: During game play the players will be called in the order added here
        self.add_player(self.cluegiver)
        self.add_player(self.guesser)
    
    def _was_target(self, word: str):
        return word in self.cluegiver.targets

    def _get_cluegiver_prompt(self) -> str:
        prompt_cluegiver = self.load_template("resources/initial_prompts/prompt_cluegiver")
        team_words = ", ".join(self.board.get_hidden_words("team"))
        opponent_words = ", ".join(self.board.get_hidden_words("opponent"))
        innocent_words = ", ".join(self.board.get_hidden_words("innocent"))
        assassin_words = ", ".join(self.board.get_hidden_words("assassin"))
        instance_prompt_cluegiver = Template(prompt_cluegiver).substitute(team_words= team_words, 
                                                                          opponent_words=opponent_words, 
                                                                          innocent_words=innocent_words, 
                                                                          assassin_words=assassin_words)
        return instance_prompt_cluegiver
    
    def _get_guesser_prompt(self) -> str:
        prompt_guesser = self.load_template("resources/initial_prompts/prompt_guesser")
        board = ", ".join(self.board.get_all_hidden_words())
        instance_prompt_guesser = Template(prompt_guesser).substitute(board=board, 
                                                                      clue=self.cluegiver.clue, 
                                                                      number=self.cluegiver.number_of_targets)
        return instance_prompt_guesser
    
    def _on_before_turn(self, current_turn):
        # add new cluegiver prompt
        self.cluegiver.retries = 0
        self.guesser.retries = 0
        self.add_user_message(self.cluegiver, self._get_cluegiver_prompt())

    def _does_game_proceed(self) -> bool:
        # Determine if the game should proceed. This is also called once initially.
        if self.invalid_response:
            return False
        
        # for the base version, a check is needed whether all team words from one team are revealed or the assassin is revealed
        continue_game = True
        if self.board.has_team_won():
            self.log_to_self("game end", "Team won!")
            continue_game = False
        elif self.board.has_opponent_won():
            self.log_to_self("game end", "Opponent won!")
            continue_game = False
        elif self.board.has_assassin_won():
            self.log_to_self("game end", "Assassin won!")
            continue_game = False
        #elif self.current_turn >= 9:
            # that would only be necessary be for Codenames Duet
        #    return False
        if not continue_game:
            self.log_to_self("board status", self.board.get_current_board())
            return False
        return True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        self.invalid_response = False
        if player == self.cluegiver:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words())
            except ValidationError as error:
                print(error.message)
                self.log_to_self("cluegiver validation error", error.message)
                self.invalid_response = True
        else:
            try:
                player.validate_response(utterance, self.board.get_all_hidden_words(), self.cluegiver.number_of_targets)
            except ValidationError as error:
                self.log_to_self("guesser validation error", error.message)
                self.invalid_response = True
        
        return not self.invalid_response
    
    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        if player == self.cluegiver:
            self.log_to_self("clue", self.cluegiver.recover_utterance() ) # already logged by logging player message but needed for score
            self.log_to_self("targets", self.cluegiver.targets)
            return player.parse_response(utterance), True
        else:
            # TODO: check this, does not seem to work
            self.log_to_self("guess", self.guesser.recover_utterance()) # already logged by logging player message but needed for score
            guesses = player.parse_response(utterance)
            for guess in guesses:
                assignment = self.board.reveal_word(guess)
                self.log_to_self("word revealed", assignment)
                if self._was_target(guess):
                    self.log_to_self("target revealed", guess)
                if not self.board.should_continue_after_revealing(guess):
                    self.log_to_self("turn end after", guess)
                    break
                
            return guesses, True
        
    def _on_before_reprompt(self, player: Player):
        logger.debug("Reprompting...")
        player.retries += 1
        self.add_user_message(player, "Your answer did not follow the requested format, please give a new answer that follows the format.")
    
    def _should_reprompt(self, player: Player):
        if player.retries < MAX_RETRIES:
            return self.invalid_response
        return False
    
    def _after_add_player_response(self, player: Player, utterance: str):
        if player == self.cluegiver:
            # put clue and number of targets into prompt for guesser
            # no intermittent turn for guesser needed, as it is short enough
            self.add_user_message(self.guesser, self._get_guesser_prompt())
        else:
            # TODO: use intermittent prompt, that also includes guess from guessing player and then prompts for new clue, with updated lists of board words
            self.add_user_message(self.cluegiver, utterance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        board_at_end = {}
        invalid_format_total = {"cluegiver": 0, "guesser": 0}
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            # Metrics per turn:
            # target-precision, target-recall, target-f1
            # team-precision
            # invalid formats per player
            turn_score = {"clue": None, "targets": [], "cluegiver invalid format": 0, 
                          "guess": None, "guesser invalid format": 0, 
                          "words revealed": {"target": 0, "team": 0, "opponent": 0, "innocent": 0, "assassin": 0, "total": 0}}
            for event in turn:
                action = event['action']
                match action["type"]:
                    case "cluegiver validation error":
                        turn_score["cluegiver invalid format"] += 1
                    case "guesser validation error":
                        turn_score["guesser invalid format"] += 1
                    case "clue" | "targets" | "guess ":
                        turn_score[action["type"]] = action["content"]
                    case "word revealed":
                        turn_score["words revealed"][action["content"]] += 1
                        turn_score["words revealed"]["total"] += 1
                    case "board status":
                        board_at_end = action["content"]

            sum_revealed_words = turn_score["words revealed"]["team"] + turn_score["words revealed"]["opponent"] + turn_score["words revealed"]["innocent"] + turn_score["words revealed"]["assassin"]
            target_precision = 0
            target_recall = 0
            target_f1 = 0
            team_precision = 0
            if sum_revealed_words:
                target_precision = turn_score["words revealed"]["target"] / sum_revealed_words
                team_precision = turn_score["words revealed"]["team"] / sum_revealed_words
            
            if len(turn_score["targets"]) > 0:
                target_recall = turn_score["words revealed"]["target"] / len(turn_score["targets"])
            if target_precision + target_recall > 0:
                target_f1 = 2 * target_precision * target_recall / (target_precision + target_recall)
            
            self.log_turn_score(turn_idx, "turn", turn_score)
            self.log_turn_score(turn_idx, "cluegiver invalid format", turn_score["cluegiver invalid format"])
            self.log_turn_score(turn_idx, "guesser invalid format", turn_score["guesser invalid format"])
            self.log_turn_score(turn_idx, "target precision", target_precision)
            self.log_turn_score(turn_idx, "target recall", target_recall)
            self.log_turn_score(turn_idx, "target f1", target_f1)
            self.log_turn_score(turn_idx, "team precision", team_precision)

            invalid_format_total["cluegiver"] += turn_score["cluegiver invalid format"]
            invalid_format_total["guesser"] += turn_score["guesser invalid format"]

        # Metrics per game:
        # win/lose
        # if lose, lost through assassin or lost through other team being faster
        # number of team words revealed vs. all words revealed by agent
        # how many turns needed
        # TODO: remove magic numbers 9 and 8, should come from word assignments somewhere
        number_of_turns = len(episode_interactions["turns"])
        self.log_episode_score("board status", board_at_end)
        self.log_episode_score("won", len(board_at_end["hidden"]["team"]) == 0)
        self.log_episode_score("lost through opponent", len(board_at_end["hidden"]["opponent"]) == 0)
        self.log_episode_score("lost through assassin", len(board_at_end["revealed"]["team"]["assassin"]) >= 1)
        self.log_episode_score("won through assassin (opponent revealed)", len(board_at_end["revealed"]["opponent"]["assassin"]) >= 1)
        self.log_episode_score("number of turns", number_of_turns)
        self.log_episode_score("team words revealed", len(board_at_end["revealed"]["team"]["team"]) / 9)
        self.log_episode_score("other words revealed", 1 - (len(board_at_end["revealed"]["team"]["assassin"]) + len(board_at_end["revealed"]["team"]["opponent"]) + len(board_at_end["revealed"]["team"]["innocent"])) / 16)
        self.log_episode_score("total invalid format", invalid_format_total)

        # Main Score: log19(1 + (#team - #opponent - assassin_true*#hidden_opponent + 9 offset) / #turns) * 100
        main_score = len(board_at_end["revealed"]["team"]["team"]) - len(board_at_end["revealed"]["team"]["opponent"]) - len(board_at_end["revealed"]["team"]["assassin"]) * len(board_at_end["hidden"]["opponent"]) + 9 # offset
        main_score = (main_score / number_of_turns) + 1
        main_score = math.log(main_score, 19) * 100
        assert 0 <= main_score <= 100, f"Main Score of {main_score} is not between 0 and 100"
        self.log_episode_score("Main Score", main_score)


class CodenamesGameBenchmark(GameBenchmark):

    def __init__(self):
        super().__init__(GAME_NAME)
        random.seed(SEED)

    def get_description(self) -> str:
        return "Codenames game between a cluegiver and a guesser"

    def create_game_master(self, experiment: Dict, player_backends: List[str]) -> GameMaster:
        return CodenamesGame(experiment, player_backends)
