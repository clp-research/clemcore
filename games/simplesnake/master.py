import re
import numpy as np
from typing import Dict, List
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player
from clemcore.clemgame.legacy.master import DialogueGameMaster
from clemcore.clemgame.legacy.scorer import GameScorer
from clemcore.backends.model_registry import Model, CustomResponseModel
from clemcore.clemgame.metrics import METRIC_REQUEST_COUNT_VIOLATED, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_COUNT, \
                                        METRIC_REQUEST_SUCCESS_RATIO, METRIC_ABORTED, BENCH_SCORE, METRIC_SUCCESS, METRIC_LOSE
import logging
logger = logging.getLogger(__name__)


class SimpleSnakeGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def get_description(self):
        return "Plays a simple game of snake with one player."

    def is_single_player(self) -> bool:
        return True

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return SimpleSnake(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return SimpleSnakeGameScorer(self.game_name, experiment, game_instance)


class Gameboard:
    def __init__(self, dim, snake_start_loc, prey_start_loc, obstacles=None):
        self.gameboard = [['' for _ in range(dim)] for _ in range(dim)]
        self.dim = dim

        # set snake start location
        self.snake_pos = [snake_start_loc // dim, snake_start_loc % dim]
        self.gameboard[self.snake_pos[0]][self.snake_pos[1]] = 's'

        # set prey start location
        self.prey_pos = [prey_start_loc // dim, prey_start_loc % dim]
        self.gameboard[prey_start_loc // dim][prey_start_loc % dim] = '*'

        # set obstacle locations
        self.obstacles = []
        if obstacles is not None:
            for ob in obstacles:
                self.obstacles.append([ob // dim, ob % dim])
                self.gameboard[ob // dim][ ob % dim] = 'X'

    def update_gameboard(self, new_loc) -> str:
        """Updates the snake location on the gameboard. Returns True if
        update results in valid game state, otherwise False.
        """
        row, col = new_loc
        # check game state
        if row == self.prey_pos[0] and col == self.prey_pos[1]:
            return 'win state'
        if (row < 0 or row >= self.dim) or (col < 0 or col >= self.dim):
            return 'lose state'
        for ob in self.obstacles:
            if row == ob[0] and col == ob[1]:
                return 'lose state'

        # move doesn't end game, so update snake pos
        self.gameboard[self.snake_pos[0]][self.snake_pos[1]] = ''
        self.snake_pos[0], self.snake_pos[1] = row, col
        self.gameboard[self.snake_pos[0]][self.snake_pos[1]] = 's'
        return self.__str__()

    def __str__(self):
        result = ''
        for i in range(self.dim):
            for j in range(self.dim):
                result += f'[{self.gameboard[i][j]}]'

            if i != self.dim - 1:
                result += '\n'
        return result


class Navigator(Player):
    def __init__(self, model: Model):
        super().__init__(model)

    def _custom_response(self):
        return "Alright, I'm ready!"


class Describer(Player):
    def __init__(self, dim, snake_start_loc, prey_start_loc, obstacle_locs, match_pattern):
        super().__init__(CustomResponseModel())
        self.gameboard = Gameboard(dim, snake_start_loc, prey_start_loc, obstacle_locs)
        self.init_game_state = True
        self.pattern = match_pattern

    def _custom_response(self, context: Dict) -> str:
        if self.init_game_state:
            self.init_game_state = False
            return self.gameboard.__str__()

        # parse direction and determine new snake location
        directions = re.findall(self.pattern, context['content'].lower())
        row, col = 0, 0
        state = None
        for direction in directions:
            if direction == 'up':
                row = self.gameboard.snake_pos[0] - 1
                col = self.gameboard.snake_pos[1]
            elif direction == 'down':
                row = self.gameboard.snake_pos[0] + 1
                col = self.gameboard.snake_pos[1]
            elif direction == 'left':
                row = self.gameboard.snake_pos[0]
                col = self.gameboard.snake_pos[1] - 1
            elif direction == 'right':
                row = self.gameboard.snake_pos[0]
                col = self.gameboard.snake_pos[1] + 1
            else:
                raise RuntimeError('Failed to parse direction.')
            state = self.gameboard.update_gameboard((row, col))

        # return self.gameboard.update_gameboard((row, col))
        return state


class SimpleSnake(DialogueGameMaster):
    """ Implements a game of snake in which one player describes the directions in which
    to move a snake within a grid to find its food.
    """
    
    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model] = None):
        super().__init__(game_spec, experiment, player_models)
        # experiment-level variables
        self.max_turns = experiment['max_turns']
        self.dim = experiment['dim']
        self.navigator_initial_prompt = experiment['navigator_initial_prompt']
        self.navigator_response_pattern = experiment['navigator_response_pattern']
        self.navigator_reprompt = experiment['navigator_reprompt']

    def _on_setup(self, **game_instance):
        self.pass_turn = True
        self.is_warned = False

        # instance-level flags
        self.invalid_response = False

        # instance-level variables
        self.game_instance = game_instance
        self.snake_location = game_instance['snake_start_loc']
        self.prey_location = game_instance['prey_start_loc']


        # instance players
        self.navigator = Navigator(self.player_models[0])
        self.add_player(self.navigator)

        # if 'obstacles' in self.game_name:
        if self.game_spec.game_name == 'simplesnake_withobstacles':
                self.obstacle_locs = game_instance['obstacle_locs']
        else:
            self.obstacle_locs = None

        self.describer = Describer(
            dim=self.dim,
            snake_start_loc=self.snake_location,
            prey_start_loc=self.prey_location,
            obstacle_locs=self.obstacle_locs,
            match_pattern=self.navigator_response_pattern,
        )
        self.add_player(self.describer)

    def _on_before_game(self):
        self.set_context_for(self.navigator, self.navigator_initial_prompt)

    def _does_game_proceed(self):
        """Proceed as long as the snake does not occupy the same gridspace as the prey."""
        if self.invalid_response:
            self.log_to_self("invalid_response", "abort game")
            return False
        if self.win_state:
            self.log_to_self("game_win", "end game")
            return False
        if self.lose_state:
            self.log_to_self("game_loss", "end game")
            return False
        if self.current_round >= self.max_turns:
            self.log_to_self("max_turns_reached", str(self.max_turns))
            return False
        return True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        # reset flags
        self.invalid_response = False
        self.win_state = False
        self.lose_state = False
        bad_response = False

        if player == self.navigator:
            if self.current_round == 0:  # ensure grid is sent during initial round
                self.log_to_self("game start", "Get initial grid.")
                return True

            # is navigator response in valid format?
            if self.game_spec.game_name == 'simplesnake_withplanning':
                bad_response = re.fullmatch(f'^(?:{self.navigator_response_pattern}\s*)+$', utterance.lower()) is None
            else:
                matches = re.findall(self.navigator_response_pattern, utterance.lower())
                bad_response = len(matches) == 0 or len(matches) > 1

            if bad_response:
                if self.is_warned:  # Reprompting failed
                    self.invalid_response = True
                    return False
                else:  # Skip describer's turn and reprompt navigator
                    self.pass_turn = False
                    self.set_context_for(self.navigator, self.navigator_reprompt)
                    self.is_warned = True
                    return True

            # Reset flags after they've been used to skip describer
            self.is_warned = False
            self.pass_turn = True
        elif player == self.describer:
            # handle end-of-game describer response
            if self.game_spec.game_name == 'simplesnake_withplanning':
                if self.current_round == 1:
                    if utterance == 'win state':
                        self.win_state = True
                    else:
                        self.lose_state = True
                    return False
            else:
                if utterance == 'lose state':
                    self.lose_state = True
                    return False
                if utterance == 'win state':
                    self.win_state = True
                    return False

        return True # valid response

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        if player == self.navigator:
            if self.current_round == 0:
                self.set_context_for(self.describer, 'Get initial grid.')
            else:
                self.set_context_for(self.describer, parsed_response)
        elif player == self.describer:
            self.set_context_for(self.navigator, parsed_response)

    def _should_pass_turn(self):
        return self.pass_turn


class SimpleSnakeGameScorer(GameScorer):
    """GameScorer subclass for SimpleSnake.
    Reads episode records, counts failures, calculates scores and stores the
    results in score files."""

    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)
        self.max_turns = experiment['max_turns']

    def compute_scores(self, episode_interactions: Dict) -> None:
        """Episode level scores.
        Writes to score file in the episode directory.
        """
        bench_score = None
        turn_scores = []
        invalid_response = False
        win_state = False
        lose_state = False
        max_turns_reached = False

        for turn_idx, turn in enumerate(episode_interactions['turns']):
            turn_score = {'request_count': 1}

            for event in turn:
                action = event['action']
                if action['type'] == 'invalid_response':
                    invalid_response = True
                if action['type'] == 'game_loss':
                    lose_state = True
                if action['type'] == 'game_win':
                    win_state = True
                if action['type'] == 'max_turns_reached':
                    max_turns_reached = True

            # check if LM followed correct formatting rules
            if invalid_response:
                turn_score['violated_request_count'] = 1
                turn_score['parsed_request_count'] = 0
            else:
                turn_score['violated_request_count'] = 0
                turn_score['parsed_request_count'] = 1

            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score["request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_VIOLATED, turn_score["violated_request_count"])
            turn_scores.append(turn_score)

        violated_request_count = sum([turn["violated_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)

        parsed_request_count = sum([turn["parsed_request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum([turn["request_count"] for turn in turn_scores])
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        self.log_episode_score(METRIC_REQUEST_SUCCESS_RATIO, parsed_request_count / request_count)

        # Common metrics
        if invalid_response:  # whether a violation of the game rules happened (response not parsable)
            self.log_episode_score(METRIC_ABORTED, 1)
            self.log_episode_score(METRIC_SUCCESS, 0)
            self.log_episode_score(METRIC_LOSE, 0)
            bench_score = np.nan  # metric not applicable
        else:
            if win_state:  # no rule violations, and reached goal
                self.log_episode_score(METRIC_SUCCESS, 1)
                self.log_episode_score(METRIC_LOSE, 0)
                bench_score = 100  # assign 100% score for game win
            elif lose_state:  # game was lost
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                bench_score = 0
            elif max_turns_reached:  # goal was never found and turns were exceeded
                self.log_episode_score(METRIC_SUCCESS, 0)
                self.log_episode_score(METRIC_LOSE, 1)
                bench_score = 0
            self.log_episode_score(METRIC_ABORTED, 0)
        self.log_episode_score(BENCH_SCORE, bench_score)
