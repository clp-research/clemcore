import statistics, math
from clemgame.clemgame import GameScorer
from clemgame.metrics import BENCH_SCORE, METRIC_ABORTED
from .constants import *

EXPECTED_WORDS_PER_TURN = 2

def f1(precision, recall, weight = 2):
    if precision + recall == 0:
        return 0
    return weight * precision * recall / (precision + recall)

class CodenamesScorer(GameScorer):
    def __init__(self, experiment_config, game_instance):
        super().__init__(GAME_NAME, experiment_config, game_instance)

    def log_turn_score(self, turn_idx, name, value, scale=False):
        if type(value) == int or type(value) == float:
            value = round(value, 3)
            value = value * 100 if scale else value
        super().log_turn_score(turn_idx, name, value)

    def log_episode_score(self, name, value, scale=False):
        value = round(value, 3)
        value = value * 100 if scale else value
        super().log_episode_score(name, value)

    def score_turns(self, episode_interactions):
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            board_status = {}
            turn_score = {CLUEGIVER: {Turn_logs.VALIDATION_ERROR: 0}, GUESSER: {Turn_logs.VALIDATION_ERROR: 0}, 
                          TARGETED: {TEAM: 0, OPPONENT: 0, INNOCENT: 0, ASSASSIN: 0, TOTAL: 0}, Turn_logs.GUESSES: [],
                          REVEALED: {TARGET: 0, TEAM: 0, OPPONENT: 0, INNOCENT: 0, ASSASSIN: 0, TOTAL: 0}}
            for event in turn:
                action = event["action"]
                if action["type"] == Turn_logs.VALIDATION_ERROR:
                    player = action["content"]["player"]
                    turn_score[player][Turn_logs.VALIDATION_ERROR] += 1
                elif action["type"] == Turn_logs.GUESSES:
                    turn_score[Turn_logs.GUESSES] =  action["content"]
                elif action["type"] == Turn_logs.WORD_TARGETED:
                    # FIXME: this should not be needed when wrong targets and guesses are removed from the utterances!
                    if not action["content"]["assignment"]:
                        continue
                    turn_score[TARGETED][action["content"]["assignment"]] += 1
                    turn_score[TARGETED][TOTAL] += 1
                    
                elif action["type"] == Turn_logs.TEAM_REVEALED:
                    turn_score[REVEALED][action["content"]["assignment"]] += 1
                    turn_score[REVEALED][TOTAL] += 1
                elif action["type"] == Turn_logs.TARGET_REVEALED:
                    turn_score[REVEALED][TARGET] += 1
                elif action["type"] == Turn_logs.BOARD_STATUS:
                    board_status = action["content"]
                    
            self.log_turn_score(turn_idx, "turn", turn_score) # TODO: needed?
            self.log_turn_score(turn_idx, f"{CLUEGIVER} {Turn_logs.VALIDATION_ERROR.value}", turn_score[CLUEGIVER][Turn_logs.VALIDATION_ERROR])
            self.log_turn_score(turn_idx, f"{GUESSER} {Turn_logs.VALIDATION_ERROR.value}", turn_score[GUESSER][Turn_logs.VALIDATION_ERROR])

            cluegiver_number_of_targets = turn_score[TARGETED][TOTAL]
            number_of_remaining_team_words = len(board_status[HIDDEN][TEAM])
            cluegiver_team_precision = 0
            cluegiver_team_recall = 0
            cluegiver_team_f1 = 0
            if cluegiver_number_of_targets > 0:
                cluegiver_team_precision = turn_score[TARGETED][TEAM] / cluegiver_number_of_targets                
            cluegiver_team_recall = turn_score[TARGETED][TEAM] / number_of_remaining_team_words
            if cluegiver_team_precision + cluegiver_team_recall > 0:
                cluegiver_team_f1 = f1(cluegiver_team_precision, cluegiver_team_recall)
            self.log_turn_score(turn_idx, Turn_Scores.CLUEGIVER_NUMBER_OF_TARGETS, cluegiver_number_of_targets)
            self.log_turn_score(turn_idx, Turn_Scores.CLUEGIVER_TEAM_PRECISION, cluegiver_team_precision)
            self.log_turn_score(turn_idx, Turn_Scores.CLUEGIVER_TEAM_RECALL, cluegiver_team_recall)
            self.log_turn_score(turn_idx, Turn_Scores.CLUEGIVER_TEAM_F1, cluegiver_team_f1) # probably not useful

            guesser_number_of_guesses = len(turn_score[Turn_logs.GUESSES])
            guesser_number_of_revealed_words = turn_score[REVEALED][TOTAL]
            guesser_number_of_unrevealed_guesses = guesser_number_of_guesses - guesser_number_of_revealed_words
            guesser_target_precision = 0
            guesser_target_recall = 0
            guesser_target_f1 = 0
            guesser_team_precision = 0
            guesser_team_recall = 0
            guesser_team_f1 = 0

            if guesser_number_of_revealed_words:
                guesser_target_precision = turn_score[REVEALED][TARGET] / guesser_number_of_revealed_words
                guesser_team_precision = turn_score[REVEALED][TEAM] / guesser_number_of_revealed_words
            if turn_score[TARGETED][TOTAL] > 0:
                guesser_target_recall = turn_score[REVEALED][TARGET] / cluegiver_number_of_targets
            if guesser_target_precision + guesser_target_recall > 0:
                guesser_target_f1 = f1(guesser_target_precision, guesser_target_recall)

            guesser_team_recall = turn_score[REVEALED][TEAM] / number_of_remaining_team_words
            guesser_team_f1 = f1(guesser_team_precision, guesser_team_recall)

            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_NUMBER_OF_GUESSES, guesser_number_of_guesses)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_NUMBER_OF_REVEALED_WORDS, guesser_number_of_revealed_words)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_NUMBER_OF_UNREVEALED_GUESSES, guesser_number_of_unrevealed_guesses)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_TARGET_PRECISION, guesser_target_precision)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_TARGET_RECALL, guesser_target_recall)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_TARGET_F1, guesser_target_f1)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_TEAM_PRECISION, guesser_team_precision)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_TEAM_RECALL, guesser_team_recall)
            self.log_turn_score(turn_idx, Turn_Scores.GUESSER_TEAM_F1, guesser_team_f1) # probably not useful

    def score_game(self, episode_interactions):
        # experiment name
        super().log_episode_score(VARIABLE, self.experiment['variable'])
        super().log_episode_score('experiment name', self.experiment[NAME])
        # game-specific scores
        for flag_name, value in self.experiment["flags"].items():
            if value:
                self.log_episode_score(f"Cluegiver {flag_name.lower()}", episode_interactions["Cluegiver engaged flags"][flag_name])
                self.log_episode_score(f"Guesser {flag_name.lower()}", episode_interactions["Guesser engaged flags"][flag_name])

        # average turn scores
        for score_const in Turn_Scores:
            score = statistics.mean([self.scores["turn scores"][x][score_const.value] for x in self.scores["turn scores"]])
            self.log_episode_score(f"Average {score_const.value}", score)

        # plus all required game scores
        super().score_game(episode_interactions)
    
    def score_game_end(self, episode_interactions):
        super().score_game_end(episode_interactions)
        # plus game specific things
        # won or lost through assassin or through revealing all words of one team

        self.log_episode_score(GAME_ENDED_THROUGH_ASSASSIN, episode_interactions[GAME_ENDED_THROUGH_ASSASSIN])
        self.board_at_end = episode_interactions[BOARD_END_STATUS]

        number_of_team_words = self.experiment[ASSIGNMENTS][TEAM]
        number_of_non_team_words = self.experiment[ASSIGNMENTS][OPPONENT] + self.experiment[ASSIGNMENTS][INNOCENT] + self.experiment[ASSIGNMENTS][ASSASSIN]
        number_of_revealed_words = len(self.board_at_end[REVEALED][TEAM][TEAM])
        self.log_episode_score(Episode_Scores.RECALL, number_of_revealed_words / number_of_team_words)
        self.log_episode_score(Episode_Scores.NEGATIVE_RECALL, 1 - (len(self.board_at_end[REVEALED][TEAM][ASSASSIN]) + len(self.board_at_end[REVEALED][TEAM][OPPONENT]) + len(self.board_at_end[REVEALED][TEAM][INNOCENT])) / number_of_non_team_words)
        number_of_turns = episode_interactions[NUMBER_OF_TURNS]
        self.log_episode_score(NUMBER_OF_TURNS, number_of_turns)

        guesser_efficiency = 0
        for turn_idx in self.scores["turn scores"]:
            guesser_efficiency += self.scores["turn scores"][turn_idx][Turn_Scores.GUESSER_NUMBER_OF_GUESSES]
        guesser_efficiency /= number_of_turns
        
        efficiency = min(1/EXPECTED_WORDS_PER_TURN * guesser_efficiency, 1)
        self.log_episode_score(Episode_Scores.EFFICIENCY, efficiency)
       
    def log_main_score(self, episode_interactions):
        # all logged scores are available via self.scores["episode scores"][score_name]
        # or self.scores["turn scores"][turn_idx][score_name]

        if self.scores["episode scores"][METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, math.nan)
            return

        # Main Score: harmonic mean of success (revealed team words / all team words (recall)) and efficiency (1/number of turns)
        progress = self.scores["episode scores"][Episode_Scores.RECALL]
        efficiency = self.scores["episode scores"][Episode_Scores.EFFICIENCY]
        main_score = statistics.harmonic_mean([progress, efficiency])
        self.log_episode_score(BENCH_SCORE, main_score, scale=True)
