import statistics, math
from clemgame.clemgame import GameScorer
from clemgame.metrics import BENCH_SCORE, METRIC_ABORTED
from .constants import Turn_logs, CLUEGIVER, GUESSER, REVEALED, TARGET, TEAM, OPPONENT, INNOCENT, ASSASSIN, GAME_NAME, NUMBER_OF_TURNS, GAME_ENDED_THROUGH_ASSASSIN

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
            # Metrics per turn:
            # target-precision, target-recall, target-f1
            # team-precision
            # invalid formats per player
            turn_score = {Turn_logs.TARGETS: [], CLUEGIVER: {Turn_logs.VALIDATION_ERROR: 0}, GUESSER: {Turn_logs.VALIDATION_ERROR: 0}, 
                          REVEALED: {TARGET: 0, TEAM: 0, OPPONENT: 0, INNOCENT: 0, ASSASSIN: 0, "total": 0}}
            for event in turn:
                action = event["action"]
                match action["type"]:
                    case Turn_logs.VALIDATION_ERROR:
                        player = action["content"]["player"]
                        turn_score[player][Turn_logs.VALIDATION_ERROR] += 1
                    case Turn_logs.TARGETS:
                        turn_score[Turn_logs.TARGETS] = action["content"]
                    case Turn_logs.TEAM_REVEALED:
                        turn_score[REVEALED][action["content"]["assignment"]] += 1
                        turn_score[REVEALED]["total"] += 1
                    case Turn_logs.TARGET_REVEALED:
                        turn_score[REVEALED][TARGET] += 1
            
            # TODO: calculate cluegiver target precision, more metrics concerning cluegiver and guesser precision, recall and f1
            # to calculate cluegiver target precision, I would need the board assignments that I do not have
            #cluegiver_target_precision = 0
            #for target in turn_score[Turn_logs.TARGETS]:
            #    if target in 
            #turn_score[]

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
            self.log_turn_score(turn_idx, f"{CLUEGIVER} {Turn_logs.VALIDATION_ERROR}", turn_score[CLUEGIVER][Turn_logs.VALIDATION_ERROR])
            self.log_turn_score(turn_idx, f"{GUESSER} {Turn_logs.VALIDATION_ERROR}", turn_score[GUESSER][Turn_logs.VALIDATION_ERROR])
            self.log_turn_score(turn_idx, "target precision", target_precision)
            self.log_turn_score(turn_idx, "target recall", target_recall)
            self.log_turn_score(turn_idx, "target f1", target_f1)
            self.log_turn_score(turn_idx, "team precision", team_precision)

    def score_game(self, episode_interactions):
        # game-specific scores

        for flag_name, value in self.experiment["flags"].items():
            if value:
                self.log_episode_score(f"Cluegiver {flag_name}", episode_interactions["Cluegiver engaged flags"][flag_name])
                self.log_episode_score(f"Guesser {flag_name}", episode_interactions["Guesser engaged flags"][flag_name])       

        number_of_turns = episode_interactions[NUMBER_OF_TURNS]
        self.log_episode_score(NUMBER_OF_TURNS, number_of_turns)
        number_of_team_words = self.experiment["assignments"]["team"]
        efficiency_multiplier = 2 # expecting two team words to be revealed each turn
        efficiency = min(1/efficiency_multiplier * number_of_team_words * 1/number_of_turns, 1)
        self.log_episode_score("efficiency", efficiency)
        target_f1s = [self.scores["turn scores"][x]["target f1"] for x in self.scores["turn scores"]]
        avg_target_f1s = statistics.mean(target_f1s)
        self.log_episode_score("avg target f1", avg_target_f1s)

        # plus all required game scores
        super().score_game(episode_interactions)
    
    def score_game_end(self, episode_interactions):
        super().score_game_end(episode_interactions)
        # plus game specific things
        # won or lost through assassin or through revealing all words of one team

        # TODO: should ratios also be 0 or NaN when the game was aborted? yes they should...

        self.log_episode_score(GAME_ENDED_THROUGH_ASSASSIN, episode_interactions[GAME_ENDED_THROUGH_ASSASSIN])

        # self.board_at_end = episode_interactions[BOARD_STATUS]
        # self.log_episode_score(BOARD_STATUS, board_at_end)

        number_of_team_words = self.experiment["assignment"]["team"]
        number_of_non_team_words = self.experiment["assignment"]["opponent"] + self.experiment["assignment"]["innocent"] + self.experiment["assignment"]["assassin"]
        self.log_episode_score("episode recall", len(self.board_at_end[REVEALED][TEAM][TEAM]) / number_of_team_words)
        self.log_episode_score("episode negative recall", 1 - (len(self.board_at_end[REVEALED][TEAM][ASSASSIN]) + len(self.board_at_end[REVEALED][TEAM][OPPONENT]) + len(self.board_at_end[REVEALED][TEAM][INNOCENT])) / number_of_non_team_words)
       
    def log_main_score(self, episode_interactions):
        # all logged scores are available via self.scores["episode scores"][score_name]
        # or self.scores["turn scores"][turn_idx][score_name]

        if self.scores["episode scores"][METRIC_ABORTED]:
            self.log_episode_score(BENCH_SCORE, math.nan)
            return

        # Main Score: harmonic mean of success (revealed team words / all team words (recall)) and efficiency (1/number of turns)
        success = self.scores["episode scores"]["episode recall"]
        efficiency = self.scores["episode scores"]["efficiency"]
        main_score = statistics.harmonic_mean([success, efficiency])
        self.log_episode_score(BENCH_SCORE, main_score, scale=True)
