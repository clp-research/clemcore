from clemgame.clemgame import Player, GameMaster, GameBenchmark, DialogueGameMaster, GameScorer
from clemgame import metrics as ms
from clemgame import get_logger
from games.matchit_ascii.instancegenerator import GAME_NAME
from backends import Model

from typing import List, Dict, Tuple

import numpy as np


logger = get_logger(__name__)


class MatchItPlayer(Player):
    def __init__(self, backend: Model):
        super().__init__(backend)


    def __init__(self, backend: Model, role: str):
        super().__init__(backend)
        self.role: str = role

        self.description: str = ""
        self.question: str = ""
        self.answer: str = ""
        self.decision: str = ""

        self.had_success: bool = False

    def _custom_response(self, messages, turn_idx) -> str:
        last_message = messages[-1]["content"]

        if "collaborative" in last_message:
            return f"DESCRIPTION: from Player {self.role}"
        elif "ask" in last_message:
            return f"QUESTION: from Player {self.role}"
        elif "QUESTION" in last_message:
            return f"ANSWER: from Player {self.role}"
        elif "decision" in last_message:
            return "DECISION: Same grid."
        else: 
            return "ANSWER: How did we land here? This is the else in the mock answers."


class MatchItAscii(DialogueGameMaster):
    def __init__(self, experiment: Dict, player_backends: List[Model]):
        super().__init__(GAME_NAME, experiment, player_backends)

        self.experiment: str = experiment["name"]
        self.flags: dict[str, str] = experiment["flags"]
        
        self.initial_prompt: str = experiment["initial_prompt"]
        self.desc_intro: str = experiment["desc_intro"] # "This is my"
        self.q_reprompt: str = experiment["q_reprompt"] # "Reprompt: Now ask a question, starting with \"QUESTION: \""
        self.d_reprompt: str = experiment["d_reprompt"] # "Make a decision." 
        self.a_request: str = experiment["a_request"] #"Start your answer with ANSWER:"

        self.solution: str = experiment["solution"]
        self.wrong_solution: str = experiment["wrong_solution"]
        
        self.final_decision: bool = False
        self.success_a: bool = True
        self.success_b: bool = True
        self.aborted: bool = False
            
        self.model_a: Model = player_backends[0]
        self.model_b: Model = player_backends[1]

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance
        self.grid_a: str = game_instance["grid_a"]
        self.grid_b: str = game_instance["grid_b"]
        self.prompt_a = self.initial_prompt.replace("$GRID$", self.grid_a)
        self.prompt_b = self.initial_prompt.replace("$GRID$", self.grid_b)

        self.decision_turn: int = game_instance["decision_turn"]

        self.player_a: Player = MatchItPlayer(self.model_a, "A")
        self.player_b: Player = MatchItPlayer(self.model_b, "B")

        self.add_player(self.player_a)
        self.add_player(self.player_b)

        self.n_turns: int = -1
        self.answer_counter: int = 0 # counts how many answers a player has given per turn -> for reprompting

    def _on_before_game(self):
        # add prompt to Player A message history
        self.add_user_message(self.player_a, self.prompt_a)

    def _on_before_turn(self, turn_idx: int):
        self.n_turns += 1

    def _does_game_proceed(self) -> bool:
        if self.aborted:
            self.log_to_self("Game over", "Aborted")
            return False        
        elif self.final_decision:
            return False
        else: 
            return True
    
    def check_flag(self, first_word: str, flag: str):
        if first_word == flag:
            self.log_to_self("valid format", "continue")
            return True
        else: 
            self.log_to_self("invalid format", f"abort, first word: {first_word}")
            self.aborted = True
            return False

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if not utterance.strip(): # check for empty message
            self.log_to_self("invalid content", "abort, empty message")
            self.aborted = True
            return False

        utt_parts = list(filter(None, utterance.strip().split("\n"))) #filter to be sure that there are no empty strings
        first_word = utt_parts[0].split(" ")[0]
        # logger.info("first word = " + first_word)
        # logger.info("utterance = " + utterance)
        
        # first turn
        if self.n_turns == 0:
            if self.answer_counter == 1:
                return self.check_flag(first_word, self.flags["question"])
            else:
                return self.check_flag(first_word, self.flags["description"])
        # decision turn
        elif self.n_turns == self.decision_turn and player == self.player_b:
            if self.answer_counter == 1:
                if self.check_flag(first_word, self.flags["decision"]):
                    if utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.solution).lower():
                        player.success = True
                        self.log_to_self(f"Decision Player {player.role}", "success")
                    elif utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.wrong_solution).lower():
                        player.success = False
                        self.log_to_self(f"Decision Player {player.role}", "loss")
                    else:
                        self.log_to_self("invalid content", "abort, wrong message content.")
                        self.aborted = True
                        return False
                    return True
                else:
                    return False
            else:
                return self.check_flag(first_word, self.flags["answer"])
        # last turn, only for Player A's decision
        elif self.n_turns == self.decision_turn + 1:
            self.final_decision = True
            if self.check_flag(first_word, self.flags["decision"]):
                if utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.solution).lower():
                        player.success = True
                        self.log_to_self(f"Decision Player {player.role}", "success")
                elif utterance.lower().strip(".\n") == (self.flags["decision"] + " " + self.wrong_solution).lower():
                    player.success = False
                    self.log_to_self(f"Decision Player {player.role}", "loss")
                else:
                    self.log_to_self("invalid content", "abort, wrong message content.")
                    self.aborted = True  
                    return False
                return True
                
        # all other turns
        else:
            if self.answer_counter == 0:
                return self.check_flag(first_word, self.flags["answer"])
            else: 
                return self.check_flag(first_word, self.flags["question"])


    def _on_parse_response(self, player: Player, utterance: str) -> Tuple[str, bool]:
        utterance = utterance.strip()
        if utterance.startswith(self.flags["description"]):
            player.description = utterance
        elif utterance.startswith(self.flags["question"]):
            player.question = utterance
        elif utterance.startswith(self.flags["answer"]):
            player.answer = utterance
        elif utterance.startswith(self.flags["decision"]):
            player.decision = utterance
        
        self.answer_counter += 1

        return utterance, False

    def _after_add_player_response(self, player: Player, utterance: str):
        # first turn
        if self.n_turns == 0:
            if player == self.player_a:
                self.add_user_message(self.player_b, self.prompt_b)
            elif player == self.player_b:
                if self.player_b.description != "" and self.player_b.question != "":
                    self.add_user_message(self.player_a, self.desc_intro + self.player_b.description + "\n" + self.player_b.question + self.a_request)
                    self.player_b.question = ""
                else:
                    logger.info(f"Warning for first turn, Player B DESC = {self.player_b.description}; QUES = {self.player_b.question}")
        # decision turn
        elif self.n_turns == self.decision_turn and player == self.player_b and self.answer_counter == 1:
            self.add_user_message(self.player_a, player.answer + "\n" + self.d_reprompt)
        # all other turns
        else:
            other_player = self.player_a if player == self.player_b else self.player_b
            
            if player.answer != "" and player.question != "":
                #self.log_to_self("note", "a+q -> A:" + player.answer + " ,Q:" + player.question + " ,D:" + player.decision )
                self.add_user_message(other_player, player.answer + "\n" + player.question + self.a_request)
                player.description = ""
                player.question = ""
                player.answer = ""
                player.decision = ""
            elif player.decision != "" and player.question != "":
                #self.log_to_self("note", "a+d -> A:" + player.answer + " ,Q:" + player.question + " ,D:" + player.decision )
                self.add_user_message(other_player, player.decision + "\n" + player.question)
                player.description = ""
                player.question = ""
                player.answer = ""
                player.decision = ""


    def _should_reprompt(self, player: Player):
        while self._does_game_proceed():
            if self.n_turns == 0 and player == self.player_a:
                self.answer_counter = 0
                return False
            elif self.n_turns == self.decision_turn + 1:
                return False
            if self.answer_counter > 1: 
                self.answer_counter = 0
                return False
            return True
        return False  
    
    def _on_before_reprompt(self, player: Player):
        if self.n_turns == self.decision_turn and player == self.player_b:
            self.add_user_message(player, self.d_reprompt)
        elif self.n_turns == 0:
            self.add_user_message(player, self.desc_intro + self.player_a.description + "\n" + self.q_reprompt)
        else:
            self.add_user_message(player, self.q_reprompt)


class MatchItScorer(GameScorer):
 
    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)        

    def compute_scores(self, episode_interactions: Dict) -> None:

        all_turn_scores = []
        success_a = False
        success_b = False
        aborted = False
        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score_dict = {"request_count": 0, "violated_request_count" : 0, "parsed_request_count" : 0} 

            for event in turn:
                action = event["action"]
                # parsed requests
                if action["type"] == "invalid format":
                    turn_score_dict["violated_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                    first_word = action["content"].split(" ")[-1]
                    with open("first_words.txt", "a") as myfile:
                        myfile.write(first_word + "\n")
                elif action["type"] == "invalid content":
                    turn_score_dict["violated_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                elif action["type"] == "valid format":
                    turn_score_dict["parsed_request_count"] += 1
                    turn_score_dict["request_count"] += 1
                elif action["content"].startswith("Abort"):
                    aborted = True
                # decision success
                elif action["type"] == "Decision Player A":
                    if action["content"] == "success":
                        success_a = True
                elif action["type"] == "Decision Player B":
                    if action["content"] == "success":
                        success_b = True

            # log turn request scores   
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_VIOLATED, turn_score_dict["violated_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT_PARSED, turn_score_dict["parsed_request_count"])
            self.log_turn_score(turn_idx, ms.METRIC_REQUEST_COUNT, turn_score_dict["request_count"])
            #calculate episode scores from turn scores
            all_turn_scores.append(turn_score_dict)

            violated_request_count = sum([turn["violated_request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_request_count)
            parsed_request_count = sum([turn["parsed_request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_request_count)
            request_count = sum([turn["request_count"] for turn in all_turn_scores])
            self.log_episode_score(ms.METRIC_REQUEST_COUNT, request_count)
            # log episode "success" scores
            if aborted:
                self.log_episode_score(ms.METRIC_ABORTED, 1)
                self.log_episode_score(ms.METRIC_SUCCESS, 0)
                self.log_episode_score(ms.METRIC_LOSE, 0)
                # Game-specific metrics
                self.log_episode_score(ms.BENCH_SCORE, np.nan)  # metric not applicable
                self.log_episode_score("Player Score", np.nan)
            else:
                # two wrong decisions:
                if not success_a and not success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 0)
                    self.log_episode_score(ms.METRIC_LOSE, 1)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 0)
                    self.log_episode_score("Player Score", 0)
                # only one decided correctly
                elif success_a != success_b:
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 0)
                    self.log_episode_score(ms.METRIC_LOSE, 1)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 0)  # current decision, may change (before: 50)
                    self.log_episode_score("Player Score", 50)

                else:   # = success_a and success_b:   
                    self.log_episode_score(ms.METRIC_ABORTED, 0)
                    self.log_episode_score(ms.METRIC_SUCCESS, 1)
                    self.log_episode_score(ms.METRIC_LOSE, 0)
                    # Game-specific metrics
                    self.log_episode_score(ms.BENCH_SCORE, 100)
                    self.log_episode_score("Player Score", 100)

            
class MatchItBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    def is_single_player(self):
        return False

    def get_description(self):
        return "A simple game in which two players have to decide whether they see the same grid or not, an ascii version of the matchit game."

    def create_game_master(self,
                           experiment: Dict,
                           player_backends: List[str]
                           ) -> GameMaster:
        return MatchItAscii(experiment, player_backends)
    
    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return MatchItScorer(experiment, game_instance)


