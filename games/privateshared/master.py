"""
A game to test the scorekeeping abilities of a model.
Implementation of a game master that control the game mechanisms. 
"""

from typing import List, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import cohen_kappa_score

import clemgame.metrics as ms
from backends import Model
from clemgame import file_utils
from clemgame.clemgame import GameMaster, GameBenchmark, GameScorer
from clemgame import get_logger

from games.privateshared.game import PrivateSharedGame
from games.privateshared.constants import (
    GAME_NAME, PROBES_PATH, RETRIES_PATH, UPDATE, WORDS_PATH,
    INVALID_LABEL, INVALID, SUCCESS, NOT_SUCCESS, NOT_PARSED, RESULT)


logger = get_logger(__name__)


class PrivateShared(GameMaster):
    """Implement mechanisms for playing PrivateShared."""
    def __init__(self, experiment: Dict, player_models: List[Model]):
        super().__init__(GAME_NAME, experiment, player_models)
        self.subtype = experiment['name']
        self.model_name = self.player_models[0].get_name()
        # load necessary texts
        probes_path = PROBES_PATH.format(self.subtype)
        self.probing_questions = self.load_json(probes_path)
        self.retries = self.load_json(RETRIES_PATH)['suffixes']
        # initialise necessary structure
        self.questioner_tag: str = None
        self.initial_prompt: str = None
        self.probe_gt: Dict = None
        self.probing: Dict = None
        self.game: PrivateSharedGame = None
        self.filled_slots: List = []
        self.n_probe_turns: int = 0
        self.aborted: bool = False
        self.played_probing_rounds: int = 0
        # initialise common metrics
        self.request_counts: List = None
        self.parsed_request_counts: List = None
        self.violated_request_counts: List = None

    def setup(self,
              game_id: int,
              initial_prompt: str,
              request_order: List[str],
              requests: Dict[str, int],
              probes: Dict[int, Dict[str, int]],
              slots: Dict[str, str],
              tag: str,
              lang: str,
              ) -> None:

        # load language specific words
        words = self.load_json(WORDS_PATH.format(lang))
        self.answer = words['ANSWER']
        self.aside = words['ASIDE']
        self.me = words['ME']
        self.dummy_prompt = words['DUMMY_PROMPT']
        self.probe_text = words['PROBE']
        self.yes = words['YES']
        self.no = words['NO']
        self.coda = words['CODA']

        self.questioner_tag = f"{tag}: "
        self.initial_prompt = initial_prompt
        self.probing = probes
        self.probe_gt = {slot: i for i, slot in enumerate(request_order)}
        self.game = PrivateSharedGame(
            self.subtype, request_order, requests, slots,
            self.player_models[0], words)
        # one probing before the game starts and one after each request
        self.n_probe_turns = self.game.max_turns + 1
        # initialise turn counters
        self.request_counts = [0] * self.n_probe_turns
        self.parsed_request_counts = [0] * self.n_probe_turns
        self.violated_request_counts = [0] * self.n_probe_turns
    
        self.log_players({
            'GM': 'Game master for privateshared',
            'Player 1': f'Answerer: {self.model_name}',
            'Player 2': 'Questioner: Programmatic'
            })

    def play(self) -> None:
        self.log_next_turn()
        all_probes = []
        # initiate game with the instructions prompt
        self.game.initiate(self.initial_prompt)
        action = {'type': 'send message', 'content': self.initial_prompt}
        self.log_event(from_='GM', to='Player 1', action=action)

        # probing round before game starts
        turn_probes, probing_successful = self.probe(self.game)
        all_probes.append(turn_probes)
        if not probing_successful:
            action = {'type': 'invalid format',
                      'content': 'Abort: invalid format in probing.'}
            self.log_event(from_='GM', to='GM', action=action)
            self.aborted = True

        # actual game: alternate real turns and probing
        while self.game.proceeds() and not self.aborted:
            self.log_next_turn()
            # slot filling turn
            turn_successful = self.turn()
            if not turn_successful:
                action = {'type': 'invalid format',
                          'content': 'Abort: invalid format in slot filling.'}
                self.log_event(from_='GM', to='GM', action=action)
                self.aborted = True
                break
            # probing round
            turn_probes, probing_successful = self.probe(self.game)
            all_probes.append(turn_probes)
            if not probing_successful:
                action = {'type': 'invalid format',
                          'content': 'Abort: invalid format in probing.'}
                self.log_event(from_='GM', to='GM', action=action)
                self.aborted = True
                break

        self.log_key('probes', all_probes)
        self.log_key('realised_slots', self.probe_gt)
        action = {'type': 'end', 'content': 'Game finished.'}
        self.log_event(from_='GM', to='GM', action=action)
        self._log_eval_assets()

    def turn(self) -> bool:
        """Perform one slot-filling turn."""
        logger.info('Game turn: %d', self.game.current_turn)

        # pseudo prompt to questioner
        action = {'type': 'send message', 'content': self.dummy_prompt}
        self.log_event(from_='GM', to='Player 2', action=action)

        # get request
        request = self.game.questioner_turn(self.questioner_tag)

        # pass it on to answerer; remove tag for logging
        clean_request = request.replace(self.questioner_tag, '').strip()
        action = {'type': 'get message', 'content': clean_request}
        self.log_event(from_='Player 2', to='GM', action=action)
        # append the instruction to be straight to the point
        request = self.coda.format(request)
        action = {'type': 'send message', 'content': request}
        self.log_event(from_='GM', to='Player 1', action=action)

        # get answer from answerer
        prompt, raw_answer, answer = self.game.answerer_turn()
        action = {'type': 'get message', 'content': answer}
        call = (prompt, raw_answer)
        self.log_event(from_='Player 1', to='GM', action=action, call=call)
        # at this point, turn count has just been increased, so it matches the
        # current probing turn
        self.request_counts[self.game.current_turn] += 1

        # parse answer
        parsed_answer = self._parse_slot_response(answer)
        action = {'type': 'parse', 'content': parsed_answer}
        self.log_event(from_='GM', to='GM', action=action)

        # if answer cannot be parsed, break immediately
        if parsed_answer == INVALID:
            self.violated_request_counts[self.game.current_turn] += 1
            return False
        self.parsed_request_counts[self.game.current_turn] += 1

        # check if the answer was correct
        slot_filled = self._is_slot_filled(answer)
        self.filled_slots.append(slot_filled)
        action = {'type': 'metadata', 'content': f'Slot filled: {slot_filled}'}
        self.log_event(from_='GM', to='GM', action=action)

        # check if the agent gave away more info than it should
        # update ground truth if necessary
        self._update_anticipated_slots(answer)

        # pseudo passing parsed answer on to questioner
        action = {'type': 'send message', 'content': parsed_answer}
        self.log_event(from_='GM', to='Player 2', action=action)

        return True

    def _has_continuation(self, response: str) -> bool:
        """Return True if the response continues after what is needed."""
        # if the answer contains a line break with some continuation after it,
        # we consider it to be an invalid response
        # we strip first to account for cases where it ends in one or many \n
        # without producing anything after it, and then check if the remaining
        # text still contains a line break
        if '\n' in response.strip('\n'):
            return True
        return False

    def _parse_slot_response(self, response: str) -> str:
        """Extract parsed answer in slot filling turn."""
        if (not response.startswith(self.answer.strip()) 
            or self._has_continuation(response)):
            logger.warning(NOT_PARSED)
            return INVALID
        clean_response = self._filter_tag(response, self.answer.strip())
        return clean_response

    def _is_slot_filled(self, answer: str) -> bool:
        """Check if answer contains the correct value for a slot."""
        slot = self.game.request_order[self.game.current_turn - 1]
        value = self.game.slots[slot]
        if value.lower() in answer.lower():
            return True
        return False

    def _update_anticipated_slots(self, answer: str) -> None:
        """Update ground truth when agent anticipates a slot value."""
        turn = self.game.current_turn - 1
        for slot, value in self.game.slots.items():
            if value.lower() in answer.lower() and turn < self.probe_gt[slot]:
                update = UPDATE.format(slot.upper(), self.probe_gt[slot], turn)
                action = {'type': 'metadata', 'content': update}
                self.log_event(from_='GM', to='GM', action=action)
                self.probe_gt[slot] = turn

    def _log_eval_assets(self) -> None:
        """Log everything needed for the evaluation."""
        self.log_key(ms.METRIC_REQUEST_COUNT,
                     self.request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_PARSED,
                     self.parsed_request_counts)
        self.log_key(ms.METRIC_REQUEST_COUNT_VIOLATED,
                     self.violated_request_counts)
        self.log_key('Filled Slots', self.filled_slots)
        self.log_key('Aborted', self.aborted)
        self.log_key('Played Probe Rounds', self.played_probing_rounds)

    def _get_gt(self, turn_idx: int, question_type: str) -> int:
        """Retrieve the ground truth value for a slot at a given turn."""
        return 1 if turn_idx > self.probe_gt[question_type] else 0

    def _create_probe_dic(self, turn_idx: int, key: str, idx: int) -> Dict:
        """Return the initialised probe dictionary."""
        question = self.probing_questions[key][idx]
        return {'target': key.upper(),
                'question': self.probe_text.format(question),
                'gt': self._get_gt(turn_idx, key)}

    def _create_turn_probes(self, turn_idx: int) -> List[Dict]:
        """Return a list of probing dictionaries."""
        return [self._create_probe_dic(turn_idx, key, idx)
                for key, idx in self.probing[str(turn_idx)].items()]

    def probe(self, game: PrivateSharedGame) -> Tuple[List[Dict], bool]:
        """Perform a round of probing."""
        action = {'type': 'info', 'content': 'Begin probing'}
        self.log_event(from_='GM', to='GM', action=action)
        turn = game.current_turn
        probes = self._create_turn_probes(turn)
        success_by_round = []
        for probe in probes:
            history = game.messages.copy()
            history.append({'role': 'user', 'content': ''})
            # perform a probing loop, with retries up to maximum retries
            probing_results = self._probing_loop(probe, history, turn, game)
            answer, parsed_response, successful, tries = probing_results
            # add results to the probe object
            probe['answer'] = answer
            probe['value'] = self._convert_response(parsed_response)
            probe['tries'] = tries
            self._log_probing_outcome(probe, successful, tries)
            success_by_round.append(successful)
            if not successful:
                # interrupt immediately
                break

        probing_successful = all(success_by_round)
        if probing_successful:
            # actual valid rounds for posterior evaluation
            self.played_probing_rounds += 1
        action = {'type': 'info', 'content': 'End probing'}
        self.log_event(from_='GM', to='GM', action=action)

        return probes, probing_successful

    def _probing_loop(self,
                      probe: Dict,
                      history: list,
                      turn: int,
                      game: PrivateSharedGame
                      ) -> Tuple[str, str, bool, int]:
        """Perform a probing round until valid response or max attempts."""
        tries = 1
        successful = False
        while tries <= len(self.retries):
            # pose probing question
            question = self._get_probe_content(probe['question'], tries)
            history[-1]['content'] = question
            action = {'type': 'probe question', 'content': question}
            self.log_event(from_='GM', to='Player 1', action=action)
            # get reply
            prompt, raw_answer, answer = game.answerer(history, turn)
            action = {'type': 'probe answer', 'content': answer}
            self.log_event(from_='Player 1', to='GM', action=action,
                           call=(prompt, raw_answer))
            self.request_counts[turn] += 1
            # parse the response
            parsed_response = self._parse_probing_response(answer)
            action = {'type': 'parse', 'content': parsed_response}
            self.log_event(from_='GM', to='GM', action=action)
            # check if valid response, otherwise try again
            if parsed_response in (self.yes, self.no):
                successful = True
                self.parsed_request_counts[turn] += 1
                break
            self.violated_request_counts[turn] += 1
            tries += 1

        return answer, parsed_response, successful, tries

    def _get_probe_content(self, question: str, tries: int) -> str:
        """Build probing question."""
        return question[:].replace(self.me, f'{self.me}{self.retries[tries - 1]} ')

    def _log_probing_outcome(self, probe: Dict, successful: bool, tries: int):
        if not successful:
            content = NOT_SUCCESS.format(probe['target'])
        else:
            content = SUCCESS.format(probe['target'], tries)
        # answer valid?
        action = {'type': 'metadata', 'content': content}
        self.log_event(from_='GM', to='GM', action=action)
        logger.info(content)
        # answer correct?
        result = '' if probe['value'] == probe['gt'] else 'in'
        action = {'type': 'check', 'content': RESULT.format(result)}
        self.log_event(from_='GM', to='GM', action=action)

    @staticmethod
    def _filter_tag(answer: str, tag: str) -> str:
        """Remove a tag from a utterance."""
        filtered = answer.replace(tag, '')
        return filtered.strip()

    def _parse_probing_response(self, response: str) -> str:
        """Extract parsed answer in probing turn."""
        if (not response.startswith(self.aside.strip())
            or self._has_continuation(response)):
            return INVALID
        clean_response = self._filter_tag(response, self.aside.strip())
        if clean_response.lower().startswith(self.yes):
            return self.yes
        if clean_response.lower().startswith(self.no):
            return self.no
        logger.warning(NOT_PARSED)
        return INVALID

    def _convert_response(self, response: str) -> bool:
        """Turn probing response into integer (0: negation, 1: affirmation)."""
        if response == self.yes:
            return 1
        if response == self.no:
            return 0
        return INVALID_LABEL

    @classmethod
    def applies_to(cls, game_name: str) -> bool:
        return game_name == GAME_NAME


class PrivateSharedScorer(GameScorer):

    def __init__(self, experiment: Dict, game_instance: Dict):
        super().__init__(GAME_NAME, experiment, game_instance)
        self.slots = game_instance["slots"]

    def compute_scores(self, episode_interactions: Dict) -> None:
        logs = episode_interactions
        gold = []
        pred = []
        aborted = logs['Aborted']
        for turn in range(logs['Played Probe Rounds']):
            turn_gt, turn_pred = self._compute_turn_scores(logs, turn)
            gold += turn_gt
            pred += turn_pred
            # n_probes = n_slots + 1, we take the third round of probing here
            if turn == int((len(self.slots) + 1) / 2) - 1:
                mid_acc = acc_score(turn_gt, turn_pred) if not aborted else np.nan
                self.log_episode_score('Middle-Accuracy', mid_acc)

        self._compute_episode_scores(gold, pred, logs, aborted)

    def _compute_turn_scores(self, logs: Dict, turn: int) -> Tuple[List, List]:
        """Compute and log turn-level scores."""
        # common scores
        reqs = logs[ms.METRIC_REQUEST_COUNT][turn]
        p_reqs = logs[ms.METRIC_REQUEST_COUNT_PARSED][turn]
        v_reqs = logs[ms.METRIC_REQUEST_COUNT_VIOLATED][turn]

        self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT, reqs)
        self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_PARSED, p_reqs)
        self.log_turn_score(turn, ms.METRIC_REQUEST_COUNT_VIOLATED, v_reqs)

        # specific scores
        turn_gt, turn_pred = self._get_gold_pred(logs['probes'][turn])
        acc = acc_score(turn_gt, turn_pred)
        self.log_turn_score(turn, 'Accuracy', acc)
        if turn != 0:
            # no slot in the first probing round
            # -1 because the probing turn ids are shifted one step
            filled = int(logs['Filled Slots'][turn - 1])
            self.log_turn_score(turn, 'Slot Filled?', filled)

        return turn_gt, turn_pred

    def _compute_episode_scores(self,
                                gold: List,
                                pred: List,
                                logs: Dict,
                                aborted: bool
                                ) -> None:
        """Compute and log episode-level scores."""
        # specific scores
        acc = acc_score(gold, pred) if not aborted else np.nan
        kappa = cohen_kappa_score(gold, pred) if not aborted else np.nan
        # we truncate kappa to be between 0 and 1
        trunc_kappa = max(0, kappa) if not aborted else np.nan
        filled = logs['Filled Slots']
        sf_acc = sum(filled) / len(filled) if not aborted else np.nan
        bench_score = PrivateSharedScorer.compute_bench_score(sf_acc, trunc_kappa)

        self.log_episode_score('Accuracy', acc)
        self.log_episode_score('Kappa', kappa)
        self.log_episode_score('Truncated Kappa', trunc_kappa)
        self.log_episode_score('Slot-Filling-Accuracy', sf_acc)
        self.log_episode_score(ms.BENCH_SCORE, bench_score)

        # common scores
        success_ratio = int(acc == 1. and sf_acc == 1.) if not aborted else 0
        lose_ratio = int(not success_ratio) if not aborted else 0
        reqs = sum(logs[ms.METRIC_REQUEST_COUNT])
        parsed_reqs = sum(logs[ms.METRIC_REQUEST_COUNT_PARSED])
        violated_reqs = sum(logs[ms.METRIC_REQUEST_COUNT_VIOLATED])

        self.log_episode_score(ms.METRIC_ABORTED, int(aborted))
        self.log_episode_score(ms.METRIC_LOSE, lose_ratio)
        self.log_episode_score(ms.METRIC_SUCCESS, success_ratio)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT, reqs)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_PARSED, parsed_reqs)
        self.log_episode_score(ms.METRIC_REQUEST_COUNT_VIOLATED, violated_reqs)
        self.log_episode_score(ms.METRIC_REQUEST_SUCCESS, parsed_reqs / reqs)

    def _get_gold_pred(self, turns: List) -> Tuple[List, List]:
        """Retrieve the gold standard and the predictions for all turns."""
        gold, pred = zip(*[(item['gt'], item['value']) for item in turns])
        return gold, pred

    @staticmethod
    def compute_bench_score(sf_acc: float, kappa: float) -> float:
        """Compute the preferred score in [0, 100] for the benchmark."""
        if np.isnan(sf_acc) or np.isnan(kappa):
            return np.nan
        if sf_acc + kappa == 0:
            return 0
        # harmonic mean between accuracy and truncated kappa
        # normalised to 0-100
        return 100 * (2 * sf_acc * kappa / (sf_acc + kappa))


class PrivateSharedGameBenchmark(GameBenchmark):
    """Integrate the game into the benchmark run."""
    def __init__(self):
        super().__init__(GAME_NAME)

    def is_single_player(self):
        return True

    def get_description(self):
        return "Questioner and answerer in scorekeeping game."

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return PrivateShared(experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return PrivateSharedScorer(experiment, game_instance)

def main():
    """Play the first episode in the instances."""
    instances = file_utils.load_json("in/instances.json", "privateshared")
    experiment = instances["experiments"][0]
    instance = experiment["game_instances"][0]
    master = PrivateShared(experiment, ["dry_run"])
    master.setup(**instance)
    master.play()


if __name__ == '__main__':
    main()
