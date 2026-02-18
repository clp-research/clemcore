import unittest

import pytest
from pettingzoo.test import api_test

from clemcore.clemgame import env, gym_env


@pytest.mark.integration
class PettingzooTestCase(unittest.TestCase):
    """PettingZoo API conformance tests.

    Note: Currently fails because we use spaces.Text which doesn't support
    full prompts with whitespaces. Requires custom space implementation to fix.
    """

    def test_api(self):
        api_test(env("taboo"), num_cycles=1000, verbose_progress=False)


@pytest.mark.integration
class WordleRewardFuncTestCase(unittest.TestCase):
    """Integration tests for reward_func with the real Wordle game.

    Requires the Wordle game to be installed (clembench repository on the path).

    Note: Wordle uses a legacy WordleGameState dataclass (success/failure/aborted booleans)
    rather than the new GameState with Outcome enum, so a game-specific reward_func is needed.
    """

    def _wordle_reward(self, observation, action, state, info):
        """Wordle-specific reward: 1 on success, -1 on abort, 0 otherwise."""
        if state.success:
            return 1.
        if state.aborted:
            return -1.
        return 0.

    def _make_guess(self, word):
        """Format a guess response in the Wordle protocol."""
        return f"explanation: test\nguess: {word}"

    def _get_target_word(self, game_env):
        """Peek at the target word from the game master state after reset."""
        return game_env.env.unwrapped.game_master.state.target_word

    def test_correct_guess_yields_reward_one(self):
        """Immediately guessing the target word should give reward 1."""
        game_env = gym_env("wordle", reward_func=self._wordle_reward)
        game_env.reset()
        target_word = self._get_target_word(game_env)

        _, reward, done, _, _ = game_env.step(self._make_guess(target_word))

        self.assertTrue(done)
        self.assertEqual(reward, 1.)
        game_env.close()

    def test_exhaust_turns_yields_reward_zero(self):
        """Exhausting all turns without guessing correctly should give reward 0."""
        game_env = gym_env("wordle", reward_func=self._wordle_reward)
        game_env.reset()
        target_word = self._get_target_word(game_env)
        # Pick a valid word that is definitely not the target
        wrong_word = "apple" if target_word != "apple" else "beach"

        reward, done = 0., False
        while not done:
            _, reward, done, _, _ = game_env.step(self._make_guess(wrong_word))

        self.assertEqual(reward, 0.)
        game_env.close()

    def test_reward_func_receives_wordle_state(self):
        """The state passed to reward_func should be WordleGameState with target_word."""
        states_seen = []

        def recording_reward(observation, action, state, info):
            states_seen.append(state)
            return 0.

        game_env = gym_env("wordle", reward_func=recording_reward)
        game_env.reset()
        target_word = self._get_target_word(game_env)

        game_env.step(self._make_guess(target_word))

        self.assertTrue(len(states_seen) > 0)
        self.assertEqual(states_seen[-1].target_word, target_word)
        self.assertTrue(states_seen[-1].success)
        game_env.close()


if __name__ == '__main__':
    unittest.main()
