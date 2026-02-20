import unittest
from unittest.mock import MagicMock

from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
from clemcore.clemgame.master import GameState, Outcome


class GameMasterEnvStepRewardsTestCase(unittest.TestCase):
    """Tests for GameMasterEnv.step() reward handling."""

    def _create_env_for_step(self, reward_func=None):
        """Helper to create GameMasterEnv ready for step() testing."""
        mock_benchmark = MagicMock()
        mock_benchmark.game_spec.game_name = "test_game"

        mock_player = MagicMock()
        mock_player.name = "Player 1"

        mock_game_master = MagicMock()
        mock_game_master.current_player = mock_player
        mock_game_master.get_context_for.return_value = {"role": "user", "content": "test"}
        mock_game_master.state = GameState()  # real GameState so outcome comparisons work

        env = GameMasterEnv(mock_benchmark, reward_func=reward_func)
        env.game_master = mock_game_master
        env.agents = ["player_0"]
        env.possible_agents = ["player_0"]
        env.player_by_agent_id = {"player_0": mock_player}
        env.player_to_agent_id = {"Player 1": "player_0"}
        env.agent_selection = "player_0"
        env.terminations = {"player_0": False}
        env.truncations = {"player_0": False}
        env.rewards = {"player_0": 0.}
        env._cumulative_rewards = {"player_0": 0.}
        env.infos = {"player_0": {}}

        return env, mock_game_master

    def test_default_reward_running_is_zero(self):
        """Default reward is 0 for non-terminal (RUNNING) steps."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (False, {})
        # state.outcome stays RUNNING (default)

        env.step("action")

        self.assertEqual(env._cumulative_rewards["player_0"], 0.)

    def test_default_reward_success_is_one(self):
        """Default reward is 1 when game ends with SUCCESS."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (True, {})
        mock_gm.state.succeed()

        env.step("action")

        self.assertEqual(env._cumulative_rewards["player_0"], 1.)

    def test_default_reward_failure_is_zero(self):
        """Default reward is 0 when game ends with FAILURE."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (True, {})
        mock_gm.state.failed()

        env.step("action")

        self.assertEqual(env._cumulative_rewards["player_0"], 0.)

    def test_default_reward_aborted_is_minus_one(self):
        """Default reward is -1 when game ends with ABORTED."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (True, {})
        mock_gm.state.abort()

        env.step("action")

        self.assertEqual(env._cumulative_rewards["player_0"], -1.)

    def test_custom_reward_func_is_called(self):
        """A custom reward_func receives (observation, action, state, info) and its return value is used."""
        recorded = {}

        def my_reward(observation, action, state, info):
            recorded["observation"] = observation
            recorded["action"] = action
            recorded["state"] = state
            recorded["info"] = info
            return 0.42

        env, mock_gm = self._create_env_for_step(reward_func=my_reward)
        mock_gm.step.return_value = (False, {"some_key": "val"})

        env.step("my_action")

        self.assertEqual(env._cumulative_rewards["player_0"], 0.42)
        self.assertEqual(recorded["action"], "my_action")
        self.assertIs(recorded["state"], mock_gm.state)
        self.assertEqual(recorded["info"], {"some_key": "val"})

    def test_custom_reward_func_can_read_game_state_fields(self):
        """Custom reward_func can access game-specific state fields (e.g. letter_matches)."""
        class WordleState(GameState):
            def __init__(self):
                super().__init__()
                self.letter_matches = 3
                self.word_length = 5

        def wordle_reward(observation, action, state, info):
            if state.outcome == Outcome.SUCCESS:
                return 1.
            if state.outcome == Outcome.ABORTED:
                return -1.
            return state.letter_matches / state.word_length

        env, mock_gm = self._create_env_for_step(reward_func=wordle_reward)
        mock_gm.state = WordleState()
        mock_gm.step.return_value = (False, {})

        env.step("crane")

        self.assertAlmostEqual(env._cumulative_rewards["player_0"], 0.6)


if __name__ == '__main__':
    unittest.main()
