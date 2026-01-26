import unittest
from unittest.mock import MagicMock

from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv


class GameMasterEnvStepRewardsTestCase(unittest.TestCase):
    """Tests for GameMasterEnv.step() reward handling with None values."""

    def _create_env_for_step(self):
        """Helper to create GameMasterEnv ready for step() testing."""
        mock_benchmark = MagicMock()
        mock_benchmark.game_spec.game_name = "test_game"

        mock_player = MagicMock()
        mock_player.name = "Player 1"

        mock_game_master = MagicMock()
        mock_game_master.current_player = mock_player
        mock_game_master.get_context_for.return_value = {"role": "user", "content": "test"}

        env = GameMasterEnv(mock_benchmark)
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

    def test_turn_score_none_defaults_to_zero(self):
        """When turn_score is explicitly None, reward defaults to 0 (no TypeError)."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (False, {"turn_score": None})

        env.step("action")  # Should not raise TypeError in _accumulate_rewards

        self.assertEqual(env._cumulative_rewards["player_0"], 0.)

    def test_response_score_none_defaults_to_zero(self):
        """When response_score is explicitly None (legacy), reward defaults to 0."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (False, {"response_score": None})

        env.step("action")  # Should not raise TypeError in _accumulate_rewards

        self.assertEqual(env._cumulative_rewards["player_0"], 0.)

    def test_episode_score_none_defaults_to_zero(self):
        """When episode_score is explicitly None on done, reward defaults to 0."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (True, {"episode_score": None})

        env.step("action")  # Should not raise TypeError in _accumulate_rewards

        self.assertEqual(env._cumulative_rewards["player_0"], 0.)

    def test_valid_turn_score_preserved(self):
        """When turn_score has a valid value, it is accumulated."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (False, {"turn_score": 0.5})

        env.step("action")

        self.assertEqual(env._cumulative_rewards["player_0"], 0.5)

    def test_valid_episode_score_preserved(self):
        """When episode_score has a valid value on done, it is accumulated."""
        env, mock_gm = self._create_env_for_step()
        mock_gm.step.return_value = (True, {"episode_score": 1.0})

        env.step("action")

        self.assertEqual(env._cumulative_rewards["player_0"], 1.0)


if __name__ == '__main__':
    unittest.main()
