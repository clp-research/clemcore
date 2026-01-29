import unittest
from unittest.mock import MagicMock

from clemcore.backends.model_registry import CustomResponseModel, ModelSpec
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    AgentControlWrapper,
    SinglePlayerWrapper,
    order_agent_mapping_by_agent_id,
)
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv


class OrderAgentMappingTestCase(unittest.TestCase):

    def test_orders_by_agent_id(self):
        mapping = {"player_2": "b", "player_0": "a", "player_1": "c"}
        ordered = order_agent_mapping_by_agent_id(mapping)
        self.assertEqual(list(ordered.keys()), ["player_0", "player_1", "player_2"])


class AgentControlWrapperTestCase(unittest.TestCase):

    def setUp(self):
        self.mock_env = MagicMock()
        self.mock_env.unwrapped = MagicMock()
        self.mock_env.unwrapped.player_by_agent_id = {
            "player_0": MagicMock(),
            "player_1": MagicMock(),
            "player_2": MagicMock(),
        }
        self.callable_agent = lambda obs: f"response to {obs}"
        self.model_agent = CustomResponseModel(ModelSpec(model_name="test"))

    def test_callable_agents_stored_separately(self):
        """Callable agents stored in callable_agents, Model agents are not."""
        wrapper = AgentControlWrapper(self.mock_env, {
            "player_0": "learner",
            "player_1": self.callable_agent,
            "player_2": self.model_agent,
        })

        self.assertIn("player_1", wrapper.callable_agents)
        self.assertNotIn("player_0", wrapper.callable_agents)
        self.assertNotIn("player_2", wrapper.callable_agents)

    def test_get_env_agent_returns_callable(self):
        """get_env_agent returns callable directly for callable agents."""
        wrapper = AgentControlWrapper(self.mock_env, {
            "player_0": "learner",
            "player_1": self.callable_agent,
        })

        env_agent = wrapper.get_env_agent("player_1")
        self.assertEqual(env_agent, self.callable_agent)
        self.assertEqual(env_agent("test"), "response to test")

    def test_get_env_agent_returns_player_for_model(self):
        """get_env_agent returns Player from unwrapped env for Model agents."""
        wrapper = AgentControlWrapper(self.mock_env, {
            "player_0": "learner",
            "player_1": self.model_agent,
        })

        env_agent = wrapper.get_env_agent("player_1")
        self.assertEqual(env_agent, self.mock_env.unwrapped.player_by_agent_id["player_1"])


class SinglePlayerWrapperTestCase(unittest.TestCase):

    def setUp(self):
        self.mock_env = MagicMock()
        self.callable_agent = lambda obs: "action"
        self.model_agent = CustomResponseModel(ModelSpec(model_name="test"))

    def test_accepts_callable_env_agent(self):
        wrapper = SinglePlayerWrapper(
            self.mock_env,
            learner_agent="player_0",
            env_agents={"player_1": self.callable_agent},
        )
        self.assertIn("player_1", wrapper.callable_agents)

    def test_mixed_agent_types(self):
        """Handles both callable and Model agents correctly."""
        wrapper = SinglePlayerWrapper(
            self.mock_env,
            learner_agent="player_0",
            env_agents={
                "player_1": self.callable_agent,
                "player_2": self.model_agent,
            },
        )

        self.assertIn("player_1", wrapper.callable_agents)
        self.assertNotIn("player_2", wrapper.callable_agents)
        self.assertEqual(wrapper.learner_agent, "player_0")
        self.assertIn("player_0", wrapper.learner_agents)


class GameMasterEnvObserveTestCase(unittest.TestCase):
    """Tests for GameMasterEnv.observe() edge cases (issue #249)."""

    def _create_env(self, get_context_return_value):
        """Helper to create GameMasterEnv with mocked game_master."""
        mock_benchmark = MagicMock()
        mock_game_master = MagicMock()
        mock_game_master.get_context_for.return_value = get_context_return_value

        mock_player = MagicMock()
        mock_player.name = "Player 1"

        env = GameMasterEnv(mock_benchmark)
        env.game_master = mock_game_master
        env.player_by_agent_id = {"player_0": mock_player}
        return env, mock_game_master, mock_player

    def test_returns_generic_message_when_no_context(self):
        """Returns generic abort message when no context has been set (e.g., early game abort)."""
        env, _, _ = self._create_env(get_context_return_value=None)
        result = env.observe("player_0")
        self.assertEqual(result, {"role": "user", "content": "The game ended before your turn."})

    def test_returns_context_when_available(self):
        """Returns context normally when set."""
        context = {"role": "user", "content": "Your turn!"}
        env, mock_gm, mock_player = self._create_env(get_context_return_value=context)

        result = env.observe("player_0")
        self.assertEqual(result, context)
        mock_gm.get_context_for.assert_called_once_with(mock_player)


if __name__ == '__main__':
    unittest.main()
