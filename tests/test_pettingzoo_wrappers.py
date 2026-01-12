import unittest
from unittest.mock import MagicMock

from clemcore.backends.model_registry import Model, CustomResponseModel, ModelSpec
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    AgentControlWrapper,
    SinglePlayerWrapper,
    order_agent_mapping_by_agent_id,
    EnvAgent
)


class TestOrderAgentMapping(unittest.TestCase):

    def test_order_agent_mapping_by_agent_id(self):
        mapping = {"player_2": "b", "player_0": "a", "player_1": "c"}
        ordered = order_agent_mapping_by_agent_id(mapping)
        self.assertEqual(list(ordered.keys()), ["player_0", "player_1", "player_2"])


class TestAgentControlWrapperCallable(unittest.TestCase):

    def test_callable_agents_are_stored(self):
        """Test that callable agents are correctly identified and stored."""
        mock_env = MagicMock()

        # Create a callable agent (lambda)
        callable_agent = lambda obs: "action"
        # Create a Model agent
        model_agent = CustomResponseModel(ModelSpec(model_name="test"))

        agent_mapping = {
            "player_0": "learner",
            "player_1": callable_agent,
            "player_2": model_agent
        }

        wrapper = AgentControlWrapper(mock_env, agent_mapping)

        # Verify callable agents are stored
        self.assertIn("player_1", wrapper.callable_agents)
        self.assertEqual(wrapper.callable_agents["player_1"], callable_agent)

        # Verify Model agents are NOT in callable_agents
        self.assertNotIn("player_2", wrapper.callable_agents)

        # Verify learner is NOT in callable_agents
        self.assertNotIn("player_0", wrapper.callable_agents)

    def test_get_env_agent_returns_callable(self):
        """Test that get_env_agent returns the callable directly for callable agents."""
        mock_env = MagicMock()
        mock_env.unwrapped = MagicMock()
        mock_env.unwrapped.player_by_agent_id = {
            "player_0": MagicMock(),
            "player_1": MagicMock()
        }

        callable_agent = lambda obs: f"response to {obs}"

        agent_mapping = {
            "player_0": "learner",
            "player_1": callable_agent
        }

        wrapper = AgentControlWrapper(mock_env, agent_mapping)

        # get_env_agent should return the callable for player_1
        env_agent = wrapper.get_env_agent("player_1")
        self.assertEqual(env_agent, callable_agent)

        # Verify the callable works
        result = env_agent("test observation")
        self.assertEqual(result, "response to test observation")

    def test_get_env_agent_returns_player_for_model(self):
        """Test that get_env_agent returns the Player for Model agents."""
        mock_env = MagicMock()
        mock_player = MagicMock()
        mock_env.unwrapped = MagicMock()
        mock_env.unwrapped.player_by_agent_id = {
            "player_0": MagicMock(),
            "player_1": mock_player
        }

        model_agent = CustomResponseModel(ModelSpec(model_name="test"))

        agent_mapping = {
            "player_0": "learner",
            "player_1": model_agent
        }

        wrapper = AgentControlWrapper(mock_env, agent_mapping)

        # get_env_agent should return the Player from the unwrapped env
        env_agent = wrapper.get_env_agent("player_1")
        self.assertEqual(env_agent, mock_player)


class TestSinglePlayerWrapperCallable(unittest.TestCase):

    def test_single_player_wrapper_accepts_callable(self):
        """Test that SinglePlayerWrapper accepts callable env_agents."""
        mock_env = MagicMock()

        callable_agent = lambda obs: "action"

        # This should not raise an error
        wrapper = SinglePlayerWrapper(
            mock_env,
            learner_agent="player_0",
            env_agents={"player_1": callable_agent}
        )

        # Verify the callable is stored
        self.assertIn("player_1", wrapper.callable_agents)
        self.assertEqual(wrapper.callable_agents["player_1"], callable_agent)

    def test_single_player_wrapper_mixed_agents(self):
        """Test SinglePlayerWrapper with mixed Model and callable agents."""
        mock_env = MagicMock()

        callable_agent = lambda obs: "callable action"
        model_agent = CustomResponseModel(ModelSpec(model_name="test"))

        wrapper = SinglePlayerWrapper(
            mock_env,
            learner_agent="player_0",
            env_agents={
                "player_1": callable_agent,
                "player_2": model_agent
            }
        )

        # Verify callable is stored
        self.assertIn("player_1", wrapper.callable_agents)

        # Verify Model is not in callable_agents
        self.assertNotIn("player_2", wrapper.callable_agents)

        # Verify learner is identified
        self.assertEqual(wrapper.learner_agent, "player_0")
        self.assertIn("player_0", wrapper.learner_agents)


class TestEnvAgentTypeAlias(unittest.TestCase):

    def test_env_agent_type_accepts_model(self):
        """Test that EnvAgent type works with Model."""
        model: EnvAgent = CustomResponseModel(ModelSpec(model_name="test"))
        self.assertIsInstance(model, Model)

    def test_env_agent_type_accepts_callable(self):
        """Test that EnvAgent type works with Callable."""
        func: EnvAgent = lambda obs: "action"
        self.assertTrue(callable(func))


if __name__ == '__main__':
    unittest.main()
