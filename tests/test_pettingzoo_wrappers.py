import unittest
from unittest.mock import MagicMock

from clemcore.backends.model_registry import CustomResponseModel, ModelSpec
from clemcore.clemgame.envs.pettingzoo.wrappers import (
    AgentControlWrapper,
    SinglePlayerWrapper,
    GameInstanceIteratorWrapper,
    order_agent_mapping_by_agent_id,
)
from clemcore.clemgame.envs.pettingzoo.master import GameMasterEnv
from clemcore.clemgame.instances import GameInstances, to_rows


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


class GameInstanceIteratorWrapperTestCase(unittest.TestCase):

    def setUp(self):
        self.mock_env = MagicMock()
        instances_data = {
            "experiments": [{
                "name": "exp1",
                "game_instances": [
                    {"game_id": 1},
                    {"game_id": 2},
                    {"game_id": 3},
                ]
            }]
        }
        self.instances = GameInstances("test_game", to_rows("test_game", instances_data))

    def _last_reset_options(self):
        return self.mock_env.reset.call_args[1]["options"]

    def test_sequential_iteration(self):
        """reset() without game_id advances through instances in order."""
        wrapper = GameInstanceIteratorWrapper(self.mock_env, self.instances, single_pass=True)
        wrapper.reset()
        self.assertEqual(self._last_reset_options()["game_instance"]["game_id"], 1)
        wrapper.reset()
        self.assertEqual(self._last_reset_options()["game_instance"]["game_id"], 2)

    def test_single_pass_raises_stop_iteration(self):
        """Single-pass iterator raises StopIteration after all instances are exhausted."""
        instances = GameInstances("test_game", to_rows("test_game", {
            "experiments": [{"name": "exp1", "game_instances": [{"game_id": 1}]}]
        }))
        wrapper = GameInstanceIteratorWrapper(self.mock_env, instances, single_pass=True)
        wrapper.reset()
        with self.assertRaises(StopIteration):
            wrapper.reset()

    def test_cycling(self):
        """Default (non-single-pass) iterator cycles infinitely."""
        instances = GameInstances("test_game", to_rows("test_game", {
            "experiments": [{"name": "exp1", "game_instances": [{"game_id": 1}]}]
        }))
        wrapper = GameInstanceIteratorWrapper(self.mock_env, instances, single_pass=False)
        wrapper.reset()
        wrapper.reset()  # cycles back to start
        self.assertEqual(self._last_reset_options()["game_instance"]["game_id"], 1)

    def test_game_id_random_access(self):
        """reset(options={"game_id": N}) selects the specific episode."""
        wrapper = GameInstanceIteratorWrapper(self.mock_env, self.instances)
        wrapper.reset(options={"game_id": 3})
        self.assertEqual(self._last_reset_options()["game_instance"]["game_id"], 3)

    def test_game_id_does_not_advance_iterator(self):
        """Random access via game_id leaves the iterator position unchanged."""
        wrapper = GameInstanceIteratorWrapper(self.mock_env, self.instances, single_pass=True)
        wrapper.reset(options={"game_id": 3})  # random access, iterator not advanced
        wrapper.reset()  # should still get first instance from iterator
        self.assertEqual(self._last_reset_options()["game_instance"]["game_id"], 1)

    def test_game_id_not_forwarded_to_env(self):
        """game_id is consumed by the wrapper and not passed to the underlying env."""
        wrapper = GameInstanceIteratorWrapper(self.mock_env, self.instances)
        wrapper.reset(options={"game_id": 1})
        self.assertNotIn("game_id", self._last_reset_options())


if __name__ == '__main__':
    unittest.main()
