import copy
import unittest
from typing import Dict
from unittest.mock import MagicMock

from clemcore.clemgame.player import Player
from clemcore.backends import CustomResponseModel, ModelSpec


class MockPlayer(Player):
    """Mock player for testing - named to avoid pytest collection."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.custom_response_value = "custom response"

    def _custom_response(self, context: Dict) -> str:
        return self.custom_response_value


class PlayerInitTestCase(unittest.TestCase):
    """Tests for Player initialization and properties."""

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.name = "test_model"

    def test_init_with_defaults(self):
        player = MockPlayer(self.mock_model)
        self.assertEqual(player.model, self.mock_model)
        self.assertIsNone(player.name)
        self.assertEqual(player.game_role, "MockPlayer")

    def test_init_with_name(self):
        player = MockPlayer(self.mock_model, name="Alice")
        self.assertEqual(player.name, "Alice")

    def test_init_with_game_role(self):
        player = MockPlayer(self.mock_model, game_role="Guesser")
        self.assertEqual(player.game_role, "Guesser")

    def test_name_setter(self):
        player = MockPlayer(self.mock_model)
        player.name = "Bob"
        self.assertEqual(player.name, "Bob")

    def test_get_description(self):
        player = MockPlayer(self.mock_model, name="Alice")
        desc = player.get_description()
        self.assertIn("Alice", desc)
        self.assertIn("MockPlayer", desc)

    def test_get_perspective_initially_empty(self):
        player = MockPlayer(self.mock_model)
        self.assertEqual(player.get_perspective(), [])

    def test_last_context_initially_none(self):
        player = MockPlayer(self.mock_model)
        self.assertIsNone(player.last_context)


class PlayerPerceiveTestCase(unittest.TestCase):
    """Tests for perceive_context and perceive_response."""

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.name = "test_model"
        self.player = MockPlayer(self.mock_model, name="TestPlayer")

    def test_perceive_context_updates_perspective(self):
        context = {"role": "user", "content": "Hello"}
        perspective = self.player.perceive_context(context, log_event=False)

        self.assertEqual(len(perspective), 1)
        self.assertEqual(perspective[0]["content"], "Hello")
        self.assertEqual(self.player.last_context, context)

    def test_perceive_context_requires_user_role(self):
        context = {"role": "assistant", "content": "Hi"}
        with self.assertRaises(AssertionError):
            self.player.perceive_context(context)

    def test_perceive_context_memorize_false(self):
        context = {"role": "user", "content": "Hello"}
        self.player.perceive_context(context, log_event=False, memorize=False)

        # Should not be added to internal messages
        self.assertEqual(self.player.get_perspective(), [])

    def test_perceive_context_forget_extras(self):
        player = MockPlayer(self.mock_model, forget_extras=["image"])
        context = {"role": "user", "content": "Look", "image": "base64data"}

        player.perceive_context(context, log_event=False)

        # Image should be removed from memorized context
        self.assertNotIn("image", player.get_perspective()[0])
        self.assertEqual(player.get_perspective()[0]["content"], "Look")

    def test_perceive_response_updates_perspective(self):
        # First add a context
        self.player.perceive_context({"role": "user", "content": "Hi"}, log_event=False)

        # Then perceive response
        self.player.perceive_response("Hello!", log_event=False)

        perspective = self.player.get_perspective()
        self.assertEqual(len(perspective), 2)
        self.assertEqual(perspective[1]["role"], "assistant")
        self.assertEqual(perspective[1]["content"], "Hello!")

    def test_perceive_response_memorize_false(self):
        self.player.perceive_response("Hello!", log_event=False, memorize=False)
        self.assertEqual(self.player.get_perspective(), [])


class PlayerDeepcopyTestCase(unittest.TestCase):
    """Tests for Player deepcopy behavior."""

    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.name = "test_model"

    def test_deepcopy_preserves_model_reference(self):
        player = MockPlayer(self.mock_model, name="Original")
        player.perceive_context({"role": "user", "content": "Hi"}, log_event=False)

        player_copy = copy.deepcopy(player)

        # Model should be same reference (not copied)
        self.assertIs(player_copy.model, player.model)

    def test_deepcopy_copies_messages(self):
        player = MockPlayer(self.mock_model, name="Original")
        player.perceive_context({"role": "user", "content": "Hi"}, log_event=False)

        player_copy = copy.deepcopy(player)

        # Messages should be independent
        player_copy.perceive_context({"role": "user", "content": "Bye"}, log_event=False)
        self.assertEqual(len(player.get_perspective()), 1)
        self.assertEqual(len(player_copy.get_perspective()), 2)

    def test_deepcopy_copies_name(self):
        player = MockPlayer(self.mock_model, name="Original")
        player_copy = copy.deepcopy(player)

        player_copy.name = "Copy"
        self.assertEqual(player.name, "Original")
        self.assertEqual(player_copy.name, "Copy")


class PlayerResetTestCase(unittest.TestCase):
    """Tests for Player reset."""

    def test_reset_calls_model_reset(self):
        mock_model = MagicMock()
        player = MockPlayer(mock_model)

        player.reset()

        mock_model.reset.assert_called_once()


class BatchResponseTestCase(unittest.TestCase):
    """Tests for Player.batch_response static method."""

    def _make_player_with_mock(self, model_name, responses):
        """Helper to create player with mocked model."""
        model = MagicMock()
        model.name = model_name
        model.generate_batch_response.return_value = responses
        return MockPlayer(model), model

    def test_basic_batch_response(self):
        player1, model1 = self._make_player_with_mock(
            "model_a", [({"prompt": "p1"}, {}, "resp1")]
        )
        player2, model2 = self._make_player_with_mock(
            "model_b", [({"prompt": "p2"}, {}, "resp2")]
        )

        result = Player.batch_response(
            [player1, player2],
            [{"role": "user", "content": "ctx1"}, {"role": "user", "content": "ctx2"}],
            row_ids=[10, 20],
        )

        self.assertEqual(result[10], ({"role": "user", "content": "ctx1"}, "resp1"))
        self.assertEqual(result[20], ({"role": "user", "content": "ctx2"}, "resp2"))
        model1.generate_batch_response.assert_called_once()
        model2.generate_batch_response.assert_called_once()

    def test_auto_row_ids(self):
        player, _ = self._make_player_with_mock(
            "model", [([{"role": "user", "content": "ctx"}], {}, "resp")]
        )

        result = Player.batch_response(
            [player],
            [{"role": "user", "content": "ctx"}],
        )

        self.assertIn(0, result)
        self.assertEqual(result[0][1], "resp")

    def test_shared_model_batched_together(self):
        model = MagicMock()
        model.name = "shared_model"
        model.generate_batch_response.return_value = [
            ([{"role": "user", "content": "ctx1"}], {}, "resp1"),
            ([{"role": "user", "content": "ctx2"}], {}, "resp2"),
        ]

        player1 = MockPlayer(model)
        player2 = MockPlayer(model)

        result = Player.batch_response(
            [player1, player2],
            [{"role": "user", "content": "ctx1"}, {"role": "user", "content": "ctx2"}],
            row_ids=[1, 2],
        )

        # Model should only be called once (batched)
        model.generate_batch_response.assert_called_once()
        self.assertEqual(result[1][1], "resp1")
        self.assertEqual(result[2][1], "resp2")

    def test_mismatched_players_contexts_raises(self):
        player, _ = self._make_player_with_mock("model", [])

        with self.assertRaises(AssertionError) as cm:
            Player.batch_response([player], [], row_ids=[1])
        self.assertIn("same length", str(cm.exception))

    def test_mismatched_row_ids_raises(self):
        player, _ = self._make_player_with_mock("model", [])

        with self.assertRaises(AssertionError) as cm:
            Player.batch_response(
                [player],
                [{"role": "user", "content": "x"}],
                row_ids=[1, 2],  # Too many
            )
        self.assertIn("same length", str(cm.exception))

    def test_missing_generate_batch_response_raises(self):
        model = MagicMock(spec=[])  # No generate_batch_response
        model.name = "bad_model"
        player = MockPlayer(model)

        with self.assertRaises(AssertionError) as cm:
            Player.batch_response(
                [player],
                [{"role": "user", "content": "x"}],
                row_ids=[1],
            )
        self.assertIn("generate_batch_response", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
