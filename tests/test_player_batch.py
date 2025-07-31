import unittest
from typing import Dict
from unittest.mock import MagicMock

from clemcore.clemgame.player import Player


class TestPlayer(Player):

    def _custom_response(self, context: Dict) -> str:
        pass


def make_player_with_mock_model(model_name, prompt_str, response_text, response_obj=None):
    model = MagicMock()
    model.name = model_name
    model.generate_batch_response.return_value = [
        ({"prompt": prompt_str}, response_obj or {}, response_text)
    ]

    player = TestPlayer(model)
    return player, model


class TestBatchResponse(unittest.TestCase):
    def test_batch_response_basic(self):
        player1, model1 = make_player_with_mock_model("model_a", "ctx1", "resp1", {"m": 1})
        player2, model2 = make_player_with_mock_model("model_b", "ctx2", "resp2", {"m": 2})

        players = [player1, player2]
        contexts = [
            {"role": "user", "content": "ctx1"},
            {"role": "user", "content": "ctx2"},
        ]
        row_ids = [10, 20]

        result = Player.batch_response(players, contexts, row_ids=row_ids)

        self.assertEqual(result, {10: "resp1", 20: "resp2"})
        model1.generate_batch_response.assert_called_once()
        model2.generate_batch_response.assert_called_once()

    def test_batch_response_mismatched_lengths(self):
        player, _ = make_player_with_mock_model("model_x", "ctx", "resp")
        players = [player]
        contexts = []  # Mismatch
        row_ids = [1]

        with self.assertRaises(AssertionError) as cm:
            Player.batch_response(players, contexts, row_ids=row_ids)
        self.assertIn("`players` and `contexts` must have the same length", str(cm.exception))

    def test_batch_response_missing_generate_batch_response(self):
        model = MagicMock()
        model.name = "bad_model"
        if hasattr(model, "generate_batch_response"):
            delattr(model, "generate_batch_response")

        player = TestPlayer(model)

        with self.assertRaises(AssertionError) as cm:
            Player.batch_response([player], [{"role": "user", "content": "x"}], row_ids=[1])
        self.assertIn("does not implement `generate_batch_response()`", str(cm.exception))

    def test_batch_response_shared_model_batched(self):
        model = MagicMock()
        model.name = "shared_model"
        model.generate_batch_response.return_value = [
            ([{"role": "user", "content": "ctx1"}], {}, "resp1"),
            ([{"role": "user", "content": "ctx2"}], {}, "resp2"),
        ]

        player1 = TestPlayer(model)
        player2 = TestPlayer(model)

        players = [player1, player2]
        contexts = [
            {"role": "user", "content": "ctx1"},
            {"role": "user", "content": "ctx2"},
        ]
        row_ids = [1, 2]

        result = Player.batch_response(players, contexts, row_ids=row_ids)

        self.assertEqual(result, {1: "resp1", 2: "resp2"})
        model.generate_batch_response.assert_called_once()

    def test_batch_response_auto_row_ids(self):
        player1, model1 = make_player_with_mock_model("model_1", "ctx1", "resp1")
        player2, model2 = make_player_with_mock_model("model_2", "ctx2", "resp2")

        players = [player1, player2]
        contexts = [
            {"role": "user", "content": "ctx1"},
            {"role": "user", "content": "ctx2"},
        ]

        result = Player.batch_response(players, contexts)

        self.assertEqual(result, {0: "resp1", 1: "resp2"})

    def test_batch_response_auto_row_ids_used_in_order(self):
        model = MagicMock()
        model.name = "test_model"
        model.generate_batch_response.return_value = [
            ([{"role": "user", "content": "ctx0"}], {}, "resp0"),
            ([{"role": "user", "content": "ctx1"}], {}, "resp1"),
        ]

        player0 = TestPlayer(model)
        player1 = TestPlayer(model)

        players = [player0, player1]
        contexts = [
            {"role": "user", "content": "ctx0"},
            {"role": "user", "content": "ctx1"},
        ]

        result = Player.batch_response(players, contexts)

        self.assertEqual(result, {0: "resp0", 1: "resp1"})
        model.generate_batch_response.assert_called_once()


if __name__ == "__main__":
    unittest.main()
