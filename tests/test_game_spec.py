import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from clemcore.clemgame.registry import GameSpec


class GameSpecTestCase(unittest.TestCase):

    def test_required_fields_game_name(self):
        """Test that game_name is required when not underspecified."""
        with self.assertRaises(KeyError) as context:
            GameSpec(game_path="/path", players=1)
        self.assertIn("game name", str(context.exception).lower())

    def test_required_fields_game_path(self):
        """Test that game_path is required when not underspecified."""
        with self.assertRaises(KeyError) as context:
            GameSpec(game_name="test_game", players=1)
        self.assertIn("game path", str(context.exception).lower())

    def test_required_fields_players(self):
        """Test that players is required when not underspecified."""
        with self.assertRaises(KeyError) as context:
            GameSpec(game_name="test_game", game_path="/path")
        self.assertIn("players", str(context.exception).lower())

    def test_allow_underspecified(self):
        """Test that underspecified GameSpec can be created."""
        spec = GameSpec(allow_underspecified=True)
        self.assertIsInstance(spec, GameSpec)

    def test_dict_like_get(self):
        """Test dict-like access to GameSpec attributes."""
        spec = GameSpec(game_name="test", game_path="/path", players=2)
        self.assertEqual(spec["game_name"], "test")
        self.assertEqual(spec["game_path"], "/path")
        self.assertEqual(spec["players"], 2)

    def test_dict_like_contains(self):
        """Test dict-like containment check."""
        spec = GameSpec(game_name="test", game_path="/path", players=2)
        self.assertTrue("game_name" in spec)
        self.assertTrue("game_path" in spec)
        self.assertFalse("nonexistent" in spec)

    def test_is_single_player(self):
        """Test single player detection."""
        spec = GameSpec(game_name="test", game_path="/path", players=1)
        self.assertTrue(spec.is_single_player())
        self.assertFalse(spec.is_multi_player())

    def test_is_multi_player(self):
        """Test multi-player detection."""
        spec = GameSpec(game_name="test", game_path="/path", players=2)
        self.assertTrue(spec.is_multi_player())
        self.assertFalse(spec.is_single_player())

    def test_from_name(self):
        """Test creating GameSpec from name only."""
        spec = GameSpec.from_name("my_game")
        self.assertEqual(spec.game_name, "my_game")
        # Should be underspecified (no path or players required)
        self.assertFalse("game_path" in spec)

    def test_from_string_json(self):
        """Test creating GameSpec from JSON string."""
        json_str = '{"game_name": "test_game", "players": 2}'
        spec = GameSpec.from_string(json_str)
        self.assertEqual(spec.game_name, "test_game")
        self.assertEqual(spec.players, 2)

    def test_from_string_simple_name(self):
        """Test creating GameSpec from simple name string."""
        spec = GameSpec.from_string("simple_game")
        self.assertEqual(spec.game_name, "simple_game")

    def test_from_dict(self):
        """Test creating GameSpec from dictionary."""
        data = {"game_name": "dict_game", "game_path": "/path", "players": 1, "extra": "value"}
        spec = GameSpec.from_dict(data)
        self.assertEqual(spec.game_name, "dict_game")
        self.assertEqual(spec.extra, "value")

    def test_to_string(self):
        """Test JSON serialization."""
        spec = GameSpec(game_name="test", game_path="/path", players=1)
        json_str = spec.to_string()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["game_name"], "test")

    def test_to_pretty_string(self):
        """Test pretty JSON serialization."""
        spec = GameSpec(game_name="test", game_path="/path", players=1)
        pretty_str = spec.to_pretty_string()
        self.assertIn("\n", pretty_str)  # Should have newlines for formatting

    def test_matches_string_attribute(self):
        """Test matching with string attributes."""
        spec = GameSpec(game_name="test", game_path="/path", players=1, category="puzzle")
        self.assertTrue(spec.matches({"category": "puzzle"}))
        self.assertFalse(spec.matches({"category": "action"}))

    def test_matches_list_attribute(self):
        """Test matching with list attributes."""
        spec = GameSpec(game_name="test", game_path="/path", players=1, tags=["easy", "fun"])
        self.assertTrue(spec.matches({"tags": "easy"}))
        self.assertTrue(spec.matches({"tags": "fun"}))
        self.assertFalse(spec.matches({"tags": "hard"}))

    def test_matches_raises_on_missing_key(self):
        """Test that matching raises KeyError for missing attributes."""
        spec = GameSpec(game_name="test", game_path="/path", players=1)
        with self.assertRaises(KeyError):
            spec.matches({"nonexistent": "value"})

    def test_deepcopy(self):
        """Test that GameSpec can be deep copied."""
        spec = GameSpec(game_name="test", game_path="/path", players=1, nested={"a": 1})
        spec_copy = copy.deepcopy(spec)
        self.assertEqual(spec_copy.game_name, "test")
        self.assertEqual(spec_copy.nested, {"a": 1})
        # Ensure it's a true copy
        spec_copy.nested["a"] = 2
        self.assertEqual(spec.nested["a"], 1)

    def test_repr(self):
        """Test string representation."""
        spec = GameSpec(game_name="test", game_path="/path", players=1)
        repr_str = repr(spec)
        self.assertIn("GameSpec", repr_str)
        self.assertIn("test", repr_str)

    def test_get_game_file(self):
        """Test getting game file path."""
        game_path = str(Path("/some/path"))
        spec = GameSpec(game_name="test", game_path=game_path, players=1)
        game_file = spec.get_game_file()
        expected = str(Path("/some/path") / "master.py")
        self.assertEqual(game_file, expected)

    def test_unify_compatible_specs(self):
        """Test unification of compatible GameSpecs."""
        spec1 = GameSpec(game_name="test", allow_underspecified=True)
        spec2 = GameSpec(game_name="test", game_path="/path", players=1)
        unified = spec1.unify(spec2)
        self.assertEqual(unified.game_name, "test")
        self.assertEqual(unified.game_path, "/path")
        self.assertEqual(unified.players, 1)

    def test_unify_incompatible_specs_raises(self):
        """Test that unification of incompatible specs raises ValueError."""
        spec1 = GameSpec(game_name="game_a", allow_underspecified=True)
        spec2 = GameSpec(game_name="game_b", allow_underspecified=True)
        with self.assertRaises(ValueError):
            spec1.unify(spec2)

    def test_from_directory(self):
        """Test loading GameSpec from directory with clemgame.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create clemgame.json
            clemgame_data = {"game_name": "dir_game", "players": 1}
            with open(os.path.join(tmpdir, "clemgame.json"), "w") as f:
                json.dump(clemgame_data, f)

            specs = GameSpec.from_directory(tmpdir)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].game_name, "dir_game")
            self.assertEqual(specs[0].game_path, tmpdir)

    def test_from_directory_list(self):
        """Test loading multiple GameSpecs from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create clemgame.json with list of specs
            clemgame_data = [
                {"game_name": "game_a", "players": 1},
                {"game_name": "game_b", "players": 2}
            ]
            with open(os.path.join(tmpdir, "clemgame.json"), "w") as f:
                json.dump(clemgame_data, f)

            specs = GameSpec.from_directory(tmpdir)
            self.assertEqual(len(specs), 2)
            self.assertEqual(specs[0].game_name, "game_a")
            self.assertEqual(specs[1].game_name, "game_b")


if __name__ == '__main__':
    unittest.main()
