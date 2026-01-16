import json
import os
import tempfile
import unittest

from clemcore.clemgame.registry import GameSpec, GameRegistry


class GameRegistryTestCase(unittest.TestCase):

    def test_empty_registry(self):
        """Test creating empty game registry."""
        registry = GameRegistry()
        self.assertEqual(len(registry), 0)

    def test_registry_with_specs(self):
        """Test creating registry with initial specs."""
        spec1 = GameSpec(game_name="game1", game_path="/path1", players=1)
        spec2 = GameSpec(game_name="game2", game_path="/path2", players=2)
        registry = GameRegistry([spec1, spec2])
        self.assertEqual(len(registry), 2)

    def test_registry_iter(self):
        """Test iterating over registry."""
        spec1 = GameSpec(game_name="game1", game_path="/path1", players=1)
        spec2 = GameSpec(game_name="game2", game_path="/path2", players=2)
        registry = GameRegistry([spec1, spec2])
        specs = list(registry)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].game_name, "game1")

    def test_get_game_specs(self):
        """Test getting all game specs."""
        spec = GameSpec(game_name="game1", game_path="/path1", players=1)
        registry = GameRegistry([spec])
        specs = registry.get_game_specs()
        self.assertEqual(len(specs), 1)
        self.assertIs(specs[0], spec)

    def test_find_game_spec_found(self):
        """Test finding an existing game spec by name."""
        spec = GameSpec(game_name="my_game", game_path="/path", players=1)
        registry = GameRegistry([spec])
        found = registry.find_game_spec("my_game")
        self.assertIsNotNone(found)
        self.assertEqual(found.game_name, "my_game")

    def test_find_game_spec_not_found(self):
        """Test finding a non-existent game spec."""
        registry = GameRegistry()
        found = registry.find_game_spec("nonexistent")
        self.assertIsNone(found)

    def test_get_game_spec_found(self):
        """Test getting a game spec by name."""
        spec = GameSpec(game_name="my_game", game_path="/path", players=1)
        registry = GameRegistry([spec])
        found = registry.get_game_spec("my_game")
        self.assertEqual(found.game_name, "my_game")

    def test_get_game_spec_not_found_raises(self):
        """Test that getting non-existent spec raises ValueError."""
        registry = GameRegistry()
        with self.assertRaises(ValueError):
            registry.get_game_spec("nonexistent")

    def test_register_from_list(self):
        """Test registering games from a list of dicts."""
        game_list = [
            {"game_name": "game_a", "game_path": "/path/a", "players": 1},
            {"game_name": "game_b", "game_path": "/path/b", "players": 2}
        ]
        registry = GameRegistry().register_from_list(game_list)
        self.assertEqual(len(registry), 2)
        self.assertIsNotNone(registry.find_game_spec("game_a"))
        self.assertIsNotNone(registry.find_game_spec("game_b"))

    def test_register_from_list_with_lookup_source(self):
        """Test that lookup_source is set when registering."""
        game_list = [{"game_name": "game_a", "game_path": "/path/a", "players": 1}]
        registry = GameRegistry().register_from_list(game_list, "test_source")
        spec = registry.find_game_spec("game_a")
        self.assertEqual(spec.lookup_source, "test_source")

    def test_register_from_list_skips_invalid(self):
        """Test that invalid specs are skipped with warning."""
        game_list = [
            {"game_name": "valid", "game_path": "/path", "players": 1},
            {"invalid": "spec"}  # Missing required fields
        ]
        registry = GameRegistry().register_from_list(game_list)
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.find_game_spec("valid").game_name, "valid")

    def test_register_from_directories(self):
        """Test registering games from directories with clemgame.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a game directory with clemgame.json
            game_dir = os.path.join(tmpdir, "test_game")
            os.makedirs(game_dir)
            clemgame_data = {"game_name": "dir_game", "players": 1}
            with open(os.path.join(game_dir, "clemgame.json"), "w") as f:
                json.dump(clemgame_data, f)

            registry = GameRegistry()
            registry.register_from_directories(tmpdir, 0, max_depth=2)
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry.find_game_spec("dir_game").game_name, "dir_game")

    def test_register_from_directories_skips_hidden(self):
        """Test that hidden directories are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a hidden game directory
            hidden_dir = os.path.join(tmpdir, ".hidden_game")
            os.makedirs(hidden_dir)
            clemgame_data = {"game_name": "hidden", "players": 1}
            with open(os.path.join(hidden_dir, "clemgame.json"), "w") as f:
                json.dump(clemgame_data, f)

            registry = GameRegistry()
            registry.register_from_directories(tmpdir, 0, max_depth=2)
            self.assertEqual(len(registry), 0)

    def test_register_from_directories_respects_max_depth(self):
        """Test that max_depth is respected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create deeply nested game directory
            deep_dir = os.path.join(tmpdir, "a", "b", "c", "d", "e", "game")
            os.makedirs(deep_dir)
            clemgame_data = {"game_name": "deep_game", "players": 1}
            with open(os.path.join(deep_dir, "clemgame.json"), "w") as f:
                json.dump(clemgame_data, f)

            registry = GameRegistry()
            registry.register_from_directories(tmpdir, 0, max_depth=2)
            # Should not find deeply nested game
            self.assertEqual(len(registry), 0)

    def test_get_game_specs_that_unify_with_name(self):
        """Test selecting games by simple name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create game with master.py
            game_path = os.path.join(tmpdir, "my_game")
            os.makedirs(game_path)
            with open(os.path.join(game_path, "master.py"), "w") as f:
                f.write("# game master")

            spec = GameSpec(game_name="my_game", game_path=game_path, players=1)
            registry = GameRegistry([spec])

            selected = registry.get_game_specs_that_unify_with("my_game", verbose=False)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].game_name, "my_game")

    def test_get_game_specs_that_unify_with_all(self):
        """Test selecting all games."""
        spec1 = GameSpec(game_name="game1", game_path="/path1", players=1)
        spec2 = GameSpec(game_name="game2", game_path="/path2", players=2)
        registry = GameRegistry([spec1, spec2])

        selected = registry.get_game_specs_that_unify_with("all", verbose=False)
        self.assertEqual(len(selected), 2)

    def test_get_game_specs_that_unify_with_json(self):
        """Test selecting games by JSON spec."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create game with master.py
            game_path = os.path.join(tmpdir, "my_game")
            os.makedirs(game_path)
            with open(os.path.join(game_path, "master.py"), "w") as f:
                f.write("# game master")

            spec = GameSpec(game_name="my_game", game_path=game_path, players=1, category="puzzle")
            registry = GameRegistry([spec])

            selected = registry.get_game_specs_that_unify_with('{"category": "puzzle"}', verbose=False)
            self.assertEqual(len(selected), 1)

    def test_get_game_specs_that_unify_with_not_found(self):
        """Test that ValueError is raised when no games match."""
        registry = GameRegistry()
        with self.assertRaises(ValueError):
            registry.get_game_specs_that_unify_with("nonexistent", verbose=False)


if __name__ == '__main__':
    unittest.main()
