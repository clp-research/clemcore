import json
import os
import tempfile
import unittest

from clemcore.clemgame.instances import GameInstances, to_instance_filter, to_rows


def make_row(game_name, experiment_name, game_id):
    return {"game_name": game_name, "experiment": {"name": experiment_name}, "game_instance": {"game_id": game_id}}


class ToInstanceFilterTestCase(unittest.TestCase):

    def test_filter_from_dataset(self):
        """Test creating filter condition from dataset."""
        dataset = [
            {"game": "game_a", "experiment": "exp1", "task_id": "1"},
            {"game": "game_a", "experiment": "exp1", "task_id": "2"},
            {"game": "game_a", "experiment": "exp2", "task_id": "3"},
            {"game": "game_b", "experiment": "exp1", "task_id": "1"},
        ]
        filter_fn = to_instance_filter(dataset)

        self.assertTrue(filter_fn(make_row("game_a", "exp1", 1)))
        self.assertTrue(filter_fn(make_row("game_a", "exp1", 2)))
        self.assertFalse(filter_fn(make_row("game_a", "exp1", 3)))
        self.assertTrue(filter_fn(make_row("game_a", "exp2", 3)))
        self.assertTrue(filter_fn(make_row("game_b", "exp1", 1)))

    def test_filter_returns_false_for_missing(self):
        """Test that filter returns False for missing game/experiment/task_id."""
        dataset = [{"game": "game_a", "experiment": "exp1", "task_id": "1"}]
        filter_fn = to_instance_filter(dataset)

        self.assertFalse(filter_fn(make_row("nonexistent", "exp1", 1)))
        self.assertFalse(filter_fn(make_row("game_a", "exp1", 99)))


class GameInstancesTestCase(unittest.TestCase):

    def get_sample_instances(self):
        """Get sample instances for testing."""
        return {
            "experiments": [
                {
                    "name": "experiment_1",
                    "param": "value1",
                    "game_instances": [
                        {"game_id": 1, "prompt": "Hello 1"},
                        {"game_id": 2, "prompt": "Hello 2"},
                    ]
                },
                {
                    "name": "experiment_2",
                    "param": "value2",
                    "game_instances": [
                        {"game_id": 3, "prompt": "Hello 3"},
                    ]
                }
            ]
        }

    def test_creation(self):
        """Test creating GameInstances."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)
        self.assertEqual(len(instances), 3)  # 2 + 1 instances

    def test_iteration(self):
        """Test iterating over instances yields row dicts."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        collected = list(instances)
        self.assertEqual(len(collected), 3)

        # Check structure: each item is a row dict
        row = collected[0]
        self.assertEqual(row["experiment"]["name"], "experiment_1")
        self.assertEqual(row["game_instance"]["game_id"], 1)

    def test_experiment_excludes_game_instances(self):
        """Test that experiment dict excludes game_instances."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        row = next(iter(instances))
        self.assertNotIn("game_instances", row["experiment"])
        self.assertIn("name", row["experiment"])
        self.assertIn("param", row["experiment"])

    def test_filter(self):
        """Test filter returns a new GameInstances with matching rows."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        filtered = instances.filter(
            lambda row: row["experiment"]["name"] == "experiment_1" and row["game_instance"]["game_id"] == 1
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(next(iter(filtered))["game_instance"]["game_id"], 1)

    def test_filter_experiment(self):
        """Test filtering to a single experiment."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        filtered = instances.filter(lambda row: row["experiment"]["name"] == "experiment_2")
        self.assertEqual(len(filtered), 1)
        self.assertEqual(next(iter(filtered))["experiment"]["name"], "experiment_2")

    def test_find_by_game_id(self):
        """Test finding a row by game_id."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        row = instances.find_by_game_id(2)
        self.assertEqual(row["game_instance"]["game_id"], 2)
        self.assertEqual(row["experiment"]["name"], "experiment_1")

    def test_find_by_game_id_accepts_string(self):
        """Test that find_by_game_id coerces string game_id to int (e.g. from HTTP callers)."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        row = instances.find_by_game_id("2")
        self.assertEqual(row["game_instance"]["game_id"], 2)

    def test_find_by_game_id_not_found(self):
        """Test that find_by_game_id raises ValueError for missing id."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)

        with self.assertRaises(ValueError):
            instances.find_by_game_id(999)

    def test_from_file(self):
        """Test loading instances from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            instances_data = self.get_sample_instances()
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump(instances_data, f)

            instances = GameInstances.from_file("test_game", tmpdir)
            self.assertEqual(len(instances), 3)

    def test_from_file_custom_name(self):
        """Test loading instances from file with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            instances_data = self.get_sample_instances()
            with open(os.path.join(tmpdir, "custom_instances.json"), "w") as f:
                json.dump(instances_data, f)

            instances = GameInstances.from_file("test_game", tmpdir, "custom_instances")
            self.assertEqual(len(instances), 3)

    def test_from_file_missing_experiments_raises(self):
        """Test that missing experiments key raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump({"not_experiments": []}, f)

            with self.assertRaises(ValueError) as context:
                GameInstances.from_file("test_game", tmpdir)
            self.assertIn("experiments", str(context.exception).lower())

    def test_from_file_experiments_not_list_raises(self):
        """Test that non-list experiments raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump({"experiments": "not_a_list"}, f)

            with self.assertRaises(ValueError) as context:
                GameInstances.from_file("test_game", tmpdir)
            self.assertIn("list", str(context.exception).lower())

    def test_from_file_empty_experiments_raises(self):
        """Test that empty experiments list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump({"experiments": []}, f)

            with self.assertRaises(ValueError) as context:
                GameInstances.from_file("test_game", tmpdir)
            self.assertIn("empty", str(context.exception).lower())

    def test_requires_game_name(self):
        """Test that game_name is required."""
        rows = to_rows("test_game", self.get_sample_instances())
        with self.assertRaises(AssertionError):
            GameInstances(None, rows)

    def test_requires_rows(self):
        """Test that rows is required."""
        with self.assertRaises(AssertionError):
            GameInstances("game", None)

    def test_str(self):
        """Test string representation."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)
        s = str(instances)
        self.assertIn("test_game", s)
        self.assertIn("2", s)  # 2 experiments
        self.assertIn("3", s)  # 3 rows

    def test_describe(self):
        """Test describe includes experiment names."""
        rows = to_rows("test_game", self.get_sample_instances())
        instances = GameInstances("test_game", rows)
        desc = instances.describe()
        self.assertIn("experiment_1", desc)
        self.assertIn("experiment_2", desc)


if __name__ == '__main__':
    unittest.main()
