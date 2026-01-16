import json
import os
import tempfile
import unittest

from clemcore.clemgame.instances import GameInstanceIterator, to_instance_filter


class ToInstanceFilterTestCase(unittest.TestCase):

    def test_filter_from_dataset(self):
        """Test creating filter from dataset."""
        dataset = [
            {"game": "game_a", "experiment": "exp1", "task_id": "1"},
            {"game": "game_a", "experiment": "exp1", "task_id": "2"},
            {"game": "game_a", "experiment": "exp2", "task_id": "3"},
            {"game": "game_b", "experiment": "exp1", "task_id": "1"},
        ]
        filter_fn = to_instance_filter(dataset)

        # Test filtering
        result = filter_fn("game_a", "exp1")
        self.assertEqual(result, [1, 2])

        result = filter_fn("game_a", "exp2")
        self.assertEqual(result, [3])

        result = filter_fn("game_b", "exp1")
        self.assertEqual(result, [1])

    def test_filter_returns_empty_for_missing(self):
        """Test that filter returns empty list for missing game/experiment."""
        dataset = [{"game": "game_a", "experiment": "exp1", "task_id": "1"}]
        filter_fn = to_instance_filter(dataset)

        result = filter_fn("nonexistent", "exp")
        self.assertEqual(result, [])


class GameInstanceIteratorTestCase(unittest.TestCase):

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

    def test_iterator_creation(self):
        """Test creating GameInstanceIterator."""
        instances = self.get_sample_instances()
        iterator = GameInstanceIterator("test_game", instances)
        self.assertEqual(len(iterator), 0)  # Not reset yet

    def test_iterator_reset(self):
        """Test resetting iterator populates queue."""
        instances = self.get_sample_instances()
        iterator = GameInstanceIterator("test_game", instances)
        iterator.reset()
        self.assertEqual(len(iterator), 3)  # 2 + 1 instances

    def test_iterator_iteration(self):
        """Test iterating over instances."""
        instances = self.get_sample_instances()
        iterator = GameInstanceIterator("test_game", instances)
        iterator.reset()

        collected = list(iterator)
        self.assertEqual(len(collected), 3)

        # Check structure: each item is (experiment, instance) tuple
        exp, inst = collected[0]
        self.assertEqual(exp["name"], "experiment_1")
        self.assertEqual(inst["game_id"], 1)

    def test_iterator_experiment_filtered(self):
        """Test that experiment dict excludes game_instances."""
        instances = self.get_sample_instances()
        iterator = GameInstanceIterator("test_game", instances)
        iterator.reset()

        exp, inst = next(iterator)
        self.assertNotIn("game_instances", exp)
        self.assertIn("name", exp)
        self.assertIn("param", exp)

    def test_iterator_with_sub_selector(self):
        """Test iterator with sub_selector filter."""
        instances = self.get_sample_instances()

        # Only select game_id 1 from experiment_1
        def sub_selector(game_name, experiment_name):
            if experiment_name == "experiment_1":
                return [1]
            return None  # Include all for other experiments

        iterator = GameInstanceIterator("test_game", instances, sub_selector=sub_selector)
        iterator.reset()

        collected = list(iterator)
        # Should have 1 from experiment_1, 1 from experiment_2
        self.assertEqual(len(collected), 2)

        game_ids = [inst["game_id"] for _, inst in collected]
        self.assertIn(1, game_ids)
        self.assertIn(3, game_ids)
        self.assertNotIn(2, game_ids)

    def test_iterator_sub_selector_empty_list(self):
        """Test that empty list from sub_selector skips experiment."""
        instances = self.get_sample_instances()

        def sub_selector(game_name, experiment_name):
            if experiment_name == "experiment_1":
                return []  # Skip this experiment
            return None

        iterator = GameInstanceIterator("test_game", instances, sub_selector=sub_selector)
        iterator.reset()

        collected = list(iterator)
        self.assertEqual(len(collected), 1)  # Only experiment_2
        exp, inst = collected[0]
        self.assertEqual(exp["name"], "experiment_2")

    def test_iterator_deepcopy(self):
        """Test deep copying iterator."""
        instances = self.get_sample_instances()
        iterator = GameInstanceIterator("test_game", instances)
        iterator.reset()

        # Consume one item
        next(iterator)
        self.assertEqual(len(iterator), 2)

        # Deep copy (note: __deepcopy__ doesn't take memo arg in this implementation)
        iterator_copy = iterator.__deepcopy__()
        self.assertEqual(len(iterator_copy), 2)

        # Consume from copy shouldn't affect original
        next(iterator_copy)
        self.assertEqual(len(iterator_copy), 1)
        self.assertEqual(len(iterator), 2)

    def test_iterator_stop_iteration(self):
        """Test that StopIteration is raised when exhausted."""
        instances = {"experiments": [{"name": "exp", "game_instances": [{"game_id": 1}]}]}
        iterator = GameInstanceIterator("test_game", instances)
        iterator.reset()

        next(iterator)  # Consume the only item
        with self.assertRaises(StopIteration):
            next(iterator)

    def test_from_file(self):
        """Test loading iterator from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            instances = self.get_sample_instances()
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump(instances, f)

            iterator = GameInstanceIterator.from_file("test_game", tmpdir)
            iterator.reset()
            self.assertEqual(len(iterator), 3)

    def test_from_file_custom_name(self):
        """Test loading iterator from file with custom name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            instances = self.get_sample_instances()
            with open(os.path.join(tmpdir, "custom_instances.json"), "w") as f:
                json.dump(instances, f)

            iterator = GameInstanceIterator.from_file("test_game", tmpdir, "custom_instances")
            iterator.reset()
            self.assertEqual(len(iterator), 3)

    def test_from_file_missing_experiments_raises(self):
        """Test that missing experiments key raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump({"not_experiments": []}, f)

            with self.assertRaises(ValueError) as context:
                GameInstanceIterator.from_file("test_game", tmpdir)
            self.assertIn("experiments", str(context.exception).lower())

    def test_from_file_experiments_not_list_raises(self):
        """Test that non-list experiments raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump({"experiments": "not_a_list"}, f)

            with self.assertRaises(ValueError) as context:
                GameInstanceIterator.from_file("test_game", tmpdir)
            self.assertIn("not a list", str(context.exception).lower())

    def test_from_file_empty_experiments_raises(self):
        """Test that empty experiments list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "instances.json"), "w") as f:
                json.dump({"experiments": []}, f)

            with self.assertRaises(ValueError) as context:
                GameInstanceIterator.from_file("test_game", tmpdir)
            self.assertIn("empty", str(context.exception).lower())

    def test_iterator_requires_game_name(self):
        """Test that game_name is required."""
        instances = self.get_sample_instances()
        with self.assertRaises(AssertionError):
            GameInstanceIterator(None, instances)

    def test_iterator_requires_instances(self):
        """Test that instances is required."""
        with self.assertRaises(AssertionError):
            GameInstanceIterator("game", None)


if __name__ == '__main__':
    unittest.main()
