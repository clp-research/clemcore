import unittest
from contextlib import ExitStack
from unittest.mock import patch, MagicMock

from clemcore.cli import main, score, run
from clemcore.clemgame.registry import GameSpec


class CLIExceptionLoggingTestCase(unittest.TestCase):
    """Test that exceptions during CLI commands are properly logged."""

    def test_run_exception_is_logged(self):
        """Verify that exceptions during 'clem run' are logged before re-raising."""
        # Use a non-existent model to trigger an error during load_models
        test_args = ["clem", "run", "-g", "nonexistent_game", "-m", "nonexistent_model"]

        with patch("sys.argv", test_args):
            with patch("clemcore.cli.logger") as mock_logger:
                with self.assertRaises(Exception):
                    main()

                # Verify logger.exception was called with the exception
                mock_logger.exception.assert_called_once()


class RunGameDeduplicationTestCase(unittest.TestCase):
    """Test that duplicate game selectors don't result in duplicate game runs."""

    def _make_spec(self, name):
        return GameSpec(game_name=name, game_path="/path", players=1)

    _IO_PATCHES = [
        "clemcore.cli.backends.load_models",
        "clemcore.cli.dispatch.run",
        "clemcore.cli.ResultsFolder",
        "clemcore.cli.Model",
        "clemcore.cli.GameBenchmarkCallbackList",
        "clemcore.cli.InstanceFileSaver",
        "clemcore.cli.ExperimentFileSaver",
        "clemcore.cli.InteractionsFileSaver",
        "clemcore.cli.RunFileSaver",
        "clemcore.cli.PlayerFileSaver",
        "clemcore.cli.GameInstances",
    ]

    def _run(self, game_selectors):
        """Call cli.run() with all I/O side effects suppressed."""
        with ExitStack() as stack:
            for target in self._IO_PATCHES:
                kw = {"return_value": [MagicMock()]} if "load_models" in target else {}
                stack.enter_context(patch(target, **kw))
            run(game_selectors, model_selectors=[], gen_args={})

    def test_duplicate_game_selectors_run_once(self):
        mock_benchmark = MagicMock()
        mock_benchmark.__enter__ = lambda s: s
        mock_benchmark.__exit__ = MagicMock(return_value=False)

        with patch("clemcore.cli.GameRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.from_directories_and_cwd_files.return_value
            mock_registry.get_game_specs_that_unify_with.return_value = [self._make_spec("taboo")]
            with patch("clemcore.cli.GameBenchmark") as mock_benchmark_cls:
                mock_benchmark_cls.load_from_spec.return_value = mock_benchmark
                self._run(["taboo", "taboo"])

        # get_game_specs called twice (once per selector) but benchmark loaded once
        self.assertEqual(mock_benchmark_cls.load_from_spec.call_count, 1)

    def test_distinct_game_selectors_each_run(self):
        taboo = self._make_spec("taboo")
        wordle = self._make_spec("wordle")

        mock_benchmark = MagicMock()
        mock_benchmark.__enter__ = lambda s: s
        mock_benchmark.__exit__ = MagicMock(return_value=False)

        def specs_for(selector):
            return [taboo] if selector == "taboo" else [wordle]

        with patch("clemcore.cli.GameRegistry") as mock_registry_cls:
            mock_registry = mock_registry_cls.from_directories_and_cwd_files.return_value
            mock_registry.get_game_specs_that_unify_with.side_effect = specs_for
            with patch("clemcore.cli.GameBenchmark") as mock_benchmark_cls:
                mock_benchmark_cls.load_from_spec.return_value = mock_benchmark
                self._run(["taboo", "wordle"])

        self.assertEqual(mock_benchmark_cls.load_from_spec.call_count, 2)


if __name__ == "__main__":
    unittest.main()
