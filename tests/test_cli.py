import unittest
from unittest.mock import patch

from clemcore.cli import main


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


if __name__ == "__main__":
    unittest.main()
