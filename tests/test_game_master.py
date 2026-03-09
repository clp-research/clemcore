import unittest
from unittest.mock import MagicMock

from clemcore.clemgame.errors import ParseError
from clemcore.clemgame.events import GameEventLogger
from clemcore.clemgame.master import DialogueGameMaster


def _make_game_master(parse_raises=False):
    """Create a minimal DialogueGameMaster with a mock logger registered."""
    game_spec = MagicMock()
    game_spec.game_name = "test_game"
    game_spec.game_path = "/tmp"
    game_spec.players = 1

    class ConcreteGM(DialogueGameMaster):
        def _on_setup(self, **kwargs):
            player = MagicMock()
            player.game_role = "Player"
            self.add_player(player, initial_context="hello")

        def _parse_response(self, player, response):
            if parse_raises:
                raise ParseError("bad format")
            return response

        def _advance_game(self, player, parsed_response):
            self.state.succeed()

    gm = ConcreteGM(game_spec, {}, [MagicMock()])
    gm.setup()
    gm.before_game()

    mock_logger = MagicMock(spec=GameEventLogger)
    gm.register(mock_logger)
    return gm, mock_logger


class TestDialogueGameMasterRequestCounting(unittest.TestCase):

    def test_step_counts_request_on_valid_response(self):
        """count_request() must be called on the GM's loggers for every step."""
        gm, mock_logger = _make_game_master()
        gm.step("valid response")
        mock_logger.count_request.assert_called_once()
        mock_logger.count_request_violation.assert_not_called()

    def test_step_counts_request_and_violation_on_parse_error(self):
        """count_request() and count_request_violation() must both be called when parsing fails."""
        gm, mock_logger = _make_game_master(parse_raises=True)
        gm.step("bad response")
        mock_logger.count_request.assert_called_once()
        mock_logger.count_request_violation.assert_called_once()


if __name__ == "__main__":
    unittest.main()
