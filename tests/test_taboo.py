import unittest

from backends import CustomResponseModel
from games.taboo.master import check_clue, Taboo


class TabooTestCase(unittest.TestCase):

    def test_taboo_messages_alternate(self):
        outer_self = self
        experiment = {"name": "test",
                      "max_turns": 3,
                      "describer_initial_prompt": "<describer-prompt",
                      "guesser_initial_prompt": "<guesser-prompt>"
                      }
        game = {
            "game_id": 0,
            "target_word": "test",
            "related_word": ["assurance", "ensure", "quality"],
            "target_word_stem": "test",
            "related_word_stem": ["assur", "ensure", "quality"]
        }

        # describer = CustomResponseModel() # not possible now, but the way to do it, could provide lambda in construct
        # describer.generate_response = lambda messages: "clue", "clue", "clue"

        # guesser = CustomResponseModel() # not possible now, but the way to do it, could provide lambda in construct
        # describer.generate_response = lambda messages: "guess", "guess", "guess"

        players = [CustomResponseModel(), CustomResponseModel()]

        class TestTaboo(Taboo):

            def _on_before_turn(self, turn_idx: int):
                # in fact this check could go into the DialogueMaster
                for name, messages in self.messages_by_names.items():
                    print("test", turn_idx, name, messages)
                    if turn_idx > 0:
                        is_same = messages[turn_idx]["role"] == messages[turn_idx - 1]["role"]
                        outer_self.assertFalse(is_same,
                                               "Consecutive roles in the messages history should not be the same")
                super()._on_before_turn(turn_idx)

        taboo = TestTaboo(experiment, players)
        taboo.setup(**game)
        taboo.play()

    def test_clue_check_issue9(self):
        errors = check_clue("A term that refers to the act of completing a task without professional help",
                            target_word="diy",
                            related_words=["do", "it", "yourself"])
        self.assertEqual(errors, [])

    def test_clue_check_similar_target(self):
        errors = check_clue("The state of doing a transition from A to B",
                            target_word="transit",
                            related_words=["transport", "cross", "traverse"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["message"],
                         "Target word 'transit' (stem=transit) "
                         "is similar to clue word 'transition' (stem=transit)")

    def test_clue_check_similar_rel(self):
        errors = check_clue("Usually local transportation especially of people by public conveyance.",
                            target_word="transit",
                            related_words=["transport", "cross", "traverse"])
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["message"],
                         "Related word 'transport' (stem=transport) "
                         "is similar to clue word 'transportation' (stem=transport)")

    def test_clue_check_ok(self):
        errors = check_clue("Conveyance of persons or things from one place to another.",
                            target_word="transit",
                            related_words=["transport", "cross", "traverse"])
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main()
