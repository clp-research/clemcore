import unittest

from games.taboo.master import check_clue


class TabooTestCase(unittest.TestCase):

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
