import unittest
from pettingzoo.test import api_test

from clemcore.clemgame import env


class PettingzooTestCase(unittest.TestCase):

    def test_api(self):
        # for now this will likely fail because of the spaces.Text which is very limited
        api_test(env("taboo"), num_cycles=1000, verbose_progress=False)


if __name__ == '__main__':
    unittest.main()
