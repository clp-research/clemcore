import unittest

import pytest
from pettingzoo.test import api_test

from clemcore.clemgame import env


@pytest.mark.integration
class PettingzooTestCase(unittest.TestCase):
    """PettingZoo API conformance tests.

    Note: Currently fails because we use spaces.Text which doesn't support
    full prompts with whitespaces. Requires custom space implementation to fix.
    """

    def test_api(self):
        api_test(env("taboo"), num_cycles=1000, verbose_progress=False)


if __name__ == '__main__':
    unittest.main()
