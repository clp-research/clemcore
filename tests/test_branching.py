import pytest

from clemcore import backends
from clemcore.clemgame import GameBenchmark, GameRegistry, GameInstances, GameBenchmarkCallbackList
from clemcore.clemgame.runners import branching

TEST_GAME = "taboo"
TEST_MODEL = "mock"


@pytest.fixture(scope="module")
def game_spec():
    return GameRegistry.from_directories_and_cwd_files().get_game_spec(TEST_GAME)


@pytest.fixture
def game_benchmark(game_spec):
    return GameBenchmark.load_from_spec(game_spec)


@pytest.fixture
def game_instances(game_spec):
    return GameInstances.from_game_spec(game_spec).filter(lambda row: row["game_instance"]["game_id"] in [0, 1])


@pytest.fixture
def player_models():
    return backends.load_models([TEST_MODEL] * 2)


@pytest.mark.integration
class TestBranchingRunner:

    def test_branching_runner(self, game_benchmark, game_instances, player_models):
        branching.run(
            game_benchmark,
            game_instances,
            player_models,
            callbacks=GameBenchmarkCallbackList(),
            branching_factor=2,
            branching_condition=lambda **_: True
        )
