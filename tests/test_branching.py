from pathlib import Path

import pytest

from clemcore import backends
from clemcore.backends import Model
from clemcore.clemgame import GameBenchmark, GameRegistry, GameInstances, GameBenchmarkCallbackList, EpochResultsFolder, \
    EpochResultsFolderCallback, InstanceFileSaver, ExperimentFileSaver, InteractionsFileSaver, SignalFileSaver
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
    return GameInstances.from_game_spec(game_spec).filter(lambda row: row["game_instance"]["game_id"] in [0])


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

    def test_branching_runner_with_results_folder(self, game_benchmark, game_instances, player_models):
        results_folder = EpochResultsFolder(Path("results-branching"), Model.to_identifier(player_models))
        model_infos = Model.to_infos(player_models)
        callbacks = GameBenchmarkCallbackList([
            EpochResultsFolderCallback(results_folder),
            InstanceFileSaver(results_folder),
            ExperimentFileSaver(results_folder, player_model_infos=model_infos),
            InteractionsFileSaver(results_folder, player_model_infos=model_infos, store_branches=True),
            SignalFileSaver(results_folder)
        ])
        # This results in 64 branches for each instance, by branching at each turn and playing 6 turns (2^6=64)
        branching.run(
            game_benchmark,
            game_instances,
            player_models,
            callbacks=callbacks,
            branching_factor=2,
            branching_condition=lambda **_: True
        )

    def test_branching_runner_with_results_folder_and_conditions(self, game_benchmark, game_instances, player_models):
        results_folder = EpochResultsFolder(Path("results-branching"), Model.to_identifier(player_models))
        model_infos = Model.to_infos(player_models)
        callbacks = GameBenchmarkCallbackList([
            EpochResultsFolderCallback(results_folder),
            InstanceFileSaver(results_folder),
            ExperimentFileSaver(results_folder, player_model_infos=model_infos),
            InteractionsFileSaver(results_folder, player_model_infos=model_infos, store_branches=True),
            SignalFileSaver(results_folder)
        ])
        # This results in 8 branches for each instance, by branching only at the first round on the describer's turn
        branching.run(
            game_benchmark,
            game_instances,
            player_models,
            callbacks=callbacks,
            branching_factor=8,
            branching_condition=branching.combined_condition(
                branching.is_player_role("WordDescriber"),
                branching.is_round(0)
            )
        )
