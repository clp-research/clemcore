"""Tests for clemcore.clemeval — covers save_clem_table and perform_evaluation."""
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

import clemcore.clemgame.metrics as clemmetrics
from clemcore.clemeval import build_df_episode_scores, load_scores, save_clem_table, perform_evaluation, PlayedScoreError

FIXTURE_RUNS_ZIP = Path(__file__).parent / "fixtures" / "clemeval" / "runs-v2-llama8b.zip"
FIXTURE_MODEL = "Meta-Llama-3.1-8B-Instruct-t0.0--Meta-Llama-3.1-8B-Instruct-t0.0"
GOLDEN_CSV = Path(__file__).parent / "fixtures" / "clemeval" / "golden_results.csv"


def _make_episode_scores_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal episode-scores DataFrame from a list of row dicts."""
    cols = ['game', 'model', 'experiment', 'episode', 'metric', 'value']
    df = pd.DataFrame(rows, columns=cols)
    return df


def _minimal_scores() -> pd.DataFrame:
    """Two episodes for one game/model: one played, one aborted."""
    return _make_episode_scores_df([
        # episode 1 — played, scored 80
        {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_1',
         'metric': clemmetrics.METRIC_ABORTED, 'value': 0},
        {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_1',
         'metric': clemmetrics.BENCH_SCORE, 'value': 80},
        # episode 2 — aborted, scored 0
        {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_2',
         'metric': clemmetrics.METRIC_ABORTED, 'value': 1},
        {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_2',
         'metric': clemmetrics.BENCH_SCORE, 'value': 0},
    ])


def _add_played_column(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate the PLAYED derivation from perform_evaluation."""
    aux = df[df["metric"] == clemmetrics.METRIC_ABORTED].copy()
    aux["metric"] = clemmetrics.METRIC_PLAYED
    aux["value"] = 1 - aux["value"]
    return pd.concat([df, aux], ignore_index=True)


class TestBuildDfEpisodeScores(unittest.TestCase):

    def test_columns_present(self):
        scores = {
            ('taboo', 'gpt-4', 'exp1', 'episode_1'): {
                'episodes': {clemmetrics.BENCH_SCORE: 80, clemmetrics.METRIC_ABORTED: 0}
            }
        }
        df = build_df_episode_scores(scores)
        self.assertListEqual(list(df.columns),
                             ['game', 'model', 'experiment', 'episode', 'metric', 'value'])

    def test_row_count(self):
        scores = {
            ('taboo', 'gpt-4', 'exp1', 'episode_1'): {
                'episodes': {clemmetrics.BENCH_SCORE: 80, clemmetrics.METRIC_ABORTED: 0}
            }
        }
        df = build_df_episode_scores(scores)
        self.assertEqual(len(df), 2)


class TestSaveClemTable(unittest.TestCase):

    def setUp(self):
        self.df = _add_played_column(_minimal_scores())
        self.tmpdir = tempfile.mkdtemp()

    def test_returns_dataframe(self):
        result = save_clem_table(self.df.copy(), self.tmpdir)
        self.assertIsInstance(result, pd.DataFrame)

    def test_csv_written(self):
        save_clem_table(self.df.copy(), self.tmpdir)
        self.assertTrue((Path(self.tmpdir) / 'results.csv').exists())

    def test_html_written(self):
        save_clem_table(self.df.copy(), self.tmpdir)
        self.assertTrue((Path(self.tmpdir) / 'results.html').exists())

    def test_clemscore_column_present(self):
        result = save_clem_table(self.df.copy(), self.tmpdir)
        clemscore_col = [c for c in result.columns if 'clemscore' in c]
        self.assertTrue(len(clemscore_col) == 1, f"clemscore column missing; got {result.columns.tolist()}")

    def test_clemscore_value(self):
        """clemscore = (% played / 100) * avg quality score.
        Episode 1: played=1, score=80. Episode 2: played=0, score=0.
        avg played = 50%, avg quality = 40. clemscore = 0.5 * 40 = 20.
        """
        result = save_clem_table(self.df.copy(), self.tmpdir)
        clemscore_col = [c for c in result.columns if 'clemscore' in c][0]
        clemscore = result.loc['gpt-4', clemscore_col]
        self.assertAlmostEqual(clemscore, 20.0, places=1)

    def test_model_is_index(self):
        result = save_clem_table(self.df.copy(), self.tmpdir)
        self.assertIn('gpt-4', result.index)

    def test_multiple_models(self):
        extra = _make_episode_scores_df([
            {'game': 'taboo', 'model': 'llama', 'experiment': 'exp1', 'episode': 'episode_1',
             'metric': clemmetrics.METRIC_ABORTED, 'value': 0},
            {'game': 'taboo', 'model': 'llama', 'experiment': 'exp1', 'episode': 'episode_1',
             'metric': clemmetrics.BENCH_SCORE, 'value': 60},
        ])
        df = _add_played_column(pd.concat([_minimal_scores(), extra], ignore_index=True))
        result = save_clem_table(df, self.tmpdir)
        self.assertIn('gpt-4', result.index)
        self.assertIn('llama', result.index)


class TestPerformEvaluation(unittest.TestCase):

    def test_raises_on_played_in_scores(self):
        """perform_evaluation should raise PlayedScoreError if METRIC_PLAYED is in scores."""
        # We test indirectly: build a df that already has METRIC_PLAYED and
        # check that the guard triggers. We monkey-patch load_scores/build_df.
        import clemcore.clemeval as clemeval

        original_load = clemeval.load_scores
        original_build = clemeval.build_df_episode_scores

        try:
            bad_df = _make_episode_scores_df([
                {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'ep1',
                 'metric': clemmetrics.METRIC_PLAYED, 'value': 1},
            ])
            clemeval.load_scores = lambda path, model_selector=None, game_selector=None: {}
            clemeval.build_df_episode_scores = lambda scores: bad_df

            with tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(PlayedScoreError):
                    clemeval.perform_evaluation(tmpdir)
        finally:
            clemeval.load_scores = original_load
            clemeval.build_df_episode_scores = original_build

    def test_full_pipeline_writes_files(self):
        """perform_evaluation should produce raw.csv, results.csv, results.html."""
        import clemcore.clemeval as clemeval

        original_load = clemeval.load_scores
        original_build = clemeval.build_df_episode_scores

        try:
            clemeval.load_scores = lambda path, model_selector=None, game_selector=None: {}
            clemeval.build_df_episode_scores = lambda scores: _minimal_scores()

            with tempfile.TemporaryDirectory() as tmpdir:
                clemeval.perform_evaluation(tmpdir)
                self.assertTrue((Path(tmpdir) / 'raw.csv').exists())
                self.assertTrue((Path(tmpdir) / 'results.csv').exists())
                self.assertTrue((Path(tmpdir) / 'results.html').exists())
        finally:
            clemeval.load_scores = original_load
            clemeval.build_df_episode_scores = original_build


class TestSaveClemTableStdFlag(unittest.TestCase):

    def setUp(self):
        # Two episodes so std is meaningful
        rows = [
            {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_1',
             'metric': clemmetrics.METRIC_ABORTED, 'value': 0},
            {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_1',
             'metric': clemmetrics.BENCH_SCORE, 'value': 80},
            {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_2',
             'metric': clemmetrics.METRIC_ABORTED, 'value': 0},
            {'game': 'taboo', 'model': 'gpt-4', 'experiment': 'exp1', 'episode': 'episode_2',
             'metric': clemmetrics.BENCH_SCORE, 'value': 60},
        ]
        self.df = _add_played_column(_make_episode_scores_df(rows))
        self.tmpdir = tempfile.mkdtemp()

    def test_std_columns_absent_by_default(self):
        result = save_clem_table(self.df.copy(), self.tmpdir)
        std_cols = [c for c in result.columns if '(std)' in c]
        self.assertEqual(std_cols, [], f"Expected no std columns by default, got {std_cols}")

    def test_std_columns_present_when_requested(self):
        result = save_clem_table(self.df.copy(), self.tmpdir, show_std=True)
        std_cols = [c for c in result.columns if '(std)' in c]
        self.assertGreater(len(std_cols), 0, "Expected at least one std column when show_std=True")


class TestSaveClemTableSortFlag(unittest.TestCase):

    def setUp(self):
        rows = []
        for model, score in [('z-model', 10), ('a-model', 90), ('m-model', 50)]:
            for ep in ['episode_1', 'episode_2']:
                rows += [
                    {'game': 'taboo', 'model': model, 'experiment': 'exp1', 'episode': ep,
                     'metric': clemmetrics.METRIC_ABORTED, 'value': 0},
                    {'game': 'taboo', 'model': model, 'experiment': 'exp1', 'episode': ep,
                     'metric': clemmetrics.BENCH_SCORE, 'value': score},
                ]
        self.df = _add_played_column(_make_episode_scores_df(rows))
        self.tmpdir = tempfile.mkdtemp()

    def test_sort_by_model_name_default(self):
        result = save_clem_table(self.df.copy(), self.tmpdir, sort_by='model_name')
        self.assertEqual(list(result.index), sorted(result.index.tolist()))

    def test_sort_by_clemscore_descending(self):
        result = save_clem_table(self.df.copy(), self.tmpdir, sort_by='clemscore')
        clemscore_col = [c for c in result.columns if 'clemscore' in c][0]
        scores = result[clemscore_col].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result.index[0], 'a-model')


class TestGoldenOutput(unittest.TestCase):
    """Regression test against real clembench-runs data.

    Compares save_clem_table output against a golden CSV captured before the
    pandas migration.
    """

    def test_results_match_golden(self):
        with tempfile.TemporaryDirectory() as unpack_dir:
            with zipfile.ZipFile(FIXTURE_RUNS_ZIP) as zf:
                zf.extractall(unpack_dir)
            results_path = Path(unpack_dir) / "runs" / FIXTURE_MODEL

            scores = load_scores(str(results_path))
            df = build_df_episode_scores(scores)
            aux = df[df["metric"] == clemmetrics.METRIC_ABORTED].copy()
            aux["metric"] = clemmetrics.METRIC_PLAYED
            aux["value"] = 1 - aux["value"]
            df = pd.concat([df, aux], ignore_index=True)

            with tempfile.TemporaryDirectory() as out_dir:
                save_clem_table(df, out_dir, show_std=True)
                actual = pd.read_csv(Path(out_dir) / "results.csv", index_col=0)

        golden = pd.read_csv(GOLDEN_CSV, index_col=0)
        pd.testing.assert_frame_equal(actual, golden, check_like=True)


if __name__ == '__main__':
    unittest.main()
