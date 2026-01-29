import unittest

from clemcore.clemgame.runners.batchwise import (
    DynamicBatchDataLoader,
    SinglePassGameSessionPoller,
    GameSession,
)


class MockPlayer:
    """Mock player for testing."""

    def __init__(self, name):
        self.name = name


class MockGameMasterEnv:
    """Mock GameMasterEnv with controllable done state."""

    def __init__(self, session_id, *, done_after=3):
        self._session_id = session_id
        self._done_after = done_after
        self._observe_count = 0
        self.agent_selection = "player_0"
        self.terminations = {"player_0": False}
        self.player_by_agent_id = {"player_0": MockPlayer(f"player_{session_id}")}

    def last(self, observe=False):
        """Return observation in PettingZoo style.

        done_after specifies how many observations before termination.
        E.g., done_after=3 means 3 observations, then done on the 4th call.
        """
        self._observe_count += 1
        context = {"turn": self._observe_count, "session": self._session_id}
        reward = 0.0
        # Terminate after done_after observations have been made
        termination = self._observe_count > self._done_after
        truncation = False
        info = {}
        if termination:
            self.terminations["player_0"] = True
        return context, reward, termination, truncation, info

    def step(self, response):
        """Step the environment."""
        pass


class GameSessionTestCase(unittest.TestCase):

    def test_session_yields_observation_when_not_done(self):
        """Test that session yields observation when game is not done."""
        env = MockGameMasterEnv(0, done_after=5)
        session = GameSession(session_id=0, game_env=env, game_instance={"id": 0})

        observations = list(session)
        self.assertEqual(len(observations), 1)
        session_id, player, context = observations[0]
        self.assertEqual(session_id, 0)
        self.assertEqual(player.name, "player_0")
        self.assertEqual(context["turn"], 1)

    def test_session_yields_nothing_when_done(self):
        """Test that session yields nothing when game is already done."""
        env = MockGameMasterEnv(0, done_after=0)
        env.terminations["player_0"] = True  # Mark as done
        session = GameSession(session_id=0, game_env=env, game_instance={"id": 0})

        observations = list(session)
        self.assertEqual(len(observations), 0)

    def test_session_can_iterate_multiple_times(self):
        """Test that session can be iterated multiple times."""
        env = MockGameMasterEnv(0, done_after=5)
        session = GameSession(session_id=0, game_env=env, game_instance={"id": 0})

        # First iteration
        obs1 = list(session)
        self.assertEqual(len(obs1), 1)
        self.assertEqual(obs1[0][2]["turn"], 1)

        # Second iteration (observe_count increases)
        obs2 = list(session)
        self.assertEqual(len(obs2), 1)
        self.assertEqual(obs2[0][2]["turn"], 2)

    def test_collate_fn(self):
        """Test the collate function unpacks batches correctly."""
        batch = [
            (0, MockPlayer("p0"), {"ctx": 0}),
            (1, MockPlayer("p1"), {"ctx": 1}),
            (2, MockPlayer("p2"), {"ctx": 2}),
        ]
        session_ids, players, contexts = GameSession.collate_fn(batch)

        self.assertEqual(session_ids, [0, 1, 2])
        self.assertEqual([p.name for p in players], ["p0", "p1", "p2"])
        self.assertEqual(contexts, [{"ctx": 0}, {"ctx": 1}, {"ctx": 2}])


class SinglePassGameSessionPollerTestCase(unittest.TestCase):

    def test_polls_all_sessions_once(self):
        """Test that poller yields one observation from each session."""
        sessions = [
            GameSession(i, MockGameMasterEnv(i, done_after=5), {})
            for i in range(3)
        ]
        poller = SinglePassGameSessionPoller(sessions)

        observations = list(poller)
        self.assertEqual(len(observations), 3)

        session_ids = [obs[0] for obs in observations]
        self.assertEqual(session_ids, [0, 1, 2])

    def test_skips_exhausted_sessions(self):
        """Test that poller skips already done sessions."""
        env_done = MockGameMasterEnv(1, done_after=0)
        env_done.terminations["player_0"] = True  # Already done
        sessions = [
            GameSession(0, MockGameMasterEnv(0, done_after=5), {}),
            GameSession(1, env_done, {}),
            GameSession(2, MockGameMasterEnv(2, done_after=5), {}),
        ]
        poller = SinglePassGameSessionPoller(sessions)

        observations = list(poller)
        self.assertEqual(len(observations), 2)

        session_ids = [obs[0] for obs in observations]
        self.assertEqual(session_ids, [0, 2])

    def test_tracks_exhausted_state(self):
        """Test that exhausted state is tracked correctly."""
        env = MockGameMasterEnv(0, done_after=1)
        sessions = [GameSession(0, env, {})]
        poller = SinglePassGameSessionPoller(sessions)

        self.assertFalse(poller.exhausted[0])

        # First poll - yields one observation (done_after=1 means 1 observation allowed)
        obs1 = list(poller)
        self.assertEqual(len(obs1), 1)
        self.assertFalse(poller.exhausted[0])  # Not exhausted yet

        # Second poll - now exhausted (no more observations)
        obs2 = list(poller)
        self.assertEqual(len(obs2), 0)
        self.assertTrue(poller.exhausted[0])

    def test_multiple_passes_track_exhaustion(self):
        """Test that multiple iterations track exhaustion properly."""
        sessions = [
            GameSession(0, MockGameMasterEnv(0, done_after=3), {}),
            GameSession(1, MockGameMasterEnv(1, done_after=1), {}),
        ]
        poller = SinglePassGameSessionPoller(sessions)

        # First pass - both active, both yield
        obs1 = list(poller)
        self.assertEqual(len(obs1), 2)
        self.assertFalse(poller.exhausted[0])
        self.assertFalse(poller.exhausted[1])

        # Second pass - session 1 becomes exhausted (done_after=1 means only 1 observation)
        obs2 = list(poller)
        self.assertEqual(len(obs2), 1)  # Only session 0 yields
        self.assertEqual(obs2[0][0], 0)
        self.assertFalse(poller.exhausted[0])
        self.assertTrue(poller.exhausted[1])


class DynamicBatchDataLoaderTestCase(unittest.TestCase):

    def test_yields_batches_up_to_batch_size(self):
        """Test that loader yields batches up to the specified size."""
        sessions = [
            GameSession(i, MockGameMasterEnv(i, done_after=2), {})
            for i in range(5)
        ]
        poller = SinglePassGameSessionPoller(sessions)
        loader = DynamicBatchDataLoader(poller, collate_fn=GameSession.collate_fn, batch_size=3)

        batches = list(loader)
        # With 5 sessions and batch_size=3, first batch has 3, second has 2
        # Then sessions exhaust after 2 observations each
        self.assertGreater(len(batches), 0)

        # First batch should have at most batch_size items
        first_batch_ids, _, _ = batches[0]
        self.assertLessEqual(len(first_batch_ids), 3)

    def test_stops_when_all_exhausted(self):
        """Test that loader stops when all sessions are exhausted."""
        sessions = [
            GameSession(i, MockGameMasterEnv(i, done_after=1), {})
            for i in range(3)
        ]
        poller = SinglePassGameSessionPoller(sessions)
        loader = DynamicBatchDataLoader(poller, collate_fn=GameSession.collate_fn, batch_size=2)

        batches = list(loader)
        # All sessions should be exhausted after iteration
        self.assertTrue(all(poller.exhausted))
        self.assertGreater(len(batches), 0)

    def test_handles_single_session(self):
        """Test loader with a single session."""
        sessions = [GameSession(0, MockGameMasterEnv(0, done_after=3), {})]
        poller = SinglePassGameSessionPoller(sessions)
        loader = DynamicBatchDataLoader(poller, collate_fn=GameSession.collate_fn, batch_size=5)

        batches = list(loader)
        self.assertEqual(len(batches), 3)  # 3 observations before done

        for batch in batches:
            session_ids, _, _ = batch
            self.assertEqual(len(session_ids), 1)
            self.assertEqual(session_ids[0], 0)

    def test_handles_empty_sessions(self):
        """Test loader with all sessions already done."""
        sessions = []
        for i in range(3):
            env = MockGameMasterEnv(i, done_after=0)
            env.terminations["player_0"] = True
            sessions.append(GameSession(i, env, {}))
        poller = SinglePassGameSessionPoller(sessions)
        loader = DynamicBatchDataLoader(poller, collate_fn=GameSession.collate_fn, batch_size=2)

        batches = list(loader)
        self.assertEqual(len(batches), 0)

    def test_batch_size_larger_than_sessions(self):
        """Test loader when batch_size exceeds number of sessions."""
        sessions = [
            GameSession(i, MockGameMasterEnv(i, done_after=2), {})
            for i in range(2)
        ]
        poller = SinglePassGameSessionPoller(sessions)
        loader = DynamicBatchDataLoader(poller, collate_fn=GameSession.collate_fn, batch_size=10)

        batches = list(loader)
        # Should still work, batches just won't reach size 10
        for batch in batches:
            session_ids, _, _ = batch
            self.assertLessEqual(len(session_ids), 2)

    def test_all_sessions_complete(self):
        """Test that all sessions eventually complete."""
        sessions = [
            GameSession(i, MockGameMasterEnv(i, done_after=3), {})
            for i in range(4)
        ]
        poller = SinglePassGameSessionPoller(sessions)
        loader = DynamicBatchDataLoader(poller, collate_fn=GameSession.collate_fn, batch_size=2)

        list(loader)  # Consume all batches

        # All game envs should be done
        for session in sessions:
            self.assertTrue(session.is_done)


if __name__ == '__main__':
    unittest.main()
