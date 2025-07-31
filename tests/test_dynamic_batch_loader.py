import random
import unittest

from clemcore.clemgame.runners.batchwise import DynamicBatchDataLoader, MultiGameRoundRobinScheduler, GameSession

ALL_DONES = set()


class ProbabilisticGameMaster:

    def __init__(self, _id):
        self._id = _id
        self._done = False

    def is_done(self):
        if self._done:
            return True
        self._done = random.randint(0, 100) > 70
        if self._done:
            ALL_DONES.add(self._id)
        return self._done

    def observe(self):
        return f"player_{self._id}", f"context_{self._id}"


def create_game_sessions(n):
    return [GameSession(idx, ProbabilisticGameMaster(idx), dict()) for idx in range(n)]


class DynamicBatchDataLoaderTest(unittest.TestCase):
    def test_something(self):
        game_sessions = create_game_sessions(16)
        scheduler = MultiGameRoundRobinScheduler(game_sessions=game_sessions)
        loader = DynamicBatchDataLoader(scheduler, collate_fn=GameSession.collate_fn, batch_size=7)
        for batch in loader:
            print("Batch:", len(batch[0]), batch[0])
            print("Done:", ALL_DONES)
            assert set(batch[0]) not in ALL_DONES
        assert all(game_session.game_master.is_done() for game_session in game_sessions), "Not all done"


if __name__ == '__main__':
    unittest.main()
