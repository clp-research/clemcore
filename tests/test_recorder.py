import unittest

from clemcore.clemgame.recorder import GameInteractionsRecorder, EventCallRecorder


class TestGameInteractionsRecorder(unittest.TestCase):

    def setUp(self):
        self.recorder = GameInteractionsRecorder(
            game_name="test_game",
            experiment_name="test_experiment",
            game_id=1,
            results_folder="/tmp/results",
            player_model_infos={"Player 1": {"model": "test-model"}}
        )

    def test_initial_meta_fields(self):
        """round_count and completed should be None initially."""
        meta = self.recorder.interactions["meta"]
        self.assertIsNone(meta["round_count"])
        self.assertIsNone(meta["completed"])

    def test_log_next_round_updates_round_count(self):
        """log_next_round should update round_count in meta."""
        self.recorder.log_next_round()
        self.assertEqual(self.recorder.interactions["meta"]["round_count"], 2)
        self.recorder.log_next_round()
        self.assertEqual(self.recorder.interactions["meta"]["round_count"], 3)

    def test_log_game_end_sets_completed(self):
        """log_game_end should set completed=True and round_count."""
        self.recorder.log_game_end()
        meta = self.recorder.interactions["meta"]
        self.assertTrue(meta["completed"])
        self.assertEqual(meta["round_count"], 1)

    def test_log_game_end_after_rounds(self):
        """log_game_end should reflect correct round_count after multiple rounds."""
        self.recorder.log_next_round()
        self.recorder.log_next_round()
        self.recorder.log_game_end()
        meta = self.recorder.interactions["meta"]
        self.assertTrue(meta["completed"])
        self.assertEqual(meta["round_count"], 3)


class TestEventCallRecorder(unittest.TestCase):

    def setUp(self):
        self.recorder = EventCallRecorder(
            game_name="test_game",
            experiment_name="test_experiment",
            game_id=1,
            player_name="Player 1",
            game_role="Guesser",
            model_name="test-model"
        )

    def test_initial_state(self):
        """Recorder should initialize with correct metadata and empty calls."""
        meta = self.recorder.requests["meta"]
        self.assertEqual(meta["game_name"], "test_game")
        self.assertEqual(meta["experiment_name"], "test_experiment")
        self.assertEqual(meta["game_id"], 1)
        self.assertEqual(meta["player_name"], "Player 1")
        self.assertEqual(meta["game_role"], "Guesser")
        self.assertEqual(meta["model_name"], "test-model")
        self.assertIsNone(meta["round_count"])
        self.assertIsNone(meta["completed"])
        self.assertEqual(self.recorder.requests["calls"], [])
        self.assertEqual(len(self.recorder), 0)

    def test_log_event_filters_by_player_name(self):
        """log_event should ignore events not from the recorder's player."""
        # Event from different player - should be ignored
        self.recorder.log_event(
            from_="Player 2",
            to="GM",
            action={"type": "get message", "content": "hello"},
            call=({"prompt": "test"}, {"response": "test"})
        )
        self.assertEqual(len(self.recorder), 0)

        # Event from GM - should be ignored
        self.recorder.log_event(
            from_="GM",
            to="Player 1",
            action={"type": "send message", "content": "hello"},
            call=({"prompt": "test"}, {"response": "test"})
        )
        self.assertEqual(len(self.recorder), 0)

    def test_log_event_records_matching_player(self):
        """log_event should record events from the recorder's player."""
        self.recorder.log_event(
            from_="Player 1",
            to="GM",
            action={"type": "get message", "content": "hello"},
            call=({"prompt": "test"}, {"response": "test"})
        )
        self.assertEqual(len(self.recorder), 1)
        call_entry = self.recorder.requests["calls"][0]
        self.assertEqual(call_entry["round"], 0)
        self.assertIn("timestamp", call_entry["call"])
        self.assertEqual(call_entry["call"]["manipulated_prompt_obj"], {"prompt": "test"})
        self.assertEqual(call_entry["call"]["raw_response_obj"], {"response": "test"})

    def test_log_event_ignores_none_call(self):
        """log_event should not record when call is None."""
        self.recorder.log_event(
            from_="Player 1",
            to="GM",
            action={"type": "get message", "content": "hello"},
            call=None
        )
        self.assertEqual(len(self.recorder), 0)

    def test_log_event_ignores_non_tuple_call(self):
        """log_event should only record when call is a tuple."""
        self.recorder.log_event(
            from_="Player 1",
            to="GM",
            action={"type": "get message", "content": "hello"},
            call="not a tuple"
        )
        self.assertEqual(len(self.recorder), 0)

    def test_log_next_round_updates_round(self):
        """log_next_round should increment round and update round_count."""
        self.assertEqual(self.recorder.round, 0)
        self.recorder.log_next_round()
        self.assertEqual(self.recorder.round, 1)
        self.assertEqual(self.recorder.requests["meta"]["round_count"], 2)
        self.recorder.log_next_round()
        self.assertEqual(self.recorder.round, 2)
        self.assertEqual(self.recorder.requests["meta"]["round_count"], 3)

    def test_log_event_uses_current_round(self):
        """log_event should tag calls with the current round number."""
        self.recorder.log_event(
            from_="Player 1", to="GM",
            action={"type": "get message", "content": "r0"},
            call=({"p": 1}, {"r": 1})
        )
        self.recorder.log_next_round()
        self.recorder.log_event(
            from_="Player 1", to="GM",
            action={"type": "get message", "content": "r1"},
            call=({"p": 2}, {"r": 2})
        )
        self.assertEqual(self.recorder.requests["calls"][0]["round"], 0)
        self.assertEqual(self.recorder.requests["calls"][1]["round"], 1)

    def test_log_game_end_sets_completed(self):
        """log_game_end should set completed=True."""
        self.assertIsNone(self.recorder.requests["meta"]["completed"])
        self.recorder.log_game_end()
        self.assertTrue(self.recorder.requests["meta"]["completed"])

    def test_log_game_end_sets_round_count(self):
        """log_game_end should set round_count even if log_next_round was never called."""
        self.recorder.log_game_end()
        self.assertEqual(self.recorder.requests["meta"]["round_count"], 1)

    def test_log_game_end_after_rounds(self):
        """log_game_end should reflect correct round_count after multiple rounds."""
        self.recorder.log_next_round()
        self.recorder.log_next_round()
        self.recorder.log_game_end()
        self.assertEqual(self.recorder.requests["meta"]["round_count"], 3)

    def test_deepcopy_in_log_event(self):
        """log_event should deepcopy call objects to prevent mutation issues."""
        prompt = {"messages": [{"role": "user", "content": "hello"}]}
        response = {"choices": [{"text": "hi"}]}
        self.recorder.log_event(
            from_="Player 1", to="GM",
            action={"type": "get message", "content": "test"},
            call=(prompt, response)
        )
        # Mutate original objects
        prompt["messages"].append({"role": "assistant", "content": "mutated"})
        response["choices"].append({"text": "mutated"})
        # Recorded call should be unchanged
        recorded = self.recorder.requests["calls"][0]["call"]
        self.assertEqual(len(recorded["manipulated_prompt_obj"]["messages"]), 1)
        self.assertEqual(len(recorded["raw_response_obj"]["choices"]), 1)


if __name__ == "__main__":
    unittest.main()
