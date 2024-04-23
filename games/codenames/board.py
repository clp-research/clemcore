from typing import Dict, List
from .constants import TEAM, INNOCENT, OPPONENT, ASSASSIN, HIDDEN, REVEALED

class CodenamesBoard:
    def __init__(self, team_words, opponent_words, innocent_words, assassin_words, random_order, flags):
        self.hidden = {TEAM: team_words, INNOCENT: innocent_words, OPPONENT: opponent_words, ASSASSIN: assassin_words}
        self.revealed = {TEAM: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []},
                         OPPONENT: {TEAM: [], INNOCENT: [], OPPONENT: [], ASSASSIN: []}}
        self.random_order = random_order
        self.flags = flags

    def get_current_board(self) -> Dict:
        return {HIDDEN: self.hidden, 
                REVEALED: self.revealed}
    
    def get_word_assignment(self, word) -> str:
        for assignment in self.hidden:
            if word in self.hidden[assignment]:
                return assignment
            
        for team in self.revealed:
            for assignment in self.revealed[team]:
                if word in self.revealed[team]:
                    return assignment

    def get_all_hidden_words(self) -> List:
        hidden_words = []
        for assignment in self.hidden:
            hidden_words.extend(self.hidden[assignment])
        randomly_ordered_hidden_words = []
        for word in self.random_order:
            if word in hidden_words:
                randomly_ordered_hidden_words.append(word)
        return randomly_ordered_hidden_words

    def get_hidden_words(self, with_assignment: str) -> List:
        return self.hidden[with_assignment]

    def get_revealed_words(self, by: str) -> List:
        revealed_words = []
        for assignment in self.revealed[by]:
            revealed_words.extend(self.revealed[by][assignment])
        return revealed_words

    def reveal_word(self, word: str, by: str = TEAM):
        for assignment in self.hidden:
            if word in self.hidden[assignment]:
                self.revealed[by][assignment].append(word)
                self.hidden[assignment].remove(word)
                return assignment

        if not self.flags["IGNORE FALSE TARGETS OR GUESSES"]:
            raise ValueError(f"Word '{word}' was not found amongst the hidden words on the board, cannot be revealed.")
    
    def should_continue_after_revealing(self, word: str, by: str = TEAM):
        return word in self.revealed[by][by]
    
    def has_team_won(self) -> bool:
        return len(self.hidden[TEAM]) == 0
    
    def has_team_won_through_assassin(self) -> bool:
        return len(self.revealed[OPPONENT][ASSASSIN]) >= 1

    def has_opponent_won(self) -> bool:
        return len(self.hidden[OPPONENT]) == 0

    def has_opponent_won_through_assassin(self) -> bool:
        return len(self.revealed[TEAM][ASSASSIN]) >= 1
    