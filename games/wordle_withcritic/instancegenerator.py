"""
Generate instances for the wordle_withcritic game.

Creates files in ./instances
"""

from games.wordle.instancegenerator import WordleGameInstanceGenerator

GAME_NAME = "wordle_withcritic"

if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()
