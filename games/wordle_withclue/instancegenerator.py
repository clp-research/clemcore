"""
Generate instances for the wordle_withclue game.

Creates files in ./instances
"""
from games.wordle.instancegenerator import WordleGameInstanceGenerator

GAME_NAME = "wordle_withclue"

if __name__ == "__main__":
    WordleGameInstanceGenerator(GAME_NAME).generate()
