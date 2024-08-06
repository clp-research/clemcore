# MatchIt ASCII

Implemented by: Antonia Schmidt

MatchIt ASCII is a dialogue game where two players have to come to an agreement whether the 5x5 grid consisting of ASCII-characters, that each of them gets as input, is the same or not.
This is the 3q/base version of matchit_ascii, meaning that each player can ask the other three questions about the other grid.

# How to sample different number of instances

The instancegenerator.py generates instances by sampling N examples per difficulty (same, two kinds of similar and different grid - see below) from the list of grid pairs in resources/grid-pairs.csv (N $\leq$ 27 due to the number of pairs for difficulty similar_grid_1). N and other version parameters (such as the number of questions) are declared in the instancegenerator file. If N should be larger or other grid pairs are wanted, change N or the seed.

# Grids
Resources for matchit_ascii contain (apart from the necessary prompt templates) all possible grids in [grids_matchit.json](resources/grid_pairs/grids_matchit.json) with IDs to identify them. They are based on the grids from reference game (only from the categories *diagonals*, *letters* and *shapes*) and were manually extended by inverting, turning and mirroring the original grids where it made sense. Additionally for each of those grids there is a grid with an edit distance of two (for difficulty category similar_grid_2).

**Note:** ID 52 and 54 are missing because they were duplicates. 

[grid-pairs.csv](resources/grid_pairs/grid-pairs.csv) is a table pairing up the aforementioned grid ids into the difficulty categories. For each pair there is also a record of the action which would transform one grid to the other. These actions are:
- vflip: mirror along the vertical axis
- hflip: mirror along the horizontal axis
- add: one of the grids can be achieved by adding more Xs to the other grid
- inv: inversion (Xs become squares and vice versa)
- turn: turning a grid by 90 degrees
- edit2: invert two random positions in the grid

For the used difficulties, the following actions were chosen:
- same_grid: none
- similar_grid_1: vflip, hflip, turn
- similar_grid_2: edit_2
- different_grid: none
