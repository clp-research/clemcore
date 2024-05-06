"""
Create string grid representations for referencegame
Creates 10 5X5 grids in each of the 6 categories
line_grids_rows
line_grids_columns
line_grids_diagonal
letter_grids
shape_grids
random_grids
"""
import json
import random

GRID_FILE = "grids_v1.5.json"


def create_line_grids_rows():

    line_grids = []
    filled_line = "X X X X X\n"
    empty_line = "\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n"

    # create all possible grids with one row line and 4 row lines
    for filled_line_id in range(5):
        grid = ""
        for line_id in range(5):
            if filled_line_id == line_id:
                grid += filled_line
            else:
                grid += empty_line
        line_grids.append(grid.rstrip())
        grid = ""
        for line_id in range(5):
            if filled_line_id == line_id:
                grid += empty_line
            else:
                grid += filled_line
        line_grids.append(grid.rstrip())
    return line_grids


def create_line_grids_columns():

    line_grids = []
    horizontal_lines_1 = ["X \u25a2 \u25a2 \u25a2 \u25a2\n", "\u25a2 X \u25a2 \u25a2 \u25a2\n",
                          "\u25a2 \u25a2 X \u25a2 \u25a2\n", "\u25a2 \u25a2 \u25a2 X \u25a2\n",
                          "\u25a2 \u25a2 \u25a2 \u25a2 X\n"]
    horizontal_lines_4 = ["\u25a2 X X X X\n", "X \u25a2 X X X\n", "X X \u25a2 X X\n", "X X X \u25a2 X\n",
                          "X X X X \u25a2\n"]
    # create all possible grids with one column line and 4 column lines
    for filled_line_id in range(5):
        grid = ""
        for line_id in range(5):
            grid += horizontal_lines_1[filled_line_id]
        line_grids.append(grid.rstrip())
        grid = ""
        for line_id in range(5):
            grid += horizontal_lines_4[filled_line_id]
        line_grids.append(grid.rstrip())
    return line_grids


def create_line_grids_diagonal():

    diagonal_grids = []
    # get diagonal line grids from
    with open("grids_v0.9.json", 'r') as f:
        grid_dict = json.load(f)
    easy_grids = grid_dict["easy_grids"]
    diagonal_ids = [0, 1, 18, 19]
    for i in diagonal_ids:
        diagonal_grids.append(easy_grids[i])
    # add additional diagonal line grids
    diagonal_grids.append(
        "X \u25a2 \u25a2 \u25a2 X\n\u25a2 X \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 X \u25a2 X \u25a2\nX \u25a2 \u25a2 \u25a2 X")  # X
    diagonal_grids.append(
        "\u25a2 X X X \u25a2\nX \u25a2 X \u25a2 X\nX X \u25a2 X X\nX \u25a2 X \u25a2 X\n\u25a2 X X X \u25a2")  # negative X
    diagonal_grids.append(
        "X \u25a2 \u25a2 \u25a2 \u25a2\nX X \u25a2 \u25a2 \u25a2\nX X X \u25a2 \u25a2\nX X X X \u25a2\nX X X X X")
    diagonal_grids.append(
        "\u25a2 \u25a2 \u25a2 \u25a2 X\n\u25a2 \u25a2 \u25a2 X X\n\u25a2 \u25a2 X X X\n\u25a2 X X X X\nX X X X X")
    diagonal_grids.append(
        "X X X X X\nX X X X \u25a2\nX X X \u25a2 \u25a2\nX X \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2")
    diagonal_grids.append(
        "X X X X X\n\u25a2 X X X X\n\u25a2 \u25a2 X X X\n\u25a2 \u25a2 \u25a2 X X\n\u25a2 \u25a2 \u25a2 \u25a2 X")
    return diagonal_grids


def create_letter_grids():

    letter_grids = []
    # create letter grids for selected letters
    letter_grids.append(
        "X X X X X\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X X X")  # E
    letter_grids.append(
        "X X X X X\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2")  # F
    letter_grids.append(
        "X \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 \u25a2 X\nX X X X X\nX \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 \u25a2 X")  # H
    letter_grids.append(
        "X \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 X \u25a2\nX X X \u25a2 \u25a2\nX \u25a2 \u25a2 X \u25a2\nX \u25a2 \u25a2 \u25a2 X")  # K
    letter_grids.append(
        "X \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X X X")  # L
    letter_grids.append(
        "X \u25a2 \u25a2 \u25a2 X\nX X \u25a2 X X\nX \u25a2 X \u25a2 X\nX \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 \u25a2 X")  # M
    letter_grids.append(
        "X \u25a2 \u25a2 \u25a2 X\nX X \u25a2 \u25a2 X\nX \u25a2 X \u25a2 X\nX \u25a2 \u25a2 X X\nX \u25a2 \u25a2 \u25a2 X")  # N
    letter_grids.append(
        "X X X X X\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2")  # T
    letter_grids.append(
        "X \u25a2 \u25a2 \u25a2 X\n\u25a2 X \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2")  # Y
    letter_grids.append(
        "X X X X X\n\u25a2 \u25a2 \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 X \u25a2 \u25a2 \u25a2\nX X X X X")  # Z
    return letter_grids


def create_shape_grids():

    shape_grids = []
    # import shape grids from older versions (manually selected)

    with open("grids_v0.9.json", 'r') as f:
        grid_dict = json.load(f)
    easy_ids = [4, 5, 10, 17]
    for i in easy_ids:
        if grid_dict["easy_grids"][i] not in shape_grids:
            shape_grids.append(grid_dict["easy_grids"][i])

    with open("grids_v1.0.json", 'r') as f:
        grid_dict = json.load(f)
    ids = [2, 3, 5, 8, 16, 17, 18]
    for grid_collection in grid_dict.keys():
        for i in ids:
            if grid_dict[grid_collection][i] not in shape_grids:
                shape_grids.append(grid_dict[grid_collection][i])
    return shape_grids


def create_random_grids(seed=123, fills=10, num_grids=10):
    """
    Create random grids.
    :param seed: change this to create a different set of random grids (previously used: v1.5: 123)
    :param fills: number of Xs in random grid
    :param num_grids: number of grids to create
    :return: list of random grids
    """
    random.seed(seed)
    random_grids = []
    empty_grid = "\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2"

    while len(random_grids) != num_grids:
        x_positions = [i for i, char in enumerate(empty_grid) if char == "\u25a2"]
        random_x_positions = random.sample(x_positions, fills)
        # convert grid to list for indexing
        random_grid = list(empty_grid)
        for i in random_x_positions:
            random_grid[i] = "X"
        # convert grid back to string representation
        grid_string = "".join(random_grid)
        # avoid duplicates (but ensure same order)
        if grid_string not in random_grids:
            random_grids.append(grid_string)
    return random_grids


def create_grids(outfile):
    """
    semi-automatically generate grids to create instances from
    and stores them in a json file
    :param outfile: filename for output grid file
    :return:
    """
    grids = dict()

    grids["line_grids_rows"] = create_line_grids_rows()
    grids["line_grids_columns"] = create_line_grids_columns()
    grids["diagonal_grids"] = create_line_grids_diagonal()
    grids["letter_grids"] = create_letter_grids()
    grids["shape_grids"] = create_shape_grids()
    grids["random_grids"]= create_random_grids()

    #save grids to file
    with open(outfile, 'w') as f:
        json.dump(grids, f, ensure_ascii=False)


def pretty_print(grid_file):
    """
    Print grids in human readable format
    :param grid_file: .json file containing the grids
    :return:
    """

    with open(grid_file, 'r') as f:
        grid_dict = json.load(f)

    for grid_collection in grid_dict.keys():
        print(grid_collection)
        for grid in grid_dict[grid_collection]:
            print(f"{grid}\n")


create_grids(GRID_FILE)
pretty_print(GRID_FILE)