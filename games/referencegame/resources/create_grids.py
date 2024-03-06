import json
from collections import defaultdict


def create_grids(outfile):
    """
    semi-automatically generate grids to create instances from
    and stores them in a json file
    :param outfile: filename for output grid file
    :return:
    """
    grids = defaultdict(list)

    # create line grids
    filled_line = "X X X X X\n"
    empty_line = "\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n"

    horizontal_lines_1 = ["X \u25a2 \u25a2 \u25a2 \u25a2\n", "\u25a2 X \u25a2 \u25a2 \u25a2\n", "\u25a2 \u25a2 X \u25a2 \u25a2\n", "\u25a2 \u25a2 \u25a2 X \u25a2\n", "\u25a2 \u25a2 \u25a2 \u25a2 X\n"]
    horizontal_lines_4 = ["\u25a2 X X X X\n", "X \u25a2 X X X\n", "X X \u25a2 X X\n", "X X X \u25a2 X\n", "X X X X \u25a2\n"]

    # create all possible grids with one row line and 4 row lines
    for filled_line_id in range(5):
        grid = ""
        for line_id in range(5):
            if filled_line_id == line_id:
                grid += filled_line
            else:
                grid += empty_line
        grids["line_grids_rows"].append(grid.rstrip())
        grid = ""
        for line_id in range(5):
            if filled_line_id == line_id:
                grid += empty_line
            else:
                grid += filled_line
        grids["line_grids_rows"].append(grid.rstrip())

    # create all possible grids with one column line and 4 column lines
    for filled_line_id in range(5):
        grid = ""
        for line_id in range(5):
            grid += horizontal_lines_1[filled_line_id]
        grids["line_grids_columns"].append(grid.rstrip())
        grid = ""
        for line_id in range(5):
            grid += horizontal_lines_4[filled_line_id]
        grids["line_grids_columns"].append(grid.rstrip())

    # get diagonal line grids from
    with open("grids_v01.json", 'r') as f:
        grid_dict = json.load(f)
    easy_grids = grid_dict["easy_grids"]
    diagonal_ids = [0,1,18,19]
    for i in diagonal_ids:
        grids["diagonal_grids"].append(easy_grids[i])
    grids["diagonal_grids"].append("X \u25a2 \u25a2 \u25a2 X\n\u25a2 X \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 X \u25a2 X \u25a2\nX \u25a2 \u25a2 \u25a2 X")  # X
    grids["diagonal_grids"].append("\u25a2 X X X \u25a2\nX \u25a2 X \u25a2 X\nX X \u25a2 X X\nX \u25a2 X \u25a2 X\n\u25a2 X X X \u25a2")  # negative X
    grids["diagonal_grids"].append("X \u25a2 \u25a2 \u25a2 \u25a2\nX X \u25a2 \u25a2 \u25a2\nX X X \u25a2 \u25a2\nX X X X \u25a2\nX X X X X")
    grids["diagonal_grids"].append("\u25a2 \u25a2 \u25a2 \u25a2 X\n\u25a2 \u25a2 \u25a2 X X\n\u25a2 \u25a2 X X X\n\u25a2 X X X X\nX X X X X")
    grids["diagonal_grids"].append("X X X X X\nX X X X \u25a2\nX X X \u25a2 \u25a2\nX X \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2")
    grids["diagonal_grids"].append("X X X X X\n\u25a2 X X X X\n\u25a2 \u25a2 X X X\n\u25a2 \u25a2 \u25a2 X X\n\u25a2 \u25a2 \u25a2 \u25a2 X")

    # create letter grids
    grids["letter_grids"].append("X X X X X\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X X X")  # E
    grids["letter_grids"].append("X X X X X\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2")  # F
    grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 \u25a2 X\nX X X X X\nX \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 \u25a2 X")  # H
    #grids["letter_grids"].append("\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2")  # I
    grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 X \u25a2\nX X X \u25a2 \u25a2\nX \u25a2 \u25a2 X \u25a2\nX \u25a2 \u25a2 \u25a2 X")  # K
    grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX \u25a2 \u25a2 \u25a2 \u25a2\nX X X X X")  # L
    grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 X\nX X \u25a2 X X\nX \u25a2 X \u25a2 X\nX \u25a2 \u25a2 \u25a2 X\nX \u25a2 \u25a2 \u25a2 X")  # M
    grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 X\nX X \u25a2 \u25a2 X\nX \u25a2 X \u25a2 X\nX \u25a2 \u25a2 X X\nX \u25a2 \u25a2 \u25a2 X")  # N
    grids["letter_grids"].append("X X X X X\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2")  # T
    #grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 X\n\u25a2 X \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 X \u25a2 X \u25a2\nX \u25a2 \u25a2 \u25a2 X")  # X
    grids["letter_grids"].append("X \u25a2 \u25a2 \u25a2 X\n\u25a2 X \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2")  # Y
    grids["letter_grids"].append("X X X X X\n\u25a2 \u25a2 \u25a2 X \u25a2\n\u25a2 \u25a2 X \u25a2 \u25a2\n\u25a2 X \u25a2 \u25a2 \u25a2\nX X X X X")  # Z

    # import shape grids from older versions (manually selected)

    with open("grids_v01.json", 'r') as f:
        grid_dict = json.load(f)
    easy_ids = [4,5,10,17]
    hard_ids = []# 1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18]
    for grid_collection in grid_dict.keys():
        if grid_collection == "easy_grids":
            for i in easy_ids:
                if grid_dict[grid_collection][i] not in grids["shape_grids"]:
                    grids["shape_grids"].append(grid_dict[grid_collection][i])
        elif grid_collection == "hard_grids":
            for i in hard_ids:
                if grid_dict[grid_collection][i] not in grids["shape_grids"]:
                    grids["shape_grids"].append(grid_dict[grid_collection][i])

    with open("grids_v02.json", 'r') as f:
        grid_dict = json.load(f)
    ids = [2, 3, 5, 8, 16, 17, 18]
    for grid_collection in grid_dict.keys():
        for i in ids:
            if grid_dict[grid_collection][i] not in grids["shape_grids"]:
                grids["shape_grids"].append(grid_dict[grid_collection][i])

    # add one "empty" grid for creating random grids
    grids["random_grids"].append("\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2\n\u25a2 \u25a2 \u25a2 \u25a2 \u25a2")

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


create_grids("grids_v03.json")
pretty_print("grids_v03.json")