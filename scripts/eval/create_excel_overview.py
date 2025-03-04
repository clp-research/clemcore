"""
Script for creating an excel summary of reference and multimodal_reference game results
for qualitative analysis
"""

import os
import pandas as pd
import json
from pathlib import Path
import re
import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("overview.py")


def natural_sort_key(s):
    """
    Generates a sorting key for natural sorting of strings.
    This is needed because, in a typical lexicographical sort, the order would be 1, 10, 11, ..., 2, 20, 21, ...
    Natural sorting ensures that numbers are sorted in a human-friendly way, such as 1, 2, 3, ..., 10, 11, ...

    How it works:
    The function splits the input string into segments of numbers and non-numbers using a regular expression.
    For each segment:
        - If the segment is a number, it is converted to an integer.
        - If the segment is not a number, it is converted to lowercase.
    This way, numeric segments are sorted as integers and non-numeric segments are sorted as strings.

    :param s: The input string to be sorted.
    :return: A list of mixed types (integers and strings) for natural sorting.
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def insert_image_in_excel(worksheet, cell, img_path, x_scale=0.25, y_scale=0.25):
    """
    Inserts an image into a specified Excel cell.

    This function takes the path to an image and inserts it into a specified cell of the given Excel worksheet.
    It allows scaling of the image using the x_scale and y_scale parameters. If the image file does not exist,
    the function returns False. If insertion is successful, it returns True.

    Args:
        worksheet (xlsxwriter.worksheet.Worksheet): The worksheet object where the image should be inserted.
        cell (str): The cell reference (e.g., "A1") where the image will be inserted.
        img_path (str or Path): The file path to the image to be inserted.
        x_scale (float, optional): The scaling factor for the image width. Default is 0.25.
        y_scale (float, optional): The scaling factor for the image height. Default is 0.25.

    Returns:
        bool: True if the image was successfully inserted, False otherwise.

    Raises:
        None: If an exception occurs during image insertion, it is caught and an error message is printed.

    """
    img_path = str(img_path)  # Convert Path object to string for compatibility with xlsxwriter
    if not Path(img_path).exists():
        logger.error(f"Image not found: {img_path}")
        return False
    try:
        # Directly insert the image into the specified cell
        worksheet.insert_image(cell, img_path, {'x_scale': x_scale, 'y_scale': y_scale})
        logger.debug(f"Image successfully inserted at {cell}: {img_path}")
        return True
    except Exception as e:  # Catch any exceptions that occur during image insertion
        logger.error(f"Failed to insert image: {e}")
        return False


def extract_instance(instance_path, base_path, game_name):
    """
    Extracts either image paths (for multimodal_referencegame) or grid data (for referencegame) and the correct answer.

    Args:
        instance_path (Path): The file path to the instance JSON file.
        base_path (Path, optional): The base directory path where the image files are stored (only used for multimodal_referencegame).
        game_name (str): The name of the game, which determines whether to extract images or grid data.

    Returns:
        tuple: A tuple containing:
            - For multimodal_referencegame:
                - target_grid (Path): The full path to the target grid image.
                - distractor_grid_1 (Path): The full path to the first distractor grid image.
                - distractor_grid_2 (Path): The full path to the second distractor grid image.
                - gold_label (str): The label for the correct target image.
            - For referencegame:
                - target_grid (str): The target grid in text form.
                - distractor_grid_1 (str): The first distractor grid in text form.
                - distractor_grid_2 (str): The second distractor grid in text form.
                - gold_label (str): The label for the correct target grid.
    """
    logger.debug(f"Game name in extract_instance: {game_name}")  # Debug-Ausgabe

    with instance_path.open("r", encoding="utf-8") as file:
        instance_data = json.load(file)

        if game_name == "multimodal_referencegame":
            # Ensure base_path is a Path object
            if isinstance(base_path, str):
                base_path = Path(base_path)

            # Use the keys for multimodal data
            target_grid = base_path / Path(instance_data.get("player_1_first_image", "")).name
            distractor_grid_1 = base_path / Path(instance_data.get("player_1_second_image", "")).name
            distractor_grid_2 = base_path / Path(instance_data.get("player_1_third_image", "")).name
            gold_label = instance_data.get("target_image_name", "")

        elif game_name == "referencegame":
            # Use the keys for referencegame data
            target_grid = instance_data.get("player_1_target_grid", "")
            distractor_grid_1 = instance_data.get("player_1_second_grid", "")
            distractor_grid_2 = instance_data.get("player_1_third_grid", "")
            gold_label = instance_data.get("target_grid_name", "")

        else:
            raise ValueError("Unsupported game name. Choose either 'multimodal_referencegame' or 'referencegame'.")

        return target_grid, distractor_grid_1, distractor_grid_2, gold_label


def extract_player_expressions(interaction_path):
    """
    Extracts the expressions of Player 1 and Player 2 from the given dataset.

    Args:
        interaction_path (Path): The path to the interactions.json file.

    Returns:
        str, str: The extracted expressions of Player 1 and Player 2, or None if no expression is found for either.
    """
    with interaction_path.open("r", encoding="utf-8") as file:
        interaction = json.load(file)
        try:
            # The expression from Player 1 is stored in interaction 1 in turn 0 in the interactions.json
            p1 = interaction["turns"][0][1]["action"]["content"]
        except IndexError:
            p1 = None
        try:
            # The expression from Player 2 is stored in interaction 4 in turn 0 in the interactions.json
            p2 = interaction["turns"][0][4]["action"]["content"]
        except IndexError:
            p2 = None

    return p1, p2


def process_triplet(triplet_path, all_target_grids, all_formula_cells, base_path, game_name):
    """
    Processes a triplet of instances, extracting image paths and player expressions,
    and appends this information to a list for generating an Excel summary.

    Args:
        triplet_path (list): A list of paths for three instances (episodes) that make up a triplet.
        all_target_grids (list): A list where the processed information for the target grids will be stored.
        all_formula_cells (list): A list for storing any formula-related information (currently unused).
        base_path (Path): The base directory where images are stored.

    Returns:
        tuple: A tuple containing the updated list of target grid information (all_target_grids)
               and the formula cells list (all_formula_cells).

    The function processes each instance in the triplet, extracting relevant data such as
    the experiment name, episode name, target grid, distractor grids, expressions of players,
    and the correct answer (ground truth). This data is then formatted as a dictionary and
    appended to the all_target_grids list. Additionally, four empty rows are added between
    processed triplets for spacing.
    """
    # loop through each instance in the triplet
    for i in [0, 1, 2]:
        instance = triplet_path[i]
        string_path = str(instance).split(os.sep)
        episode = string_path[-1]
        experiment_name = string_path[-2]

        # Separate extraction and structure for each game type
        if game_name == "referencegame":
            # For referencegame, extract grid data and use text-only structure
            target_grid, distractor_grid_1, distractor_grid_2, gold_answer = extract_instance(
                Path(os.path.join(instance, "instance.json")), None, game_name
            )
            player1_expression, player2_answer = extract_player_expressions(
                Path(os.path.join(instance, "interactions.json"))
            )
            # Append data in referencegame structure
            all_target_grids.append({
                "Experiment": experiment_name if i == 0 else "",
                "Episode": episode,
                "Target Grid": target_grid if i == 0 else "",
                "Distractor Grid 1": distractor_grid_1 if i == 0 else "",
                "Distractor Grid 2": distractor_grid_2 if i == 0 else "",
                "Target Position": i + 1,
                "Player 1 Expression": player1_expression,
                "Player 2 Answer": player2_answer,
                "Ground Truth": gold_answer,
            })

            # FÃ¼ge Leerzeilen fÃ¼r AbstÃ¤nde zwischen den Triplets hinzu
            all_target_grids.extend([{}] * 4)


        elif game_name == "multimodal_referencegame":
            # For multimodal, extract image paths and use structure with images
            target_grid, distractor_grid_1, distractor_grid_2, gold_answer = extract_instance(
                Path(os.path.join(instance, "instance.json")), base_path, game_name
            )
            player1_expression, player2_answer = extract_player_expressions(
                Path(os.path.join(instance, "interactions.json"))
            )
            # Append data in multimodal structure, allowing for image insertions
            all_target_grids.append({
                "Experiment": experiment_name if i == 0 else "",
                "Episode": episode,
                "Target Grid": target_grid if i == 0 else "",
                "T1": "", "T2": "", "T3": "", "T4": "", "T5": "",
                " ": "",
                "Distractor Grid 1": distractor_grid_1 if i == 0 else "",
                "D1": "", "D2": "", "D3": "", "D4": "", "D5": "",
                "  ": "",
                "Distractor Grid 2": distractor_grid_2 if i == 0 else "",
                "D6": "", "D7": "", "D8": "", "D9": "", "D10": "",
                "Target Position": i + 1,
                "Player 1 Expression": player1_expression,
                "Player 2 Answer": player2_answer,
                "Ground Truth": gold_answer,
            })

    # Add empty rows between triplets for spacing
    empty_rows = [{}] * (3 if game_name == "referencegame" else 4)
    all_target_grids.extend(empty_rows)

    return all_target_grids, all_formula_cells


def process_folders(result_path, image_path):
    """
    Processes all model folders in the given results folder.

    This function iterates over each directory in the provided path. If an image_path id given, it process all
    'multimodal_referencegame' results, otherwise it looks for 'referencegame' subdirectories.
    It processes all experiments. For each experiment, it sorts and groups the interaction files into triplets.

    Args:
        result_path (Path): The path to the 'results' directory containing the model folders to process.
        image_path (Path): the path to the image files for 'multimodal_referencegame', or None for 'referencegame'
    """

    if image_path:
        game_name = "multimodal_referencegame"
    else:
        game_name = "referencegame"

    for model_folder in result_path.glob('*/'):
        game_folder = Path(os.path.join(model_folder, game_name))
        if game_folder.exists():
            logger.debug(f"Processing game folder: {game_folder}")  # Debug-Ausgabe

            all_target_grids = []
            all_formula_cells = []

            # loop through all experiments
            for experiment_folder in sorted(game_folder.glob('*/'), key=natural_sort_key):
                logger.debug(f"Processing {experiment_folder}")
                # sort episodes (otherwise order would be 1, 10, 11, ..., 2, 20, 21, ... )
                episode_paths = sorted(
                    experiment_folder.glob("episode_*"), key=natural_sort_key
                )

                # Check if episodes were found
                if not episode_paths:
                    logger.warning(f"No episodes found in {experiment_folder}. Skipping...")
                    continue

                # Creating triples from the episode file paths
                triplets = [
                    episode_paths[i: i + 3] for i in range(0, len(episode_paths), 3)
                ]

                # Check if valid triples exist
                if not triplets:
                    logger.warning(f"No valid triplets found in {experiment_folder}. Skipping...")
                    continue

                for triplet in triplets:
                    all_target_grids, all_formula_cells = process_triplet(triplet, all_target_grids, all_formula_cells,
                                                                          image_path, game_name)

            # Check if data is available before writing the Excel file
            if all_target_grids:
                output_file = model_folder / f"{model_folder.name}.xlsx"
                write_excel(output_file, all_target_grids, all_formula_cells, game_name)
                logger.info(f"Excel file created: {output_file}")  # Debug-Ausgabe
            else:
                logger.error(f"No data to write for {model_folder}.")

                # Player 2 Counts
                # counts the answers for each experiment
                # answers_count = defaultdict(int)  # Dictionary to count answers
                # total_answers = 0  # Total number of answers processed
                #
                # # Geht durch jede Interaktionsdatei
                # for episode_file in sorted(
                #         experiment_folder.glob("episode_*/interactions.json"), key=natural_sort_key
                # ):
                #     with episode_file.open("r") as file:
                #         data = json.load(file)
                #         extract_player2_answer(data, episode_file, experiment_name, answers_count)
                #         total_answers += 1  # counts every file that has been processed
                #
                # Calculate and display the answer distribution
                # if total_answers > 0:
                #     print(f"\nAnswer distribution for {experiment_name}:")
                #     for answer, count in answers_count.items():
                #         percentage = (count / total_answers) * 100
                #         print(f"{answer}: {percentage:.2f}%")
                # else:
                #     print(f"No answers found in {experiment_name}.")
                # # Define the expected distribution
                # # TODO: extract from gold answers!
                # right_distribution = {"First": 33.3, "Second": 33.3, "Third": 33.3}
                #
                # # Display the expected distribution
                # print(f"\nRight distribution for {experiment_name}:")
                # for answer, percentage in right_distribution.items():
                #     print(f"{answer}: {percentage}%")
                #
                # print("\n")
                # interactions_paths = sorted(
                #     experiment_folder.glob("episode_*/interactions.json"), key=natural_sort_key
                # )
                #
                # if not interactions_paths:
                #     print(f"No interaction files found in {experiment_folder}. Skipping...")
                #     continue

            # Write output to excel file for manual annotation
            output_file = model_folder / f"{model_folder.name}.xlsx"
            write_excel(output_file, all_target_grids, all_formula_cells, game_name)
        else:
            logger.warning(f"No game transcripts found for {game_name} in {model_folder}")


def write_excel(output_file, all_target_grids, all_formula_cells, game_name):
    """
    Creates an Excel file and inserts images or grid text based on the game_name.

    Args:
        output_file (str): The path to the output Excel file.
        all_target_grids (list): A list containing all the data to be written to the Excel file.
        all_formula_cells (list): A list of cells with formulas (currently unused).
        game_name (str): The name of the game, which determines whether to insert images or grid text.
    """
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        # Only include essential columns for referencegame
        if game_name == "referencegame":
            df = pd.DataFrame(all_target_grids, columns=[
                "Experiment", "Episode", "Target Grid", "Distractor Grid 1",
                "Distractor Grid 2", "Target Position", "Player 1 Expression",
                "Player 2 Answer", "Ground Truth"
            ])
        else:
            df = pd.DataFrame(all_target_grids, columns=[
                "Experiment", "Episode", "Target Grid", "Distractor Grid 1",
                "Distractor Grid 2", "Target Position", "Player 1 Expression",
                "Player 2 Answer", "Ground Truth"
            ])

        # Write the DataFrame to the Excel file
        df.to_excel(writer, index=False, header=True)

        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        # Column widths adjustment
        column_widths = {
            "A": 15, "B": 10, "C": 25, "D": 25, "E": 25,
            "F": 15, "G": 60, "H": 20, "I": 15
        }
        for col, width in column_widths.items():
            worksheet.set_column(f"{col}:{col}", width)

        # Set uniform row height
        row_height = 20
        for row_index in range(1, len(all_target_grids) + 1):
            worksheet.set_row(row_index, row_height)

        # Insert images or grids as text
        row_index = 1
        for index, original_row in enumerate(all_target_grids):
            if game_name == "multimodal_referencegame":
                if original_row.get("Target Grid"):
                    insert_image_in_excel(worksheet, f"C{row_index + 1}", original_row["Target Grid"], x_scale=0.25,
                                          y_scale=0.25)
                if original_row.get("Distractor Grid 1"):
                    insert_image_in_excel(worksheet, f"D{row_index + 1}", original_row["Distractor Grid 1"],
                                          x_scale=0.25, y_scale=0.25)
                if original_row.get("Distractor Grid 2"):
                    insert_image_in_excel(worksheet, f"E{row_index + 1}", original_row["Distractor Grid 2"],
                                          x_scale=0.25, y_scale=0.25)
            elif game_name == "referencegame":
                # For referencegame, split the grid text into lines for each grid type
                target_grid_lines = original_row.get("Target Grid", "").split('\n')
                distractor_grid_1_lines = original_row.get("Distractor Grid 1", "").split('\n')
                distractor_grid_2_lines = original_row.get("Distractor Grid 2", "").split('\n')

                # Ensure all three grids are written in a 5x5 format
                for line_index in range(5):  # Assuming each grid has exactly 5 lines
                    worksheet.write(row_index + line_index, 2,
                                    target_grid_lines[line_index] if line_index < len(target_grid_lines) else "")
                    worksheet.write(row_index + line_index, 3, distractor_grid_1_lines[line_index] if line_index < len(
                        distractor_grid_1_lines) else "")
                    worksheet.write(row_index + line_index, 4, distractor_grid_2_lines[line_index] if line_index < len(
                        distractor_grid_2_lines) else "")

                row_index += 5  # Adding a space after each 5x5 grid set

            row_index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", type=Path, help="Path where the result files are stored by model")
    parser.add_argument("-i", "--images_path", help="Base path for images (required for multimodal_referencegame)",
                        default=None)
    args = parser.parse_args()

    if args.result_path.exists() and args.result_path.is_dir():
        process_folders(args.result_path, args.images_path)
    else:
        raise FileNotFoundError(f"Path doesn't exist: {args.result_path}")