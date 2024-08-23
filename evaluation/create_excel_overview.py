"""
Script for creating an excel summary of reference game results
for qualitative analysis
"""

import os
import pandas as pd
import json
from pathlib import Path
import re
from collections import defaultdict
import argparse


def natural_sort_key(s):
    """
    Generiert einen Sortierschlüssel für natürliche Sortierung von Zeichenketten
    Nötig, da Ordnung der Episoden-Ordner sonst 1, 10, 11, ..., 2, 20, 21, ... wäre
    # TODO: kurze Beschreibung, was hier genau passiert
    :param s:
    :return:
    """
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def extract_instance(instance_path):
    """
    Extrahiert das 'Target Grid', die 'Distractor Grids' und die korrrekte Antwort.

    Args:
        instance_path (Path): Der Pfad zur Datei instance.json

    Returns:
        tuple: Ein Tupel bestehend aus dem extrahierten 'Target Grid', den beiden 'Distractor Grids' und der korrekten Antwort.
    """
    with instance_path.open("r", encoding="utf-8") as file:
        instance_data = json.load(file)

        target_grid = instance_data["player_1_target_grid"]
        distractor_grid_1 = instance_data["player_1_second_grid"]
        distractor_grid_2 = instance_data["player_1_third_grid"]
        gold_label = instance_data["target_grid_name"]
        # fix for te_google: accidentally inserted '(' in target_grid_name -> ["(ప్రధమ", "1"] (only mixtral and llama-3-70b)
        if "te_google" in str(instance_path):
            gold_label[0] = gold_label[0].lstrip("(")
        return target_grid, distractor_grid_1, distractor_grid_2, gold_label


def extract_player_expressions(interaction_path):
    """
    Extrahiert die Expression von Player 1 und 2 aus dem gegebenen Datensatz.

    Args:
        interaction_path (Path): Der Pfad zur Dati interctions.json.

    Returns:
        str, str: Die extrahierten Expressions von Player 1 und 2 oder jeweils None, falls keine gefunden wurde.
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
        try:
            # The parsed expression without tag, if valid (is stored in interaction 2; GM to GM)
            if interaction["turns"][0][2]["action"]["type"] == "parse":
                p1_parsed = interaction["turns"][0][2]["action"]["expression"]
            elif interaction["turns"][0][2]["action"]["type"] == "invalid format":
                p1_parsed = interaction["turns"][0][2]["action"]["content"]
        except IndexError:
            p1_parsed = None
        try:
            # The answer without tag, if valid (is stored in interaction 5)
            if interaction["turns"][0][5]["action"]["type"] == "parse":
                p2_parsed = interaction["turns"][0][5]["action"]["answer"]
            elif interaction["turns"][0][5]["action"]["type"] == "invalid format":
                p2_parsed = interaction["turns"][0][5]["action"]["content"]
        except IndexError:
            p2_parsed = None
    return p1, p2, p1_parsed, p2_parsed


def process_triplet(triplet_path, all_target_grids, all_formula_cells):
    # loop through instances in triplet, extracting the grids only once from first instance
    for i in [0, 1, 2]:
        instance = triplet_path[i]
        string_path = str(instance).split(os.sep) # split directories by os-dependent separator ("/" or "\")
        episode = string_path[-1]
        experiment_name = string_path[-2]
        if i == 0:
            #define formatting
            excel_row = len(all_target_grids) + 2
            all_formula_cells.append((f"D{excel_row}", f'=TEXTSPLIT(C{excel_row}," ",CHAR(10), TRUE)'))
            all_formula_cells.append((f"K{excel_row}", f'=TEXTSPLIT(J{excel_row}," ",CHAR(10), TRUE)'))
            all_formula_cells.append((f"R{excel_row}", f'=TEXTSPLIT(Q{excel_row}," ",CHAR(10), TRUE)'))

        # extract instance content and model responses
        target_grid, distractor_grid_1, distractor_grid_2, gold_answer = extract_instance(
            Path(os.path.join(instance, "instance.json")))
        player1_text, player2_text, player1_expression, player2_answer = extract_player_expressions(
            Path(os.path.join(instance, "interactions.json")))

        # collect all information in specific cells
        all_target_grids.append(
            {
                "Experiment": experiment_name if i == 0 else "",
                "Episode": episode,
                "Target Grid": target_grid if i == 0 else "",
                "T1": "",
                "T2": "",
                "T3": "",
                "T4": "",
                "T5": "",
                " ": "",
                "Distractor Grid 1": distractor_grid_1 if i == 0 else "",
                "D1": "",
                "D2": "",
                "D3": "",
                "D4": "",
                "D5": "",
                "  ": "",
                "Distractor Grid 2": distractor_grid_2 if i == 0 else "",
                "D6": "",
                "D7": "",
                "D8": "",
                "D9": "",
                "D10": "",
                "Target Position": i + 1,
                "Player 1 Text": player1_text,  # using the captured expression
                "Player 2 Text": player2_text,
                "Ground Truth": gold_answer,
                "Player 1 Parsed Expression": player1_expression,
                "Player 2 Parsed Answer": player2_answer
            }
        )
    # Add three empty rows
    all_target_grids.extend([{}] * 3)

    return all_target_grids, all_formula_cells


def process_folders(result_path, game_name):
    """
    Processes all model folders in the given 'results' folder for the given game_name.

    This function iterates over each directory in the provided path. If the directory contains a game_name subdirectory,
    it processes all experiments. For each experiment, it sorts and groups the interaction files into triplets.

    Args:
        result_path (Path): The path to the 'results' directory containing the model folders to process.
        game_name: the name of the game to process ('referencegame' for text only, 'referencegame_multimodal' for images)
    """

    for model_folder in result_path.glob('*/'):
        game_folder = Path(os.path.join(model_folder, game_name))
        if game_folder.exists():

            all_target_grids = []
            all_formula_cells = []  # AB: What are these for? Formatting?

            # loop through all experiments
            for experiment_folder in sorted(game_folder.glob('*/'), key=natural_sort_key):
                print(f"Processing {experiment_folder}")
                # sort episodes (otherwise order would be 1, 10, 11, ..., 2, 20, 21, ... )
                episode_paths = sorted(
                    experiment_folder.glob("episode_*"), key=natural_sort_key
                )

                # Erstellen von Tripeln aus den Episodendateipfaden
                triplets = [
                    episode_paths[i: i + 3] for i in range(0, len(episode_paths), 3)
                ]

                for triplet in triplets:
                    all_target_grids, all_formula_cells = process_triplet(triplet, all_target_grids, all_formula_cells)

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
            write_excel(output_file, all_target_grids, all_formula_cells)
        else:
            print(f"No game transcripts found for {game_name} in {model_folder}")


def write_excel(output_file, all_target_grids, all_formula_cells):
    """
    TODO: Description
    :param output_file:
    :param all_target_grids:
    :param all_formula_cells:
    :return:
    """
    # Using 'xlsxwriter' as the engine to write to an existing Excel file
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:

        df = pd.DataFrame(all_target_grids)
        df.to_excel(writer, index=False)

        workbook = writer.book
        worksheet = writer.sheets["Sheet1"]

        # Anpassen der Spaltenbreite und Hinzufügen von Formeln usw.
        for i, col in enumerate(df.columns):
            max_length = max(df[col].astype(str).map(len).max(), len(col))
            worksheet.set_column(i, i, max_length * 0.8)

        worksheet.set_column("C:C", 0)
        worksheet.set_column("J:J", 0)
        worksheet.set_column("Q:Q", 0)

        worksheet.merge_range('D1:H1', "Target Grid")
        worksheet.merge_range('K1:O1', "Distractor 1")
        worksheet.merge_range('R1:V1', "Distractor 2")


        for cell, formula in all_formula_cells:
            worksheet.write_formula(cell, formula)

        #colour-code (in)correct answerd
        format_red = workbook.add_format({"bg_color": "#FFC7CE"})
        format_green = workbook.add_format({"bg_color": "#C6EFCE"})
        for i in range(400):
            worksheet.conditional_format(f'Y{i}', {'type': 'text',
                                                   'criteria': 'containing',
                                                   'value': f'Z{i}',
                                                   'format': format_green})

        print(f"Created excel file: {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("result_path", help="Path where the result files are stored by model")
    parser.add_argument("-game_name", choices=['referencegame', 'multimodal_referencegame'], default="referencegame", help="name of the game, choices are 'referencegame' or 'multimodal_referencegame'")
    args = parser.parse_args()

    result_path = Path(args.result_path)
    if result_path.exists() and result_path.is_dir():
        print(f"Starting processing for results folder: {result_path} and game: {args.game_name}")
        process_folders(result_path, args.game_name)
    else:
        print(f"Path doesn't exist: {result_path}")
