import glob
import json
import logging
import os
import html
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

import clemcore.clemgame.transcripts.constants as constants
import clemcore.clemgame.transcripts.patterns as patterns
from clemcore.utils import file_utils
from clemcore.clemgame.resources import store_file, load_json

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def _get_class_name(event):
    """Get a string representation of the direction of a message.
    Example: A message from the game's GM to Player 1 is represented by the string 'gm-a'.
    Args:
        event: The interaction record event to get the message direction for.
    Returns:
        The string representation of the direction of the message in the passed interaction event.
    """
    if event['from'] == 'GM' and event['to'].startswith('Player 1'):
        return "gm-a"
    if event['from'] == 'GM' and event['to'].startswith('Player 2'):
        return "gm-b"
    if event['from'].startswith('Player 1') and event['to'] == 'GM':
        return "a-gm"
    if event['from'].startswith('Player 2') and event['to'] == 'GM':
        return "b-gm"
    if event['from'] == 'GM' and event['to'] == 'GM':
        return "gm-gm"
    raise RuntimeError(f"Cannot handle event entry {event}")


def build_transcripts(top_dir: str, filter_games: List = None):
    """
    Create and store readable HTML and LaTeX episode transcripts from the interactions.json.
    Transcripts are stored as sibling files in the directory where the interactions.json is found.
    Args:
        top_dir: Path to a top directory.
        filter_games: Transcribe only interaction files which are part of the given games.
                      A game is specified by its name e.g. ['taboo']
    """
    if filter_games is None:
        filter_games = []
    interaction_files = glob.glob(os.path.join(top_dir, '**', 'interactions.json'), recursive=True)
    if filter_games:
        interaction_files = [interaction_file for interaction_file in interaction_files
                             if any(game_name in interaction_file for game_name in filter_games)]
    stdout_logger.info(f"Found {len(interaction_files)} interaction files to transcribe. "
                       f"Games: {filter_games if filter_games else 'all'}")
    error_count = 0
    for interaction_file in tqdm(interaction_files, desc="Building transcripts"):
        try:
            game_interactions = load_json(interaction_file)
            interactions_dir = Path(interaction_file).parent
            transcript = build_transcript(game_interactions)
            store_file(transcript, "transcript.html", interactions_dir)
            transcript_tex = build_tex(game_interactions)
            store_file(transcript_tex, "transcript.tex", interactions_dir)
        except Exception:  # continue with other episodes if something goes wrong
            module_logger.exception(f"Cannot transcribe {interaction_file} (but continue)")
            error_count += 1
    if error_count > 0:
        stdout_logger.error(f"'{error_count}' exceptions occurred: See clembench.log for details.")


def build_transcript(interactions: Dict):
    """Create an HTML file with the interaction transcript.
    The file is stored in the corresponding episode directory.
    Args:
        interactions: An episode interaction record dict.
        experiment_config: An experiment configuration dict.
        game_instance: The instance dict the episode interaction record is based on.
        dialogue_pair: The model pair descriptor string for the Players.
    """
    meta = interactions["meta"]
    transcript = patterns.HTML_HEADER.format(constants.CSS_STRING)
    title = f"Interaction Transcript for {meta['experiment_name']}, " \
            f"episode {meta['game_id']} with {meta['dialogue_pair']}."
    transcript += patterns.TOP_INFO.format(title)
    # Collect all events over all turns (ignore turn boundaries here)
    events = [event for turn in interactions['turns'] for event in turn]
    for event in events:
        class_name = _get_class_name(event)
        msg_content = event['action']['content']
        msg_raw = html.escape(f"{msg_content}").replace('\n', '<br/>')
        if event['from'] == 'GM' and event['to'] == 'GM':
            speaker = f'Game Master: {event["action"]["type"]}'
        else:
            speaker = f"{event['from'].replace('GM', 'Game Master')} to {event['to'].replace('GM', 'Game Master')}"
        # in case the content is a json BUT given as a string!
        # we still want to check for image entry
        if isinstance(msg_content, str):
            try:
                msg_content = json.loads(msg_content)
            except:
                ...
        style = "border: dashed" if "label" in event["action"] and "forget" == event["action"]["label"] else ""
        # in case the content is a json with an image entry
        if isinstance(msg_content, dict):
            if "image" in msg_content:
                transcript += f'<div speaker="{speaker}" class="msg {class_name}" style="{style}">\n'
                transcript += f'  <p>{msg_raw}</p>\n'
                for image_src in msg_content["image"]:
                    if not image_src.startswith("http"):  # take the web url as it is
                        if "IMAGE_ROOT" in os.environ:
                            image_src = os.path.join(os.environ["IMAGE_ROOT"], image_src)
                        else:
                            # CAUTION: this only works when the project is checked out (dev mode)
                            image_src = os.path.join(file_utils.project_root(), image_src)
                    transcript += (f'  <a title="{image_src}">'
                                   f'<img style="width:100%" src="{image_src}" alt="{image_src}" />'
                                   f'</a>\n')
                transcript += '</div>\n'
            else:
                transcript += patterns.HTML_TEMPLATE.format(speaker, class_name, style, msg_raw)
        else:
            transcript += patterns.HTML_TEMPLATE.format(speaker, class_name, style, msg_raw)
    transcript += patterns.HTML_FOOTER
    return transcript


def build_tex(interactions: Dict):
    """Create a LaTeX .tex file with the interaction transcript.
    The file is stored in the corresponding episode directory.
    Args:
        interactions: An episode interaction record dict.
    """
    tex = patterns.TEX_HEADER
    # Collect all events over all turns (ignore turn boundaries here)
    events = [event for turn in interactions['turns'] for event in turn]
    for event in events:
        class_name = _get_class_name(event).replace('msg ', '')
        msg_content = event['action']['content']
        if isinstance(msg_content, str):
            msg_content = msg_content.replace('\n', '\\\\ \\tt ')
        rgb, speakers, cols_init, cols_end, ncols, width = constants.TEX_BUBBLE_PARAMS[class_name]
        tex += patterns.TEX_TEMPLATE.substitute(cols_init=cols_init,
                                                rgb=rgb,
                                                speakers=speakers,
                                                msg=msg_content,
                                                cols_end=cols_end,
                                                ncols=ncols,
                                                width=width)
    tex += patterns.TEX_FOOTER
    return tex
