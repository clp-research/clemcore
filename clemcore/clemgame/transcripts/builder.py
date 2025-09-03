import glob
import json
import logging
import os
import html
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import markdown as md
from pylatex.utils import escape_latex

import clemcore.clemgame.transcripts.html_templates as html_templates
import clemcore.clemgame.transcripts.tex_templates as tex_templates
from clemcore.utils import file_utils
from clemcore.clemgame.resources import store_file, load_json

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")

def get_css(num_players) -> str:
    """
    Get the CSS template for the given number of players, including GM.
    Args:
        players: A dict of players as passed by the interaction record.
    Returns:
        The CSS template to be used for the transcript as a String.
    """
    stylesheet = html_templates.CSS_BASIC
    if num_players == 3:
        stylesheet += html_templates.CSS_TWO_TRACKS
    else:
        stylesheet += html_templates.CSS_ONE_TRACK
        for i in range(1, num_players):
            color = html_templates.CSS_COLORS[(i - 1) % len(html_templates.CSS_COLORS)]
            stylesheet += f".msg.player-gm.p{i} {{ background: {color}; }}\n"
    return stylesheet

def get_css_player_dict(players: Dict) -> Dict[str, str]:
    """Get a dict of player names as keys and css abbreviations as values.
    Args:
        players: A dict of players as passed by the interaction record.
    Returns:
        A dict of player names and their css abbreviations.
    """
    player_dict = { "GM": "gm" }
    i = 1
    for player in players:
        if player == "GM":
            continue
        else:
            player_dict[player] = f"p{i}"
            i += 1
    return player_dict

def _get_class_name(event: Dict, css_player_dict: Dict[str, str]) -> str:
    """Get the CSS class name for a given event based on its 'from' and 'to' fields.
    Args:
        event: A dict representing a game event with 'from' and 'to' fields.
        css_player_dict: A dict mapping player names to their CSS abbreviations.
    Returns:
        The CSS class name as a string.
    """
    player = None
    if event['from'] == 'GM':
        from_ = 'gm'
    else:
        from_ = 'player'
        player = css_player_dict[event['from']]
    if event['to'] == 'GM':
        to = 'gm'
    else:
        to = 'player'
        player = css_player_dict[event['to']]
    class_string = f"{from_}-{to}"
    return class_string, player

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
    """
    meta = interactions["meta"]
    players = interactions["players"]
    markdown = interactions.get("markdown", False)

    css_player_dict = get_css_player_dict(players)
    transcript = html_templates.HEADER.format(get_css(len(players)))
    pair_descriptor = meta["results_folder"] if "results_folder" in meta else meta["dialogue_pair"]
    title = f"Interaction Transcript for game '{meta['game_name']}', experiment '{meta['experiment_name']}', " \
            f"episode {meta['game_id']} with {pair_descriptor}."
    transcript += html_templates.TOP_INFO.format(title)
    for turn_idx, turn in enumerate(interactions['turns']):
        transcript += f'<div class="game-round" data-round="{turn_idx}">'
        for event in turn:
            class_name, player = _get_class_name(event, css_player_dict)
            if player is not None:
                class_name += f" {player}"
            msg_content = event['action']['content']
            if markdown:
                msg_raw = msg_content.strip()
                while msg_raw.startswith('`') and msg_raw.endswith('`'):
                    # remove code block markers if whole message is wrapped in them
                    msg_raw = msg_raw[1:-1]
                msg_raw = md.markdown(msg_raw, extensions=['fenced_code'])
            else:
                msg_raw = html.escape(f"{msg_content}").replace('\n', '<br/>')
            if event['from'] == 'GM' and event['to'] == 'GM':
                speaker_attr = f'Game Master: {event["action"]["type"]}'
            else:
                from_player = event['from']
                to_player = event['to']
                if "game_role" in players[from_player] and "game_role" in players[to_player]:
                    from_game_role = players[from_player]["game_role"]
                    to_game_role = players[to_player]["game_role"]
                    speaker_attr = f"{from_player} ({from_game_role}) to {to_player} ({to_game_role})"
                else:  # old mode (before 2.4)
                    speaker_attr = f"{event['from'].replace('GM', 'Game Master')} to {event['to'].replace('GM', 'Game Master')}"
            # in case the content is a json BUT given as a string!
            # we still want to check for image entry
            if isinstance(msg_content, str):
                try:
                    msg_content = json.loads(msg_content)
                except:
                    ...
            style = "border: dashed" if "label" in event["action"] and "forget" == event["action"]["label"] else ""

            images = []
            # check if images are in the action dict (new structure)
            if "image" in event["action"]:
                images += event["action"]["image"]
            # check if images are in the content field (old structure)
            if isinstance(msg_content, dict) and "image" in msg_content:
                images += msg_content["image"]

            if images:
                transcript += f'<div speaker="{speaker_attr}" class="msg {class_name}" style="{style}">\n'
                transcript += f'  <p>{msg_raw}</p>\n'
                for image_src in images:
                    if not image_src.startswith("http"):  # take the web url as it is
                        if "IMAGE_ROOT" in os.environ:
                            image_src = os.path.join(os.environ["IMAGE_ROOT"], image_src)
                        elif image_src.startswith("/"):
                            pass  # keep absolute path to image
                        else:
                            # CAUTION: this only works when the project is checked out (dev mode)
                            image_src = os.path.join(file_utils.project_root(), image_src)
                    transcript += (f'  <a title="{image_src}">'
                                   f'<img style="width:100%" src="{image_src}" alt="{image_src}" />'
                                   f'</a>\n')
                transcript += '</div>\n'
            else:
                transcript += html_templates.EVENT_TEMPLATE.format(speaker_attr, class_name, style, msg_raw)
        transcript += "</div>"
    transcript += html_templates.FOOTER
    return transcript

def build_tex(interactions: Dict):
    """Create a LaTeX .tex file with the interaction transcript.
    The file is stored in the corresponding episode directory.
    Args:
        interactions: An episode interaction record dict.
    """
    css_player_dict = get_css_player_dict(interactions["players"])
    track_type = "two_tracks" if len(interactions["players"]) == 3 else "one_track"
    column_header = tex_templates.COLUMN_HEADER[track_type]
    meta = interactions["meta"]
    pair_descriptor = meta["results_folder"] if "results_folder" in meta else meta["dialogue_pair"]
    title = escape_latex(f"Interaction Transcript for game `{meta['game_name']}', experiment `{meta['experiment_name']}', " \
            f"episode {meta['game_id']} with {pair_descriptor}.")
    tex = tex_templates.HEADER.replace("$title", title).replace("$column_header", column_header)
    tex_bubble = tex_templates.BUBBLE[track_type]
    # Collect all events over all turns (ignore turn boundaries here)
    events = [event for turn in interactions['turns'] for event in turn]
    for event in events:
        class_name, player = _get_class_name(event, css_player_dict)
        if track_type == "two_tracks" and player is not None:
            class_name += f" {player}"
        elif player:
            player = player.upper()
        msg_content = event['action']['content']
        if isinstance(msg_content, str):
            lines = msg_content.splitlines()
            msg_content = ""
            for line in lines:
                if line.strip() == "":
                    msg_content += "\\\\ \n"
                else:
                    msg_content += "\\texttt{" + escape_latex(line) + "} \\\\\n"
            msg_content = msg_content[:-1]
        rgb, speakers, cols_init, cols_end, ncols, width = tex_bubble[class_name]
        if track_type == "one_track":
            speakers = speakers.replace("$player_name", player if player else "")
        if rgb is None and player is not None:
            rgb = tex_templates.COLORS[int(player[1:]) - 1 % len(tex_templates.COLORS)]
        else:
            rgb = "0.9,0.9,0.9" if "gm" in class_name else "0.95,0.95,0.95"
        tex += tex_templates.EVENT_TEMPLATE.substitute(cols_init=cols_init,
                                                rgb=rgb,
                                                speakers=speakers,
                                                msg=msg_content,
                                                cols_end=cols_end,
                                                ncols=ncols,
                                                width=width)
    tex += tex_templates.FOOTER
    return tex
