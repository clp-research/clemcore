import glob
import json
import logging
import os
import html
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import hashlib

import clemcore.clemgame.transcripts.patterns as patterns
from clemcore.utils import file_utils
from clemcore.clemgame.resources import store_file, load_json
from clemcore.clemgame.resources import load_packaged_file

module_logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.run")


def _get_class_name(event, players):
    """
    Get a string representation of the direction of a message.

    Example:
        A message from the game's GM to Player 1 is represented by the string 'gm-player_1'.

    Args:
        event: The interaction record event to get the message direction for.
        players: A dictionary of players as given in the game config.

    Returns:
        The string representation of the direction of the message.
    """
    def normalize(name):
        if name == "GM":
            return "gm"
        elif name in players:
            return name.lower().replace(" ", "_")
        else:
            raise RuntimeError(f"Cannot handle event entry {event}. Unknown player name: {name}")

    from_label = normalize(event["from"])
    to_label = normalize(event["to"])
    return f"{from_label}-{to_label}"


class TranscriptStyles:
    def __init__(self, players: Dict[str, Dict]):
        self.players = players
        self.css_base = load_packaged_file("utils/chat-two-tracks.css")
        self.tex_params = self._generate_tex_params()
        self.generated_css = self._generate_css()

    def _hash_to_rgb(self, label: str) -> str:
        """Generate a unique pastel RGB color in decimal format from a string label."""
        hash_bytes = hashlib.md5(label.encode()).digest()
        r = 0.6 + (hash_bytes[0] / 255) * 0.3
        g = 0.6 + (hash_bytes[1] / 255) * 0.3
        b = 0.6 + (hash_bytes[2] / 255) * 0.3
        return f"{round(r, 2)},{round(g, 2)},{round(b, 2)}"

    def _add_direction(self, from_p, to_p):
        """Add bubble style only for gm-player or player-gm directions."""
        if from_p == "gm" and to_p != "gm":
            rgb = "0.9,0.9,0.9"
            speakers = f"GM$\\rangle${to_p.replace('_', ' ').title()}"
            cols_init, cols_end = "& &", "&"
        elif to_p == "gm" and from_p != "gm":
            rgb = self._hash_to_rgb(from_p)
            speakers = f"{from_p.replace('_', ' ').title()}$\\rangle$GM"
            cols_init, cols_end = "&", "& &"
        else:
            return {}
        return {f"{from_p}-{to_p}": (rgb, speakers, cols_init, cols_end, 4, 0.6)}

    def _generate_tex_params(self):
        """Generate LaTeX-style parameters for color, speaker format, and layout based on participants."""
        tex_params = {
            "gm-gm": ("0.95,0.95,0.95", "GM$|$GM", "& & &", "& &", 2, 0.3)
        }
        for player in self.players:
            if player == "GM":
                continue
            normalized = player.lower().replace(" ", "_")
            tex_params.update(self._add_direction("gm", normalized))
            tex_params.update(self._add_direction(normalized, "gm"))
        return tex_params

    def _generate_css(self):
        """Generate CSS rules for message bubbles using color parameters shared with LaTeX output."""
        rules = []
        for class_name, (rgb, *_rest) in self.tex_params.items():
            rgb_vals = [int(float(x) * 255) for x in rgb.split(",")]
            rgb_str = f"rgb({rgb_vals[0]},{rgb_vals[1]},{rgb_vals[2]})"
            rules.append(f""".{class_name} {{
                background-color: {rgb_str};
                color: black;
                padding: 12px 16px;
                border-radius: 10px;
                margin: 20px 0;
                clear: both;
                overflow-wrap: break-word;
                display: block;
            }}""")
        return self.css_base + "\n" + "\n".join(rules)


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
            players = game_interactions["players"]
            transcript = build_transcript(game_interactions)
            store_file(transcript, "transcript.html", interactions_dir)
            transcript_tex = build_tex(game_interactions, players)
            store_file(transcript_tex, "transcript.tex", interactions_dir)
        except Exception:
            module_logger.exception(f"Cannot transcribe {interaction_file} (but continue)")
            error_count += 1
    if error_count > 0:
        stdout_logger.error(f"'{error_count}' exceptions occurred: See clembench.log for details.")


def build_transcript(interactions: Dict):
    """
    Create an HTML file with the interaction transcript.
    The file is stored in the corresponding episode directory.
    Args:
        interactions: An episode interaction record dict.
        experiment_config: An experiment configuration dict.
        game_instance: The instance dict the episode interaction record is based on.
        dialogue_pair: The model pair descriptor string for the Players.
    """
    meta = interactions["meta"]
    players = interactions["players"]
    styles = TranscriptStyles(players)
    TEX_BUBBLE_PARAMS = styles.tex_params
    combined_css = styles.generated_css
    transcript = patterns.HTML_HEADER.format(combined_css)
    title = f"Interaction Transcript for {meta['experiment_name']}, " \
            f"episode {meta['game_id']} with {meta['dialogue_pair']}."
    transcript += patterns.TOP_INFO.format(title)
    for turn_idx, turn in enumerate(interactions['turns']):
        transcript += f'<div class="game-round" data-round="{turn_idx}">'
        for event in turn:
            class_name = _get_class_name(event, players)
            msg_content = event['action']['content']
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
                else: # old mode (before 2.4)
                    speaker_attr = f"{event['from'].replace('GM', 'Game Master')} to {event['to'].replace('GM', 'Game Master')}"
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
                    transcript += f'<div speaker="{speaker_attr}" class="msg {class_name}" style="{style}">\n'
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
                    transcript += patterns.HTML_TEMPLATE.format(speaker_attr, class_name, style, msg_raw)
            else:
                transcript += patterns.HTML_TEMPLATE.format(speaker_attr, class_name, style, msg_raw)
        transcript += "</div>"
    transcript += patterns.HTML_FOOTER
    return transcript


def build_tex(interactions: Dict, players: Dict):
    """
    Create a LaTeX .tex file with the interaction transcript.
    The file is stored in the corresponding episode directory.
    Args:
        interactions: An episode interaction record dict.
    """

    styles = TranscriptStyles(players)
    TEX_BUBBLE_PARAMS = styles.tex_params

    tex = patterns.TEX_HEADER
    events = [event for turn in interactions['turns'] for event in turn]
    for event in events:
        class_name = _get_class_name(event, players).replace('msg ', '')
        msg_content = event['action']['content']
        if isinstance(msg_content, str):
            msg_content = msg_content.replace('\n', '\\ \\tt ')
        rgb, speakers, cols_init, cols_end, ncols, width = TEX_BUBBLE_PARAMS[class_name]
        tex += patterns.TEX_TEMPLATE.substitute(cols_init=cols_init,
                                                rgb=rgb,
                                                speakers=speakers,
                                                msg=msg_content,
                                                cols_end=cols_end,
                                                ncols=ncols,
                                                width=width)
    tex += patterns.TEX_FOOTER
    return tex