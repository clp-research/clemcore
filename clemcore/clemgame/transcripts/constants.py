from clemcore.clemgame.resources import load_packaged_file

CSS_STRING = load_packaged_file("utils/chat-two-tracks.css")
TEX_BUBBLE_PARAMS = {
    "a-gm": ("0.8,1,0.9", "A$\\rangle$GM", "&", "& &", 4, 0.6),
    "b-gm": ("1,0.85,0.72", "GM$\\langle$B", "& & &", "", 4, 0.6),
    "gm-a": ("0.9,0.9,0.9", "A$\\langle$GM", "& &", "&", 4, 0.6),
    "gm-b": ("0.9,0.9,0.9", "GM$\\rangle$B", "& &", "&", 4, 0.6),
    "gm-gm": ("0.95,0.95,0.95", "GM$|$GM", "& & &", "& &", 2, 0.3)
}
