import glob
import re

HTML_HEAD = """
<!DOCTYPE html>
<html>
<head>
    <title>Navigator</title>
</head>
<body>
    <h1></h1>
    <div id="transcript-links">
"""

TRANSCRIPT_LINK = """
        <a style="display: none;" target="transcript-frame" href="{}">Next episode</a>
"""

HTML_TAIL = """
    </div>

    <!-- Iframe zum Anzeigen der Transcripts -->
    <iframe id="transcript-frame" name="transcript-frame" width="90%" height="600px"></iframe>

    <script>
        const transcript_links = document.getElementById("transcript-links").getElementsByTagName("a");

        transcript_links[0].style.display = "block";

        for (const link of transcript_links) {
            link.addEventListener("click", function () {
                this.style.display = "none";
                let prev = this.previousElementSibling;
                let next = this.nextElementSibling;
                if (prev) {
                    prev.style.display = "block";
                    prev.innerHTML = "Previous episode";
                    let preprev = prev.previousElementSibling;
                    if (preprev) {
                        preprev.style.display = "none";
                        preprev.innerHTML = "Next episode";
                    }
                }
                if (next) {
                    next.style.display = "block";
                    next.innerHTML = "Next episode";
                    let afternext = next.nextElementSibling;
                    if (afternext) {
                        afternext.style.display = "none";
                        afternext.innerHTML = "Next episode";
                    }
                }
            });
        }
    </script>

</body>
</html>
"""


def natural_sort_key(s):
    # von Anne Beyer
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


def get_transcript_htmls(results_path, games):
    filename = "transcript.html"
    paths_per_game = []
    for game in games:
        paths = glob.glob(f"{results_path}/**/{game}/**/{filename}", recursive=True)
        paths = sorted(paths, key=natural_sort_key)
        paths_per_game.append(paths)
    paths_per_game = [path for game in paths_per_game for path in game]
    return paths_per_game


if __name__ == "__main__":
    results_path = "results/v1.5_multiling_debug"
    games = ["referencegame", "imagegame"]
    transcript_paths = get_transcript_htmls(results_path, games)
    html = HTML_HEAD
    for path in transcript_paths:
        html_link = TRANSCRIPT_LINK.format(path)
        html += html_link
    html += HTML_TAIL

    with open(f"transcript_navigator.html", "w") as file:
        file.write(html)