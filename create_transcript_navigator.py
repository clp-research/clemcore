import glob
import re

from evaluation.create_excel_overview import natural_sort_key

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


def get_transcript_htmls(results_path, games, languages=None, episode_sample=None):
    filename = "transcript.html"
    paths_per_game = []
    for game in games:
        paths = glob.glob(f"{results_path}/**/{game}/**/{filename}", recursive=True)
        paths = sorted(paths, key=natural_sort_key)
        if languages:
            paths = [path for path in paths
                     if re.search(f"{results_path}/(\w+)/", path).group(1) in languages]
        if episode_sample:
            paths = [path for path in paths
                     if int(re.search("episode_(\d*)", path).group(1)) in episode_sample]
        paths_per_game.append(paths)
    paths_per_game = [path for game in paths_per_game for path in game]
    return paths_per_game


if __name__ == "__main__":
    results_path = "results/v1.5_multiling_liberal"
    games = ["imagegame"]  # "imagegame" "referencegame"
    transcript_paths = get_transcript_htmls(results_path, games, languages=["en", "es", "ru", "te", "tk", "tr"])
    html = HTML_HEAD
    for path in transcript_paths:
        html_link = TRANSCRIPT_LINK.format(path)
        html += html_link
    html += HTML_TAIL

    with open("transcript_navigator_imagegame_liberal_human.html", "w") as file:
        file.write(html)
