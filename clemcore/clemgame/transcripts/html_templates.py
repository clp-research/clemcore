from clemcore.clemgame.resources import load_packaged_file

CSS_BASIC = load_packaged_file("utils/chat-basic.css")
CSS_TWO_TRACKS = load_packaged_file("utils/chat-two-tracks.css")
CSS_ONE_TRACK = load_packaged_file("utils/chat-one-track.css")
CSS_COLORS = ["darkseagreen", "cadetblue", "indianred", "goldenrod", "thistle", "wheat", "sandybrown", "rebeccapurple"]


HEADER = '''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
    <style>
        {}
    </style>
</head>
<body>

<br/>
'''

TOP_INFO = '''
<div class="top-info">
    <p>{}</p>
</div>

<br/>

<div class="chat">
'''

EVENT_TEMPLATE = '''
    <div speaker="{}" class="msg {}" style="{}">
        <p>{}</p>
    </div>
'''

FOOTER = '''
</div>

</body>
</html>
'''