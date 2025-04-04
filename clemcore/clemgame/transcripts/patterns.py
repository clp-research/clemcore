from string import Template

HTML_HEADER = '''
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

HTML_TEMPLATE = '''
    <div speaker="{}" class="msg {}" style="{}">
        <p>{}</p>
    </div>
'''

HTML_FOOTER = '''
</div>

</body>
</html>
'''

TEX_HEADER = '''
\\documentclass{article}
\\usepackage{colortbl}
\\usepackage{makecell}
\\usepackage{multirow}
\\usepackage{supertabular}

\\begin{document}

\\newcounter{utterance}

\\twocolumn

{ \\footnotesize  \\setcounter{utterance}{1}
\\setlength{\\tabcolsep}{0pt}
\\begin{supertabular}{c@{$\;$}|p{.15\\linewidth}@{}p{.15\\linewidth}p{.15\\linewidth}p{.15\\linewidth}p{.15\\linewidth}p{.15\linewidth}}

    \\# & $\\;$A & \\multicolumn{4}{c}{Game Master} & $\\;\\:$B\\\\
    \\hline 
'''

TEX_TEMPLATE = Template('''
    \\theutterance \\stepcounter{utterance}  

    $cols_init \\multicolumn{$ncols}{p{$width\\linewidth}}{\\cellcolor[rgb]{$rgb}{%\n\t\\makecell[{{p{\\linewidth}}}]{% \n\t  \\tt {\\tiny [$speakers]}  \n\t $msg \n\t  } \n\t   } \n\t   } \n\t $cols_end \\\\ \n 
''')

TEX_FOOTER = '''
\\end{supertabular}
}

\\end{document}
'''
