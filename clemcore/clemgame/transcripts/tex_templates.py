from string import Template

HEADER = '''
\\documentclass{article}
\\usepackage{colortbl}
\\usepackage{makecell}
\\usepackage{multirow}
\\usepackage{supertabular}

\\begin{document}

\\newcounter{utterance}

\\centering \\large $title
\\vspace{24pt}

{ \\footnotesize  \\setcounter{utterance}{1}
\\setlength{\\tabcolsep}{0pt}
\\begin{supertabular}{c@{$\;$}|p{.15\linewidth}@{}p{.15\linewidth}p{.15\linewidth}p{.15\linewidth}p{.15\linewidth}p{.15\linewidth}}
$column_header
    \\hline
'''

# 2 is for 2 players, 1 for all other cases
COLUMN_HEADER = {
    1: "    \\# & \\multicolumn{2}{c}{Player} && \\multicolumn{2}{c}{Game Master} \\\\",
    2: "    \\# & $\\;$A & \\multicolumn{4}{c}{Game Master} & $\\;\\:$B\\\\",
}

BUBBLE = {
    1: {
        "player-gm": (None, "$player_name$\\rangle$GM", "&", "& &", 4, 0.6),
        "gm-player": ("0.9,0.9,0.9", "$player_name$\\langle$GM", "& & &", "", 4, 0.6),
        "gm-gm": ("0.95,0.95,0.95", "GM$|$GM", "& & &", "& &", 2, 0.3)
    },
    2: {
        "player-gm p1": (None, "P1$\\rangle$GM", "&", "& &", 4, 0.6),
        "player-gm p2": (None, "GM$\\langle$P2", "& & &", "", 4, 0.6),
        "gm-player p1": ("0.9,0.9,0.9", "P1$\\langle$GM", "& &", "&", 4, 0.6),
        "gm-player p2": ("0.9,0.9,0.9", "GM$\\rangle$P2", "& &", "&", 4, 0.6),
        "gm-gm": ("0.95,0.95,0.95", "GM$|$GM", "& & &", "& &", 2, 0.3)
    }
}

COLORS = ["0.561,0.737,0.561", "0.373,0.62,0.627", "0.804,0.361,0.361", "0.804,0.498,0.196", "0.718,0.561,0.718", "0.961,0.871,0.702", "0.957,0.643,0.376", "0.412,0.353,0.804"]

EVENT_TEMPLATE = Template('''
    \\theutterance \\stepcounter{utterance}  
    $cols_init \\multicolumn{$ncols}{p{$width\\linewidth}}{
        \\cellcolor[rgb]{$rgb}{
            \\makecell[{{p{\\linewidth}}}]{
                \\texttt{\\tiny{[$speakers]}}
                $msg
            }
        }
    }
    $cols_end \\\\ \\\\
''')

FOOTER = '''
\\end{supertabular}
}

\\end{document}
'''
