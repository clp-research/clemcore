import pandas, latextable, os
from texttable import Texttable

# split flag table for generous results into cluegiver and guesser?
# make largest or smallest bold?

STRICT_MODE = "in the Strict Mode"
GENEROUS_MODE = "in the Generous Mode"
MIXED_MODE = "with mixed player teams"

results_folders = ['results/', 'strict_results/', 'mixed_results/']
results_naming = {
    'results/' : 'generous',
    'strict_results/': 'strict',
    'mixed_results/' : 'mixed'
}
mode_naming = {
    'results/' : GENEROUS_MODE,
    'strict_results/': STRICT_MODE,
    'mixed_results/' : MIXED_MODE
}
captions = {
    'flags': "The number of flag-enabled behaviours triggered $MODE$.",
    'requests': 'The amount of requests that were made $MODE$, which were either successfully parsed or violated the rules. The amount of parsed requests and the amount of total requests constitute the Request Success Ratio.',
    'specific results': 'Game-specific results $MODE$.',
    'errors': 'All errors that occurred $MODE$.',
    'results' : "The main results of all models $MODE$."
}


def main():
    for results in results_folders:
        print(f'==={results}===========================')
        csvs = [file for file in os.listdir(results) if file.endswith('.csv')]
        print(csvs)
        for csv in csvs:
            print(f'---{results}: {csv}-----------------------')
            if csv == 'raw results.csv' or csv == 'codenames-turn scores.csv':
                continue
            df = pandas.read_csv(results + csv, index_col=0)
            table = create_tex_table(df, results, csv)
            print(table)
            save_tex_table(table, csv, results)

def create_tex_table(df, results_path, name):
    table = Texttable()
    table.set_precision(2)
    header = [df.index.name]
    header.extend(df.columns)
    print(header)
    table.set_cols_align(["l"] + ["c"] * (len(header) - 1))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows([header])

    for index, row in df.iterrows():
        table_row = [index]
        table_row.extend(row)
        table.add_row(table_row)
    
    alias = {
        '%'  : '\%',
        '+-' : '$\\pm$',
        '(+' : '\\textcolor{ForestGreen}{(+',
        '(-' : '\\textcolor{BrickRed}{(-',
        ')'  : ')}',
        ' ('  : ' \\textcolor{BurntOrange}{(',
        '\\textcolor{BurntOrange}{(Std)}' : '(std)',
        'No Correct Guess': '\\thead{No Correct\\\\Guess}',
        'Wrong Number Of Guesses': '\\thead{Wrong Number\\\\Of Guesses}',
        'Clue Is Morphologically Related To Word On The Board' : '\\thead{Clue Is\\\\Morphologically\\\\Related}',
        'Clue Contains Spaces': '\\thead{Clue\\\\Contains\\\\Spaces}',
        'Parsed Request Count': '\\thead{Parsed\\\\Request\\\\Count}',
        'Violated Request Count': '\\thead{Violated\\\\Request\\\\Count}',
        'Request Success Ratio': '\\thead{Request\\\\Success\\\\Ratio}',
        'Request Count': '\\thead{Request\\\\Count}',
        'Game Ended Through Assassin': '\\thead{Game\\\\Ended\\\\Through\\\\Assassin}', 
        'Number Of Turns': '\\thead{Number\\\\Of Turns}', 
        'Episode Recall': '\\thead{Episode\\\\Recall}', 
        'Episode Negative Recall': '\\thead{Episode\\\\Negative\\\\Recall}',
        'Target Is Hallucination': '\\thead{Target Is\\\\Hallucination}',
        'Guess Was Already Guessed': '\\thead{Guess Was\\\\Already\\\\Guessed}',
        'Target Was Already Guessed': '\\thead{Target\\\\Was\\\\Already\\\\Guessed}',
        'Guess Is Clue Word': '\\thead{Guess Is\\\\Clue}',
        'Guess Word Is Hallucination': '\\thead{Guess Is\\\\Hallucination}',
        'Rambling Error' : '\\thead{Rambling\\\\Error}',
        'Guess Appears More Than Once In Utterance': '\\thead{Guess Made\\\\Twice\\\\In Turn}',
        'Prefix Error': '\\thead{Prefix\\\\Error}'
    }

    if name == 'codenames-specific results.csv' or name == 'codenames-specific results-difference.csv':
        alias['Quality Score'] = '\\thead{Quality\\\\Score}' 

    caption = make_caption(results_path, name)
    label = make_label(results_path, name)
    print(label)
    table_string = latextable.draw_latex(table, position='htb', alias = alias, caption=caption, label=label)
    table_string = table_string.replace('\\begin{center}', '\\centering\n\t\\resizebox{\\textwidth}{!}{%')
    table_string = table_string.replace('\\end{center}', '}')
    table_string = table_string.replace('\n\t\t\tideal mock', '\\hline\n\t\t\tideal mock')
    return table_string

def make_caption(results_path, name):
    name = name.removesuffix('.csv')
    if name.endswith('-difference'):
        short_name = name.removesuffix('-difference')
    elif name.endswith('-status'):
        short_name = name.removesuffix('-status')
    else:
        short_name = name

    if short_name.startswith('codenames-'):
        short_name = short_name.removeprefix('codenames-')

    caption = captions[short_name]
    caption = caption.replace('$MODE$', mode_naming[results_path])

     # check whether it ends in -difference or in -status
    if name.endswith('-difference') or name.endswith('-status'):
        if results_path == 'results/':
            difference_mode = STRICT_MODE
        elif results_path == 'mixed_results/':
            difference_mode = GENEROUS_MODE
        if name.endswith('difference'):
            caption += f' The numbers in brackets denote the differences to the same scores achieved {difference_mode}.'
        elif name.endswith('status'):
            unit = name.removesuffix('-status')
            caption += f' The numbers in brackets denote the amount of {unit} {difference_mode}.'    

    return caption

def make_label(results_path, name):
    mode_name = results_naming[results_path]
    name = name.removesuffix('.csv')
    if name.startswith('codenames-'):
        name = name.removeprefix('codenames-')
    return f"tab: {mode_name} {name}"

def save_tex_table(text, csv_name, results_path):
    name = csv_name.removesuffix('.csv')
    if name.startswith('codenames-'):
        name = name.removeprefix('codenames-')
    with open(f"{results_path}{name}.tex", "w") as file:
        file.write(text)
        #file.close()

if __name__ == '__main__':
    main()