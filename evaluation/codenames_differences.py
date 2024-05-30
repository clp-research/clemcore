import pandas as pd
from pathlib import Path

difference_tables = ["results", "codenames-requests", "codenames-specific results"]     #  "codenames-turn scores"
status_tables = ["errors", "codenames-flags"]
strict_results_path = "./strict_results/"
generous_results_path = "./results/"

def main():
    for table in difference_tables:
        strict_table = pd.read_csv(f"{strict_results_path}{table}.csv").set_index('Model')
        generous_table = pd.read_csv(f"{generous_results_path}{table}.csv").set_index('Model')
        df = make_difference_table(strict_table, generous_table)
        save_table(df, generous_results_path, f"{table}-difference")
    for table in status_tables:
        strict_table = pd.read_csv(f"{strict_results_path}{table}.csv").set_index('Model')
        generous_table = pd.read_csv(f"{generous_results_path}{table}.csv").set_index('Model')
        df = make_status_table(strict_table, generous_table)
        save_table(df, generous_results_path, f"{table}-status")


def make_difference_table(strict_table, generous_table):
    difference_table = pd.DataFrame(columns=generous_table.columns)
    for column in generous_table.columns:
        for row in generous_table.index:
            if column not in strict_table.columns:
                difference_table.loc[row, column] = generous_table.loc[row, column]
            else:
                difference = round(generous_table.loc[row, column] - strict_table.loc[row, column], 2)
                if difference == 0:
                    difference_table.loc[row, column] = generous_table.loc[row, column]
                elif difference > 0:
                    difference_table.loc[row, column] = f"{generous_table.loc[row, column]} (+{difference})"
                else:
                    difference_table.loc[row, column] = f"{generous_table.loc[row, column]} ({difference})"
    print(difference_table)
    return difference_table

def make_status_table(strict_table, generous_table):
    status_table = pd.DataFrame(columns=generous_table.columns)
    for column in generous_table.columns:
        for row in generous_table.index:
            if column not in strict_table.columns:
                status_table.loc[row, column] = generous_table.loc[row, column]
            else:
                status = strict_table.loc[row, column]
                if status == 0:
                    status_table.loc[row, column] = generous_table.loc[row, column]
                else:
                    status_table.loc[row, column] = f"{generous_table.loc[row, column]} ({status})"
    print(status_table)
    return status_table
    
def save_table(df, path, table_name):
    df.columns = [column.title() for column in df.columns]
    Path(path).mkdir(parents=True, exist_ok=True)
    df = df.rename_axis('Model')
    df.to_csv(Path(path) / f'{table_name}.csv')
    # df.to_html(Path(path) / f'{table_name}.html')
    print(f'\n Saved results into {path}/{table_name}.csv')


if __name__ == '__main__':
    main()