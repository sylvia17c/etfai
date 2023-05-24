import pandas as pd

def split_df(df, percentages):
    df = df.sort_index()
    total_rows = len(df)
    indices = [int(total_rows * p) for p in percentages]

    # Split the DataFrame into multiple parts based on the indices
    dfs = []
    start_index = 0
    for index in indices:
        dfs.append(df[start_index:index])
        start_index = index
    dfs.append(df[start_index:])

    return dfs