import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns
import configparser
import numpy as np

CONFIG = configparser.ConfigParser()
CONFIG.read('./etfai/config.ini')
DATA_DIR = Path(CONFIG['DATA']['DataDir'])
RES_DIR = DATA_DIR / '../result'

def plot_ret_line():
    # plot graph of four HSI return-based variables
    target = pd.read_csv(DATA_DIR / 'target/HSI.csv', index_col='Date', parse_dates=True)
    ret = target[['OO', 'OC', 'CO', 'CC']]
    range = ret.max() - ret.min()

    fig, ax = plt.subplots(figsize=(30,10))

    # Control plotting order
    sorted_columns = range.sort_values(ascending=False).index
    zorder = 1
    for column in sorted_columns:
        ret[column].plot(ax=ax, label=column, zorder=zorder)
        zorder += 1

    # Adjust marker frequency
    plt.xticks(rotation=45)
    ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Color the background of the plot
    area1_end_date = ret.index[int(len(ret) * 0.7)]
    area2_end_date = ret.index[int(len(ret) * 0.85)]
    ax.fill_between(ret.index, ret.min().min()*1.1, ret.max().max()*1.1, where=ret.index < area1_end_date, facecolor='lightblue', alpha=0.5)
    ax.fill_between(ret.index, ret.min().min()*1.1, ret.max().max()*1.1, where=(ret.index >= area1_end_date) & (ret.index < area2_end_date), facecolor='lightgreen', alpha=0.5)
    ax.fill_between(ret.index, ret.min().min()*1.1, ret.max().max()*1.1, where=ret.index >= area2_end_date, facecolor='lightpink', alpha=0.5)

    # Adjust grid line
    ax.grid(linewidth=0.5, color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlim(ret.index[0], ret.index[-1])
    plt.ylim(ret.min().min()*1.1, ret.max().max()*1.1)

    plt.ylabel('Returns')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot to a file
    RES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{RES_DIR}/four_returns.png', dpi=300, bbox_inches='tight')
    print(f'please find the plot at {RES_DIR}/four_returns.png')


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


def plot_ret_freq(df, file_path):
    fig, ax = plt.subplots()
    prefix = ['OO', 'OC', 'CC', 'CO']
    suffix = ['bi_0', 'ter_0.005', 'ter_0.01']
    target_cols = [f'{p}_{s}' for s in suffix for p in prefix]

    # dfs[0][f'{p}_{s}'].value_counts().plot(ax=ax, kind='bar')
    value_counts = df[target_cols].apply(pd.Series.value_counts)
    stack_plt = value_counts.T.plot(kind='bar', stacked=True, ax=ax, width=0.8)
    for container in stack_plt.containers:
        for rect in container.patches:
            height = rect.get_height()
            if height > 0:  # Skip labeling for bars with 0 frequency
                ax.annotate(f'{int(height)}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 3), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8, color='white')

    plt.xticks(rotation=45)
    plt.ylabel('Frequency')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f'please find the plot at {file_path}')


def plot_missing(df, file_name):
    plot = sns.heatmap(df.isnull(), cmap='viridis', yticklabels=False, cbar=False)
    plot.set_xticks(np.arange(df.shape[1]) + 0.5)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    plot.figure.savefig(RES_DIR / f'{file_name}.png', dpi=300, bbox_inches='tight')