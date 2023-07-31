import numpy as np
import pandas as pd
from math import floor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from pathlib import Path


CAPITAL = 100000
LSTM_P = "./etfai/backtesting/signals/lstm_stock_prediction_classification_feature.csv"
LSTM_PP = "./etfai/backtesting/signals/lstm_stock_prediction_classification_nlp.csv"

XGBOOST_P = "./etfai/backtesting/signals/xgboost_stock_prediction_classification_feature.csv"
XGBOOST_PP = "./etfai/backtesting/signals/xgboost_stock_prediction_classification_nlp.csv"

LLM = "./etfai/backtesting/signals/bert_stock_prediction_classification.csv"

PORT_NAMES = ['BuyNHold', 'LSTM+', 'LSTM++', 'XGBOOST+', 'XGBOOST++', 'LLM']

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

HIBOR_ON = "./etfai/backtesting/historical_data/raw/HIBOR.csv"

START = "2019-12-16"
END = "2022-11-29"

def get_hist_data():
    file_path = './etfai/backtesting/historical_data/raw/Tracker Fund of Hong Kong (2800.HK).csv'
    price_data = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    price_data = price_data[(price_data.index >= pd.to_datetime('2019-12-01')) & (price_data.index <= pd.to_datetime('2022-11-30'))]

    return price_data

# def get_daily_Rf(start, end):
#     df = pd.read_csv(HIBOR_ON, index_col='Date', parse_dates=True)
#     df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
#     return df['Close'].mean()

def get_Rf(start, end):
    df = pd.read_csv(HIBOR_ON, index_col='Date', parse_dates=True)
    df = df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    return (df['Close']/100).mean()


def calculate_daily_capital(s, last_s, multiplier=500, transaction_cost_rate=0.002):

    if s['signal']==0 or s['signal']*last_s['target_pos'] > 0:
        s['target_pos'] = last_s['target_pos']
    else:
        s['target_pos'] = floor(last_s['capital']/(s['Open']*multiplier))*(s['signal'])

    s['intraday_PnL'] = (s['Adj Close']-s['Open'])*s['target_pos']*multiplier
    s['NAV'] = abs(s['target_pos'])*s['Adj Close']*multiplier
    s['transaction_cost'] = s['NAV'] *transaction_cost_rate/365
    s['capital'] = last_s['capital'] - s['transaction_cost'] + s['intraday_PnL']
    s['cash'] = s['capital'] - s['NAV']

    return s


def trading_sim(dataset, init_capital=100000, multiplier=500, transaction_cost_rate=0.0008):
    dataset['target_pos'] = pd.Series()
    dataset['capital'] = pd.Series()
    dataset['NAV'] = pd.Series()
    dataset['cash'] = pd.Series()
    dataset['transaction_cost'] = pd.Series()
    dataset['intraday_PnL'] = pd.Series()

    dummy_row = dataset.iloc[0].shift(1)
    dummy_row['target_pos'] = 0
    dummy_row['capital'] = init_capital
    dummy_row['NAV'] = 0
    dummy_row['cash'] = init_capital
    dummy_row['transaction_cost'] = 0
    dummy_row['intraday_PnL'] = 0
    dummy_row = dummy_row.to_frame().T
    dataset = pd.concat([dummy_row, dataset])

    for i in range(1, len(dataset)):
        row = dataset.iloc[i]
        last_row = dataset.iloc[i-1]
        dataset.iloc[i] = calculate_daily_capital(row, last_row)

    return dataset.iloc[1:]

    # return dataset.iloc[1:].apply(calculate_daily_capital, axis=1)


def eval_performance(dataset, init_capital=100000):
    # Calculate cumulative PnL
    daily_pnl = dataset['capital']-dataset['capital'].shift(1)
    daily_pnl.iloc[0] = dataset['capital'][0] - init_capital
    cumulative_pnl = daily_pnl.cumsum()

    # Calculate total return and annualized return
    total_pnl = cumulative_pnl[-1]
    
    annualized_return = (dataset['capital'].iloc[-1]/dataset['capital'].iloc[0]) ** (252 / len(daily_pnl)) - 1
    cumulative_return = dataset['capital'][-1] / init_capital - 1

    # Calculate maximum drawdown
    rolling_max = cumulative_pnl.cummax()
    drawdown = cumulative_pnl - rolling_max
    max_drawdown = drawdown.min()

    # Calculate Sharpe ratio (using risk-free rate of 0% for simplicity)
    daily_returns = dataset['capital'].pct_change()
    daily_returns.iloc[0] = dataset['capital'][0] / init_capital - 1

    rf = get_Rf(START, END)
    ann_std = (daily_returns.std(ddof=0) * np.sqrt(252))
    
    sharpe_ratio = (annualized_return - rf) / ann_std


    # Print results
    print("Total PnL:", total_pnl)
    print("Annualized Return:", annualized_return)
    print("Max Drawdown:", max_drawdown)
    print("Sharpe Ratio:", sharpe_ratio)

    return {'Total PnL': total_pnl, 'Annualized Return': annualized_return, 'Cumulative Return': cumulative_return, 'Annualized Standard Deviation': ann_std, 'Max Drawdown': max_drawdown, 'Sharpe Ratio': sharpe_ratio}


def plot_graph():

    # plot graph of four HSI return-based variables
    cols = ['capital',  'target_pos', 'signal', 'target_pos']
    # fig, axs = plt.subplots(ncols=len(cols), figsize=(20,10), sharex=True)


    sims = [s for s in Path('./etfai/backtesting/simulation').iterdir() if (s.is_file() and not s.name.startswith('.'))] 
    
    
    plot_names = ['Trading Simulation', 'Position', 'Signals','Position_sep']
    y_labels = ['Capital (HKD)', 'Volume', 'Signal (Buy=1, Neutral=0, Sell=-1)','Volume']

    plt.figure(1, figsize=(20,10))
    
    plt.figure(2, figsize=(20,10))
    # plt.subplot(int(f'{len(sims)}1{len(sims)}'))
    


    plt.figure(3, figsize=(20,10))
    plt.yticks([-1, 0, 1])
    plt.subplots(len(sims), 1, figsize=(20, 10), sharex=True, sharey=True)


    plt.figure(4, figsize=(20,10))
    # plt.subplot(int(f'{len(sims)}1{len(sims)}'))
    plt.subplots(len(sims), 1, figsize=(20, 10), sharex=True, sharey=True)

    # [plt.figure(i+1, figsize=(20,10)) for i in range(len(cols))]


    for i, s in enumerate(sims):
        sim = pd.read_csv(s, index_col='Date',parse_dates=True)

        label = s.stem
        
        for j, plot in enumerate(cols):
            plt.figure(j+1)
                
            if j ==2 or j==3:
                # plt.subplot(int(f'{len(sims)}1{j+1}'))
                plt.subplot(int(len(sims)), 1, i+1)


                
            
            # plt.xticks(rotation=45)
            
            # plt.xlim(sim.index[0], sim.index[-1])

            # ax = plt.gca()
            # plt.ylabel(y_labels[j])
            # plt.xlabel('Date')
            ax = plt.gca()
            ax.yaxis.set_major_locator(plt.MaxNLocator(20))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot(sim.index, sim[plot], label=label, color=f'C{i}')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            # plt.ylabel(y_labels[j])
            plt.grid(linewidth=0.5)
    # for j, plot in enumerate(cols):
    #     axs[j].set_ylabel(plot)

    for j, plot in enumerate(cols):
        plt.figure(j+1)
        plt.tight_layout()
        plt.xlabel('Date')
        plt.ylabel(y_labels[j])
        plt.xticks(rotation=45)
        # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.grid(linewidth=0.5)
        plt.xlim(sim.index[0], sim.index[-1])
        # fig = plt.gcf()
        # fig.text(0.5, 0.04, 'Date', va='center', ha='center', fontsize=rcParams['axes.labelsize'])
        # fig.text(0.04, 0.5, y_labels[j], va='center', ha='center', rotation='vertical', fontsize=rcParams['axes.labelsize'])

        # # ax = plt.gca()
        # plt.ylabel(y_labels[j])
        # plt.xlabel('Date')

        
        plt.savefig(f'./etfai/backtesting/simulation/plots/{plot_names[j]}.png',dpi=300, bbox_inches='tight')




def backtest(price_data, signal, port_name):
    dataset = price_data.join(signal, how='left')
    dataset = dataset.ffill()
    sim_df = trading_sim(dataset)
    sim_df.to_csv(f'./etfai/backtesting/simulation/{port_name}.csv', index=True, index_label='Date')
    performance_metrics = eval_performance(sim_df)
    return sim_df, performance_metrics



price_data = get_hist_data()

price_data = price_data[(price_data.index >= pd.to_datetime(START)) & (price_data.index <= pd.to_datetime(END))][['Open','Adj Close']]

BuyNHold_signal = pd.DataFrame({'signal': [0]*len(price_data)}, index=price_data.index)
BuyNHold_signal.iloc[0,0] = 1

signals_fp = [LSTM_P, LSTM_PP, XGBOOST_P, XGBOOST_PP, LLM]
signals = [BuyNHold_signal]
signals.extend([pd.read_csv(fp, index_col='Date', parse_dates=True).rename(columns={'Predicted': 'signal'}) for fp in signals_fp])
signals[-1]['signal'] = signals[-1]['signal'].map({0:1, 1:0, 2:-1})


all_performance = []
for i, s in enumerate(signals):
    port_name = PORT_NAMES[i]
    sim_df, performance = backtest(price_data, s, port_name)
    performance['Portfolio'] = port_name
    all_performance.append(performance)

all_df = pd.DataFrame(all_performance)
all_df.set_index('Portfolio', inplace=True)
all_df.to_csv('./etfai/backtesting/performance/result.csv')

plot_graph()

print('Done')

# valid_data = price_data[(price_data.index >= pd.to_datetime('2019-12-01')) & (price_data.index <= pd.to_datetime('2021-05-31'))]
# test_data = price_data[(price_data.index >= pd.to_datetime('2021-06-01')) & (price_data.index <= pd.to_datetime('2022-11-30'))]

# BuyNHold_signal = pd.DataFrame({'signal': [0]*len(price_data)}, index=price_data.index)
# BuyNHold_signal.iloc[0,0] = 1
# BuyNHold_dataset = price_data.join(BuyNHold_signal, how='left')

# BuyNHold_backtest = backtest(BuyNHold_dataset)
# BuyNHold_backtest.to_csv('./etfai/backtesting/simulation/BuyNHold_sim.csv')

# eval_performance(BuyNHold_backtest)


# LSTM_PP_signal = pd.read_csv(LSTM_SIGNALS_PATH, index_col='Date', parse_dates=True)
# LSTM_PP_signal.rename(columns={'Predicted':'signal'}, inplace=True)
# LSTM_PP_dataset = price_data.join(LSTM_PP_signal, how='left')
# LSTM_dataset['signal'] = LSTM__PP_dataset['signal'].ffill()

# LSTM_backtest = backtest(LSTM_dataset)
# LSTM_backtest.to_csv('./etfai/backtesting/simulation/LSTM_NLP_sim.csv')

# eval_performance(LSTM_backtest)

 
# XGBOOST_signal = pd.read_csv(XGBOOST_SIGNALS_PATH, index_col='Date', parse_dates=True)
# XGBOOST_signal.rename(columns={'Predicted':'signal'}, inplace=True)
# XGBOOST_dataset = price_data.join(XGBOOST_signal, how='left')
# XGBOOST_dataset['signal'] = XGBOOST_dataset['signal'].ffill()

# XGBOOST_backtest = backtest(XGBOOST_dataset)
# XGBOOST_backtest.to_csv('./etfai/backtesting/simulation/XGBOOST_NLP_sim.csv')

# eval_performance(XGBOOST_backtest)

print('DONE')




