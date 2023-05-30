import configparser
import json
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from sklearn.preprocessing import LabelEncoder

class Indicators:
    INTRADAY_RET = 1
    OVERNIGHT_RET = 2
    DAILY_RET = 8
    INTRADAY_OVERNIGHT_RET_DIFF = 3
    RET_RANGE = 4
    INTRADAY_MOM = 5
    OVERNIGHT_MOM = 6
    INTRADAY_OVERNIGHT_MOM_DIFF = 7
    STOCH_K = 9
    DISC_STOCH_K_1 = 10
    DISC_STOCH_K_3 = 11
    MACD = 12
    DISC_MACD_1 = 13
    DISC_MACD_9 = 14
    RSI = 15
    DISC_RSI_TER = 16
    DISC_RSI_BI = 17
    DISC = {10:9, 11:9, 13:12, 14:12, 16:15, 17:15}

    @classmethod
    def is_discrete(cls, indicator):
        if indicator in cls.DISC:
            return True
        return False

    @classmethod
    def calculate_indicator(cls, indicator, data=None, raw=None):
        # check indicator value and calculate corresponding indicator
        if indicator == cls.INTRADAY_RET:
            return cls.intraday_ret(data)
        elif indicator == cls.OVERNIGHT_RET:
            return cls.overnight_ret(data)
        elif indicator == cls.DAILY_RET:
            return cls.daily_ret(data)
        elif indicator == cls.INTRADAY_OVERNIGHT_RET_DIFF:
            return cls.intraday_overnight_ret_diff(data)
        elif indicator == cls.RET_RANGE:
            return cls.ret_range(data)
        elif indicator == cls.INTRADAY_MOM:
            df = data.join(cls.intraday_ret(data))
            return pd.Series(cls.momentum(df, 1), name='intraday_mom')
        elif indicator == cls.OVERNIGHT_MOM:
            df = data.join(cls.overnight_ret(data))
            return pd.Series(cls.momentum(df, 1), name='overnight_mom')
        elif indicator == cls.INTRADAY_OVERNIGHT_MOM_DIFF:
            df = data.join(cls.intraday_ret(data))
            intra_mom = cls.momentum(df, 1)
            df = data.join(cls.overnight_ret(data))
            overn_mon = cls.momentum(df, 1)
            return intra_mom - overn_mon
        elif indicator == cls.STOCH_K:
            return cls.stoch_k(data)
        elif indicator == cls.DISC_STOCH_K_1:
            return cls.disc_stoch_k_1(data, 1, stoch_k_n=raw)
        elif indicator == cls.DISC_STOCH_K_3:
            return cls.disc_stoch_k_3(data, 3, stoch_k_n=raw)
        elif indicator == cls.MACD:
            return cls.macd(data)
        elif indicator == cls.DISC_MACD_1:
            return cls.disc_macd_1(data, 1, macd=raw)
        elif indicator == cls.DISC_MACD_9:
            return cls.disc_macd_9(data, 9, macd=raw)
        elif indicator == cls.RSI:
            return cls.rsi(data)
        elif indicator == cls.DISC_RSI_TER:
            return cls.disc_rsi_ter(data, rsi=raw)
        elif indicator == cls.DISC_RSI_BI:
            return cls.disc_rsi_bi(data, rsi=raw)
        else:
            return None

    @classmethod
    def intraday_ret(cls, df):
        return pd.Series((df['Close'] - df['Open']) / df['Open'], name='intraday_ret')

    @classmethod
    def overnight_ret(cls, df):
        return pd.Series((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1), name='overnight_ret')

    @classmethod
    def daily_ret(cls, df):
        return pd.Series(df['Close'].pct_change(), name='daily_ret')

    @classmethod
    def intraday_overnight_ret_diff(cls, df):
        return pd.Series(cls.intraday_ret(df) - cls.overnight_ret(df), name='intraday_overnight_ret_diff')

    @classmethod
    def ret_range(cls, df):
        return pd.Series((df['High'] - df['Low']) / df['Close'].shift(1), name='ret_range')
    
    @classmethod
    def momentum(cls, df, n):
        if n == 0:
            return pd.Series(name=f'mom_{n}')
        return pd.Series(df['Close'] / df['Close'].shift(n) - 1, name=f'mom_{n}')

    @classmethod
    def stoch_k(cls, df, n=14):
        low_n = df['Low'].rolling(n).min()
        high_n = df['High'].rolling(n).max()
        return pd.Series((df['Close'] - low_n) / (high_n - low_n), name='stoch_k')

    @classmethod
    def disc_stoch_k_1(cls, df=None, n=14, stoch_k_n=None):
        if stoch_k_n is None: 
            stoch_k_n = cls.stoch_k(df, n)
        return pd.Series((stoch_k_n - stoch_k_n.shift(1)).apply(lambda x: 1 if x > 0 else 0), name='disc_stoch_k_1')

    @classmethod
    def disc_stoch_k_3(cls, df=None, n=14, stoch_k_n=None):
        if stoch_k_n is None:
            stoch_k_n = cls.stoch_k(df, n)
        stoch_k_n_3 = stoch_k_n.rolling(3).mean()
        return pd.Series((stoch_k_n - stoch_k_n_3).apply(lambda x: 1 if x > 0 else 0), name='disc_stoch_k_3')

    @classmethod
    def macd(cls, df, n_fast=12, n_slow=26):
        ema_fast = df['Close'].ewm(span=n_fast, min_periods=n_fast).mean()
        ema_slow = df['Close'].ewm(span=n_slow, min_periods=n_slow).mean()
        return pd.Series(ema_fast - ema_slow, name='macd')

    @classmethod
    def disc_macd_1(cls, df=None, n_fast=12, n_slow=26, macd=None):
        if macd is None:
            macd = cls.macd(df, n_fast, n_slow)
        return pd.Series((macd - macd.shift(1)).apply(lambda x: 1 if x > 0 else 0), name='disc_macd_1')

    @classmethod
    def disc_macd_9(cls, df=None, n_fast=12, n_slow=26, macd=None):
        if macd is None:
            macd = cls.macd(df, n_fast, n_slow)
        macd_signal = macd.ewm(span=9, min_periods=9).mean()
        return pd.Series((macd - macd_signal).apply(lambda x: 1 if x > 0 else 0), name='disc_macd_9')

    @classmethod
    def rsi(cls, df, period=14):
        delta = df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1*delta.clip(upper=0)
        ema_up = up.ewm(com=period-1, adjust=True).mean()
        ema_down = down.ewm(com=period-1, adjust=True).mean()
        rs = ema_up/ema_down
        rsi = 100 - (100/(1+rs))
        return pd.Series(rsi, name='rsi')

    @classmethod
    def disc_rsi_bi(cls, df=None, period=14, rsi=None):
        if rsi is None:
            rsi = cls.rsi(df, period)
        discrete_rsi = np.where(rsi <= 30, 'oversold', 'overbought')
        return pd.Series(discrete_rsi, name='disc_rsi_bi')

    @classmethod
    def disc_rsi_ter(cls, df=None, period=14, rsi=None):
        if rsi is None:
            rsi = cls.rsi(df, period)
        bins = [-np.inf, 30, 70, np.inf]
        labels = ['oversold', 'neutral', 'overbought']
        discrete_rsi = pd.cut(rsi, bins=bins, labels=labels)
        return pd.Series(discrete_rsi, name='disc_rsi_ter')
    

class Target:
    OPEN_TO_OPEN = 1
    OPEN_TO_CLOSE = 2
    CLOSE_TO_OPEN = 3
    CLOSE_TO_CLOSE = 4
    BINARY_TARGET = 5
    TERNARY_TARGET = 6

    @classmethod
    def calculate_target(cls, target, data=None, **kwargs):
        if target == cls.OPEN_TO_OPEN:
            return cls.open_to_open(data)
        elif target == cls.OPEN_TO_CLOSE:
            return cls.open_to_close(data)
        elif target == cls.CLOSE_TO_OPEN:
            return cls.close_to_open(data)
        elif target == cls.CLOSE_TO_CLOSE:
            return cls.close_to_close(data)
        elif target == cls.BINARY_TARGET:
            data_col = kwargs['data_col']
            cut_off = kwargs['cut_off']
            return cls.binary_label(data_col, cut_off)
        elif target == cls.TERNARY_TARGET:
            data_col = kwargs['data_col']
            cut_off = kwargs['cut_off']
            return cls.ternary_label(data_col, cut_off)
    
    @classmethod
    def open_to_open(cls, data):
        return pd.Series(data['Open'].pct_change(), name='OO')
    
    @classmethod
    def open_to_close(cls, data):
        return pd.Series((data['Close'] - data['Open']) / data['Open'], name='OC')
    
    @classmethod
    def close_to_open(cls, data):
        return pd.Series((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1), name='CO')
    
    @classmethod
    def close_to_close(cls, data):
        return pd.Series(data['Close'].pct_change(), name='CC')
    
    @classmethod
    def binary_label(cls, data_col, cut_off):
        s = np.where(data_col > cut_off, 'buy', 'sell')
        return pd.Series(s, name=f'{data_col.name}_bi_{cut_off}', index=data_col.index)
    
    @classmethod
    def ternary_label(cls, data_col, cut_off):
        bins = [-np.inf, -cut_off, cut_off, np.inf]
        labels = ['sell', 'neutral', 'buy']
        s = pd.cut(data_col, bins=bins, labels=labels)
        return pd.Series(s, name=f'{data_col.name}_ter_{cut_off}')


class DataFetcher:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read('./etfai/config.ini')
        with open(self.config['DATA']['Variable'], 'r') as f:
            self.tickers = json.load(f)
        self.start_date = self.config['DATA']['StartDate']
        self.end_date = self.config['DATA']['EndDate']
        self.eff_date = self.config['DATA']['EffectiveDate']
        self.data_dir = Path(self.config['DATA']['DataDir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, ticker, attr):
        if attr['source'] == 'yfinance':
            df = yf.download(attr['symbol'], start=self.start_date, end=self.end_date)
        else:
            df = pd.read_csv(f'{self.data_dir}/raw/{ticker}.csv', index_col='Date')
            df.index = pd.to_datetime(df.index)
        return df
        
    def get_vars(self):
        for ticker, attr in self.tickers.items():
            df = self.fetch(ticker, attr)
            df_var = pd.DataFrame(index=df.index)
            if df.empty: 
                continue

            vars_ = attr['variable']
            for var in vars_:
                df_var = df_var.join(Indicators.calculate_indicator(data=df, indicator=var))
            df_var = df_var[(df_var.index >= pd.to_datetime(self.eff_date)) & (df_var.index <= pd.to_datetime(self.end_date))]
            df_var.to_csv(f'{self.data_dir}/variable/{ticker}.csv')

            target = attr.get('target')
            if target:
                df_target = pd.DataFrame(index=df.index)
                for tar in target:
                    s_tar = Target.calculate_target(tar, df)
                    df_target = df_target.join(s_tar)
                    df_target = df_target.join(Target.calculate_target(Target.BINARY_TARGET, data_col=s_tar, cut_off=0))
                    df_target = df_target.join(Target.calculate_target(Target.TERNARY_TARGET, data_col=s_tar, cut_off=0.005))
                    df_target = df_target.join(Target.calculate_target(Target.TERNARY_TARGET, data_col=s_tar, cut_off=0.01))
                
                df_target = df_target[(df_target.index >= pd.to_datetime(self.eff_date)) & (df_target.index <= pd.to_datetime(self.end_date))]
                df_target.to_csv(f'{self.data_dir}/target/{ticker}.csv')

def load_data(dataset_dir, prefix=False):
    files = glob(f'{Path(dataset_dir)}/*.csv')
    dfs = []
    if prefix:
        for f in files:
            indicator = Path(f).stem.replace(' ','').replace('_', '2')
            df = pd.read_csv(f, index_col='Date', parse_dates=True)
            df.columns = [f'{indicator}_{col}' for col in df.columns]
            dfs.append(df)
    else:
        for f in files:
            df = pd.read_csv(f, index_col='Date', parse_dates=True)
            dfs.append(df)
    return dfs

def fill_missing(df):
    df.interpolate(method='linear', inplace=True)

    # recalculate discrete indicators after interploation
    for col in df.columns:
        ticker, indicator = col.split('_', 1)
        indicator = getattr(Indicators, indicator.upper())
        if indicator in Indicators.DISC:
            v = Indicators.DISC[indicator]
            for attr_name in dir(Indicators):
                attr_v = getattr(Indicators, attr_name)
                if attr_v == v:
                    raw = df[f'{ticker}_{attr_name.lower()}']
            df[col] = Indicators.calculate_indicator(indicator, raw=raw)
            
    return df

def one_hot_encode(df):
    label_encoder = LabelEncoder()
    for col in df.columns:
        _, indicator = col.split('_', 1)
        indicator = getattr(Indicators, indicator.upper())
        if indicator in Indicators.DISC:
            df[col] = label_encoder.fit_transform(df[col])
    return df