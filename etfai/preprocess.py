import pandas as pd
from pathlib import Path
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.data import DataFetcher, load_data, fill_missing, one_hot_encode
from data_preprocessing.eda import split_df, plot_ret_line, plot_ret_freq, plot_missing
from data_preprocessing.feature_select import RFImportance

# from data_preprocessing.feature_select import abs_corr_weight

# # gather data
# data_obj = DataFetcher()
# data_obj.get_vars()

# plot_ret_line()

# # split all data into train, validate and test set
# data_dir = Path('./etfai/data_preprocessing/data/')
# # split all variables and target into training, validation and test set
# train_dir = Path(data_dir / 'train/')
# validate_dir = Path(data_dir / 'validate/')
# test_dir = Path(data_dir / 'test/')
# (train_dir / 'variable').mkdir(parents=True, exist_ok=True)
# (train_dir / 'target').mkdir(parents=True, exist_ok=True)
# (validate_dir / 'variable').mkdir(parents=True, exist_ok=True)
# (validate_dir / 'target').mkdir(parents=True, exist_ok=True)
# (test_dir / 'variable').mkdir(parents=True, exist_ok=True)
# (test_dir / 'target').mkdir(parents=True, exist_ok=True)

# data = pd.read_csv(data_dir / 'target/HSI.csv')
# dfs = split_df(data, [0.7, 0.85, 1])
# dfs[0].to_csv(train_dir / 'target/HSI.csv', index=False)
# dfs[1].to_csv(validate_dir / 'target/HSI.csv', index=False)
# dfs[2].to_csv(test_dir / 'target/HSI.csv', index=False)

# # plot frequence of different signals
# plot_ret_freq(dfs[0], f'{data_dir}/../result/four_ret_freq_train.png')
# plot_ret_freq(dfs[1], f'{data_dir}/../result/four_ret_freq_validate.png')
# plot_ret_freq(dfs[2], f'{data_dir}/../result/four_ret_freq_test.png')
variables = load_data('./etfai/data_preprocessing/data/variable/', prefix=True)
variables = pd.concat(variables, axis=1)
plot_missing(variables, 'missing_variables')
variables = fill_missing(variables)
variables = one_hot_encode(variables)

# # calculate absolute correlation value of each feature with all other features
# corr_matrix = variables.corr()

target = load_data('./etfai/data_preprocessing/data/target/', prefix=True)
target = pd.concat(target, axis=1)
target = target[['HSI_OO_ter_0.005']]
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
dataset = variables.join(target['HSI_OO_ter_0.005'], how='right')
RF_importance = RFImportance(dataset.iloc[:,:-1], dataset.iloc[:,-1])
RF_importance.to_csv('./etfai/data_preprocessing/result/RFImportance.csv')
top_ten = RF_importance.index[:10]
extracted_dataset = dataset[list(top_ten)+[dataset.columns[-1]]]

dataset_dir = Path('./etfai/data_preprocessing/dataset')
dataset_dir.mkdir(parents=True, exist_ok=True)
extracted_dataset.to_csv(f'{dataset_dir}/{extracted_dataset.columns[-1]}.csv')
pass