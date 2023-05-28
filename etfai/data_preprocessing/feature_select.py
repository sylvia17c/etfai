from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, kruskal
# from data_.data import Indicators


def RFImportance(X, y, num_runs=10):
    importances_list = []

    for _ in range(num_runs):
        model = RandomForestClassifier()
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        importances_list.append(importances)
        
    average_importances = np.mean(importances_list, axis=0)
    indices = np.argsort(average_importances)[::-1]

    feature_importances = pd.Series(average_importances[indices], index=X.columns[indices], name='RFImportance')

    for i, index in enumerate(indices):
        print(f"{i+1}. {X.columns[index]}: {average_importances[index]}")

    return feature_importances


# def abs_corr_weight(df):
#     corr_matrix = df.corr(numeric_only=True)
#     numeric_cols = corr_matrix.columns
#     for col in df.columns:
#         ticker, indicator = col.split('_', 1)
#         indicator = getattr(Indicators, indicator.upper())
#         if indicator in Indicators.DISC:
#             distinct_ct = df[col].nunique()
#             corr = []
#             if distinct_ct == 2:
#                 one_hot = pd.get_dummies(df[col], drop_first=True)
                
#                 for ncol in numeric_cols:
#                     correlation, p_value = pointbiserialr(one_hot, df[ncol])
#                     corr.append(correlation)
#             else:
#                 # To correlation between numeric columns and categorical columns with more than 2 categories
#                 pass
                

