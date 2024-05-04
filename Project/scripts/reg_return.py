import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing


def forward_stepwise(X, y):
    included = []
    excluded = list(X.columns)

    while len(excluded) > 0:
        best_pval = float('inf')
        best_var = None

        for var in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [var]])).fit()
            pval = model.pvalues[var]

            if pval < best_pval:
                best_pval = pval
                best_var = var

        if best_pval < 0.1:
            included.append(best_var)
            excluded.remove(best_var)
        else:
            break

    return included


df = pd.read_csv('../document/17_combine.csv', header=0)
df2 = pd.read_csv('../document/Preprocessed_17.csv', header=0)
df3 = pd.read_csv('../document/18_combine.csv', header=0)
df4 = pd.read_csv('../document/Preprocessed_18.csv', header=0)
dfx1 = pd.concat([df[['FvsP', 'PvsN', 'final_score']],
                 df2[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share',
                      'market_abnormal_return']]]
                , axis=1)
dfx2 = pd.concat([df3[['FvsP', 'PvsN', 'final_score']],
                 df4[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share',
                      'market_abnormal_return']]]
                , axis=1)
dfx = pd.concat([dfx1, dfx2], axis=0)

dfx = dfx.replace([np.inf, -np.inf], np.nan)
dfx = dfx.dropna()
dfx['FvsP'] = preprocessing.scale(dfx['FvsP'])
dfx['PvsN'] = preprocessing.scale(dfx['PvsN'])
dfx['final_score'] = preprocessing.scale(dfx['final_score'])
dfx['nasdq'] = preprocessing.scale(dfx['nasdq'])
dfx['btm'] = preprocessing.scale(dfx['btm'])
dfx['pre_rmse'] = preprocessing.scale(dfx['pre_rmse'])
dfx['market_value'] = preprocessing.scale(dfx['market_value'])
dfx['pre_alpha'] = preprocessing.scale(dfx['pre_alpha'])
dfx['InstOwn_Perc'] = preprocessing.scale(dfx['InstOwn_Perc'])
dfx['log_share'] = preprocessing.scale(dfx['log_share'])

X = dfx[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share']]
y = dfx['market_abnormal_return']
X_wContant = sm.add_constant(X)
Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
y_pred = Model_all_index.predict(X_wContant)
residuals = y - y_pred
print("RMSE:", np.sqrt(np.mean(residuals**2)))
print('R2:', Model_all_index.rsquared)
print('AR2:', Model_all_index.rsquared_adj)
print("F-value:", Model_all_index.fvalue)
print("p-value:", Model_all_index.f_pvalue)
print("Coeff:", Model_all_index.params)

# selected_vars = forward_stepwise(X, y)
# print(selected_vars)
