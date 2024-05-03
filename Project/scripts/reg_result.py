import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing

df_ti = pd.read_csv('../document/16_tense_index.csv', header=0)
df_tp = pd.read_csv('../document/16_tense_position.csv', header=0)
df_wp = pd.read_csv('../document/16_word_position.csv', header=0)
df_re = pd.read_csv('../document/16_result_score1.csv', header=0)

X = df_ti['FvsP']
y = df_re['final_score']
X_wContant = sm.add_constant(X)
Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
print(Model_all_index.summary())

X = df_ti['PvsN']
y = df_re['final_score']
X_wContant = sm.add_constant(X)
Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
print(Model_all_index.summary())

X = preprocessing.scale(df_tp['avg_tense_position'])
y = df_re['final_score']
X_wContant = sm.add_constant(X)
Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
print(Model_all_index.summary())

X = preprocessing.scale(df_wp['avg_word_position'])
y = df_re['final_score']
X_wContant = sm.add_constant(X)
Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
print(Model_all_index.summary())

