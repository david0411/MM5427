import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv('../document/16_combine.csv', header=0)

# X = df['FvsP']
# y = df['final_score']
# X_wContant = sm.add_constant(X)
# Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
# print(Model_all_index.summary())
#
# X = df['PvsN']
# y = df['final_score']
# X_wContant = sm.add_constant(X)
# Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
# print(Model_all_index.summary())
#
# X = preprocessing.scale(df['avg_tense_position'])
# y = df['final_score']
# X_wContant = sm.add_constant(X)
# Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
# print(Model_all_index.summary())
#
# X = preprocessing.scale(df['avg_word_position'])
# y = df['final_score']
# X_wContant = sm.add_constant(X)
# Model_all_index = sm.OLS(endog=y, exog=X_wContant).fit(maxiter=5000)
# print(Model_all_index.summary())

print(df['final_score'].describe())
print(df['FvsP'].describe())
print(df['PvsN'].describe())
print(df['avg_tense_position'].describe())
print(df['avg_word_position'].describe())

X = df[['final_score', 'FvsP', 'PvsN']]
vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)
