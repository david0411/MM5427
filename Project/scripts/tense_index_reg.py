# regression-FvsP on return
import pandas as pd
import statsmodels.api as sm
from itertools import combinations

df = pd.read_csv('../document/16_tense_index.csv', header=0)
df2 = pd.read_csv('../document/filtered_16.csv', header=0)
df3 = pd.read_csv('../document/16_result_score1.csv', header=0)

df = pd.concat([df[['FvsP', 'PvsN']],
                df2[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share',
                     'market_abnormal_return']],
                df3['result']], axis=1)
df = df.dropna()

X = df[['FvsP', 'PvsN']]
C = df[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share']]
y = df['market_abnormal_return']

best_model = None
best_features = None
best_aic = float('inf')

# 逐步选择自变量
for x in combinations(X, 2):
    for c in combinations(C, 7):
        X = df[list(x) + list(c)]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_features = list(x) + list(c)
            best_aic = aic

if best_model is not None:
    print(best_model.summary())
    print("Best features:", best_features)
else:
    print("No model found.")

# regression-FvsP + PvsN + control variabels on result
X = df[['FvsP', 'PvsN']]
C = df[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share']]
y = df['result']

best_model = None
best_features = None
best_aic = float('inf')

# 逐步选择自变量
for x in combinations(X, 2):  # 从X中选择2个变量的组合
    for c in combinations(C, 7):
        # 构建自变量
        X = df[list(x) + list(c)]
        X = sm.add_constant(X)
        # 拟合模型
        model = sm.OLS(y, X).fit()
        # 计算AIC
        aic = model.aic
        # 保存最佳模型
        if aic < best_aic:
            best_model = model
            best_features = list(x) + list(c)
            best_aic = aic

# 输出最佳模型结果
if best_model is not None:
    print(best_model.summary())
    print("Best features:", best_features)
else:
    print("No model found.")
