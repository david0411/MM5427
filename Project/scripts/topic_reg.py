from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd


lin_reg = LinearRegression()
dataset = pd.read_csv('AnnualReports16_topic.csv', header=0)
dataset = dataset.dropna()
x = pd.concat([pd.get_dummies(dataset['topic'], prefix='Topic', dtype=int),
               dataset[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share']]], axis=1)
print(x.head())
y = dataset['market_abnormal_return']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=99)

lin_reg.fit(x_train, y_train)

mse_cv = -cross_val_score(lin_reg, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
score_cv = cross_val_score(lin_reg, x_train, y_train, cv=10)

print("Training MSE: %.2f" % mse_cv.mean())
print("Training R square: %.6f" % score_cv.mean())
print("Testing MSE: %.2f" % mean_squared_error(y_test, lin_reg.predict(x_test)))
print("Testing R square: %.6f" % lin_reg.score(x_test, y_test, sample_weight=None))
