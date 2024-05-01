import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


def sentiment_score(text, sen_list):
    temp_list = []
    for t in text:
        if isinstance(t, str):
            temp = 0
            for w in sen_list:
                temp += t.count(w)
            if len(t) != 0:
                temp_list.append(temp / len(t))
            else:
                temp_list.append(0)
        else:
            temp_list.append(0)
    return temp_list


df = pd.read_csv('../document/AnnualReports16_processed2.csv')
sen_df = pd.read_csv('../document/emot_score_16.csv')

X = np.array(df['pre_alpha']).reshape(-1, 1)
y = np.array(sen_df['result']).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

y_train = np.array(y_train)
model.fit(X_train, y_train)
coefficient = model.coef_[0]
print("Coefficient:", coefficient)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

X_train_sm = sm.add_constant(X_train)  # Add a constant term to X2_train
model_sm = sm.OLS(y_train, X_train_sm)
results = model_sm.fit()
p_value = results.pvalues[1]
print("P-value:", p_value)

df2 = pd.DataFrame(sen_df['result']).copy()
df2['unc_Dic'] = sentiment_score(df['item7'], unc_list)
df2['stg_Dic'] = sentiment_score(df['item7'], stg_list)
df2['weak_Dic'] = sentiment_score(df['item7'], weak_list)

df2['lit_Dic'] = sentiment_score(df['item7'], lit_list)
df2['ctr_Dic'] = sentiment_score(df['item7'], ctr_list)

df2['unc_risk'] = df2['unc_Dic'] + df2['weak_Dic'] - df2['stg_Dic']
df2['lit_risk'] = df2['lit_Dic'] + df2['ctr_Dic']

features = df2.loc[:, 'unc_Dic':'lit_risk']
X = features
y = df2['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

# Iterate over each feature and evaluate the linear regression model
for feature in features:
    # Create the linear regression model
    model = sm.OLS(y_train, sm.add_constant(X_train[[feature]]))
    results_single = model.fit()

    # Predict on the test set
    X_test_const = sm.add_constant(X_test[[feature]])
    y_pred = results_single.predict(X_test_const)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Get the regression coefficient and p-value
    coef = results_single.params[1]
    p_value = results_single.pvalues[1]

    # Append the results to the list
    results.append({'Feature': feature, 'Coefficient': coef, 'P-value': p_value, 'MSE': mse, 'R2 Score': r2})

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)
sorted_results = results_df.sort_values(by=['P-value', 'R2 Score'], ascending=[True, True])
print(sorted_results)
