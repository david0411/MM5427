import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm


def count_sentences(text):
    if pd.isnull(text):
        return 0
    sentences = nltk.sent_tokenize(text)
    return len(sentences)


def delete_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) > 2:
        return ' '.join(sentences[2:])
    else:
        return text


df = pd.read_csv('../document/AnnualReports_16.csv')
sen_df = pd.read_csv('../document/16_result_score1.csv')
df2 = pd.read_csv('../document/16_result_score2.csv')

df['sentence_count'] = df['item7'].apply(count_sentences)
df = df[df['sentence_count'] > 10]

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

X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm)
results = model_sm.fit()
p_value = results.pvalues[1]
print("P-value:", p_value)

features = df2.loc[:, 'unc_Dic':'lit_risk']
X = features
y = df2['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

results = []

for feature in features:
    model = sm.OLS(y_train, sm.add_constant(X_train[[feature]]))
    results_single = model.fit()

    X_test_const = sm.add_constant(X_test[[feature]])
    y_pred = results_single.predict(X_test_const)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    coef = results_single.params[1]
    p_value = results_single.pvalues[1]

    results.append({'Feature': feature, 'Coefficient': coef, 'P-value': p_value, 'MSE': mse, 'R2 Score': r2})

results_df = pd.DataFrame(results)
sorted_results = results_df.sort_values(by=['P-value', 'R2 Score'], ascending=[True, True])
print(sorted_results)
