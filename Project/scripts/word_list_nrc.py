import re
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def sentiment_score(text, sen_list):
    temp_list = []
    for t in text:
        if isinstance(t, str):
            t = re.sub("[^A-Za-z0-9]+", " ", t)
            t = t.lower()
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


dataset = pd.read_csv('../document/AnnualReports16_processed.csv', header=0)
lexicon = pd.read_csv('../word_list/NRC-Emotion-Lexicon.txt', sep='\t', names=['term', 'category', 'associated'])

category_list = lexicon['category'].unique().tolist()
filtered_df = lexicon[lexicon['associated'] == 1]
grouped_df = filtered_df.groupby('category')['term'].apply(list)

dataset['Pos_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['positive'])
dataset['Neg_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['negative'])
dataset['Ang_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['anger'])
dataset['Anti_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['anticipation'])
dataset['Dis_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['disgust'])
dataset['Fear_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['fear'])
dataset['Joy_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['joy'])
dataset['Sad_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['sadness'])
dataset['Surp_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['surprise'])
dataset['Tru_Dic'] = sentiment_score(dataset['processed_text'], grouped_df.loc['trust'])

dataset['Sent_Dic_pos_surp'] = (
            dataset['Pos_Dic'] + dataset['Anti_Dic'] + dataset['Joy_Dic'] + dataset['Surp_Dic'] + dataset['Tru_Dic']
            - dataset['Neg_Dic'] - dataset['Ang_Dic'] - dataset['Dis_Dic'] - dataset['Fear_Dic'] - dataset['Sad_Dic'])

dataset['Sent_Dic_neg_surp'] = (
            dataset['Pos_Dic'] + dataset['Anti_Dic'] + dataset['Joy_Dic'] + dataset['Tru_Dic'] - dataset['Surp_Dic']
            - dataset['Neg_Dic'] - dataset['Ang_Dic'] - dataset['Dis_Dic'] - dataset['Fear_Dic'] - dataset['Sad_Dic'])

dataset.to_csv('../document/AnnualReports16_nrc.csv', index=False)

features = dataset.loc[:, 'Pos_Dic':'Sent_Dic_neg_surp']
X = features
y = dataset['market_abnormal_return']
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
    results.append({'Feature': feature, 'p-value': p_value, 'Coefficient': coef, 'MSE': mse, 'R2 Score': r2})

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)
sorted_results = results_df.sort_values(by=['MSE', 'R2 Score'], ascending=[True, False])
print(sorted_results)

results = []

# Iterate over each feature and evaluate the linear regression model
for feature in features:
    # Create the linear regression model
    model = LinearRegression()
    model.fit(X_train[[feature]], y_train)

    # Predict on the test set
    y_pred = model.predict(X_test[[feature]])

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Get the regression coefficient
    coef = model.coef_[0]

    # Append the results to the list
    results.append({'Feature': feature, 'Coefficient': coef, 'MSE': mse, 'R2 Score': r2})

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)
sorted_results = results_df.sort_values(by=['MSE', 'R2 Score'], ascending=[True, False])

new_X_train = X_train.drop(['Sent_Dic_pos_surp', 'Sent_Dic_neg_surp'], axis=1)
new_X_test = X_test.drop(['Sent_Dic_pos_surp', 'Sent_Dic_neg_surp'], axis=1)


# Add a constant column to the new_X_train data frame
X_train_const = sm.add_constant(new_X_train)

# Create and fit the linear regression model using statsmodels
model = sm.OLS(y_train, X_train_const)
results = model.fit()

# Predict on the test set
X_test_const = sm.add_constant(new_X_test)
y_pred = results.predict(X_test_const)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('MSE of multiple linear regression model:', mse)
print('R2 of multiple linear regression model:', r2)

# Get the coefficients and p-values from the results
coefficients = results.params[1:]
p_values = results.pvalues[1:]

# Create a DataFrame with coefficients and p-values
coeff_df = pd.DataFrame({'Coefficient': coefficients, 'p-value': p_values})

# Print the coefficients and p-values
print(coeff_df)
