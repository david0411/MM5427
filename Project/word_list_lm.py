import pandas as pd
import re
import nltk
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def sentiment_score(text, sen_list):
    # 确保文本不为空或None
    if pd.isnull(text) or text == "":
        return 0
    # 计算文本中情感词汇的总数
    total_count = sum(text.lower().count(word) for word in sen_list if word in text.lower())
    # 归一化得分，使用文本长度作为分母
    return total_count / max(len(text), 1)


def stepwise_regression(X_LM, y, significance_level=0.1):
    features = X_LM.columns.tolist()
    while (len(features) > 0):
        # Add a constant term to the features
        features_with_const = sm.add_constant(X_LM[features])

        # Fit the model and get p-values for the features
        p_values = sm.OLS(y, features_with_const).fit().pvalues[1:]  # Exclude constant

        # Check if p_values contains only NaNs or is empty
        if p_values.isnull().all() or p_values.empty:
            break

        # Get the feature with the maximum p-value
        max_p_feature = p_values.idxmax() if not p_values.isna().any() else None

        # If the maximum p-value is greater than or equal to the significance level, remove the feature
        if (max_p_feature is not None) and (p_values[max_p_feature] >= significance_level):
            features.remove(max_p_feature)
        else:
            break

    # Create a new DataFrame with the remaining features
    X_stepwise = X_LM[features]

    # Add a constant term to the final model
    X_stepwise = sm.add_constant(X_stepwise)

    return X_stepwise

dataset = pd.read_csv('AnnualReports16_processed.csv', encoding='latin-1')
dataset_cleaned = dataset.dropna()

y = dataset_cleaned['market_abnormal_return']

lexicon_LM = pd.read_csv('Loughran-McDonald_MasterDictionary_1993-2023.csv')
lexicon_LM['Word'] = lexicon_LM['Word'].str.lower()

Negative = set(lexicon_LM[lexicon_LM['Negative'] != 0]['Word'])
Positive = set(lexicon_LM[lexicon_LM['Positive'] != 0]['Word'])
Uncertainty = set(lexicon_LM[lexicon_LM['Uncertainty'] != 0]['Word'])
Litigious = set(lexicon_LM[lexicon_LM['Litigious'] != 0]['Word'])
Strong_Modal = set(lexicon_LM[lexicon_LM['Strong_Modal'] != 0]['Word'])
Weak_Modal = set(lexicon_LM[lexicon_LM['Weak_Modal'] != 0]['Word'])
Constraining = set(lexicon_LM[lexicon_LM['Constraining'] != 0]['Word'])
Complexity = set(lexicon_LM[lexicon_LM['Complexity'] != 0]['Word'])

for sentiment, score_name in zip(
        [Negative, Positive, Uncertainty, Litigious, Strong_Modal, Weak_Modal, Constraining, Complexity],
        ['Negative_score', 'Positive_score', 'Uncertainty_score', 'Litigious_score', 'Strong_Modal_score', 'Weak_Modal_score', 'Constraining_score', 'Complexity_score']
):
    dataset_cleaned.loc[:, score_name] = dataset_cleaned['processed_text'].apply(lambda x: sentiment_score(x, list(sentiment)))

print(dataset_cleaned.head())

X_LM = dataset_cleaned[['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share','Negative_score', 'Positive_score', 'Uncertainty_score', 'Litigious_score', 'Strong_Modal_score', 'Weak_Modal_score','Constraining_score','Complexity_score']]

x_LM_train, x_LM_test, y_LM_train, y_LM_test = train_test_split(X_LM, y, test_size=0.2, random_state=42)

model_LM = LinearRegression()
model_LM.fit(x_LM_train, y_LM_train)

y_LM_pred = model_LM.predict(x_LM_test)

mse_LM = mean_squared_error(y_LM_test, y_LM_pred)
mae_LM = mean_absolute_error(y_LM_test, y_LM_pred)
r2_LM= r2_score(y_LM_test, y_LM_pred)
medae_LM = median_absolute_error(y_LM_test, y_LM_pred)

print(f"Model_count_vector reply count MSE: {mse_LM}")
print(f'MAE_reply count: {mae_LM}')
print(f'R²__reply count: {r2_LM}')
print(f'MedAE_reply count: {medae_LM}')
print(f"model_LM.coef_: {model_LM.coef_}")

X_LM = sm.add_constant(X_LM)
model = sm.OLS(y, X_LM).fit()
print(model.summary())

vif_data = pd.DataFrame()
vif_data["feature"] = X_LM.columns
vif_data["VIF"] = [variance_inflation_factor(X_LM.values, i) for i in range(len(X_LM.columns))]

print(vif_data)

X_LM_stepwise = stepwise_regression(X_LM, y)

model_stepwise = sm.OLS(y, X_LM_stepwise).fit()
print(model_stepwise.summary())
