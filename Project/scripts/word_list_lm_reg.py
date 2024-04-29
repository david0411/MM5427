import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor


def stepwise_regression(X_LM, y, significance_level=0.1):
    features = X_LM.columns.tolist()
    while len(features) > 0:
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


dataset = pd.read_csv('../document/AnnualReports16_lm.csv')

y = dataset['market_abnormal_return']

X_LM = dataset[
    ['nasdq', 'market_value', 'btm', 'pre_alpha', 'pre_rmse', 'InstOwn_Perc', 'log_share', 'Negative_score',
     'Positive_score', 'Uncertainty_score', 'Litigious_score', 'Strong_Modal_score', 'Weak_Modal_score',
     'Constraining_score', 'Complexity_score']]

x_LM_train, x_LM_test, y_LM_train, y_LM_test = train_test_split(X_LM, y, test_size=0.2, random_state=42)

model_LM = LinearRegression()
model_LM.fit(x_LM_train, y_LM_train)

y_LM_pred = model_LM.predict(x_LM_test)

mse_LM = mean_squared_error(y_LM_test, y_LM_pred)
mae_LM = mean_absolute_error(y_LM_test, y_LM_pred)
r2_LM = r2_score(y_LM_test, y_LM_pred)
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
