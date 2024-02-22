# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from sklearn.ensemble import StackingRegressor

# Load the sales data
sales_data = pd.read_csv('sales_data.csv')

# Let's assume the data has 'Date', 'Sales', 'Marketing Channel', 'Marketing Spend', 'Promotions', 'Competitor Price', and 'Weather' columns
features = ['Date', 'Marketing Channel', 'Marketing Spend', 'Promotions', 'Competitor Price', 'Weather']
X = sales_data[features]
y = sales_data['Sales']

# Convert date to datetime
X['Date'] = pd.to_datetime(X['Date'])

# Extract additional date features
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day
X.drop(columns=['Date'], inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['Marketing Channel']
numerical_features = ['Marketing Spend', 'Promotions', 'Competitor Price', 'Weather', 'Year', 'Month', 'Day']

# Create preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Initialize base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
    ('lgbm', LGBMRegressor(n_estimators=100, random_state=42))
]

# Initialize meta-estimator
meta_estimator = Ridge()

# Initialize stacking regressor with hyperparameter tuning
stacking_regressor = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_estimator,
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)

# Create a pipeline including preprocessing, feature selection, and stacking regressor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(GradientBoostingRegressor())),
    ('stacking_regressor', stacking_regressor)
])

# Define hyperparameters for tuning
param_grid = {
    'feature_selection__estimator__n_estimators': [50, 100, 200],
    'stacking_regressor__ridge__alpha': [0.1, 1.0, 10.0]
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", grid_search.best_params_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Predicted vs Actual Sales')
plt.show()

# Feature Importance
importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = importance.importances_mean.argsort()
plt.barh(X.columns[sorted_idx], importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")
plt.title("Feature Importance")
plt.show()
