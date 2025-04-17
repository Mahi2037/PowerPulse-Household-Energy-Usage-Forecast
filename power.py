import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Data Loading and Initial Exploration ---
print("Loading data...")
try:
    df = pd.read_csv(r'L:\Guvi\Power\household_power_consumption.txt', sep=';', na_values=['?'], infer_datetime_format=True, parse_dates=['Date', 'Time'])
except FileNotFoundError:
    print("Error: household_power_consumption.txt not found.  Make sure the file is in the same directory as the script, or provide the correct path.")
    exit()

print("Data loaded successfully.")
print(f"Shape of the dataset: {df.shape}")
print(df.head())
print(df.info())

# --- 2. Data Preprocessing ---
print("\nStarting Data Preprocessing...")

# 2.1 Handling Missing Values
print("Handling missing values...")
print(f"Missing values before imputation:\n{df.isnull().sum()}")
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True) # Impute with mean
print(f"Missing values after imputation:\n{df.isnull().sum()}")

# 2.2 Combining Date and Time
print("Combining Date and Time...")
df['DateTime'] = df['Date'] + pd.to_timedelta(df['Time'].astype(str))
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

# 2.3 Data Type Conversion
print("Converting data types...")
numeric_cols = df.columns.drop('DateTime') # DateTime is already handled
df[numeric_cols] = df[numeric_cols].astype('float32')
print(df.info())

# 2.4 Feature Engineering (Basic)
print("Performing basic feature engineering...")
df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month

print("Data Preprocessing complete.")

# --- 3. Exploratory Data Analysis (EDA) ---
print("\nStarting Exploratory Data Analysis (EDA)...")

# 3.1 Summary Statistics
print("Summary Statistics:")
print(df.describe())

# 3.2 Visualizations
print("Creating visualizations...")

# Time series plot of Global_active_power
plt.figure(figsize=(12, 4))
plt.plot(df['Global_active_power'], label='Global Active Power')
plt.title('Global Active Power Over Time')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.legend()
plt.show()

# Distribution of Global_active_power
plt.figure(figsize=(8, 4))
sns.histplot(df['Global_active_power'], kde=True)
plt.title('Distribution of Global Active Power')
plt.xlabel('Global Active Power')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("EDA complete.")

# --- 4. Feature Selection and Data Splitting ---
print("\nPreparing data for modeling...")

X = df.drop('Global_active_power', axis=1)
y = df['Global_active_power']

# 4.1 Data Scaling
print("Scaling the data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4.2 Train-Test Split
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Data preparation complete.")

# --- 5. Model Training and Evaluation ---
print("\nStarting Model Training and Evaluation...")

def train_and_evaluate(model, model_name, X_train, y_train, X_test, y_test):
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)

    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"{model_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R-squared: {r2:.4f}")

    # Cross-validation (optional, but good practice)
    print("Performing cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)
    print(f"Cross-validation RMSE scores: {cv_rmse_scores}")
    print(f"Mean cross-validation RMSE: {cv_rmse_scores.mean():.4f}")

    return model, y_pred

# 5.1 Linear Regression
lr_model = LinearRegression()
lr_model, lr_y_pred = train_and_evaluate(lr_model, "Linear Regression", X_train, y_train, X_test, y_test)

# 5.2 Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42) # Example hyperparameters
rf_model, rf_y_pred = train_and_evaluate(rf_model, "Random Forest Regressor", X_train, y_train, X_test, y_test)

# 5.3 Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42) # Example hyperparameters
gb_model, gb_y_pred = train_and_evaluate(gb_model, "Gradient Boosting Regressor", X_train, y_train, X_test, y_test)

# 5.4 Neural Network (MLPRegressor)
nn_model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, max_iter=200) # Example hyperparameters
nn_model, nn_y_pred = train_and_evaluate(nn_model, "Neural Network", X_train, y_train, X_test, y_test)

print("\nModel Training and Evaluation complete.")

# --- 6. Feature Importance (for Random Forest and Gradient Boosting) ---
print("\nAnalyzing Feature Importance...")

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importances = pd.Series(importances, index=feature_names)
        feature_importances = feature_importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=feature_importances.index)
        plt.title(f'Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.show()
    else:
        print(f"Model {model_name} does not support feature importance analysis.")

plot_feature_importance(rf_model, X.columns, "Random Forest Regressor")
plot_feature_importance(gb_model, X.columns, "Gradient Boosting Regressor")

# --- 7. Results Visualization (Example) ---
print("\nVisualizing Results (Example)...")

# Scatter plot of actual vs. predicted values for Random Forest
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Values (Random Forest)')
plt.xlabel('Actual Global Active Power')
plt.ylabel('Predicted Global Active Power')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Add a diagonal line for reference
plt.show()

print("\nProject completed.  See the generated plots and the printed evaluation metrics.")
