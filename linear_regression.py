import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/hamdanzoghbor/Desktop/Machine Learning/BKB_WaterQualityData_2020084.csv"
data = pd.read_csv(file_path)

# Data Cleaning
columns_to_drop = ['Unit_Id', 'Air Temp-Celsius', 'DateVerified', 'WhoVerified', 'Air Temp (?F)']
data_cleaned = data.drop(columns=columns_to_drop)
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].median())

# Data Preparation
features = data_cleaned[['Salinity (ppt)', 'Dissolved Oxygen (mg/L)', 'Secchi Depth (m)', 'Water Depth (m)', 'Water Temp (?C)', 'AirTemp (C)']]
target = data_cleaned['pH (standard units)']

# Splitting the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Hyperparameter choices
lambdas = [0.01, 0.1, 1, 10, 100, 1000]  # Regularization parameter choices
degrees = [1, 2, 3]  # Polynomial degrees for feature transformation
kernels = ['linear', 'poly', 'rbf']

# Ridge Regression
results_ridge = []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    for l in lambdas:
        model_ridge = Ridge(alpha=l)
        pipeline = Pipeline([
            ('imputer', imputer),
            ('polynomial_features', poly),
            ('standard_scaler', scaler),
            ('ridge_regression', model_ridge)
        ])
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        results_ridge.append({
            'Degree': degree,
            'Lambda': l,
            'Training MSE': mean_squared_error(y_train, y_train_pred),
            'Validation MSE': mean_squared_error(y_val, y_val_pred),
            'Training R² Score': r2_score(y_train, y_train_pred),
            'Validation R² Score': r2_score(y_val, y_val_pred),
            'Feature Weights': pipeline['ridge_regression'].coef_
        })

# Lasso Regression
results_lasso = []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    for l in lambdas:
        model_lasso = Lasso(alpha=l)
        pipeline = Pipeline([
            ('imputer', imputer),
            ('polynomial_features', poly),
            ('standard_scaler', scaler),
            ('lasso_regression', model_lasso)
        ])
        pipeline.fit(X_train, y_train)
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        results_lasso.append({
            'Degree': degree,
            'Lambda': l,
            'Training MSE': mean_squared_error(y_train, y_train_pred),
            'Validation MSE': mean_squared_error(y_val, y_val_pred),
            'Training R² Score': r2_score(y_train, y_train_pred),
            'Validation R² Score': r2_score(y_val, y_val_pred),
            'Feature Weights': pipeline['lasso_regression'].coef_
        })

# Support Vector Regression (SVR)
results_svr = []
for kernel in kernels:
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    model_svr = SVR(kernel=kernel)
    pipeline = Pipeline([
        ('imputer', imputer),
        ('standard_scaler', scaler),
        ('svr', model_svr)
    ])
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    results_svr.append({
        'Kernel': kernel,
        'Training MSE': mean_squared_error(y_train, y_train_pred),
        'Validation MSE': mean_squared_error(y_val, y_val_pred),
        'Training R² Score': r2_score(y_train, y_train_pred),
        'Validation R² Score': r2_score(y_val, y_val_pred)
    })

# Converting results to DataFrame
results_table_ridge = pd.DataFrame(results_ridge)
results_table_lasso = pd.DataFrame(results_lasso)
results_table_svr = pd.DataFrame(results_svr)

print("Ridge Regression Results:")
print(results_table_ridge)
print("\nLasso Regression Results:")
print(results_table_lasso)
print("\nSVR Results:")
print(results_table_svr)

# Plotting feature weights and errors for Ridge Regression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for degree in degrees:
    lambda_values = [result['Lambda'] for result in results_ridge if result['Degree'] == degree]
    feature_weights = [result['Feature Weights'] for result in results_ridge if result['Degree'] == degree]
    validation_mse = [result['Validation MSE'] for result in results_ridge if result['Degree'] == degree]

    ax1.plot(lambda_values, feature_weights, label=f'Degree {degree}')
    ax2.plot(lambda_values, validation_mse, label=f'Degree {degree}')

ax1.set_title('Ridge Regression - Feature Weights')
ax1.set_xlabel('Lambda')
ax1.set_ylabel('Feature Weights')
ax1.set_xscale('log')
ax1.legend()

ax2.set_title('Ridge Regression - Validation MSE')
ax2.set_xlabel('Lambda')
ax2.set_ylabel('Validation MSE')
ax2.set_xscale('log')
ax2.legend()

plt.tight_layout()
plt.show()

# Plotting feature weights and errors for Lasso Regression
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

for degree in degrees:
    lambda_values = [result['Lambda'] for result in results_lasso if result['Degree'] == degree]
    feature_weights = [result['Feature Weights'] for result in results_lasso if result['Degree'] == degree]
    validation_mse = [result['Validation MSE'] for result in results_lasso if result['Degree'] == degree]

    ax1.plot(lambda_values, feature_weights, label=f'Degree {degree}')
    ax2.plot(lambda_values, validation_mse, label=f'Degree {degree}')

ax1.set_title('Lasso Regression - Feature Weights')
ax1.set_xlabel('Lambda')
ax1.set_ylabel('Feature Weights')
ax1.set_xscale('log')
ax1.legend()

ax2.set_title('Lasso Regression - Validation MSE')
ax2.set_xlabel('Lambda')
ax2.set_ylabel('Validation MSE')
ax2.set_xscale('log')
ax2.legend()

plt.tight_layout()
plt.show()

# Select the best performing model based on validation metrics
best_model_type = 'ridge'  # I am just assuming ridge regression performs  best
best_degree = 2
best_lambda = 0.1

if best_model_type == 'ridge':
    best_poly = PolynomialFeatures(degree=best_degree)
    best_scaler = StandardScaler()
    best_imputer = SimpleImputer(strategy='median')
    best_model = Ridge(alpha=best_lambda)
    best_pipeline = Pipeline([
        ('imputer', best_imputer),
        ('polynomial_features', best_poly),
        ('standard_scaler', best_scaler),
        ('ridge_regression', best_model)
    ])
elif best_model_type == 'lasso':
    best_poly = PolynomialFeatures(degree=best_degree)
    best_scaler = StandardScaler()
    best_imputer = SimpleImputer(strategy='median')
    best_model = Lasso(alpha=best_lambda)
    best_pipeline = Pipeline([
        ('imputer', best_imputer),
        ('polynomial_features', best_poly),
        ('standard_scaler', best_scaler),
        ('lasso_regression', best_model)
    ])
else:  # SVR
    best_kernel = 'rbf'  # Assuming RBF kernel performs the best
    best_scaler = StandardScaler()
    best_imputer = SimpleImputer(strategy='median')
    best_model = SVR(kernel=best_kernel)
    best_pipeline = Pipeline([
        ('imputer', best_imputer),
        ('standard_scaler', best_scaler),
        ('svr', best_model)
    ])

# Train the best model on the combined training and validation sets
X_train_val = pd.concat([X_train, X_val])
y_train_val = pd.concat([y_train, y_val])
best_pipeline.fit(X_train_val, y_train_val)

# Evaluate the best model on the test set
y_test_pred = best_pipeline.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nBest Model: {best_model_type}")
print("Test Set Metrics:")
print(f"MSE: {test_mse:.4f}")
print(f"R² Score: {test_r2:.4f}")

# Function to make a prediction based on input feature values
def predict_ph_level(input_features):
    input_df = pd.DataFrame([input_features], columns=features.columns)
    predicted_ph = best_pipeline.predict(input_df)
    return predicted_ph[0]

input_features = {
    'Salinity (ppt)': 1,
    'Dissolved Oxygen (mg/L)': 11.7,
    'Secchi Depth (m)': 0.4,
    'Water Depth (m)': 0.4,
    'Water Temp (C)': 5.9,
    'AirTemp (C)': 8
}

predicted_ph = predict_ph_level(input_features)
print(f"\nPredicted pH level: {predicted_ph:.2f}")