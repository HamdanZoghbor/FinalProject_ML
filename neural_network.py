import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}

# Load the dataset
file_path = "/Users/hamdanzoghbor/Desktop/Machine Learning/BKB_WaterQualityData_2020084.csv"
data = pd.read_csv(file_path)

# Data Cleaning and same processing
columns_to_drop = ['Unit_Id', 'Air Temp-Celsius', 'DateVerified', 'WhoVerified', 'Air Temp (?F)']
data_cleaned = data.drop(columns=columns_to_drop)
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].median())

# Convert pH levels to categorical classes and binarize for multi-class AUROC
data_cleaned['pH Category'] = pd.cut(data_cleaned['pH (standard units)'],
                                     bins=[0, 6.5, 7.5, 14],
                                     labels=['Low', 'Medium', 'High'])
labels = ['Low', 'Medium', 'High']
y_binarized = label_binarize(data_cleaned['pH Category'], classes=labels)

# Data Preparation
features = data_cleaned[['Salinity (ppt)', 'Dissolved Oxygen (mg/L)', 'Secchi Depth (m)',
                         'Water Depth (m)', 'Water Temp (?C)', 'AirTemp (C)']]
target = data_cleaned['pH Category']

# Splitting the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp, y_train_bin, y_temp_bin = train_test_split(features, target, y_binarized, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test, y_val_bin, y_test_bin = train_test_split(X_temp, y_temp, y_temp_bin, test_size=0.5, random_state=42)

# Define different hidden layer sizes, regularization strengths, and polynomial degrees for neural networks
hidden_layer_sizes = [(100, 50, 100)]
alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10]
degrees = [1, 2, 3]

# Results dictionary to store results for neural networks with regularization and polynomial features
results_nn_reg_poly = []

# Create a model for each combination of hidden layer size, regularization strength, and polynomial degree
for hidden_size in hidden_layer_sizes:
    for alpha in alphas:
        for degree in degrees:
            poly = PolynomialFeatures(degree=degree)
            scaler = StandardScaler()
            model_nn = MLPClassifier(hidden_layer_sizes=hidden_size, alpha=alpha, max_iter=1000, random_state=42)
            pipeline = Pipeline([
                ('polynomial_features', poly),
                ('standard_scaler', scaler),
                ('neural_network', model_nn)
            ])
            pipeline.fit(X_train, y_train)
            y_train_pred = pipeline.predict(X_train)
            y_val_pred = pipeline.predict(X_val)
            y_val_proba = pipeline.predict_proba(X_val)

            # Store the weights for each layer separately
            layer_weights = []
            if isinstance(hidden_size, int):
                layer_weights.append(pipeline['neural_network'].coefs_[0].flatten())
            else:
                for i in range(len(hidden_size)):
                    layer_weights.append(pipeline['neural_network'].coefs_[i].flatten())

            results_nn_reg_poly.append({
                'Hidden Layer Size': hidden_size,
                'Alpha': alpha,
                'Degree': degree,
                'Training Accuracy': accuracy_score(y_train, y_train_pred),
                'Validation Accuracy': accuracy_score(y_val, y_val_pred),
                'Training Precision': precision_score(y_train, y_train_pred, average='macro'),
                'Validation Precision': precision_score(y_val, y_val_pred, average='macro'),
                'Training Recall': recall_score(y_train, y_train_pred, average='macro'),
                'Validation Recall': recall_score(y_val, y_val_pred, average='macro'),
                'Training F1 Score': f1_score(y_train, y_train_pred, average='macro'),
                'Validation F1 Score': f1_score(y_val, y_val_pred, average='macro'),
                'Validation AUROC': roc_auc_score(y_val_bin, y_val_proba, multi_class='ovr', average='macro'),
                'Layer Weights': layer_weights
            })

# Convert results to DataFrame for better visualization
results_table_nn_reg_poly = pd.DataFrame(results_nn_reg_poly)
print("Neural Network Results with Regularization and Polynomial Features:")
print(results_table_nn_reg_poly)

# Plot the impact of regularization and polynomial degree on layer weights for each layer separately
num_layers = 1 if isinstance(hidden_size, int) else len(hidden_size)
fig, axes = plt.subplots(len(degrees), num_layers, figsize=(15, 15))
fig.suptitle('Layer Weights vs. Regularization Strength and Polynomial Degree')

for i, degree in enumerate(degrees):
    for layer in range(num_layers):
        ax = axes[i, layer] if num_layers > 1 else axes[i]
        for alpha in alphas:
            weights = results_table_nn_reg_poly[(results_table_nn_reg_poly['Alpha'] == alpha) & (results_table_nn_reg_poly['Degree'] == degree)]['Layer Weights'].values[0][layer]
            ax.plot(weights, marker='o', label=f'Alpha: {alpha}')
        ax.set_xlabel(f'Layer {layer+1} Weight Index')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Degree {degree} - Layer {layer+1} Weights')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.show()