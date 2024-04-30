import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ValidationCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tabulate import tabulate
import time

# Load the dataset
file_path = "BKB_WaterQualityData_2020084.csv"
data = pd.read_csv(file_path)

# Data Cleaning
# Even though for further model training only two columns will be required, we still leave 
# other columns intact in case we need them in the future
columns_to_drop = ['Unit_Id', 'Site_Id', 'Read_Date', 'Time (24:00)', 'Field_Tech', 'Year', 
                   'Air Temp-Celsius', 'DateVerified', 'AirTemp (C)', 'WhoVerified', 'Air Temp (?F)']
data_cleaned = data.drop(columns=columns_to_drop)
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].median())

# Data Preparation
features = data_cleaned[['Secchi Depth (m)', 'Water Depth (m)']]
target = data_cleaned['pH (standard units)']

# Label Enconding
# The task is to identify how safe a water sample is to drink based on its pH value. It is recommended to drink
# water that has pH value in the range from 6.5 to 8.5. The values below and above that bound are considered
# unsafe to drink and may cause different unwanted effects. Since the effect of drinking water that has pH below
# and above the standard level differs, we distinguish between the two.
# Label encoding is used instead of one-hot encoding because there is a natural ordering in pH levels,
# and we want to preserve it
target.loc[target <= 6.5] = 1
target.loc[(target > 6.5) & (target <= 8.5)] = 2
target.loc[target > 8.5] = 3

# The range distribution was as follows: [0, 6.5] - 755 samples, (6.5, 8.5] - 1472 samples, and (8.5, above) - 144 samples
# After the initial training of the model, by insepcting the confusion matrix we noticed that imbalance in the number of samples
# resulted in a very low recall for category 3. To improve the results, it was decided to randomly sample the set to equalize the
# number of samples belonging to each group

# Random under-sampling 
# n_group_1_to_drop = len(target[target == 1]) - len(target[target == 3]) 
# n_group_2_to_drop = len(target[target == 2]) - len(target[target == 3]) 
# indices_to_drop_1 = target[target == 1].sample(n_group_1_to_drop).index
# indices_to_drop_2 = target[target == 2].sample(n_group_2_to_drop).index
# features = features.drop(indices_to_drop_1)
# features = features.drop(indices_to_drop_2)
# target = target.drop(indices_to_drop_1)
# target = target.drop(indices_to_drop_2)

# Random over-sampling
ros = RandomOverSampler(random_state=0, shrinkage=0.3)
features, target = ros.fit_resample(features, target)

# Splitting the data into training, validation, and testing sets
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.4, random_state=42)


#---------------------------------Section 1------------------------------------
# To get an initial understanding of what model parameters would suit best for the given problem,
# we decided to perform a grid search
# grid = {'C':np.logspace(-4, 4, 50), 'penalty':['l1', 'l2'], 'solver':['newton-cg', 'sag', 'saga', 'lbfgs']}
# logistic = LogisticRegression()
# logistic_grid = GridSearchCV(logistic, grid)
# logistic_grid.fit(X_train,y_train)
# print("The best parameters ", logistic_grid.best_params_)
# print("Accuracy :", logistic_grid.best_score_)
# time.sleep(5000)

#---------------------------------Section 2------------------------------------
# To see how weights change as we iterate over parameter C
# A table to store training performance metrics
# lambdas = np.logspace(-5,5,25)
# logistic_coef = []
# for l in lambdas:
#     logistic = LogisticRegression(penalty ='l2', multi_class = 'multinomial', solver = 'newton-cg', C=1/l)
#     logistic.fit(X_train, y_train)
#     logistic_coef.append(logistic.coef_)

# # Printing the weights for each class at every iteration
# for index, coef in enumerate(logistic_coef):
#     print(f"Lambda: {lambdas[index]} - Coefficients:")
#     print(coef)
# colors = ['blue', 'green', 'red']
# for class_index in range(logistic.coef_.shape[0]):
#     plt.plot(lambdas, [coef[class_index] for coef in logistic_coef], label=f'Class {class_index+1}', color=colors[class_index])
# plt.xscale("log")
# plt.gca().invert_xaxis()  # reverse axis
# plt.xlabel("C")
# plt.ylabel("Weights")
# plt.title("Weights for 2 features of each class vs C")
# plt.legend()
# plt.show()


#---------------------------------Section 3------------------------------------
# To see how validation curve changes as we iterate through different values for C and different polynomial
# transformation degrees
# Define lambda for regularization
# lambdas = np.logspace(-5, 5, 25)
# polynomial_transform = PolynomialFeatures(degree = 3)
# features_poly = polynomial_transform.fit_transform(features)
# display = ValidationCurveDisplay.from_estimator(
#    LogisticRegression(penalty="l2", solver="newton-cg"), features_poly, target, param_name="C", param_range=lambdas
# )
# display.plot()
# plt.show()
# time.sleep(5000)

#---------------------------------Section 4------------------------------------
#Create a modelt for each combination of lambda and degree of polynomial features
results_logistic_test = []
results_logistic_validation = []
lambdas = np.logspace(-4, 4, 10)
degrees = [1, 2, 3]
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    scaler = StandardScaler() ##read into the instaiatin
    for l in lambdas:
        # Multi_class is set to be “multinomial”, so that the softmax function 
        # is used to find the predicted probability of each class. 
        logistic = LogisticRegression(penalty ='l2', multi_class = 'multinomial', solver = 'newton-cg', C=l)
        pipeline = Pipeline([
           ('polynomial_features', poly),##understadn the defintions of trasnformers
            ('standard_scaler', scaler),
            ('logistic', logistic),
        ])
        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        y_pred_val = pipeline.predict(X_val)
        #print(accuracy_score(y_test, predictions))
        #cm = confusion_matrix(y_val, y_pred, labels=pipeline.classes_)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
        #disp.plot()
        results_logistic_test.append({
            'Degree': degree,
            'Lambda': l,
            'Test accuracy': accuracy_score(y_train, y_pred_train),
            'Test precision': precision_score(y_train, y_pred_train, average = 'macro', zero_division=np.nan),
            'Test recall': recall_score(y_train, y_pred_train, average = 'macro')
        })
        results_logistic_validation.append({
            'Degree': degree,
            'Lambda': l,
            'Validation accuracy': accuracy_score(y_val, y_pred_val),
            'Validation precision': precision_score(y_val, y_pred_val, average = 'macro', zero_division=np.nan),
            'Validation recall': recall_score(y_val, y_pred_val, average = 'macro')
        })
results_table1 = pd.DataFrame(results_logistic_test)
print(tabulate(results_table1, headers = 'keys', tablefmt = 'latex'))       
results_table2 = pd.DataFrame(results_logistic_validation)
print(tabulate(results_table2, headers = 'keys', tablefmt = 'latex'))
#plt.show()
