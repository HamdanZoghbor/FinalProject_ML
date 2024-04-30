import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
file_path = "BKB_WaterQualityData_2020084.csv"
data = pd.read_csv(file_path)

# Data Cleaning
columns_to_drop = ['Unit_Id', 'Site_Id', 'Read_Date', 'Time (24:00)', 'Field_Tech', 'Year', 'Air Temp-Celsius', 'DateVerified', 'WhoVerified', 'Air Temp (?F)']
#data_cleaned = data.drop(columns=columns_to_drop).sample(int(len(data)/16))
data_cleaned = data.drop(columns=columns_to_drop)
columns = list(data_cleaned.head(0))
numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
data_cleaned[numeric_columns] = data_cleaned[numeric_columns].fillna(data_cleaned[numeric_columns].median())
#Data Preparation
features = data_cleaned[['Salinity (ppt)', 'Dissolved Oxygen (mg/L)', 'Secchi Depth (m)', 
                         'Water Depth (m)', 'Water Temp (?C)', 'AirTemp (C)']]
target = data_cleaned['pH (standard units)']




for i in range(0, len(columns)-1):
    for j in range (i+1, len(columns)):
        feature_to_plot = data_cleaned[[columns[i], columns[j], 'pH (standard units)']].to_numpy()
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(feature_to_plot)
        centroids = kmeans.predict(feature_to_plot)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(feature_to_plot[:, 0], feature_to_plot[:, 1], feature_to_plot[:, 2], c=centroids, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)
        #plt.scatter(feature_to_plot[:, 0], feature_to_plot[:, 1], s=50)
        ax.set_xlabel(columns[i])
        ax.set_ylabel(columns[j])
        ax.set_zlabel('pH (standard units)')
plt.show()       


