import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.discrete.discrete_model import Logit
import statsmodels.api as sm

def logit_mod(features, df):
    '''
    Classification metrics
    ARGS: 
        features - features
        df - dataframe
    RETURNS
        Metrics
    '''
    X = features
    y = df['Severity'].values

    logreg = LogisticRegression()
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    return classification_report(y, y_pred, target_names=['Non-Severe', 'Severe'])

def vif(features, df):
    '''
    Determine multicollinearity with variance inflation factors
    ARGS: 
        features - features
        df - dataframe
    RETURNS
        List of features ordered by VIF
    '''
    vif_data = pd.DataFrame() 
    vif_data["Feature"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i)
        for i in range(len(features.columns))]
    return vif_data

def accuracy(features, df):
    '''
    Determine accuracy score
    ARGS: 
        features - features
        df - dataframe
    RETURNS
        Metric Score
    '''
    X= features
    y = df['Severity'].values
    logreg = LogisticRegression(max_iter=150)
    logreg.fit(X, y)
    y_pred = logreg.predict(X)
    return accuracy_score(y, y_pred)

if __name__ == '__main__':
    # Read csv file into a pandas dataframe
    df = pd.read_csv('../data/logistic_data.csv')

    # Features to input in model
    full_features = df[['Temperature(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
       'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
       'Traffic_Calming', 'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend', 'Region_Midwest',
       'Region_Northeast', 'Region_Pacific', 'Region_Rockies',
       'Region_Southwest', 'Side_R', 'Season_Spring', 'Season_Summer',
       'Season_Winter', 'Weather_Condition_Clear', 'Weather_Condition_Fog',
       'Weather_Condition_Other', 'Weather_Condition_Rain',
       'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm']]

    # Display the main classification metrics 
    logit_mod(full_features, df)

    # Calculating VIF for each feature 
    vif(full_features, df)

    # Display Accuracy Metric
    accuracy(full_features, df)
    