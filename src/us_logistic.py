import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble 
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

if __name__ == '__main__':
    # Read csv file into a pandas dataframe
    df = pd.read_csv('../data/nonlinear_data.csv')

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
    print(logit_mod(full_features, df))

