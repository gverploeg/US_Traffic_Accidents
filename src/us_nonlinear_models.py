import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def default_random_forest(features, df):
    """Generate a default Random Forest
    Args:
        features - list of chosen features to include
        df - dataframe
    Returns:
        Classification report with accuracy, precision and recall scores
    """    
    X= features
    y = df['Severity'].values

    classify = RandomForestClassifier(n_estimators = 100)
    classify.fit(X, y)
    y_pred = classify.predict(X)
    
    return classification_report(y, y_pred, target_names=['Non-Severe', 'Severe'])

def 

def random_forest_grid(features, df):
    """Using CV to find the most optimal hyperparameters for random forest model
    Args:
        features - list of chosen features to include
        df - DataFrame
    Returns:
        Score and list of best hyperparameters
    """   
    rf = RandomForestClassifier(random_state=42, class_weight = 'balanced')
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions = grid_param, n_iter = 500,
                                cv=5, verbose=2, random_state=42, n_jobs=-1)
    y = df.attrition
    X = df.drop('attrition',axis=1)



if __name__ == '__main__':
    # Read csv file into a pandas dataframe
    df = pd.read_csv('../data/nonlinear_data.csv')

    # Features to input in model
    full_features = df[['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
       'Wind_Speed(mph)', 'Precipitation(in)', 'Bump', 'Crossing', 'Junction',
       'Railway', 'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend',
       'Weather_Condition_Clear', 'Weather_Condition_Cloudy',
       'Weather_Condition_Fog', 'Weather_Condition_Other',
       'Weather_Condition_Rain', 'Weather_Condition_Snow',
       'Weather_Condition_Thunderstorm', 'Season_Fall', 'Season_Spring',
       'Season_Summer', 'Season_Winter', 'Region_Midwest', 'Region_Northeast',
       'Region_Pacific', 'Region_Rockies', 'Region_Southeast',
       'Region_Southwest', 'Side_R']]

    

    default_random_forest(full_features, df)


    