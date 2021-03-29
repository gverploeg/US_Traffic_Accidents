import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn.tree import plot_tree
from sklearn import tree
from xgboost import XGBClassifier

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

def random_forest_grid(features, df, param_dict):
    """Using CV to find the most optimal hyperparameters for random forest model
    Args:
        features - list of chosen features to include
        df - DataFrame
        param_dict - parameter grid to sample from during fitting
    Returns:
        Score and list of best hyperparameters
    """   
    X= features
    y = df['Severity'].values
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_dict, n_iter = 70, cv = 5, scoring='f1',
                               verbose=2, random_state=42, n_jobs = -1)
    result = rf_random.fit(X, y)
    return result.best_score_, result.best_params_

def feature_importance(features, df):
    """Feature Importances of Random Forest Model
    Args:
        features - list of chosen features to include
        df - dataframe
    Returns:
        feature importance plot
    """
    feature_name = features.columns.values
    X= features
    y = df['Severity'].values

    model = RandomForestClassifier(n_estimators = 300, max_features='sqrt', max_depth=60, random_state=42,
                                    min_samples_leaf = 2, bootstrap = True, min_samples_split=5)
    model.fit(X, y)
    results = permutation_importance(model, X, y, scoring='f1')
    importance = results.importances_mean
    std = results.importances_std
    indices = np.argsort(importance)[::-1][:20]
    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importance[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), feature_name[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()
    
def partial_dependence(features, df, feat):
    """Partial Dependence of Random Forest Model
    Args:
        features - list of chosen features to include
        df - dataframe
        feat - feature(s) to input for dependence
    Returns:
        Partial dependence plot
    """
    plt.rcParams['figure.figsize'] = 16, 9
    X= features
    y = df['Severity'].values

    model = RandomForestClassifier(n_estimators = 300, max_features='sqrt', max_depth=60, random_state=42,
                                    min_samples_leaf = 2, bootstrap = True, min_samples_split=5)
    model.fit(X, y)
    plot_partial_dependence(model, X, feat, line_kw={"c": "m"})
    plt.show()

def decision_tree(features, df):
    """Plot Decision Tree
    Args:
        features - list of chosen features to include
        df - dataframe
    Returns:
        decision tree
    """
    X= features
    y = df['Severity']
    clf = DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=2, max_depth=4, 
                             criterion = 'gini', random_state=42)
    clf.fit(X, y)
    plt.figure(figsize=(25,10))
    a = tree.plot_tree(clf, 
              feature_names=X.columns.to_list(), 
              filled=True, 
              rounded=True, 
              fontsize=14)
    plt.show()

def xgboost_model(features, df):
    """Generate eXtreme Gradient Boosting model
    Args:
        features - list of chosen features to include
        df - dataframe
    Returns:
        Classification report with accuracy, precision and recall scores
    """
    X= features
    y = df['Severity'].values

    xg_model = XGBClassifier(subsample= .7, reg_lambda = 5, n_estimators=900, min_child_weight=1, max_depth=20,
                        learning_rate=.01, gamma = .5, colsample_bytree = .6, colsample_bylevel=.7)
    xg_model.fit(X, y)
    y_pred = xg_model.predict(X)
    
    return classification_report(y, y_pred, target_names=['Non-Severe', 'Severe'])



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

    # Define a grid of hyperparameter ranges and randomly sample from the grid
    small_df = df.sample(frac =.05)
    reduced_features = small_df[['Temperature(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)', 
       'Junction', 
       'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend', 'Region_Midwest',
       'Region_Northeast', 'Region_Pacific', 'Region_Rockies',
       'Region_Southwest', 'Side_R', 'Season_Spring', 'Season_Summer',
       'Season_Winter', 'Weather_Condition_Clear', 'Weather_Condition_Fog',
       'Weather_Condition_Other', 'Weather_Condition_Rain',
       'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm']]

    n_estimators = [50, 100, 200, 400, 500, 700]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    param_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
    random_forest_grid(reduced_features, small_df, param_grid)

    # Display Feature Importance
    feature_importance(reduced_features, small_df)

    # Display Partial Dependence Plot
    partial_dependence(reduced_features, small_df, feat=['Temperature(F)', 'Humidity(%)', 'Pressure(in)'])

    # Display Decision Tree
    decision_tree(full_features, df)

    # Display Classification Report for XGBoost
    xgboost_model(full_features, df)

    