import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.inspection import plot_partial_dependence
from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier

def feature_importance(df):
    """Plotted overlap between Random Forest and Decision Tree for Rockies
    Args:
        df - dataframe
    Returns:
        Plot of Feature Importance
    """
    column_list = df[['Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
       'Wind_Speed(mph)', 'Precipitation(in)', 'Bump', 'Crossing', 'Junction',
       'Railway', 'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend',
       'Weather_Condition_Fog', 'Weather_Condition_Other',
       'Weather_Condition_Rain', 'Weather_Condition_Snow',
       'Weather_Condition_Thunderstorm', 'Season_Spring',
       'Season_Summer', 'Season_Winter', 'Side_R']]
    column_names = column_list.rename(columns={'Temperature(F)': 'Temperature(F)', 'Humidity(%)': 'Humidity(%)', 
                        'Civil_Twilight': 'During Day',
                        'Rush Hour': 'During Rush Hour',
                         'Wind_Speed(mph)': 'Wind Speed(mph)', 'Traffic_Signal': "Traffic Signal",
                        'Side_R': "Right Side of Road", 'Wind_Speed(mph)': 'Wind Speed(mph)', 
                        'Season_Spring':'Season Spring', 'Season_Summer': 'Season Summer',
                        'Season_Winter': 'Season Winter'
                        })

    feature_name = column_names.columns.values 
    X= column_list
    y = df['Severity'].values
    model = RandomForestClassifier(n_estimators = 50, max_features='sqrt', max_depth=70, random_state=42,
                                 min_samples_leaf = 4, bootstrap = True, min_samples_split=5)
    model.fit(X, y)
    results = permutation_importance(model, X, y, scoring='f1')
    #Plot Imp 
    importance = results.importances_mean
    std = results.importances_std
    indices = np.argsort(importance)[::-1][:12]
    plt.figure(figsize=(10,7))
    plt.title("Random Forest Feature Importances", fontsize=20)
    patches = plt.bar(range(len(indices)), importance[indices], color="dimgray", yerr=std[indices], align="center")
    patches[0].set_fc('r')
    patches[1].set_fc('r')
    patches[3].set_fc('r')
    patches[4].set_fc('r')
    patches[5].set_fc('r')
    patches[6].set_fc('r')
    patches[7].set_fc('r')
    plt.xticks(range(len(indices)), feature_name[indices], rotation='45', fontsize=16, horizontalalignment='right')
    plt.xlim([-1, len(indices)])
    plt.ylabel("Importance", fontsize=17)
    plt.yticks(fontsize=16)
    plt.savefig('../Images/rf_featureimportance.png', transparent=False, bbox_inches='tight', format='png', dpi=200)
    plt.show()

def partial_dep(df, feat, label, save_as):
    """Partial Dependence of Random Forest Model
    Args:
        df - dataframe
        feat - feature to plot
        label - feature to label in plot
        save_as - name to save image under
    Returns:
        Partial dependence plot
    """
    column_names = df[['Temperature(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)', 
       'Junction', 
       'Traffic_Calming', 'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend', 
                         'Side_R', 'Season_Spring', 'Season_Summer',
       'Season_Winter', 'Weather_Condition_Clear', 'Weather_Condition_Fog',
       'Weather_Condition_Other', 'Weather_Condition_Rain',
       'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm']]
    X= column_names
    y = df['Severity'].values
    model = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=70, random_state=42,
                                 min_samples_leaf = 4, bootstrap = True, min_samples_split=10)
    model.fit(X, y)

    fig, ax = plt.subplots(figsize=(9, 6))
    mlp_disp = plot_partial_dependence(model, column_names, [feat], ax=ax, line_kw={"color": "red"})
    plt.ylabel("Partial Dependence", fontsize=21)
    plt.xlabel(label,fontsize=21)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.savefig(save_as, transparent=False, bbox_inches='tight', format='png', dpi=200)
    plt.show()

def decision_tree(df):
    """Plot Rockies Decision Tree
    Args:
        df - dataframe
    Returns:
        Decision tree visualization
    """
    features = df[['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)', 'Amenity', 'Bump', 'Crossing', 'Give_Way',
       'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
       'Traffic_Calming', 'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend',
                     'Side_R', 'Season_Spring', 'Season_Summer',
       'Season_Winter', 'Weather_Condition_Clear', 'Weather_Condition_Fog',
       'Weather_Condition_Other', 'Weather_Condition_Rain',
       'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm']]
    X= features
    y = df['Severity']
    clf = DecisionTreeClassifier(min_samples_split=6, min_samples_leaf=2, max_depth=3, 
                                criterion = 'gini', random_state=42)
    clf.fit(X, y)

    plt.figure(figsize=(25,10))
    a = plot_tree(clf, 
                feature_names=X.columns.to_list(), 
                filled=True, 
                rounded=True, 
                fontsize=14)
    plt.savefig("rockies_decision_tree.png")
    plt.show()

if __name__ == '__main__':
    # Read csv file into a pandas dataframe
    rockies = pd.read_csv('../data/nonlinear_rockies_data.csv')

    feature_importance(rockies)
    partial_dep(rockies, 'Humidity(%)', 'Humidity(%)', '../Images/partial_humid.png')
    partial_dep(rockies, 'Wind_Speed(mph)', 'Wind Speed(mph)', '../Images/partial_wind.png')
    decision_tree(rockies)
