import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_features(df, features):
    '''
    Principal component analysis features
    ARGS: 
        df - dataframe
        features - dataframe features to include
    RETURNS
        Top 5 features
    '''
    X = df[features]
    feature_name = X.columns.values
    pca_ = PCA(n_components=2)
    principal_components = pca_.fit_transform(X)
    principal_df = pd.DataFrame(data=principal_components
                               , columns=['Principal component 1', 'Principal component 2'])
    pca_df = pd.concat([principal_df, df[['Severity']]], axis=1)
    features = []
    for i in feature_name:
        features.append(i)
    top_5 = pca_.components_[0].argsort()[-5:]
    return ('Top 5 most important questions: ', np.array(features)[top_5])

def two_dim_pca(df, features, save_loc):
    '''
    Plots 2D PCA factors
    ARGS: 
        df - dataframe
        features - dataframe features to include
        save_loc - location to save plot
    RETURNS
        2 Dimensional PCA plot
    '''
    X = df[features]
    feature_name = X.columns.values
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X)
    principal_df = pd.DataFrame(data = principal_components
                               , columns=['Principal Component 1', 'Principal Component 2'])
    pca_df = pd.concat([principal_df, df[['Severity']]], axis=1)
    final_df = pca_df
    not_severe_df = final_df[final_df['Severity'] == 0]
    is_sever_df = final_df[final_df['Severity'] == 1]
    plot1 = is_sever_df[['Principal Component 1', 'Principal Component 2']]
    plot2 = not_severe_df[['Principal Component 1', 'Principal Component 2']]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Two components PCA', fontsize=20)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    targets = ['Severe', 'Non-Severe']
    ax.scatter(plot1['Principal Component 1'], plot1['Principal Component 2'], color='r')
    ax.scatter(plot2['Principal Component 1'], plot2['Principal Component 2'], color='b')
    ax.legend(targets)
    ax.grid()
    plt.savefig(save_loc, dpi=150, bbox_inches = 'tight')
    plt.show()

def three_dim_pca(X, y, save_loc):
    '''
    Plots 3D PCA factors
    ARGS: 
        df - dataframe
        x_features - dataframe features to include
        target_var - target variable
    RETURNS
        new dataframe
    '''
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('PCA with 3 Components', fontsize = 22)
    ax.scatter(X[:,0], X[:,1], X[:,2], c=y,cmap='bwr_r')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    plt.savefig(save_loc, dpi=150, bbox_inches = 'tight')
    plt.show()

if __name__ == '__main__':

    # Read csv file into a pandas dataframe
    df_full = pd.read_csv('../data/logistic_data.csv')

    df_features = ['Temperature(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)', 
       'Junction', 
       'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend', 'Region_Midwest',
       'Region_Northeast', 'Region_Pacific', 'Region_Rockies',
       'Region_Southwest', 'Side_R', 'Season_Spring', 'Season_Summer',
       'Season_Winter', 'Weather_Condition_Clear', 'Weather_Condition_Fog',
       'Weather_Condition_Other', 'Weather_Condition_Rain',
       'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm']


    # print(pca_features(df_full, df_features))
    # two_dim_pca(df_full, df_features, '../Images/two_pca_plot.png')

    y = df_full['Severity'].values
    three_pca = PCA(n_components=3)
    X_pca = three_pca.fit_transform(df_full[df_features])
    three_dim_pca(X_pca, y, '../Images/three__pca__plot.png')

    