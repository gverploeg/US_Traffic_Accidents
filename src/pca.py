import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca(df, x_features):
    '''
    Principal component analysis to better understand the correlation between the data columns.
    Used data that was already normalized through min max scaler
    ARGS: 
        df - dataframe
        x_features - dataframe features to include
    RETURNS
        new dataframe and prints important features
    '''
    X = df[x_features]
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
    print('Top 5 most important questions: ', np.array(features)[top_5])
    return pca_df

def two_dim_pca(df, x_features, save_loc):
    '''
    Plots 2D PCA factors
    ARGS: 
        df - dataframe
        save_loc - location to save plot
    RETURNS
        new dataframe
    '''
    final_df = pca(df, x_features)
    not_severe_df = final_df[final_df['Severity'] == 0]
    is_sever_df = final_df[final_df['Severity'] == 1]
    plot1 = is_sever_df[['principal component 1', 'principal component 2']]
    plot2 = not_severe_df[['principal component 1', 'principal component 2']]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('Two components PCA', fontsize=20)
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    targets = ['Severe', 'Non-Severe']
    ax.scatter(plot2['principal component 1'], plot2['principal component 2'], color='b')
    ax.scatter(plot1['principal component 1'], plot1['principal component 2'], color='r')
    ax.legend(targets)
    ax.grid()
    plt.savefig(save_loc, dpi=150, bbox_inches = 'tight')

def three_dim_pca(df, x_features, target_var, save_loc):
    '''
    Plots 3D PCA factors
    ARGS: 
        df - dataframe
        x_features - dataframe features to include
        target_var - target variable
    RETURNS
        new dataframe
    '''
    X = df[[x_features]]
    y = df[target_var].values

    fig = plt.figure(figsize=(10,12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('PCA with 3 Components', fontsize = 24)
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

if __name__ == '__main__':

    # Read csv file into a pandas dataframe
    df_full = pd.read_csv('../data/logistic_data.csv')

    features = ['Temperature(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
       'Precipitation(in)', 
       'Junction', 
       'Traffic_Signal', 'Civil_Twilight', 'Rush Hour', 'Weekend', 'Region_Midwest',
       'Region_Northeast', 'Region_Pacific', 'Region_Rockies',
       'Region_Southwest', 'Side_R', 'Season_Spring', 'Season_Summer',
       'Season_Winter', 'Weather_Condition_Clear', 'Weather_Condition_Fog',
       'Weather_Condition_Other', 'Weather_Condition_Rain',
       'Weather_Condition_Snow', 'Weather_Condition_Thunderstorm']

    feature_df = pca(df_full, features)

    two_dim_pca(df_full, features, '../images/two_pca_plot.png')

    three_pca = PCA(n_components=3)
    X_pca = three_pca.fit_transform(df_full[features])
    three_dim_pca(df_full, features, 'Severity', '../images/two_pca_plot.png')

    
 