import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from us_cleaned_data import *

def one_hot_encoding(df, categorical_feature, dropped_features):
    '''
    Converts categorical variables to numerical in an interpretable format
    Drops basecase to deal with collinearity if logistic
    ARGS: 
        df - dataframe
        categorical_feature - feature to create dummies
        dropped_features -  features to drop
    RETURNS
        dataframe with added features
    '''
    dummies = pd.get_dummies(df[[categorical_feature]])
    new_df = pd.concat([df, dummies], axis=1)
    new_df.drop(dropped_features,axis=1,inplace=True)
    return new_df

def balance_data(df, feature):
    '''
    Undersampling from the majority class to balance the data
    ARGS: 
        df - dataframe
        feature - feature to determine sampling 
    RETURNS
        new dataframe with balanced classes
    '''
    non_severe = df[(df[feature] == 0)]
    severe = df[(df[feature] == 1)]
    non_severe_samp = non_severe.sample(n=len(severe))
    df_model = pd.concat([severe, non_severe_samp], axis=0)
    return df_model

def normalize(df, features):
    '''
    Normalizes the data through min max scaler
    ARGS: 
        df - dataframe
        features - continuous features to normalize 
    RETURNS
        new dataframe with normalized features
    '''
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df


if __name__ == '__main__':

    # Read csv file into a pandas dataframe
    df = pd.read_csv('../data/total_cleaned_data.csv')
    df_log = df.copy()
    df_nonlin  = df.copy()

    # Create Logistic Model Data
    df_l = one_hot_encoding(df_log,'Weather_Condition', ['Weather_Condition', 'Weather_Condition_Cloudy'])
    df_l = one_hot_encoding(df_l,'Season', ['Season', 'Season_Fall'])
    df_l = one_hot_encoding(df_l,'Region', ['Region', 'Region_Southeast'])
    df_l = one_hot_encoding(df_l,'Side', ['Side', 'Side_L'])

        # Create Target Groups
    df_l["Severity"].replace({1:0, 2:0, 3:0, 4:1}, inplace=True)

    list_of_features_norm = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                    'Wind_Speed(mph)', 'Distance(mi)', 'Precipitation(in)']
    normalize(df_l, list_of_features_norm)

    final_log_df = balance_data(df_l, 'Severity')

    # Save as CSV
    final_log_df.to_csv('../data/logistic_data.csv')

    # Create Nonlinear Model Data
    df_nl = one_hot_encoding(df_nonlin,'Weather_Condition', ['Weather_Condition'])
    df_nl = one_hot_encoding(df_nl,'Season', ['Season'])
    df_nl = one_hot_encoding(df_nl,'Region', ['Region'])
    df_nl = one_hot_encoding(df_nl,'Side', ['Side', 'Side_L'])

        # Create Target Groups
    df_nl["Severity"].replace({1:0, 2:0, 3:0, 4:1}, inplace=True)

    final_nonlin_df = balance_data(df_nl, 'Severity')

    # Save as CSV
    final_nonlin_df.to_csv('../data/nonlinear_data.csv')