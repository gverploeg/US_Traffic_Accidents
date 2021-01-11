import pandas as pd
import numpy as np

def show_missing_data(df):
    '''
    Shows percent of missing data for each column
    
    ARGS:
        df - dataframe
    RETURNS:
        dataframe with missing column percentages
    '''
    missing_data = pd.DataFrame(df.isnull().sum()).reset_index()
    missing_data.columns = ['Feature', 'Missing_Percent(%)']
    missing_data['Missing_Percent(%)'] = missing_data['Missing_Percent(%)'].apply(lambda x: x / df.shape[0] * 100)
    return missing_data.loc[missing_data['Missing_Percent(%)']>0,:]

def display_row_nulls(df):
    '''
    Shows the rows with null values 
    
    ARGS: 
        df - dataframe
    RETURNS
        dataframe that has rows with null vals
    '''
    is_NaN = df.isnull()
    row_has_Nan = is_NaN.any(axis = 1)
    return df[row_has_Nan].head(15)

def drop_col(df, col):
    '''
    Drops null or insignificant data

    ARGS: 
        df - dataframe
        col - column to be dropped
    RETURNS
        dataframe with removed null column
    '''
    return df.drop([col], axis=1, inplace= True)

def drop_duplicates(df):
    '''
    Drops duplicates

    ARGS: 
        df - dataframe
    RETURNS
        dataframe with duplicates dropped
    '''
    df.drop_duplicates(inplace=True)
    return df

def remove_null_rows(df, col):
    '''
    Takes the rows where the rows are not NAN

    ARGS: 
        df - dataframe
        col - column to be dropped
    RETURNS
        dataframe with removed null rows
    '''
    df = df[df[col].notna()]
    return df

def combine_weather_conditions(df, col, weath_items, new_weather):
    '''
    There are 127 unique columns in weather - lets merge similar conditions

    ARGS: 
        df - dataframe
        col - column 
        weath_items - weather conditions to be combined
        new_weather - new category name of merged weather
    RETURNS
        dataframe with combined weather conditions
    '''
    df.loc[df[col].str.contains(weath_items), col] = new_weather
    return df


def create_other_weather(df, col):
    '''
    Create other category for remaining weather conditions based on frequency threshold

    ARGS: 
        df - dataframe
        col - weather column 
        new_weather - new category name of merged weather
    RETURNS
        dataframe with other weather condition
    '''
    # Step 1: count the frequencies
    frequencies = df[col].value_counts(normalize = True)
    frequencies

    # Step 2: establish your threshold and filter the smaller categories
    threshold = 0.004
    small_categories = frequencies[frequencies < threshold].index
    small_categories

    # Step 3: replace the values
    df[col] = df[col].replace(small_categories, "Other")
    df[col].value_counts(normalize = True)
    return df

def replace_nulls(df, feature, condition):
    '''
    Replace nulls depending on condition

    ARGS: 
        df - dataframe
        feature - column to be replaced
        condition - replace with average of column or 0
    RETURNS
        dataframe with replaced nans
    '''
    if condition == 'zero':
        df[feature] = df[feature].fillna(0)
    else:
        df[feature] = df[feature].fillna((df[feature].mean()))
    return df

def drop_values(df, feature, replacement):
    '''
    Drop certain values from column
    For example, the 0 values from Pressure & Visibility - NANs could have been inserted as 0

    ARGS: 
        df - dataframe
        feature - column to be dropped
    RETURNS
        dataframe with dropped values
    '''
    df = df[df[feature] != replacement]
    return df

def day_night_encoding(df, feature):
    '''
    Replace Day and night with 1 and 0

    ARGS: 
        df - dataframe
        feature - column 
    RETURNS
        dataframe with updated feature
    '''
    df[feature].replace({'Night':0, 'Day':1}, inplace=True)
    return df

def to_bool(df, feature):
    '''
    Replace True and False columns with int values

    ARGS: 
        df - dataframe
        feature - column 
    RETURNS
        dataframe with updated feature
    '''
    df[feature] = df[feature].astype(int)
    return df

def one_hot_encoding(df, categorical_feature):
    '''
    Converts categorical variables to numerical in an interpretable format
    ARGS: 
        df - dataframe
        categorical_feature - categorical column
    RETURNS
        dataframe with added features
    '''
    dummies = pd.get_dummies(df[[categorical_feature]])
    new_df = pd.concat([df, dummies], axis=1)
    return new_df

if __name__ == '__main__':

    # Read csv file into a pandas dataframe
    df_full = pd.read_csv('../data/US_Accidents_June20.csv')

    # Dealing with nulls
    df_full = drop_duplicates(df_full)
    df_full = remove_null_rows(df_full, 'Weather_Condition')
    df_full = remove_null_rows(df_full, 'Civil_Twilight')
    
    df_full = replace_nulls(df_full, 'Precipitation(in)', 'zero')
    list_of_features = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
                    'Wind_Speed(mph)', 'Distance(mi)']
    for i in list_of_features:
        replace_nulls(df_full, i, 'mean')

    df_full = drop_values(df_full, "Pressure(in)", 0)
    df_full = drop_values(df_full, "Visibility(mi)", 0)
    df_full = drop_values(df_full, "Side", " ")

    # Combine weather
    df_full = combine_weather_conditions(df_full, 'Weather_Condition', 'Cloud|Overcast', 'Cloudy')
    df_full = combine_weather_conditions(df_full, 'Weather_Condition', 'Clear|Fair', 'Clear')
    df_full = combine_weather_conditions(df_full, 'Weather_Condition', 'Rain|Drizzle', 'Rain')
    df_full = combine_weather_conditions(df_full, 'Weather_Condition', 'Snow|Sleet|Wintry', 'Snow')
    df_full = combine_weather_conditions(df_full, 'Weather_Condition', 'Mist|Haze|Fog', 'Fog')
    df_full = combine_weather_conditions(df_full, 'Weather_Condition', "Thunder|T-Storm", 'Thunderstorm')

    df_full = create_other_weather(df_full, "Weather_Condition")

    # Convert Data to multiple columns
    df_full["Start_Time"] = pd.to_datetime(df_full["Start_Time"])

        # Extract year, month, weekday, day, and hour
    df_full["Year"] = df_full["Start_Time"].dt.year
    df_full["Month"] = df_full["Start_Time"].dt.month
    df_full["Weekday"] = df_full["Start_Time"].dt.weekday
    df_full["Day"] = df_full["Start_Time"].dt.day
    df_full["Hour"] = df_full["Start_Time"].dt.hour

    # Feature Engineering
        # Create Binary Rush Hour Column
    rush = []
    for row in df_full['Hour']:
        if row >= 0 and row <=6:        rush.append(0)
        elif row >= 7 and row <= 10:    rush.append(1)
        elif row > 10 and row <=15:     rush.append(0)
        elif row >15 and row <= 19:     rush.append(1)
        else: rush.append(0)
    df_full['Rush Hour'] = rush

        # Create Binary Weekend column
    weekend_or_not = []
    for row in df_full['Weekday']:
        if row == 5 or row ==6:         weekend_or_not.append(1)
        else:                           weekend_or_not.append(0)
    df_full['Weekend'] = weekend_or_not
    
        # Create Binary Season Column
    df_full['Season'] = df_full["Month"].replace({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer',
                                        7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'})

    # Convert Boolean Values to Ints
    list_of_bool_feats = ['Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for i in list_of_bool_feats:
        to_bool(df_full, i)

    # Create Groupings by State Region
    Pacific = ['OR', 'WA', 'CA']
    Rockies = ['NV', 'UT', 'CO', 'WY', 'ID', 'MT']
    Southwest = ['AZ', 'NM', 'TX', 'OK']
    Midwest = ['ND', 'SD', 'NE', 'KS', 'MO', 'IA', 'MN', 'WI', 'IL', 'IN', 'OH', 'MI']
    Southeast = ['AR', 'LA', 'MS', 'AL', 'GA', 'FL', 'SC', 'NC', 'VA', 'WV', 'KY', 'TN', 'MD', 'DC', 'DE']
    Northeast = ['PA', 'NJ', 'NY', 'CT', 'RI', 'MA', 'NH', 'VT', 'ME']

    df_full['Region'] = df_full['State']

    df_full['Region'] = df_full['Region'].replace([Pacific],'Pacific')
    df_full['Region'] = df_full['Region'].replace(Rockies,'Rockies')
    df_full['Region'] = df_full['Region'].replace(Southwest,'Southwest')
    df_full['Region'] = df_full['Region'].replace(Midwest,'Midwest')
    df_full['Region'] = df_full['Region'].replace(Southeast,'Southeast')
    df_full['Region'] = df_full['Region'].replace(Northeast,'Northeast')

    # Features to Drop
    features_to_drop = ["Source", "TMC", "End_Lat", "End_Lng", "Number", "Street", "Airport_Code", 
                    "Weather_Timestamp", "Wind_Chill(F)", "Turning_Loop", "Sunrise_Sunset",
                    "Nautical_Twilight", "Astronomical_Twilight"]
    df_full = df_full.drop(features_to_drop, axis=1)

    # One_Hot Features
    '''
    Dropped the original feature and dropped one of new data frames
    ''' 
    df_full = one_hot_encoding(df_full,'Road_Type')
    df_full.drop(['Road_Type'],axis=1,inplace=True)
    df_full.drop(['Road_Type_Single carriageway'],axis=1,inplace=True)

    df_full = one_hot_encoding(df_full,'Road_Surface_Conditions')
    df_full.drop(['Road_Surface_Conditions'],axis=1,inplace=True)
    df_full.drop(['Road_Surface_Conditions_Dry'],axis=1,inplace=True)

    df_full = one_hot_encoding(df_full,'Pedestrian_Crossing-Physical_Facilities')
    df_full.drop(['Pedestrian_Crossing-Physical_Facilities'],axis=1,inplace=True)
    df_full.drop(['Pedestrian_Crossing-Physical_Facilities_No physical crossing within 50 meters'],axis=1,inplace=True)

    df_full = one_hot_encoding(df_full,'Light_Conditions')
    df_full.drop(['Light_Conditions'],axis=1,inplace=True)
    df_full.drop(['Light_Conditions_Daylight: Street light present'],axis=1,inplace=True)

    df_full = one_hot_encoding(df_full,'Weather_Conditions')
    df_full.drop(['Weather_Conditions'],axis=1,inplace=True)
    df_full.drop(['Weather_Conditions_Fine without high winds'],axis=1,inplace=True)


    # Save as CSV
    # df_full.to_csv('../data/Cleaned_data.csv')