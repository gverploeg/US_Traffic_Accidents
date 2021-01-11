
![](images/cover_image.jpg)  

# United Kingdom Traffic Accidents and their Influencing Factors
## Background & Goal:
Road Traffic Accidents (RTA)s are a major cause of death globally, leading to around 1.25 million deaths annually. In the United Kingdom, the UK Department of Transport [(gov.uk)](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) provides detailed statistics about personal injury road accidents, vehicles and casualties involved. Most of the statistics are based on road accidents reported to the police (Stats19). 

The goal of this repository is to help emergency services identify the key elements of severe accidents, helping determine staffing needs of where and when they are most needed. 

## Data:

This data is made up of two datasets that recorded traffic accidents around the UK from 2009-2011 and 2012-2014. There are over 900,000 records and over 30 different features.

The features are broken down into Categorical and Numerical Data. After combining datasets and dealing with nulls, I looked at Accident Severity as my target variable. Initially, the breakdown of Accident_Severity was Fatal, Serious, Slight, so I merged that into a binary severe (1) or not severe / minor (0). In picking Accident Severity as my target, I concluded that I would need to exclude certain features from my analysis that would result from the accident already taking place such as Number of Casualties, Number of Vehicles, and whether Police attended the scene. The table below shows my sorted and filtered dataframe. 

|   | Accident_Index | Police_Force | Longitude | Latitude  | Accident_Severity | Number_of_Vehicles | Number_of_Casualties | Date   | Time  | Road_Type          | Speed_limit | Weather_Conditions      | Pedestrian_Crossing-Physical_Facilities     | Light_Conditions               | Road_Surface_Conditions | Urban_or_Rural_Area | Did_Police_Officer_Attend_Scene_of_Accident |
|---|----------------|--------------|-----------|-----------|-------------------|--------------------|----------------------|--------|-------|--------------------|-------------|-------------------------|---------------------------------------------|--------------------------------|-------------------------|---------------------|---------------------------------------------|
| 0 | 200901BS70001  | 1            | -0.201349 | 51.512273 | 1                 | 2                  | 1                    | 1/1/09 | 15:11 | One way street     | 30          | Fine without high winds | No physical crossing within 50 meters       | Daylight: Street light present | Dry                     | 1                   | Yes                                         |
| 1 | 200901BS70002  | 1            | -0.199248 | 51.514399 | 1                 | 2                  | 11                   | 5/1/09 | 10:59 | Single carriageway | 30          | Fine without high winds | Zebra crossing                              | Daylight: Street light present | Wet/Damp                | 1                   | Yes                                         |
| 2 | 200901BS70003  | 1            | -0.179599 | 51.486668 | 0                 | 2                  | 1                    | 4/1/09 | 14:19 | Single carriageway | 30          | Fine without high winds | No physical crossing within 50 meters       | Daylight: Street light present | Dry                     | 1                   | Yes                                         |
| 3 | 200901BS70004  | 1            | -0.20311  | 51.507804 | 1                 | 2                  | 1                    | 5/1/09 | 8:10  | Single carriageway | 30          | Other                   | Pedestrian phase at traffic signal junction | Daylight: Street light present | Frost/Ice               | 1                   | Yes                                         |


## EDA:
Start and see where outside research and intuition could take me

Performed Feature Engineering with regards to Inferential Regression

Inferential Assumptions:
1. Independence
2. Normality 
3. No multicollinearity: the independent variables are not highly correlated with each other 


![](images/hour_count.png)
![](images/hour_pt.png)

Applied Feature Engineering to the Time column, where the hour was extracted and used to create a binary Rush Hour feature, which in the UK is generally considered to be between 07:00-10:00 and 16:00-19:00
* Appears to show lower proportion during typical busy hours

![](images/day_count.png)
![](images/day_pt.png)

Used Day of Week to create a binary Weekend feature
* Higher proportion during Saturday and Sunday


 Urban             |  Rural
:-------------------------:|:-------------------------:
![](images/urban_map.png)  |  ![](images/rural_map.png)

More accidents in Urban areas, but there is higher proportion of severe accidents in rural areas.
Speed Limit shares a similar trend, that as the limit increase, so does the proportion of Severe Accidents.

## Inferential Logistic Regression

Determined Multicolinearity with Variance Inflation Factor (VIF). As the name suggests, a variance inflation factor (VIF) quantifies how much the variance is inflated. A variance inflation factor exists for each of the predictors in a multiple regression model. A VIF of 1 means that there is no correlation, while VIFs exceeding 10 are signs of serious multicollinearity requiring correction. In the data, I dropped Latitude and Longitude due to their high values, both exceeding 20. 

Due to the abudance of categorical features, there are several columns that need to be one-hot encoded or changed to a binary value in order to utilize there features. By dropping one of the one-hot encoded columns from each categorical feature, we ensure there are no "reference" columns — the remaining columns become linearly independent. These features included: 

* Road Surface: Dry, Wet or damp, Snow, Frost or ice, Flood over 3cm. deep, Oil or diesel
* Road Type: Roundabout, One way street, Dual carriageway, Single carriageway, Slip road
* Weather: Fine no high winds, Raining no high winds, Snowing no high winds, Fine + high winds, Fog or mist
* Light Conditions: Daylight, Darkness - lights lit, Darkness - lights unlit
* Pedestrian Crossing Physical: Zebra, Footbridge or subway, Pedestrian phase at traffic signal junction

Additional Steps:
* Balanced data so Severe and Minor were equal
* Standarized the data in order to be able to compare coefficients
    * Only standarized Speed Limit


### Logit Model for Feature Importance
![](images/coeffs.png)

| Features                                        | Coeff - Log Odds | Coeff - Odds |
|-------------------------------------------------|------------------|----------------------|
| Road Type - Dual Carriageway                    | -0.376           | 0.686                |
| Road Surface Conditions - Frost/Ice             | -0.385           | 0.68                 |
| Light Conditions - Dark with No street lighting | 0.421            | 1.523                |
| Road Type - Roundabout                          | -0.584           | 0.557                |
| Road Type - Slip Road                           | -0.769           | 0.463                |


Pseudo R-squared: 0.020

* Above, the positive scores indicate a feature that influences class 1 "Severe", whereas the negative scores indicate a feature that influences class 0 "Minor"
* The further from zero, the more impact it has in determining severe or minor
* With the features that are one-hot encoded, you’re comparing to the feature dropped. 




## Conclusion & Future Direction

* Continue training this model to improve its performance and maximize its potential - an R-squared of 0.02 is not enough
    * Regularize, manipulating features

* Compare feature importance with other models such as random forest and XGBoost

* Look into other features as potential target variables such as Number of Casualties or Number of Vehicles to see if it improves the model. 

* Once model improves, perform a predictive regression to determine future accidents

