import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go

def feature_breadown(df, feat, title, save_loc):
    """Plot Breakdown of Accidents depending on specific feature to input
    Args:
        df - dataframe
        feat - feature to input
        title - graph title
        save_as - name to save image under
    """
    grouped_df = df.groupby([feat])['ID'].count().reset_index()
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, figsize=(10, 8))
    x = grouped_df.iloc[:, 0]
    y = grouped_df.iloc[:,1]
    ax.bar(x, y, color='royalblue', align='center')
    ax.set_xlabel("Severity Level", fontsize=14)
    ax.set_xticks(x)
    plt.xticks(rotation=45, fontsize=12, horizontalalignment='center')
    ax.set_ylabel("Number of Accidents", fontsize=14)
    ax.set_title(title, fontsize=16)
    plt.savefig(save_loc, dpi=150, bbox_inches = 'tight')
    fig.tight_layout()
    plt.show()

def time_total(df, save_loc):
    """Plot Total amount of Accidents depending on time of day
    Args:
        df - dataframe
        save_as - name to save image under
    """
    tm = df.groupby(['Hour'])['ID'].count().reset_index()
    hours=['12am', '1am', '2am', '3am', '4am','5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm',
            '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm',]

    fig, ax = plt.subplots(1, figsize=(14, 4))
    x1 = tm.iloc[:, 0]
    y1 = tm.iloc[:,1]
    ax.bar(x1, y1, color='royalblue', width=.3, align='edge')
    plt.yticks( fontsize=13)
    plt.xticks(x1, fontsize=13)
    ax.set_xticklabels(labels=hours, fontsize=16, rotation=45,)
    ax.set_ylabel("Number of Accidents", fontsize=18)
    ax.set_title("Total Accidents by Time of Day", fontsize=18)
    fig.tight_layout()
    plt.savefig(save_loc,dpi=150, bbox_inches = 'tight')
    plt.show()

def binary_sev(df):
    """Create data frame with binary target variable: severe or not severe
    Args:
        df - dataframe
    """
    new_df = df.copy()
    new_df["Severity"].replace({1:0, 2:0, 3:0, 4:1}, inplace=True)
    return new_df

def time_prop(df, save_loc):
    """Plot Proportion of Severe Accidents depending on time of day
    Args:
        df - dataframe
        save_as - name to save image under
    """
    sev_0 = df[(df['Severity'] == 0)]
    sev_1 = df[(df['Severity'] == 1)]
    tm_0 = sev_0.groupby(['Hour'])['ID'].count().reset_index()
    tm_1 = sev_1.groupby(['Hour'])['ID'].count().reset_index()
    hours=['12am', '1am', '2am', '3am', '4am','5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm',
            '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm',]

    fig, ax = plt.subplots(1, figsize=(14, 4))
    x1 = tm_0.iloc[:, 0]
    x2 = tm_1.iloc[:,0]
    y1 = tm_0.iloc[:,1]
    y2 = tm_1.iloc[:,1]
    ax.bar(x2, (y2/(y1+y2)), color='tomato', align='edge', width=.3)
    plt.yticks( fontsize=13)
    plt.xticks(x2, fontsize=13)
    ax.set_xticklabels(labels=hours, fontsize=16, rotation=45,)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel("Percent of Accidents", fontsize=18)
    ax.set_title("Percent of Severe Accidents by Time of Day", fontsize=18)
    fig.tight_layout()
    plt.savefig(save_loc,dpi=150, bbox_inches = 'tight')
    plt.show()

def geographic_dist(df):
    """Plot Geographic distribution of severe accidents totals by state
    Args:
        df - Total dataframe
    """
    sev_1 = df[(df['Severity'] == 1)]
    state_counts = sev_1["State"].value_counts()
    fig = go.Figure(data=go.Choropleth(locations=state_counts.index, z=state_counts.values.astype(float), 
                                    locationmode="USA-states", colorscale="Reds"))
    fig.update_layout(title_text="Total Severe Accidents for each State", geo_scope="usa")
    fig.show()


def geographic_dist_prop(df, df_total):
    """Plot Geographic distribution of severe accidents proportions by state
    Args:
        df - severe/ non-severe dataframe
        df_total - Total dataframe
    """
    sev_1 = df[(df['Severity'] == 1)]
    state_counts = sev_1["State"].value_counts()/ df_total['State'].value_counts()
    fig = go.Figure(data=go.Choropleth(locations=state_counts.index, z=state_counts.values.astype(float), 
                                    locationmode="USA-states", colorscale="Reds"))

    fig.update_layout(title_text="Proportion of Severe Accidents by Total Accidents in each State", geo_scope="usa")
    fig.show()

if __name__ == '__main__':
    # Read csv file into a pandas dataframe
    df_full = pd.read_csv('../data/total_cleaned_data.csv')

    feature_breadown(df_full, 'Severity', 'Severity Level', '../Images/severity_breakdown.png')
    feature_breadown(df_full, 'Region', 'Accidents by Region', '../Images/region_breakdown.png')
    time_total(df_full, '../Images/total_accident_time.png')
    new_df = binary_sev(df_full)
    time_prop(new_df, '../Images/accident_time_proportion.png')
    geographic_dist(new_df)
    geographic_dist_prop(new_df, df_full)


