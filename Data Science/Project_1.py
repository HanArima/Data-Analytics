import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Facebook_Marketplace_data.csv")
df = df.drop('Column1', axis=1)
df = df.drop('Column2', axis=1)
df = df.drop('Column3', axis=1)
df = df.drop('Column4', axis=1)

df['status_published'] = pd.to_datetime(df['status_published'])
df['date'] = df['status_published'].dt.date
df['month'] = df['status_published'].dt.month
df['year'] = df['status_published'].dt.year
df['time'] = df['status_published'].dt.time
df['hour_of_day'] = df['status_published'].dt.hour

def main():
    correlation_matrix()

def display_table(df):
    print(df.head(10))


def correlation_matrix():
    temp_df = df[['num_reactions','num_comments',	'num_shares','num_likes']]
    display_table(temp_df)
    corr = temp_df.corr()
    plt.figure(figsize=(30,20), edgecolor="red")
    sns.heatmap(corr, fmt=".2f", annot=True, linewidths=1.5, xticklabels='auto', yticklabels='auto')
    plt.show()

def relation_bw_reaction_timestamp():
    temp_df = df[['hour_of_day', 'num_reactions']]
    display_table(temp_df)
    fig1 = plt.figure(figsize=(10,30) )
    sns.scatterplot(data=temp_df, x="hour_of_day", y="num_reactions")
    plt.title('Scatter Plot for Reactions by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Reactions')
    plt.xlim(0,24)

    fig2 = plt.figure(figsize=(10,20))
    sns.lineplot(data=temp_df, x="hour_of_day", y="num_reactions")
    plt.title('Line Plot for Reactions by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Reactions')
    plt.ylim(0,1000)
    plt.xlim(0,24)
    
    fig3 = plt.figure(figsize=(10,30))
    sns.scatterplot(data=df, x="month", y="num_reactions")
    plt.title('Scatter Plot for Reactions by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Reactions')
    plt.xlim(0,13)


    fig4 = plt.figure(figsize=(10,30))
    sns.scatterplot(data=df, x="month", y="num_reactions")
    plt.title('Scatter Plot for Reactions by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Reactions')

    fig5 = plt.figure(figsize=(10,30))
    sns.lineplot(data=df, x="month", y="num_reactions")
    plt.title('Line Plot for Reactions by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of Reactions')
    plt.xlim(0,13)


    fig6 = plt.figure(figsize=(10,30))
    sns.lineplot(data=df, x="year", y="num_reactions")
    plt.title('Line Plot for Reactions by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Reactions')

    plt.show()

if __name__ == "__main__":
    main()