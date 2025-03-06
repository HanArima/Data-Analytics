import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

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
    kmeansclustering_scaling()

def display_table(df):
    print(df.head(10))

def kmeansclustering():
    # Converting the status type from float to Categorical Data
    df['status_type'] = pd.Categorical(df['status_type']).codes
    # Selecting features for clustering
    features = df.drop(['status_published', 'date', 'time', 'year', 'month', 'hour_of_day', 'status_id'], axis=1)

    display_table(features)
    # Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10,5))
    sns.lineplot(x=range(1,11), y=wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

def kmeansclustering_scaling():
    # Converting the status type from float to Categorical Data
    df['status_type'] = pd.Categorical(df['status_type']).codes
    # Selecting features for clustering
    features = df.drop(['status_published', 'date', 'time', 'year', 'month', 'hour_of_day', 'status_id'], axis=1)
    # Scaling the features
    scal = StandardScaler()
    scaled_features = scal.fit_transform(features)
    # Display the first few rows of the scaled features
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns)
    display_table(scaled_df)

    ''' Elbow Method '''
    wcss = []
    for i in range(1,21):
        kmean = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmean.fit(scaled_df)
        wcss.append(kmean.inertia_)
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, 21), y=wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.show()

    ''' Silhoutte Method '''
    silhoutte = []
    for i in range(2,11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_df)

        silhoutte_avg = silhouette_score(scaled_df, cluster_labels)
        silhoutte.append(silhoutte_avg)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(2,11), y=silhoutte, marker='o')
    plt.title('Silhouette Method for Optimal k')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.show()

    ''' Davies-Bouldin Index '''
    dbi = []
    for i in range(2,11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_df)

        db_index = davies_bouldin_score(scaled_df, cluster_labels)
        dbi.append(db_index)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(2,11), y=dbi, marker='o')
    plt.title('Davies-Bouldin Index Method for Optimal k')
    plt.xlabel('Number of Clusters') 
    plt.ylabel('Davies-Bouldin Index Score')
    plt.show()

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