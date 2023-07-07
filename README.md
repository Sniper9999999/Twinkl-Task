# Twinkl-Graduate_Statistician_Task

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Read csv file
df = pd.read_csv('twinkl_ab_data.csv')

df.head()

#conduct A/B test
popup_1 = df[df['group'] == 'popup_1']
popup_2 = df[df['group'] == 'popup_2']

# check missing values in click_yes
missing_values = df['click_yes'].isnull().any()

#check whether some users have been sampled multiple times
ab_test = pd.concat([popup_1, popup_2], axis=0)
ab_test.reset_index(drop=True, inplace=True)
ab_test
ab_test.info()

ab_test['group'].value_counts()

# Visualising the results

CTR = ab_test.groupby('group')['click_yes']

std_p = lambda x: np.std(x, ddof=0)   # Std. deviation of the proportion           

se_p = lambda x: stats.sem(x, ddof=0)  # Std. error of the proportion (std / sqrt(n))          

CTR = CTR.agg([np.mean, se_p])
CTR.columns = ['CRT', 'std_error']

plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['click_yes'], ci=False)

plt.ylim(0, 0.17)
plt.title('Click Through Rates (CTR) by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Click Through Rates', labelpad=15)

# Testing the hypothesis

from statsmodels.stats.proportion import proportions_ztest, proportion_confint

popup_1_results = ab_test[ab_test['group'] == 'popup_1']['click_yes']
popup_2_results = ab_test[ab_test['group'] == 'popup_2']['click_yes']
n_con = popup_1_results.count()
n_treat = popup_2_results.count()
successes = [popup_1_results.sum(), popup_2_results.sum()]
nobs = [n_con, n_treat]

z_stat, pval = proportions_ztest(successes, nobs=nobs)
(lower_con, lower_treat), (upper_con, upper_treat) = proportion_confint(successes, nobs=nobs, alpha=0.05)

print(f'z statistic: {z_stat:.2f}')
print(f'p-value: {pval:.3f}')
print(f'ci 95% for popup_1 group: [{lower_con:.3f}, {upper_con:.3f}]')
print(f'ci 95% for popup_2 group: [{lower_treat:.3f}, {upper_treat:.3f}]')

# calculate the action rate 

df['Click'] = df['click_yes'] + df['click_no']
popup_1_c = df[df['group'] == 'popup_1']
popup_2_c = df[df['group'] == 'popup_2']
ab_test = pd.concat([popup_1_c, popup_2_c], axis=0)
ab_test.reset_index(drop=True, inplace=True)
ab_test['group'].value_counts()

CTR_c = ab_test.groupby('group')['Click']
std_p = lambda x: np.std(x, ddof=0)          
se_p = lambda x: stats.sem(x, ddof=0)  
CTR_c = CTR_c.agg([np.mean, se_p])
CTR_c.columns = ['CRT_c', 'std_error']

popup_1_results_c = ab_test[ab_test['group'] == 'popup_1']['Click']
popup_2_results_c = ab_test[ab_test['group'] == 'popup_2']['Click']
n_con_c = popup_1_results_c.count()
n_treat_c = popup_2_results_c.count()
successes_c = [popup_1_results_c.sum(), popup_2_results_c.sum()]
nobs_c = [n_con_c, n_treat_c]

z_stat_c, pval_c = proportions_ztest(successes_c, nobs=nobs_c)
(lower_con_c, lower_treat_c), (upper_con_c, upper_treat_c) = proportion_confint(successes_c, nobs=nobs_c, alpha=0.05)


# Segment Analysis


from sklearn.metrics import silhouette_score


# data preperation

## missing data

# drop rows which contain the missing values in country, career and subscription columns.
df = df.dropna(subset=['country','career','subscription','group'])

# replace missing values by median (numeric data)
df['hour'] = df['hour'].fillna(df['hour'].median())
df['days_since_account_creation'] = df['days_since_account_creation'].fillna(df['days_since_account_creation'].median())
df['num_downloads'] = df['num_downloads'].fillna(df['num_downloads'].median())
df['num_searches'] = df['num_searches'].fillna(df['num_searches'].median())

## generate the new variables (make new categorical varibles)

df['England'] = np.where(df['country'] == 'England', 1, 0)
df['Scotland'] = np.where(df['country'] == 'Scotland', 1, 0)
df['Wales'] = np.where(df['country'] == 'Wales', 1, 0)
df['other'] = np.where((df['country'] != 'England') & (df['country'] != 'Wales') & (df['country'] != 'Scotland'), 1, 0)
df['KS1'] = np.where(df['career'] == 'KS1', 1, 0)
df['Paid'] = np.where(df['subscription'] == 'paid', 1, 0)
df['popup_1'] = np.where(df['group'] == 'popup_1', 1, 0)
df['popup_2'] = np.where(df['group'] == 'popup_2', 1, 0)

# seperate hour, days_since_account_creation, num_downloads, num_searches into several categories
hour_quantiles = df['hour'].quantile([0.25, 0.5, 0.75])
days_quantiles = df['days_since_account_creation'].quantile([0.25, 0.5, 0.75])
download_quantiles = df['num_downloads'].quantile([0.25, 0.5, 0.75])
searches_quantiles = df['num_searches'].quantile([0.25, 0.5, 0.75])

# Create ranges to group the data
hour_cut = [0, 13, np.inf]
days_cut = [0, 1800, np.inf]
num_downloads_cut = [0, 25, np.inf]
num_searches_cut = [0, 20, np.inf]

labels_hour = ['0', '1']
labels_days = ['0', '1']
labels_download = ['0', '1']
labels_searches = ['0', '1']

# generate the new variables

df['hour_cut'] = pd.cut(df['hour'], bins=hour_cut, labels=labels_hour, right=False)
df['days_cut'] = pd.cut(df['days_since_account_creation'], bins=days_cut, labels=labels_days, right=False)
df['num_downloads_cut'] = pd.cut(df['num_downloads'], bins=num_downloads_cut, labels=labels_download, right=False)
df['num_searches_cut'] = pd.cut(df['num_searches'], bins=num_searches_cut, labels=labels_searches, right=False)


# Conduct K-modes clustering

from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score


# Select the variables for clustering
selected_columns = ['England', 'Scotland', 'Wales', 'other', 'KS1', 'Paid', 'hour_cut', 'days_cut', 'num_downloads_cut', 'num_searches_cut','popup_1','popup_2']

# Subset the data with the selected columns
data_select = df[selected_columns]
data_select = data_select.dropna()


# Find the best number of clusters
cost = []
silhouette_scores = []
for k in range(2, 11):
    km = KModes(n_clusters=k, n_init=5, verbose=0)
    clusters = km.fit_predict(data_select)
    cost.append(km.cost_)
    silhouette_scores.append(silhouette_score(data_select, clusters))

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), cost, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Iteration')
plt.title('Elbow Curve')
plt.show()

# Plot the silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.show()


# Create an instance of the KModes algorithm
km = KModes(n_clusters=5, n_init=5, verbose=1)

# Perform clustering on the categorical data
clusters = km.fit_predict(data_select)

# Access the cluster centroids
centroids = km.cluster_centroids_

# Print the cluster labels and centroids
print("Cluster Labels:", clusters)
print("Cluster Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i+1}: {centroid}")

# calculate the counts of each clusters
quantity = pd.Series(km.labels_).value_counts()

