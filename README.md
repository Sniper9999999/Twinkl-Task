# Twinkl-Graduate_Statistician_Task

# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Read csv file
df = pd.read_csv('twinkl_ab_data.csv')
df.head()

# check no users that have been sampled multiple times
session_counts = df['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts > 1].count()

print(f'There are {multi_users} users that appear multiple times in the dataset')

# conduct A/B test
popup_1 = df[df['group'] == 'popup_1']
popup_2 = df[df['group'] == 'popup_2']

ab_test = pd.concat([popup_1, popup_2], axis=0)
ab_test.reset_index(drop=True, inplace=True)
ab_test
ab_test.info()

ab_test['group'].value_counts()

# Visualising the results

yes_rates = ab_test.groupby('group')['click_yes']

std_p = lambda x: np.std(x, ddof=0)  # Std. deviation of the proportion              

se_p = lambda x: stats.sem(x, ddof=0)  # Std. error of the proportion (std / sqrt(n))           

yes_rates = yes_rates.agg([np.mean, std_p, se_p])
yes_rates.columns = ['conversion_rate', 'std_deviation', 'std_error']

yes_rates.style.format('{:.3f}')

# Plot the results

plt.figure(figsize=(8,6))

sns.barplot(x=ab_test['group'], y=ab_test['click_yes'], ci=False)

plt.ylim(0, 0.17)
plt.title('Yes rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Click_yes (proportion)', labelpad=15)

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

