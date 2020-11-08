# PRICING: What should be price for the item ?

# A game company gave gift coins to its users for purchasing items in a game.Using these virtual coins,
# users buy various items for their characters.The game company did not specify a price for an item and provided users
# to buy this item at the price they wanted.For example, for the item named shield, users will buy this shield by paying
# the amounts they see fit.A user can pay with 30 units of virtual money given to his/her, while the other user can pay with
# 45 units. Therefore, users can buy this item with the amounts they can afford to pay.


# Problems to be solved:
# Does the price of the item differ by category? Express it statistically.
# Depending on the first question, what should the item cost? Explain why?
# It is desirable to be "flexible" in terms of price. Create a decision support system for the price strategy.
# Simulate item purchases and income for possible price changes.

# Data Analysis

# loading necessary libraries
import pandas as pd
import itertools
import statsmodels.stats.api as sms
from scipy.stats import shapiro
import scipy.stats as stats

# reading the data
df = pd.read_csv("Datasets/pricing.csv", sep=";")

# Analyze the dataframes
def analyze_df (df):
    print("Shape of dataframe: {0}".format(df.shape), "\n") # shape of dataframe
    print("There are {0} observations and {1} features".format(len(df),len(df.columns)),"\n") # number of observations and features
    print(df.head(),"\n") # first 5 observation
    print("Number of unique categories:{0}".format(df["category_id"].nunique()),"\n")
    print("Names of categories:{0}".format(df["category_id"].unique()),"\n")
    for col in df.columns:
        print(" Number of null value in the {0} column: {1}".format(col,df[col].isnull().sum())) # is there a null value in any columns
    print(df.describe().T,"\n") # for observe the outliers

analyze_df (df)

# When the average price by categories is analyzed, we can make comparisons for groups,
# but we need to prove this statistically.
df.groupby("category_id").agg({"price":"mean"})

# When we look at the average of the categories, we can some observe. But this observations are not a statistically significant results.
# So, we will test all hypotheses of categories in pairs and obtain statistical results.

# Definition of the Hypothesis
# H0 : There is no statistically significant difference between price average of two category
# H1 : There is statistically significant difference between price average of two category

# Assumptions of the Hypothesis
# 1- Normal Distribution
# 2- Homogeneity of Variances

# Checking Assumptions
# 1-Normal Distribution
# Non-normal population distributions, especially those that are thick-tailed or heavily skewed, considerably reduce the power of the test

# The Shapiro-Wilks Test for Normality
# H0: There is no statistically significant difference between sample distribution and theoretical normal distribution
# H1: There is statistically significant difference between sample distribution and theoretical normal distribution

# We apply shapiro-wilks test to each group and test their normal distribution.
print(" Shapiro-Wilks Test Result")
for category in df["category_id"].unique():
    test_statistic , pvalue = shapiro(df.loc[df["category_id"] ==  category,"price"])
    if(pvalue<0.05):
        print('\n','{0} -> '.format(category),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue),"H0 is rejected.")
    else:
         print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue),"H0 is not rejected.")


# 489756 -> H0 is rejected.
# 361254 -> H0 is rejected.
# 874521 -> H0 is rejected.
# 326584 -> H0 is rejected.
# 675201 -> H0 is rejected.
# 201436 -> H0 is rejected.
# Normal distribution is not provided so, we can analyze outliers.

# To determine the threshold value for outliers
def outlier_thresholds(dataframe, variable, low_quantile=0.05, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Threshold values are determined for the price variable.
low_limit,up_limit = outlier_thresholds(df, "price")
print("Low Limit : {0}  Up Limit : {1}".format(low_limit,up_limit))

# Are there any outlier observations? if any, how many?
def has_outliers(dataframe, numeric_columns, plot=False):
   # variable_names = []
    for col in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, " : ", number_of_outliers, "outliers")
            #variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    #return variable_names

has_outliers(df, ["price"])

# removing outliers
def remove_outliers(dataframe, numeric_columns):
    for variable in numeric_columns:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return dataframe_without_outliers

df = remove_outliers(df, ["price"])

# new shape of dataframe
df.shape

# We apply shapiro-wilks test to each group and test their normal distribution.
print(" Shapiro-Wilks Test Result")
for category in df["category_id"].unique():
    test_statistic , pvalue = shapiro(df.loc[df["category_id"] ==  category,"price"])
    if(pvalue<0.05):
        print('\n','{0} -> '.format(category),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue),"H0 is rejected.")
    else:
         print('Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue),"H0 is not rejected.")


# 489756 -> H0 is rejected.
# 361254 -> H0 is rejected.
# 874521 -> H0 is rejected.
# 326584 -> H0 is rejected.
# 675201 -> H0 is rejected.
# 201436 -> H0 is rejected.
# Normal distribution was not achieved even after the outliers were removed.

## Homogeneity of Variances
#Levene’s Test for Homogeneity of variances
#Levene’s test is an equal variance test. It can be used to check if our data sets fulfill the homogeneity of variance assumption before we perform the t-test or Analysis of Variance

#H0: the compared categories have equal variance.
#H1: the compared categories do not have equal variance.

# category pairs for hypothesis
pairs = []
for pair in itertools.combinations(df["category_id"].unique(),2):
    pairs.append(pair)
pairs

print("  Levene Test Result")
for pair in pairs:
    test_statistic,pvalue = stats.levene(df.loc[df["category_id"] ==  pair[0],"price"],df.loc[df["category_id"] ==  pair[1],"price"] )
    if(pvalue < 0.05):
        print('\n',"({0} - {1}) -> ".format(pair[0],pair[1]),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue), "  H0 is rejected")
    else:
         print('\n',"({0} - {1}) -> ".format(pair[0],pair[1]),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue), "  H0 is not rejected")


# Pairs that do not have equal variance (H0 is rejected):
# (489756 - 361254)
# (489756 - 874521)
# (489756 - 326584)
# (489756 - 675201)
# (489756 - 201436)
# (361254 - 874521)
# (361254 - 326584)
# (361254 - 675201)

# Pairs that have equal variance (H0 is not rejected):
# (361254 - 201436)
# (874521 - 326584)
# (874521 - 675201)
# (874521 - 201436)
# (326584 - 675201)
# (326584 - 201436)
# (675201 - 201436)


# Implementing Hypothesis Test
# We decide which test to apply according to the assumptions of normality and variance. Normal distribution hypotheses of the groups were rejected. Therefore, we need to apply a non-parametric method.

# Non-Parametrik İndependet Two Sample Test

# Mann-Whitney U test: It is a non-parametric method used to compare the means of two independent groups in a distribution that does not show normal distribution.
# H0 : There is no statistically significant difference between price average of two category
# H1 : There is statistically significant difference between price average of two category

listofResult = []
print(" Mann-Whitney U test Result")
for pair in pairs:
    test_statistic,pvalue = stats.stats.mannwhitneyu(df.loc[df["category_id"] ==  pair[0],"price"],df.loc[df["category_id"] ==  pair[1],"price"] )
    if(pvalue < 0.05):
        listofResult.append((pair[0],pair[1], "H0 is Rejected"))
        print('\n',"({0} - {1}) -> ".format(pair[0],pair[1]),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue), "  H0 is rejected")
    else:
        print('\n',"({0} - {1}) -> ".format(pair[0],pair[1]),'Test statistic = %.4f, p-Value = %.4f' % (test_statistic, pvalue), "  H0 is not rejected")
        listofResult.append((pair[0],pair[1], "H0 is not Rejected"))


result_df = pd.DataFrame()
result_df["Category 1"] = [pair[0] for pair in listofResult]
result_df["Category 2"] = [pair[1] for pair in listofResult]
result_df["H0"] = [pair[2] for pair in listofResult]

result_df

# Does the price of the item differ by category?¶
# When we examine the table above, there is no statistically significant difference average price between 6 categorical pairs,
# while there is a statistically significant difference average price** between 12 categorical pairs.

# What should the item cost?

result_df[result_df["H0"] == "H0 is not Rejected"]

# Categorical groups with no statistically significant difference :
# 361254
# 874512
# 675201
# 201436
# We can make the prices of these groups which do not differ statistically, are the same. we may apply the same price to the
# remaining two groups, let's examine their averages.

df.groupby("category_id").agg({"price":"mean"})

# Category 326584 is very close to the price average of the other 4 groups we are considering to make the same price. Category 489756 differs,
# but we will not set a separate price for it, and we will continue with a common price scenario for all categories.
#
# The average of 4 statistically identical categories will be the price we will determine. Since the average price paid by the other two category
# is high or close, not including them will not affect the purchase negatively.

signif_cat = [361254,874521,675201,201436]
sum = 0
for i in signif_cat:
    sum += df.loc[df["category_id"]== i,"price"].mean()
PRICE = sum/4

print("PRICE :{%.4f}"%PRICE)

# Confidence Intervals: It is desirable to be "flexible" in terms of price.

# We list the prices of the 4 categories that selected for pricing
prices = []
for category in signif_cat:
    for i in df.loc[df["category_id"]== category,"price"]:
        prices.append(i)

print("Felexible Price Range: ", sms.DescrStatsW(prices).tconfint_mean())

# Simulation For Item Purchases
# We will calculate the incomes that can be obtained from the minimum, maximum values of the confidence
# interval and the prices we set.

# Assumption 1 : Price(36.71096)
#For minimum price in confidence interval
freq = len(df[df["price"]>=36.7109597897918]) #number of sales equal to or greater than this price
income = freq * 36.71096 #income
print("Income: ", income)

# Assumption 2: Price(37.0923)
# For decided price
freq = len(df[df["price"]>=37.09238177238653]) #number of sales equal to or greater than this price
income = freq * 37.09238177238653 #income
print("Income: ", income)

# Assumption 3 : Price(38.17576)
# For maximum price in confidence interval
freq = len(df[df["price"]>=38.17576299427283])
income = freq * 38.17576299427283
print("Income: ",income)

# SUMMARY
# - A statistical test was applied to see if prices varied categorically.
#    - The assumptions for the test were checked
#    - All Categories rejected the normal distribution hypothesis, so it was decided to apply non-parametric independent two sample test.
# - It was observed whether there was a significant statistical difference between the categories and pricing has been made for the item.
# - Confidence interval was determined as flexibility was desired in terms of price.
# - Product purchases were simulated for possible price changes according to the confidence interval.
