#!/usr/bin/env python
# coding: utf-8

# *Keywords:* Loan Default, Python, Machine Learning, Predictive Modeling, Logistic Regression, Random Forest, Decision Tree, Correlation Matrix

# ## Project Overview and Objectives

# **Research Question:** 
# 
# What borrower details, financial attributes, and potential interactions predict loan default for loans issued between 2012 and 2019 using Advanced Machine Learning techniques?
# 
# **Alternative Hypothesis (H1):** 
# 
# Suggests that there is a significant relationship between at least one of the borrower details, financial attributes, or potential interactions and loan defaults for loans issued between 2012 and 2019 using Advanced Machine Learning techniques.
# 
# **Null Hypothesis (H0):** 
# 
# Suggests no significant relationship between borrower details, financial attributes, or potential interactions and loan defaults for loans issued between 2012 and 2019 using Advanced Machine Learning techniques.
# 
# **Methodology**
# 
# This project employs a structured methodology consisting of several key stages: data cleaning, Exploratory Data Analysis (EDA), feature engineering, and feature selection. Following those steps, predictive modeling is conducted utilizing Logistic Regression, Random Forest, and Decision Tree algorithms. Finally, the project culminates with a comprehensive feature importance analysis, using techniques such as Confusion Matrix and ROC Curve evaluation. 

# **Dataset:** Lending Club Loan Data; located on Kaggle
# 
# **Second Research Question to this project:** Analyzing if natural disasters can be used as a predictor for Loan Default. Located on Kaggle "Natural Disasters as Predictor for Loan Default"
# 
# [Kaggle.com: rosaaestrada - Natural Disasters as Predictor for Loan Default](https://www.kaggle.com/code/rosaaestrada/natural-disasters-as-predictor-for-loan-default)

# # Importing the data

# In[1]:


#Import libraries
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn import under_sampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV


# In[2]:


#Loading dataset
df = pd.read_csv('/kaggle/input/lending-club-loan-data-csv/loan.csv', low_memory = False)


# In[3]:


#Loading the dictionary file
description = pd.read_excel('/kaggle/input/lending-club-loan-data-csv/LCDataDictionary.xlsx').dropna()


# In[4]:


df.shape


# In[5]:


#Show first 10 rows
df.head(5)


# In[6]:


# Selecting the first 10 variables/columns
first_10_variables = df.iloc[:, :10]

# Displaying the first 5 rows of the selected variables
print(first_10_variables.head())


# In[7]:


# Selecting the last 10 variables/columns
last_10_variables = df.iloc[:, -10:]

# Displaying the first 5 rows of the selected variables
print(last_10_variables.head())


# In[8]:


#Show all column names
df.columns


# In[9]:


#Numbers of each data type
data_types_count = df.dtypes.value_counts()

#Define color palette as a dictionary
color_palette = {'float64': 'steelblue', 'int64': 'darkorange', 
                 'object': 'seagreen'}

#Get colors for each data type
colors = [color_palette[dt] for dt in data_types_count.index.astype(str)]

plt.figure(figsize=(18,8))
bars = plt.bar(data_types_count.index.astype(str), data_types_count.values, color=colors)
plt.ylabel('Number of columns', fontsize=15)
plt.xlabel('Data type', fontsize=15)

plt.show()


# In[10]:


#Print data type counts
data_types_count


# In[11]:


#Check the description for each column
description.style.set_properties(subset=['Description'], **{'width': '1000px'})


# # Data Cleaning

# ## Missing Values

# In[12]:


#Create a function to calculate the percentage of missing values for each column
def missing_values(df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

#check missing values
miss_values = missing_values(df)
miss_values.head(10)


# In[13]:


miss_values.tail(10)


# In[14]:


#Function to keep only columns with less than 30 missing values
def filter_columns_with_less_than_30_percent_missing(df):
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    cols_to_keep = mis_val_percent[mis_val_percent < 30].index.tolist()
    filtered_df = df[cols_to_keep]
    return filtered_df

filtered_df = filter_columns_with_less_than_30_percent_missing(df)


# In[15]:


#Verify if we have successfully dropped columns had more than 70% missing values
filtered_df.shape   #It was (2260668, 145)


# In[16]:


##### Make a copy
df1 = filtered_df.copy()


# ### Check missing values(categorical)

# In[17]:


#list categorical features
cat_col = df1.select_dtypes(include=['object']).columns

#Check missing values in categorical features
cat_mv = df1[cat_col].isnull().sum()

print("Missing values in categorical features:\n", cat_mv)


# In[18]:


#Imputing with the most frequent value(mode)
df1['last_pymnt_d'].fillna(df1['last_pymnt_d'].mode()[0], inplace=True)
df1['earliest_cr_line'].fillna(df1['earliest_cr_line'].mode()[0], inplace=True)
df1['last_credit_pull_d'].fillna(df1['last_credit_pull_d'].mode()[0], inplace=True)


# In[19]:


#Drop the columns with more than ten thousand missing values, as they contribute significantly to the overall missing data
#Missing values in these variables are going to drop
var_drop= ['emp_title', 'emp_length', 'title', 'zip_code']

#Drop rows with missing values in any of the specified variables
df1 = df1.dropna(subset=var_drop)

#Verify
print("Missing values in categorical features:\n", df1[cat_col].isnull().sum())


# ### Check missing values(numeric)

# In[20]:


#list numeric features
num_col = df1.select_dtypes(exclude=['object']).columns

#Check missing values in numeric features
num_mv = df1[num_col].isnull().sum().to_string()

print("Missing values in numeric features:\n", num_mv)


# In[21]:


#Imputing missing values with mean
df1['dti'].fillna(df1['dti'].mean(), inplace=True)


# In[22]:


#Drop other missing values
df1 = df1.dropna(subset=num_col)

#Verify
print("Missing values in numeric features after cleaning:")
print(df1[num_col].isnull().sum().to_string())


# In[23]:


#Verify
df1.shape  #It was (2260668, 87)


# In[24]:


### Make a copy
df2 = df1.copy()


# ## Outliers

# In[25]:


#list numeric features along with their statistical description
pd.set_option('display.float_format', lambda x: '%.2f' % x)
des = df2.select_dtypes(exclude=['object']).describe().round(decimals=2).transpose()
print(des.to_string())


# ### loan_amnt, funded_amnt, funded_amnt_inv, and installment

# In[26]:


#Plotting histograms
df2[['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment']].hist(bins=50, figsize=(20, 15), 
                                                                            layout=(2,2), color='skyblue', edgecolor='black')

plt.suptitle('Histograms for Loan Amounts and Installment', fontsize=20)
plt.show()


# In[27]:


#We observed that loan_amnt, funded_amnt and funded_amnt_inv have similar distributions, so we only keep one of them
#Also we keep installment
df2.drop(columns=['funded_amnt', 'funded_amnt_inv'], inplace=True)


# In[28]:


#The installment shows a different distribution pattern compared to the loan amount variables.
#It has a right-skewed distribution indicating that most loans have smaller monthly payments.
#Drop the payments larger than 1100
df2= df2[df2['installment'] <= 1100]


# ### int_rate

# In[29]:


#Plotting histogram for int_rate
df2['int_rate'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Interest Rate', fontsize=10)
plt.show()


# In[30]:


#Drop the int_rate larger than 25
df2= df2[df2['int_rate'] <= 25]


# ### annual_inc

# In[31]:


#Plotting histogram for annual_inc
df2['annual_inc'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Annual Income', fontsize=10)
plt.show()


# In[32]:


#Remove outliers using IQR
#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df2['annual_inc'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[33]:


#Drop outliers in annual_inc
df2 = df2[(df2['annual_inc'] >= 17500) & (df2['annual_inc'] <=162500) ]


# ### dti

# In[34]:


#Plotting histogram for dti
df2['dti'].hist(bins=80, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Debt to interest ratio', fontsize=10)
plt.show()


# In[35]:


#Drop outlier in dti
df2 = df2[(df2['dti'] >0) & (df2['dti'] <=50) ]


# ### open_acc and total_acc

# In[36]:


#Plotting histograms
df2[['open_acc', 'total_acc']].hist(bins=50, figsize=(20, 15), layout=(2,2), color='skyblue', edgecolor='black')


# In[37]:


#Drop outliers in both variables
df2 = df2[df2['total_acc'] <=60]
df2 = df2[df2['open_acc'] <=30]


# ### pub_rec, 
# number of derogatory public records

# In[38]:


#Plotting histogram for puc_rec
df2['pub_rec'].hist(bins=80, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of derogatory public records', fontsize=10)
plt.show()


# In[39]:


#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df2['pub_rec'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[40]:


#Drop the variable due to its low variations
df2.drop(columns=['pub_rec'], inplace=True)


# ### delinq_2yrs,
# The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years

# In[41]:


#Plotting histogram for delinq_2yrs
df2['delinq_2yrs'].hist(bins=80, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of Delinquency for the Past 2 Years', fontsize=10)
plt.show()


# In[42]:


#Since majority values are clustered in 0, we decide to drop the column
df2.drop(columns=['delinq_2yrs'], inplace=True)


# ### inq_last_6mths,
# The number of inquiries in past 6 months (excluding auto and mortgage inquiries)

# In[43]:


#Plotting histogram for delinq_2yrs
df2['inq_last_6mths'].hist(bins=80, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('The number of inquiries in past 6 months', fontsize=10)
plt.show()


# In[44]:


#It looks like a categorical variable(object for convinience), so convert it
df2['inq_last_6mths'] = df2['inq_last_6mths'].astype('object')


# ### revol_bal, 
# Total credit revolving balance
# ### revol_util, 
# Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.

# In[45]:


#Plotting histograms
df2[['revol_bal', 'revol_util']].hist(bins=50, figsize=(20, 15), layout=(2,2), color='skyblue', edgecolor='black')


# In[46]:


#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df2['revol_bal'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[47]:


#Drop column(revol_bal), notes if we remove outliers, we will lose more than 1 million rows, better drop the column
df2.drop(columns=['revol_bal'], inplace=True)


# In[48]:


#Drop outlier in revol_util
df2 = df2[df2['revol_util'] <= 100]


# ### out_prncp, 
# Remaining outstanding principal for total amount funded
# ### out_prncp_inv, 
# Remaining outstanding principal for portion of total amount funded by investors

# In[49]:


#Plotting histograms
df2[['out_prncp', 'out_prncp_inv']].hist(bins=50, figsize=(20, 15), layout=(2,2), color='skyblue', edgecolor='black')


# In[50]:


#Drop out_prncp_inv (Not going to drop outliers since a large outstanding pricipal might be an indicator of a loan that's 
#at risk of defaulting)
df2.drop(columns = ['out_prncp_inv'], inplace = True)


# ### total_pymnt,
# Payments received to date for total amount funded
# ### total_pymnt_inv,
# Payments received to date for portion of total amount funded by investors
# ### last_pymnt_amnt
# Last total payment amount received

# In[51]:


#Plotting histograms
df2[['total_pymnt', 'total_pymnt_inv', 'last_pymnt_amnt']].hist(bins=50, figsize=(20, 15), layout=(2,2), 
                                                                color='skyblue', edgecolor='black')


# In[52]:


#Drop total_pymnt_inv
df2.drop(columns = ['total_pymnt_inv'], inplace = True)


# ### total_rec_prncp  
# Principal received to date
# ### total_rec_int 
# Interest received to date
# ### total_rec_late_fee 
# Late fees received to date

# In[53]:


#Plotting histograms
df2[['total_rec_prncp', 'total_rec_int', 'total_rec_late_fee']].hist(bins=50, figsize=(20, 15), layout=(2,2), 
                                                                color='skyblue', edgecolor='black')


# In[54]:


#Drop total_rec_late_fee
df2.drop(columns = ['total_rec_late_fee'], inplace = True)


# ### recoveries
# post charge off gross recovery
# ### collection_recovery_fee
# post charge off collection fee

# In[55]:


#Plotting histograms
df2[['recoveries', 'collection_recovery_fee']].hist(bins=50, figsize=(20, 15), layout=(2,2), color='skyblue', edgecolor='black')


# In[56]:


#Drop both columns, not relevant to loan default
df2.drop(columns = ['recoveries', 'collection_recovery_fee'], inplace = True)


# ### collections_12_mths_ex_med
# Number of collections in 12 months excluding medical collections

# In[57]:


#Plot
df2['collections_12_mths_ex_med'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of collections in 12 months', fontsize=10)
plt.show()


# In[58]:


#It may have an impact on loan default, so covert it to category(object)
df2['collections_12_mths_ex_med'] = df2['collections_12_mths_ex_med'].astype('object')


# ### policy_code
# publicly available policy_code=1 new products not publicly available policy_code=2

# In[59]:


#Plot
df2['policy_code'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Policy Code', fontsize=10)
plt.show()


# In[60]:


#Drop it
df2.drop(columns = ['policy_code'], inplace = True)


# ### acc_now_delinq
# The number of accounts on which the borrower is now delinquent.

# In[61]:


#Plot
df2['acc_now_delinq'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('The number of accounts on which the borrower is now delinquent', fontsize=10)
plt.show()


# In[62]:


#Could be an indicator, cover it to categorical variable(object)
df2['acc_now_delinq'] = df2['acc_now_delinq'].astype('object')


# ### tot_coll_amt   
# Total collection amounts ever owed
# ### tot_cur_bal
# Total current balance of all accounts
# ### avg_cur_bal
# Average current balance of all accounts

# In[63]:


#Plotting histograms
df2[['tot_coll_amt', 'tot_cur_bal', 'avg_cur_bal']].hist(bins=50, figsize=(20, 15), layout=(2,2), 
                                          color='skyblue', edgecolor='black')


# In[64]:


#Drop tot_coll_amt, clustered at low amount, drop avg_cur_bal(an average of all accounts)
df2.drop(columns = ['tot_coll_amt', 'avg_cur_bal'], inplace = True)


# In[65]:


#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df2['tot_cur_bal'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[66]:


#Considering to drop values larger than upper cutoff
df2 = df2[df2['tot_cur_bal'] <= 473862]


# ### total_rev_hi_lim
# Total high credit/credit limit

# In[67]:


#Plot
df2['total_rev_hi_lim'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Total high credit/credit limit', fontsize=10)
plt.show()


# In[68]:


#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df2['total_rev_hi_lim'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[69]:


#Considering to drop values larger than upper cutoff
df2 = df2[df2['total_rev_hi_lim'] <= 78276.875]


# ### acc_open_past_24mths
# Number of trades opened in past 24 months

# In[70]:


#Plot
df2['acc_open_past_24mths'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of trades opened in past 24 months', fontsize=10)
plt.show()


# In[71]:


#Remove outliers
df2 = df2[df2['acc_open_past_24mths'] <= 13]


# ### bc_open_to_buy 
# Total open to buy on revolving bankcards
# ### bc_util
# Ratio of total current balance to high credit/credit limit for all bankcard accounts

# In[72]:


#Plotting histograms
df2[['bc_open_to_buy', 'bc_util']].hist(bins=50, figsize=(20, 15), layout=(2,2), 
                                          color='skyblue', edgecolor='black')


# In[73]:


#remove outliers
df2 = df2[df2['bc_open_to_buy'] <= 40000]
df2 = df2[df2['bc_util'] <= 100]


# ### chargeoff_within_12_mths
# Number of charge-offs within 12 months

# In[74]:


#Plot
df2['chargeoff_within_12_mths'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of charge-offs within 12 months', fontsize=10)
plt.show()


# In[75]:


#Could be a potential feature to predcit loan default, cover it to categorical
df2['chargeoff_within_12_mths'] = df2['chargeoff_within_12_mths'].astype('object')


# ### delinq_amnt
# The past-due amount owed for the accounts on which the borrower is now delinquent

# In[76]:


#Plot
df2['delinq_amnt'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Amount owed for accounts', fontsize=10)
plt.show()


# In[77]:


#Low variations, drop it
df2.drop(columns = ['delinq_amnt'], inplace = True)


# ### mo_sin_old_il_acct 
# Months since oldest bank installment account opened
# ### mo_sin_old_rev_tl_op
# Months since oldest revolving account opened
# ### mo_sin_rcnt_rev_tl_op 
# Months since most recent revolving account opened
# ### mo_sin_rcnt_tl
# Months since most recent account opened

# In[78]:


#Plotting histograms
df2[['mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl']].hist(bins=50, figsize=(20, 15), 
                                                              layout=(2,2), color='skyblue', edgecolor='black')


# In[79]:


#Remove outliers(These variables maybe not relevant to our target variable)
df2 = df2[df2['mo_sin_old_il_acct'] <= 250]
df2 = df2[df2['mo_sin_old_rev_tl_op'] <= 400]
df2 = df2[df2['mo_sin_rcnt_rev_tl_op'] <= 50]
df2 = df2[df2['mo_sin_rcnt_tl'] <= 25]


# ### num_accts_ever_120_pd 
# Number of accounts ever 120 or more days past due
# ### num_actv_bc_tl 
# Number of currently active bankcard accounts
# ### num_actv_rev_tl 
# Number of currently active revolving trades

# In[80]:


#Plotting histograms
df2[['num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl']].hist(bins=50, figsize=(20, 15), 
                                                              layout=(2,2), color='skyblue', edgecolor='black')


# In[81]:


#Drop num_accts_ever_120_pd, clustered at "0", also remove outliers in other two variables
df2.drop(columns = ['num_accts_ever_120_pd'], inplace = True)

df2 = df2[df2['num_actv_bc_tl'] <= 10]
df2 = df2[df2['num_actv_rev_tl'] <= 15]


# ### num_bc_sats
# Number of satisfactory bankcard accounts
# ### num_bc_tl
# Number of bankcard accounts
# ### num_il_tl 
# Number of installment accounts

# In[82]:


#Plotting histograms
df2[['num_bc_sats', 'num_bc_tl', 'num_il_tl']].hist(bins=50, figsize=(20, 15), layout=(2,2), color='skyblue', edgecolor='black')


# In[83]:


#Remove outliers in all variables
df2 = df2[df2['num_bc_sats'] <= 10]
df2 = df2[df2['num_bc_tl'] <= 20]
df2 = df2[df2['num_il_tl'] <= 25]


# ### num_op_rev_tl
# Number of open revolving accounts
# ### num_rev_accts 
# Number of revolving accounts
# ### num_rev_tl_bal_gt_0
# Number of revolving trades with balance >0
# ### num_sats    
# Number of satisfactory accounts

# In[84]:


#Plotting histograms
df2[['num_op_rev_tl', 'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats']].hist(bins=50, figsize=(20, 15),
                                                      layout=(2,2), color='skyblue', edgecolor='black')


# In[85]:


#Remove outliers (They have similar distributions)
df2 = df2[df2['num_op_rev_tl'] <= 20]
df2 = df2[df2['num_rev_accts'] <= 30]
df2 = df2[df2['num_rev_tl_bal_gt_0'] <= 13]
df2 = df2[df2['num_sats'] <= 22]


# ### num_tl_120dpd_2m
# Number of accounts currently 120 days past due (updated in past 2 months)
# ### num_tl_30dpd 
# Number of accounts currently 30 days past due (updated in past 2 months)
# ### num_tl_90g_dpd_24m
# Number of accounts 90 or more days past due in last 24 months
# ### num_tl_op_past_12m 
# Number of accounts opened in past 12 months

# In[86]:


#Plotting histograms
df2[['num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m', 'num_tl_op_past_12m']].hist(bins=50, figsize=(20, 15),
                                                      layout=(2,2), color='skyblue', edgecolor='black')


# In[87]:


#Only keep num_tl_op_past_12, drop rest
df2.drop(columns = ['num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m'], inplace = True)
df2 = df2[df2['num_tl_op_past_12m'] <= 6]


# In[88]:


#Cover it to categorical 
df2['num_tl_op_past_12m'] = df2['num_tl_op_past_12m'].astype('object')


# ### pct_tl_nvr_dlq
# Percent of trades never delinquent

# In[89]:


#Plot
df2['pct_tl_nvr_dlq'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Percent of trades never delinquent', fontsize=10)
plt.show()


# In[90]:


#remove outliers
df2 = df2[df2['pct_tl_nvr_dlq'] >= 80]


# ### percent_bc_gt_75
# Percentage of all bankcard accounts > 75% of limit

# In[91]:


#Plot
df2['percent_bc_gt_75'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Percentage of all bankcard accounts > 75% of limit', fontsize=10)
plt.show()


# ### pub_rec_bankruptcies
# Number of public record bankruptcies

# In[92]:


#Plot
df2['pub_rec_bankruptcies'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of public record bankruptcies', fontsize=10)
plt.show()


# In[93]:


#Could be an indicator, cover it to categorical variable(object)
df2['pub_rec_bankruptcies'] = df2['pub_rec_bankruptcies'].astype('object')


# ### tax_liens
# Number of tax liens

# In[94]:


#Plot
df2['tax_liens'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Number of tax liens', fontsize=10)
plt.show()


# In[95]:


#drop it
df2.drop(columns = ['tax_liens'], inplace = True)


# ### tot_hi_cred_lim
# Total high credit/credit limit

# In[96]:


#Plot
df2['tot_hi_cred_lim'].hist(bins=50, figsize=(6, 4), color='skyblue', edgecolor='black')
plt.title('Total high credit/credit limit', fontsize=10)
plt.show()


# In[97]:


#Drop
df2.drop(columns = ['tot_hi_cred_lim'], inplace = True)


# ### total_bal_ex_mort 
# Total credit balance excluding mortgage
# ### total_bc_limit  
# Total bankcard high credit/credit limit
# ### total_il_high_credit_limit 
# Total installment high credit/credit limit

# In[98]:


#Plotting histograms
df2[['total_bal_ex_mort', 'total_bc_limit', 'total_il_high_credit_limit']].hist(bins=50, figsize=(20, 15),
                                                      layout=(2,2), color='skyblue', edgecolor='black')


# In[99]:


#remove outliers
df2 = df2[df2['total_bal_ex_mort'] <= 200000]
df2 = df2[df2['total_bc_limit'] <= 60000]


# In[100]:


#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df2['total_il_high_credit_limit'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[101]:


df2 = df2[df2['total_il_high_credit_limit'] <= 108223]


# In[102]:


df2.shape # It was (1654107, 87)


# In[103]:


#Double check
df2.select_dtypes(exclude=['object']).describe().round(decimals=2).transpose()


# In[104]:


###### Make a copy
df3 = df2.copy()


# ### Check duplicated rows

# In[105]:


#check if there are some duplicate rows
number_of_duplicates = df3.duplicated().sum()

print(f"Number of duplicate rows: {number_of_duplicates}")


# In[106]:


df3.head(7)


# # Exploratory Data Analysis(EDA)

# ## Target Variable Analysis

# In[107]:


#Check our target variable loan_status
df3['loan_status'].unique()


# In[108]:


# Create a color palette with multiple colors
custom_palette = {
    'Current'            : 'darkorange',
    'Fully Paid'         : 'seagreen',
    'In Grace Period'    : 'tomato',
    'Late (31-120 days)' : 'mediumpurple',
    'Late (16-30 days)'  : 'orange',
    'Charged Off'        : 'steelblue',
    'Default'            : 'firebrick'
}

# Use value_counts() to count the occurrences of each value in the loan_status column
status_counts = df3['loan_status'].value_counts()

# Plotting the bar graph with custom colors
plt.figure(figsize=(12, 6))
sns.barplot(x=status_counts.index, y=status_counts.values, palette=custom_palette)
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()


# In[109]:


#Calculate the value counts of each category in 'loan_status'
loan_status_value_counts = df3['loan_status'].value_counts()

#Calculate the percentage of each category in 'loan_status'
percentage_loan_status = (loan_status_value_counts / len(df3['loan_status'])) * 100

#Combine the value counts and percentages into a DataFrame
loan_status_summary = pd.DataFrame({'Count': loan_status_value_counts, 'Percentage': percentage_loan_status})

#Print the results
print("Values in 'loan_status' and their percentages:")
print(loan_status_summary)


# #### Considering the values and their percentage in our target variable, we will only utilize 'Fully paid' and 'Charged off' for our project. The 'Charged off' category is defined by Lending Club, as they believe borrowers in this group are most likely to default.

# In[110]:


#Drop other values for the target variable
df3 = df3[df3['loan_status'].isin(['Fully Paid', 'Charged Off'])]


# In[111]:


#Pie chart for target variable loan_status
loan_status_counts = df3['loan_status'].value_counts()

plt.figure(figsize=(8, 4))
plt.pie(loan_status_counts.values, labels=loan_status_counts.index, autopct='%1.1f%%', 
        colors=['lightgreen', 'grey'])
plt.show()


# In[112]:


df3.shape #It was (932003, 67)


# ## Target Variable Imbalance
# #### Due to the imbalance in the target variable, we will employ the random under sampling technique to reduce the majority class (fully paid).

# In[113]:


#Separate features and target variable
X = df3.drop('loan_status', axis=1)
y = df3['loan_status']

#Initialize and fit to the data
rus = RandomUnderSampler(random_state = 123)
X_resampled, y_resampled = rus.fit_resample(X, y)


# In[114]:


#Plot the pie chart
loan_status_resampled_counts = y_resampled.value_counts()

plt.figure(figsize=(8, 4))
plt.pie(loan_status_resampled_counts, labels=loan_status_resampled_counts.index, 
        autopct='%1.1f%%',colors=['grey', 'lightgreen'])
plt.show()


# In[115]:


#Combine X_resampled and y_resampled back into a new DataFrame
resampled_df = pd.concat([X_resampled, y_resampled], axis=1)

#Verify
print(resampled_df['loan_status'].value_counts())


# In[116]:


#Re-label target variable: 'fully paid' to 0 and 'charged off' to 1
resampled_df['loan_status'] = resampled_df['loan_status'].replace({'Fully Paid': 0, 'Charged Off': 1})


# In[117]:


#Verify
resampled_df['loan_status'].value_counts()


# In[118]:


resampled_df.shape  #It was (560048, 67)


# In[119]:


################ Make a copy 
df4 = resampled_df.copy()


# ## Category Variables and Target Variable Analysis

# In[120]:


#Unique levels for each categorical variable
df4.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[121]:


# Filter categorical variables
categorical_df = df4.select_dtypes('object')

# Select first 10 variables
categorical_df_subset = categorical_df.iloc[:, :10]

# Print first 5 rows and first 10 categorical variables
print(categorical_df_subset.head())


# #### We noticed that those variables have more than 50 or even more categories. It's uncommon to have categorical variables with such an extensive range of categories in modeling scenarios. Having thousands of encoded variables could significantly complicate the modeling process and may lead to issues like overfitting.

# In[122]:


#Drop the specified columns
df4.drop(columns=['emp_title', 'earliest_cr_line', 'title', 'zip_code', 'issue_d', 'last_credit_pull_d'], 
         inplace=True)


# #### If a categorical variable only has one category, it does not provide any variability or information to the predictive model. Therefore, keeping such a variable in the model is not beneficial and would not contribute to the model's predictive performance.

# In[123]:


#Drop categorical variables with only one category
df4.drop(columns= ['pymnt_plan', 'hardship_flag'], inplace=True)


# In[124]:


#Verify the categorical variables
df4.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# ## Two cells below not used

# In[125]:


# Let's assume loan_status is the target variable
target_variable = 'loan_status'

# Selecting categorical variables
categorical_vars = df4.select_dtypes(include=['object']).columns

# Calculate the number of rows and columns for subplots
num_plots = len(categorical_vars)
num_cols = 3  # You can change the number of columns as per your preference
num_rows = (num_plots - 1) // num_cols + 1

# Set up the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
axes = axes.flatten()

# Define colors for target variable values
colors = {0: 'seagreen', 1: 'steelblue'}

# Plot each categorical variable
for i, cat_var in enumerate(categorical_vars):
    # Create a DataFrame for plotting
    plot_data = df4[[cat_var, target_variable]].value_counts().reset_index(name='count')
    # Scatter plot
    for val in plot_data[target_variable].unique():
        data = plot_data[plot_data[target_variable] == val]
        axes[i].scatter(data[cat_var], data['count'], color=colors[val], label=f"{target_variable}={val}", s=100, alpha=0.5)
    axes[i].set_title(cat_var + " vs " + target_variable)
    axes[i].set_xlabel(cat_var)
    axes[i].set_ylabel("Count")
    axes[i].legend()

# Hide any empty subplots
for j in range(i+1, num_rows*num_cols):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# In[126]:


target_variable = 'loan_status'

# Selecting categorical variables
categorical_vars = df4.select_dtypes(include=['object']).columns

# Calculate the number of rows and columns for subplots
num_plots = len(categorical_vars)
num_cols = 3  # You can change the number of columns as per your preference
num_rows = (num_plots - 1) // num_cols + 1

# Set up the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
axes = axes.flatten()

# Define colors for target variable values
colors = {0: 'seagreen', 1: 'steelblue'}

# Plot each categorical variable
for i, cat_var in enumerate(categorical_vars):
    # Create a DataFrame for plotting
    plot_data = df4[[cat_var, target_variable]].value_counts().unstack().fillna(0)
    # Count plot
    plot_data.plot(kind='bar', stacked=True, color=[colors[col] for col in plot_data.columns], ax=axes[i])
    axes[i].set_title(cat_var + " vs " + target_variable)
    axes[i].set_xlabel(cat_var)
    axes[i].set_ylabel("Count")
    axes[i].legend(title=target_variable)

# Hide any empty subplots
for j in range(i+1, num_rows*num_cols):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()


# ### Term and loan_status

# In[127]:


#Set the style of the plot
sns.set(style="whitegrid")

#Create the bar plot
plt.figure(figsize=(8, 4))
sns.countplot(x='term', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Distribution of Loan Status by Term')
plt.xlabel('Term')
plt.ylabel('Count')

plt.show()


# #### The term feature should likely be kept in the model since it could be a significant predictor of loan status. The clear difference in outcomes between the two terms suggests that term length influences the likelihood of a loan being fully paid or charged off.

# ### Grade and Sub_grade with loan_status

# In[128]:


#Bar chart for grade
plt.figure(figsize=(8, 4))
sns.countplot(x='grade', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'], order=sorted(df4['grade'].unique()))

#Set the title and labels
plt.title('Distribution of Loan Status by Grade')
plt.xlabel('Grade')
plt.ylabel('Count')

plt.show()


# In[129]:


df4['grade'].value_counts()


# In[130]:


#sub_grade
plt.figure(figsize=(12, 6))
sns.countplot(x='sub_grade', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'], 
              order=sorted(df4['sub_grade'].unique()))

#Set the title and labels
plt.title('Distribution of Loan Status by Sub_grade')
plt.xlabel('Sub_grade')
plt.ylabel('Count')

plt.show()


# #### We do not require both 'Grade' and 'Sub_grade' since they convey similar information about the loan grading. Additionally, we will consolidate the 'F' and 'G' grades into 'E', and subsequently rename this combined category to 'E_FG'.

# #### Both 'Grade' and 'Sub_grade' appear to be important features for predicting loan default and should be considered for inclusion in the predictive model. They show a clear differentiation in loan performance, which is valuable for risk assessment.

# In[131]:


#Combine 'F' and 'G' grades into 'E'
df4['grade'] = df4['grade'].replace(['F', 'G'], 'E')

#Rename 'E' to 'E_FG'
df4['grade'] = df4['grade'].replace('E', 'E_FG')


# In[132]:


#Drop sub_grade column
df4.drop(columns=['sub_grade'], inplace=True)


# ### emp_length and loan_status

# In[133]:


#Create the bar plot
plt.figure(figsize=(8, 4))
sns.countplot(x='emp_length', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Distribution of Loan Status by Employment Length')
plt.xticks(rotation=45, ha='right') 
plt.xlabel('Emp_length')
plt.ylabel('Count')

plt.show()


# In[134]:


for year in df4.emp_length.unique():
    print(f"{year} years in this position:")
    print(f"{df4[df4.emp_length == year].loan_status.value_counts(normalize=True)}")
    print('==========================================')


# #### The 10+ years category has the highest count for both 'Fully Paid' and 'Charged Off' loans. This suggests that a significant portion of the loan applicants are those who have been employed for a longer duration. However, both fully paid and charge off rates are extremely similar across all employment lengths. So we are going to drop the emp_length column.

# In[135]:


#Drop emp_length column
df4.drop(columns=['emp_length'], inplace=True)


# ### home_ownership and loan_status

# In[136]:


#Create the bar plot
plt.figure(figsize=(8, 4))
sns.countplot(x='home_ownership', hue='loan_status', palette = ['lightgreen', 'grey'], data=df4)

#Set the title and labels
plt.title('Distribution of Loan Status by Home Ownership')
plt.xticks(rotation=45, ha='right') 
plt.xlabel('home_ownership')
plt.ylabel('Count')

plt.show()


# In[137]:


df4['home_ownership'].value_counts()


# #### MORTGAGE and RENT are the most common categories in home ownership status, with MORTGAGE being the most frequent. This suggests that most borrowers either have a mortgage or rent their home. OWN is less common than MORTGAGE and RENT, but still represents a significant number of borrowers. Since there aren't too many values in ANY, NONE, OTHER, we will drop these classes.

# In[138]:


#Drop ANY, NONE, OTHER classes
df4 = df4[~df4['home_ownership'].isin(['ANY', 'NONE', 'OTHER'])]


# ### purpose and loan_status

# In[139]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='purpose', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Distribution of Loan Status by Purpose')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Purpose')
plt.ylabel('Count')

plt.show()


# In[140]:


df4['purpose'].value_counts()


# #### Upon reviewing the 'purpose' variable, it was observed that the majority of observations fell into categories such as DEBT CONSOLIDATION, CREDIT CARD, and HOME IMPROVEMENT, while other categories had significantly fewer observations. To simplify the analysis, only the top four categories were retained, and all other categories were grouped into a single category labeled 'Other.' This approach is commonly used to handle categorical variables with numerous categories. 

# In[141]:


#Since the primary reasons for obtaining a loan are for debt consolidation, credit card, home improvement, and other unspecified
#purposes, we will reclassify the remaining categories under a general 'other' category.

#List of categories to keep
categories_to_keep = ['debt_consolidation', 'credit_card', 'home_improvement']

#Reclassify all other categories into 'other'
df4['purpose'] = df4['purpose'].apply(lambda x: x if x in categories_to_keep else 'other')


# ### loan and addr_state

# In[142]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='addr_state', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Distribution of Loan Status by States')
plt.xticks(rotation=45, ha='right')
plt.xlabel('States')
plt.ylabel('Count')
plt.xticks(rotation=70)

plt.show()


# In[143]:


df4['addr_state'].value_counts()


# ### inq_last_6_mths and loan_staus

# In[144]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='inq_last_6mths', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('The number of inquiries in past 6 months ')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Inquiries')
plt.ylabel('Count')

plt.show()


# In[145]:


#Create a mapping dictionary
category_mapping = {
    0.0: 'none',
    1.0: 'one',
    2.0: 'two',
    3.0: 'three',
    4.0: 'four',
    5.0: 'five',
    6.0: 'six'
}

# Replace the integer values with string labels
df4['inq_last_6mths'] = df4['inq_last_6mths'].replace(category_mapping)


# In[146]:


df4['inq_last_6mths'].value_counts()


# In[147]:


#Combine 3, 4, 5, 6 categories into 2
df4['inq_last_6mths'] = df4['inq_last_6mths'].replace(['three', 'four', 'five', 'six'], 'two')

#Rename 2 to gt_2
df4['inq_last_6mths'] = df4['inq_last_6mths'].replace('two', 'two_or_more')


# ### collections_12_mths_ex_med and loan_staus

# In[148]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='collections_12_mths_ex_med', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Number of collections in 12 months excluding medical collections ')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Numbers')
plt.ylabel('Count')

plt.show()


# In[149]:


#Drop it
df4.drop(columns = ['collections_12_mths_ex_med'], inplace = True)


# ### acc_now_delinq and loan_staus

# In[150]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='acc_now_delinq', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('The number of accounts on which the borrower is now delinquent')
plt.xticks(rotation=45, ha='right')
plt.xlabel('numbers')
plt.ylabel('Count')

plt.show()


# In[151]:


#Drop it
df4.drop(columns = ['acc_now_delinq'], inplace = True)


# ### chargeoff_within_12_mths and loan_staus

# In[152]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='chargeoff_within_12_mths', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Number of charge-offs within 12 months')
plt.xticks(rotation=45, ha='right')
plt.xlabel('numbers')
plt.ylabel('Count')

plt.show()


# In[153]:


#Drop it
df4.drop(columns = ['chargeoff_within_12_mths'], inplace = True)


# ### num_tl_op_past_12m and loan_staus

# In[154]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='num_tl_op_past_12m', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Number of accounts opened in past 12 months')
plt.xticks(rotation=45, ha='right')
plt.xlabel('numbers')
plt.ylabel('Count')

plt.show()


# In[155]:


df4['num_tl_op_past_12m'].value_counts()


# In[156]:


#Relabel each category
#Create a mapping dictionary
category_mapping1 = {
    0.0: 'none',
    1.0: 'one',
    2.0: 'two',
    3.0: 'three',
    4.0: 'four',
    5.0: 'five',
    6.0: 'six'
}

# Replace the integer values with string labels
df4['num_tl_op_past_12m'] = df4['num_tl_op_past_12m'].replace(category_mapping1)


# ### pub_rec_bankruptcies and loan_status

# In[157]:


#Create bar chart
plt.figure(figsize=(8, 4))
sns.countplot(x='pub_rec_bankruptcies', hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])

#Set the title and labels
plt.title('Number of public record bankruptcies')
plt.xticks(rotation=45, ha='right')
plt.xlabel('numbers')
plt.ylabel('Count')

plt.show()


# In[158]:


df4['pub_rec_bankruptcies'].value_counts()


# In[159]:


#Create a mapping dictionary
category_mapping = {
    0.0: 'none',
    1.0: 'one',
}
# Replace the integer values with string labels
df4['pub_rec_bankruptcies'] = df4['pub_rec_bankruptcies'].replace(category_mapping)


# In[160]:


#Only keep vaule 'none' and 'one'
df4 = df4[df4['pub_rec_bankruptcies'].isin(['none', 'one'])]


# ### Bar charts for the rest category variabels

# In[161]:


#Bar charts
cat = ['verification_status', 'initial_list_status', 'application_type', 'disbursement_method', 'debt_settlement_flag' ] 

for column in cat:
    plt.figure(figsize=(8, 4))  
    sns.countplot(x=column, hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])  
    plt.title(f'Distribution of {column} by Loan Status')  
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()  
    plt.show()


# #### The features like 'application_type', 'disbursement_method', 'debt_settlement_flagmay' may not provide much predictive power because there is little variation for a model to learn from. So, we will be dropping them.

# In[162]:


#Drop columns
df4.drop(columns=['application_type', 'disbursement_method', 'debt_settlement_flag'], inplace=True)


# In[163]:


#Check 
df4.shape   #It was (222748, 67)


# # Feature Engineering

# ## New Features

# ### Date- Extract year ('last_pymnt_d')

# In[164]:


df4['last_pymnt_d'].unique()


# In[165]:


# Count the occurrences of each unique value in the column
value_counts = df4['last_pymnt_d'].value_counts()

# Plot the distribution using seaborn
plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')  # Change color palette if needed
plt.xlabel('last_pymnt_d')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')  # Rotate x-labels for better readability and align them to the right
plt.xticks(range(0, len(value_counts.index), 3), value_counts.index[::3])  # Show every 3rd tick to spread out
plt.tight_layout()  # Adjust layout
plt.show()


# In[166]:


#Extract
df4['last_pymnt_d_year'] = pd.to_datetime(df4['last_pymnt_d'], format='%b-%Y').dt.year

counts = df4['last_pymnt_d_year'].value_counts().sort_values(ascending=False)
print(counts)


# #### As observed, the "last_pymnt_d" variable only contains data from January to Febrary in 2019, rendering it insufficient to represent the entire year. Consequently, we opt to exclude January 2019 from our analysis. 
# #### Additionally, due to the relatively limited number of observations in 2012 and 2013 compared to other years,  we will focus on the "last_pymnt_d_year" values from 2014 to 2018. 
# #### This decision is based on the fact that the last payment date typically indicates whether the loan has been fully paid or charged off. By examining the payments made within this timeframe, we can effectively determine the status of the loan, particularly whether it has defaulted or not.

# In[167]:


#Drop orignal last_pymnt_d columns
df4 = df4.drop(columns=['last_pymnt_d'])


# In[168]:


df4 = df4[df4['last_pymnt_d_year'].between(2014, 2018)]


# In[169]:


#Convert it to categorical(object)
df4['last_pymnt_d_year'] = df4['last_pymnt_d_year'].astype('object')


# In[170]:


#See what categorical variables look like
df4.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# ## Modified Features

# In[171]:


#Bar charts
cat1 = df4.select_dtypes(include='object')

for column in cat1:
    plt.figure(figsize=(8, 4))  
    sns.countplot(x=column, hue='loan_status', data=df4, palette = ['lightgreen', 'grey'])  
    plt.title(f'Distribution of {column} by Loan Status')  
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()  
    plt.show()


# ## One-hot encoding

# In[172]:


#One-hot encoding
cat1_dummies = pd.get_dummies(cat1)
#Drop original categorical columns
df4_encoded = df4.drop(cat1.columns, axis=1)

df4 = pd.concat([df4_encoded, cat1_dummies], axis=1)


# In[173]:


df4.shape   #It was (220316, 51)


# In[174]:


#Check dataset see if we have successfully done one-hot encoding
df4.head(5)


# In[175]:


###########Make a copy 
df5 = df4.copy()


# # Feature Selection

# ## Correlations matrix

# In[176]:


corr=df4.corr()
plt.figure(figsize=(15, 15))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True)

# Set the title
plt.title('Correlation Heatmap for All Indep endent Variables')

# Show the plot
plt.show()


# In[177]:


#Define the threshold
threshold = 0.75  

#Find features that are highly correlated (absolute value)
highly_correlated_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)): 
        if abs(corr.iloc[i, j]) > threshold:
            col1 = corr.columns[i]
            col2 = corr.columns[j]
            highly_correlated_pairs.append((col1, col2, corr.iloc[i, j]))

#Print out highly correlated pairs with their correlation coefficient
for col1, col2, corr_value in highly_correlated_pairs:
    print(f"{col1} and {col2} have a correlation of {corr_value:.2f}")


# ## Correlation Analysis with target variable

# In[178]:


#Set up threshold
target_variable = 'loan_status'
threshold = 0.2 

#Find features with significant correlation with the target variable
significant_correlations = corr[target_variable].drop(target_variable).where(lambda x: abs(x) > threshold).dropna()

#Print
for feature, corr_value in significant_correlations.items():
    print(f"{feature} and {target_variable} have a correlation of {corr_value:.2f}")


# In[179]:


#Filter features based on absolute correlation matrix (abs correlation > 0.2)
#We noticed that total_pymnt and loan_status have a correlation of -0.42, but total_rec_prncp and total_pymnt are highly 
#correlated, so we only keep one of them
features = ['int_rate', 'total_rec_prncp', 'last_pymnt_amnt', 'loan_status'] 

#Create a new DataFrame 
df5 = df4[features]

df5.head()


# ## Correaltion Matrix for selected features

# In[180]:


#Calculate the correlation matrix for the selected features
corr_subset = df5.corr()

#Plot the heatmap
plt.figure(figsize=(10, 8))
heatmap_subset = sns.heatmap(corr_subset, vmin=-1, vmax=1, annot=True, cmap= 'YlGnBu')

#Set the title
plt.title('Correlation Heatmap for Specific Features')

#Show the plot
plt.show()


# In[181]:


df5.shape    # It was (211209, 126)


# # Predicting modeling (Logistic Regression, Random Forest, Decision Tree )

# ## Split data into training and test set
# ## We will use three features (int_rate, total_rec_prncp, and last_pymnt_amnt)

# In[182]:


X = df5.drop('loan_status', axis=1)
y = df5['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)


# ## Train the models

# In[183]:


#Create a pipeline that standardizes the data then applies logistic regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('logistic', LogisticRegression(max_iter=1000))])

pipeline.fit(X_train, y_train)


# In[184]:


#Random forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=1234)
random_forest_model.fit(X_train, y_train)


# In[185]:


#Decision tree
decision_tree_model = DecisionTreeClassifier(random_state=1234)
decision_tree_model.fit(X_train, y_train)


# ## Make predictions

# In[186]:


#Predict on the test set
y_pred_logistic = pipeline.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_dt = decision_tree_model.predict(X_test)


# ## Print out coefficients for logistic model

# In[187]:


feature_names = X.columns
coefficients = pipeline.named_steps['logistic'].coef_
coef_df = pd.DataFrame(coefficients.flatten(), index=feature_names, columns=['Coefficient'])

print(coef_df)


# ## Evaluate the models

# In[188]:


print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


# #### The classification reports for Logistic Regression, Random Forest, and Decision Tree models show very high performance across all metrics for this particular dataset. Here are some insights:
# 
# The Random Forest model has the best performance across all metrics, making it the most reliable model of the three for predicting loan status.
# 
# The Logistic Regression model, while generally good, does not perform as well as the tree-based models, particularly in terms of precision for class 1 (charged off loans).
# 
# The Decision Tree model also performs well, with a slight edge over Logistic Regression but is not quite as effective as the Random Forest model.
# 
# The high recall for class 1 in the Logistic Regression model is notable, as it indicates a strong ability to identify the majority of charged off loans, which could be particularly valuable in a lending scenario where identifying potential defaults is critical.
# 
# The high overall accuracy of the Random Forest and Decision Tree models indicates that they are both quite effective at distinguishing between the two classes of loan status.

# ## Feature importance

# ### Random Forest

# In[189]:


#Get feature importances
importances = random_forest_model.feature_importances_

#Create a DataFrame to view the features and their importance scores
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

#Sort the DataFrame to see the most important features at the top
features_df.sort_values(by='Importance', ascending=False, inplace=True)

print(features_df)


# In[190]:


#Visualize the feature importances for Random Forest
plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.gca().invert_yaxis()  
plt.show


# ### Decision tree

# In[191]:


#Get feature importances
impt = decision_tree_model.feature_importances_

#Create a DataFrame to view the features and their importance scores
features_df1 = pd.DataFrame({'Feature': X.columns, 'Importance': impt})

#Sort the DataFrame to see the most important features at the top
features_df1.sort_values(by='Importance', ascending=False, inplace=True)

print(features_df1)


# In[192]:


#Visualize the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features_df1['Feature'], features_df1['Importance'], color='steelblue')
plt.xlabel('Importance')
plt.gca().invert_yaxis()  
plt.show


# ## Model Comparison

# ### Confusion Matrix

# In[193]:


#Confusion matrix
y_preds = {
    "Logistic Regression": y_pred_logistic,
    "Random Forest": y_pred_rf,
    "Decision Tree": y_pred_dt
}

for model_name, y_pred in y_preds.items():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='GnBu', cbar=True)
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Prediction')
    plt.ylabel('True')
    plt.show()


# ### ROC curve

# In[194]:


#Calculate ROC curve points
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_pred_logistic)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)

#ROC-AUC scores for the plots
roc_auc_logistic = roc_auc_score(y_test, y_pred_logistic)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)

#Create plots for ROC curves
plt.figure(figsize=(10, 8))

#Plot Logistic Regression ROC
plt.plot(fpr_logistic, tpr_logistic, label=f'Logistic Regression (area = {roc_auc_logistic:.2f})')

#Plot Random Forest ROC
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (area = {roc_auc_rf:.2f})')

#Plot Decision Tree ROC
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (area = {roc_auc_dt:.2f})')

#Plot Base Rate ROC
plt.plot([0, 1], [0, 1], linestyle='--', label='Base Rate')

#Customizing the plot
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

plt.show()


# #### The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test. Hence, both the Random Forest and Decision Tree models perform slightly better than Logistic Regression in this case, as indicated by their curves being closer to the top-left corner.

# ## Now, we are going to drop ''int_rate'' see if model accuracy changes

# In[195]:


#Selecting the features and target
X = df5[['total_rec_prncp', 'last_pymnt_amnt']]
y = df5['loan_status']

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

#Create a pipeline that standardizes the data then applies logistic regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('logistic', LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)

#Initialize the Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=1234)
random_forest_model.fit(X_train, y_train)

#Decision tree
decision_tree_model = DecisionTreeClassifier(random_state=1234)
decision_tree_model.fit(X_train, y_train)

#Predict on the test set
y_pred_logistic = pipeline.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_dt = decision_tree_model.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


# ## Removing features ''int_rate'' and "total_rec_prncp" see if model accuracy changes

# In[196]:


#Selecting the features and target
X = df5[['last_pymnt_amnt']]
y = df5['loan_status']

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

#Create a pipeline that standardizes the data then applies logistic regression
pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('logistic', LogisticRegression(max_iter=1000))])
pipeline.fit(X_train, y_train)

#Initialize the Random Forest Classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=1234)
random_forest_model.fit(X_train, y_train)

#Decision tree
decision_tree_model = DecisionTreeClassifier(random_state=1234)
decision_tree_model.fit(X_train, y_train)

#Predict on the test set
y_pred_logistic = pipeline.predict(X_test)
y_pred_rf = random_forest_model.predict(X_test)
y_pred_dt = decision_tree_model.predict(X_test)

print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logistic))
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))


# ## Correlation Matrix Explanation for Natural Disasters

# In[197]:


target_variable = 'loan_status'

#Find features with significant correlation with the target variable
significant_correlations = corr[target_variable].drop(target_variable).where(lambda x: abs(x) > threshold)

#Print
for feature, corr_value in significant_correlations.items():
    print(f"{feature} and {target_variable} have a correlation of {corr_value:.2f}")


# In[198]:


# 'significant_correlations' is a dictionary containing correlation values
significant_correlations = {
    'addr_state_AK': 0.00, 'addr_state_AL': 0.01, 'addr_state_AR': 0.01, 'addr_state_AZ': -0.00,
    'addr_state_CA': 0.00, 'addr_state_CO': -0.02, 'addr_state_CT': -0.01, 'addr_state_DC': -0.01,
    'addr_state_DE': 0.00, 'addr_state_FL': 0.01, 'addr_state_GA': -0.01, 'addr_state_HI': 0.01,
    'addr_state_ID': -0.00, 'addr_state_IL': -0.01, 'addr_state_IN': 0.00, 'addr_state_KS': -0.01,
    'addr_state_KY': 0.00, 'addr_state_LA': 0.01, 'addr_state_MA': -0.00, 'addr_state_MD': 0.01,
    'addr_state_ME': -0.01, 'addr_state_MI': -0.00, 'addr_state_MN': -0.00, 'addr_state_MO': 0.00,
    'addr_state_MS': 0.01, 'addr_state_MT': -0.01, 'addr_state_NC': 0.00, 'addr_state_ND': 0.00,
    'addr_state_NE': 0.01, 'addr_state_NH': -0.02, 'addr_state_NJ': 0.01, 'addr_state_NM': 0.01,
    'addr_state_NV': 0.01, 'addr_state_NY': 0.03, 'addr_state_OH': 0.00, 'addr_state_OK': 0.01,
    'addr_state_OR': -0.02, 'addr_state_PA': 0.01, 'addr_state_RI': -0.01, 'addr_state_SC': -0.01,
    'addr_state_SD': 0.00, 'addr_state_TN': 0.00, 'addr_state_TX': -0.00, 'addr_state_UT': -0.01,
    'addr_state_VA': 0.00, 'addr_state_VT': -0.01, 'addr_state_WA': -0.02, 'addr_state_WI': -0.01,
    'addr_state_WV': -0.01, 'addr_state_WY': -0.00,
    'last_pymnt_d_year_2014': 0.04, 'last_pymnt_d_year_2015': 0.03, 'last_pymnt_d_year_2016': 0.07,
    'last_pymnt_d_year_2017': 0.01, 'last_pymnt_d_year_2018': -0.11
}

# Create a DataFrame from significant_correlations
corr_df = pd.DataFrame(significant_correlations.items(), columns=['Feature', 'Correlation'])

# Reshape DataFrame to prepare for heatmap
corr_matrix = corr_df.set_index('Feature')

# Plot heatmap
plt.figure(figsize=(12, 12))  # Change the values here to adjust width and height
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', center=0, fmt=".2f")
plt.show()


# # Final Evaluation

# **Coefficients in the Logistic Regression:**
# * 'int_rate' has a positive coefficient of 0.78 (indicating that as the interest rate increases, the likelihood of loan default also increases)
# * 'total_rec_prncp' (-1.49) and 'last_pymnt_amnt' (-7.63) have negative coefficients (indicating that higher values of these variables are associated with a lower likelihood of default)
# 
# This supports the research question by providing insights into the relationship between financial attributes and loan defaults.

# **Feature Importance Score for Random Forest and Decision Tree models:**
# 
# *Random Forest*
# * 'last_pymnt_amnt' (0.54) is the most important feature
# * 'total_rec_prncp' (0.38) is the next most important feature
# * 'int_rate' (0.08) is the least important feature
# 
# *Decision Tree*
# * 'last_pymnt_amnt' (0.67) is the most important feature
# * 'total_rec_prncp' (0.30) is the next most important feature
# * 'int_rate' (0.03) is the least important feature
# 
# Overall, these scores suggest that 'last_pymnt_amnt' is the most important feature for predicting loan defaults, followed by 'total_rec_prncp,' and 'int_rate'
# 
# This supports the research question by identifying that 'last_pymnt_amnt,' a financial attribute, is highly influential in predicting loan default. 

# **Research Question:**
# 
# Financial attributes such as the last payment amount ('last_pymnt_amnt') and the total received principal ('total_rec_prncp') are significant predictors of loan defaults, with higher values of these attributes associated with a lower likelihood of default, as determined by Advanced Machine Learning techniques applied to loans issued between 2012 and 2019.

# **Alternative Hypothesis (H1):**
# 
# We accept the Alternative Hypothesis (H1) as there is evidence suggesting a significant relationship between at least one of the borrower details, financial attributes, or potential interactions and loan defaults for loans issued between 2012 and 2019 using Advanced Machine Learning techniques.

# **Null Hypothesis (H0):**
# 
# We reject the Null Hypothesis (H0) as evidence supports the existence of a significant relationship between the identified features and loan defaults.

# **Best Performing Model:**
# 
# The Random Forest model where we excluded the variable 'int_rate' from the analysis, appears to be the best for predicting loan defaults, as it achieves the highest overall performance in metrics in precision, recall, F1-score, and accuracy. 
