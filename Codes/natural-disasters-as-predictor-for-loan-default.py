#!/usr/bin/env python
# coding: utf-8

# *Keywords:* Loan Default, Python, Machine Learning, Predictive Modeling, Logistic Regression, Random Forest, Decision Tree, Correlation Matrix

# ## Project Overview and Objectives

# **Resaerch Question:** 
# 
# Could the occurence of natural disasters, as witness in the United States from 2012 to 2019, be used as a predictor for loans defaults?
# 
# **Alternative Hypothesis (H1):** 
# 
# Suggests that the occurence of natural disasters in the United States from 2012-2019 is a significant predictor for loan default.
# 
# **Null Hypothesis (H0):** 
# 
# Suggests that the occurence ofnatural disasters in the United States from 2012 to 2019 has no predictive power for loan defualts.
# 
# **Methodology**
# 
# This project is an extension to the "Predicting What Features Influence Loan Default" notebook. It employs a structured methodology consisting of several key stages: data cleaning, Exploratory Data Analysis (EDA), feature engineering, and feature selection. Following those steps, predictive modeling is conducted utilizing Logistic Regression, Random Forest, and Decision Tree algorithms. Finally, the project culminates with a comprehensive feature importance analysis, using techniques such as Confusion Matrix and ROC Curve evaluation. 

# **Dataset:**
# - Lending Club Loan Data; located on Kaggle
# - US Natural Disaters Declarations; located on Kaggle
# 
# **First research question to this project:** Analyzing what factors influence loan default in the U.S. Located on Kaggle "Predicting Factors Influencing Loan Default in U.S"
# 
# [Kaggle.com: rosaaestrada - Predicting Factors Influencing Loan Default in U.S](https://www.kaggle.com/code/rosaaestrada/predicting-factors-influencing-loan-default-in-u-s)

# ## Importing the data

# In[1]:


#Import libaries
get_ipython().run_line_magic('matplotlib', 'inline')

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn import under_sampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:


#Loading loan dataset
df_loan = pd.read_csv('/kaggle/input/lending-club-loan-data-csv/loan.csv', low_memory = False)


# In[3]:


#Select features
cols = ['last_pymnt_amnt', 'total_rec_prncp', 'last_pymnt_d', 'loan_status']

df_loan = df_loan[cols]
df_loan.head(5)


# In[4]:


# Set the size of the plot
plt.figure(figsize=(12, 8))

# Define selected categories for specific variables
selected_ticks = {
    'last_pymnt_d': ['Aug-2017', 'December-2018', 'May-2014'],
    'loan_status': ['Current', 'Fully Paid', 'Charged Off', 'Default']
}

# Iterate through each variable and create a histogram
for i, col in enumerate(df_loan.columns):
    plt.subplot(2, 2, i+1)  # Create subplots in a 2x2 grid
    df_loan[col].hist(color='steelblue', bins=20)  # Create histogram
    plt.title(col)  # Set title for each subplot
    
    # Show only selected categories for specific variables
    if col in selected_ticks:
        plt.xticks(selected_ticks[col])  # Show selected category names under the plots
    else:
        plt.xticks(rotation=0)  # Rotate category names for better readability

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[5]:


df_loan.shape


# # Data Cleaning

# ## Missing Values

# In[6]:


#Checking missing values and their percentage
missing_values = df_loan.isnull().sum()
missing_percentage = (missing_values / len(df_loan)) * 100

#Combining both into a DataFrame
missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})

missing_df


# In[7]:


#Imputing with the most frequent value(mode)
df_loan['last_pymnt_d'].fillna(df_loan['last_pymnt_d'].mode()[0], inplace=True)


# In[8]:


#list numeric features along with their statistical description
des = df_loan.select_dtypes(exclude=['object']).describe().round(decimals=2).transpose()
print(des.to_string())


# In[9]:


#Plotting histograms
df_loan[['last_pymnt_amnt', 'total_rec_prncp']].hist(bins=50, figsize=(20, 15), layout=(2,2), color='skyblue', 
                                                     edgecolor='black')


# In[10]:


#Find the 25 and 75 percentiles
q1, q3 = np.percentile(df_loan['last_pymnt_amnt'],[25, 75])

#Find the IQR and the cutoffs for the outliers
IQR = q3 - q1
lowerOutliers= q1 - 1.5*IQR  
upperOutliers= q3 + 1.5*IQR   
print(lowerOutliers, upperOutliers)


# In[11]:


#Drop outliers in last_pymnt_amnt
df_loan = df_loan[df_loan['last_pymnt_amnt'] < 8374.4525 ]


# ## Check Duplicate Rows

# In[12]:


#check if there are some duplicate rows
number_of_duplicates = df_loan.duplicated().sum()

print(f"Number of duplicate rows: {number_of_duplicates}")


# In[13]:


df_loan.drop_duplicates(inplace=True)


# In[14]:


df_loan.shape


# # Exploratory Data Analysis (EDA)

# ## Target Variable Analysis

# In[15]:


#Drop other classes for the target variable
df_loan = df_loan[df_loan['loan_status'].isin(['Fully Paid', 'Charged Off'])]


# In[16]:


#Bar chart for target variable loan_status
loan_status_counts = df_loan['loan_status'].value_counts()

plt.figure(figsize=(8, 4))
plt.pie(loan_status_counts.values, labels=loan_status_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'grey'])
plt.title('Loan Status Distribution')
plt.show()  


# ### Target Variable Imbalance

# In[17]:


#Separate features and target variable
X = df_loan.drop('loan_status', axis=1)
y = df_loan['loan_status']

#Initialize and fit to the data
rus = RandomUnderSampler(random_state = 123)
X_resampled, y_resampled = rus.fit_resample(X, y)

#Plot the pie chart
loan_status_resampled_counts = y_resampled.value_counts()

plt.figure(figsize=(8, 4))
plt.pie(loan_status_resampled_counts, labels=loan_status_resampled_counts.index, autopct='%1.1f%%',
        colors=['grey', 'lightgreen'])
plt.title('Loan Status Distribution After Resampling')
plt.show()


# In[18]:


#Combine X_resampled and y_resampled back into a new DataFrame
resampled_df = pd.concat([X_resampled, y_resampled], axis=1)

#Verify
print(resampled_df['loan_status'].value_counts())


# In[19]:


#Re-label target variable: 'fully paid' to 0 and 'charged off' to 1
resampled_df['loan_status'] = resampled_df['loan_status'].replace({'Fully Paid': 0, 'Charged Off': 1})


# In[20]:


#Verify
resampled_df['loan_status'].value_counts()


# In[21]:


df1 = resampled_df.copy()


# ### Categorical Variables

# In[22]:


df1['last_pymnt_d'].unique()


# In[23]:


#Extract
df1['year'] = pd.to_datetime(df1['last_pymnt_d'], format='%b-%Y').dt.year

counts = df1['year'].value_counts().sort_values(ascending=False)
print(counts)


# In[24]:


#Drop orignal last_pymnt_d columns
df1 = df1.drop(columns=['last_pymnt_d'])


# In[25]:


df1 = df1[df1['year'].between(2016, 2018)]


# ### Focus on the years between 2016-2018

# In[26]:


df1.head(5)


# In[27]:


df1.info()


# ### df_loan dataset is ready for merging

# # Load Disaster Dataset

# In[28]:


#Loading disaster dataset
df_dis = pd.read_csv('/kaggle/input/us-natural-disaster-declarations/us_disaster_declarations.csv')


# In[29]:


df_dis.head(5)


# In[30]:


df_dis.shape


# In[31]:


#Numbers of each data type
data_types_count = df_dis.dtypes.value_counts()

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


# In[32]:


#Print data type counts
data_types_count


# In[33]:


df_dis.info()


# In[34]:


df_dis.columns.tolist()


# ## EDA: ND

# ### Examine Variable 'state'

# In[35]:


#Select features
features = ['incident_type', 'incident_begin_date', 'state']

df_dis = df_dis[features]


# In[36]:


#check number of values in each state
counts = df_dis['state'].value_counts(ascending=False)

counts


# In[37]:


# Extract the counts for each state
counts = df_dis['state'].value_counts(ascending=False)

# Create a custom color palette using 'viridis'
colors = plt.cm.viridis.colors

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar plot with the custom color palette
counts.plot(kind='bar', color=colors)

# Set title and labels
plt.xlabel('State')
plt.ylabel('Count')

# Rotate x-axis labels for better readability and increase font size
plt.xticks(rotation=90, fontsize=8)

# Show plot
plt.show()


# ### Randomly select 6 states with 1,000 or more disaster counts

# In[38]:


# set seed for reproducibility
random.seed(42)

# List of categories
categories = ['TX', 'MO', 'KY', 'VA', 'LA', 'OK', 'FL', 
              'GA', 'NC', 'PR', 'MS', 'IA', 'KS', 'AL', 
              'TN', 'CA', 'AR', 'MN', 'NY', 'NE', 'IN', 
              'SD', 'ND', 'IL', 'OH', 'PA', 'WV', 'ME']

# Randomly select a category
random_categories = random.sample(categories, 3)

print("Randomly selected category:", random_categories)


# In[39]:


# Filter the DataFrame to retain only the desired categories in the 'state' variable
desired_categories = ['IN', 'VA', 'TX']
df_dis = df_dis[df_dis['state'].isin(desired_categories)]


# In[40]:


#check number of values in each type
counts = df_dis['state'].value_counts(ascending=False)

counts


# In[41]:


# Extract the counts for each state
counts = df_dis['state'].value_counts(ascending=False)

# Create a custom color palette using 'viridis'
colors = plt.cm.viridis.colors

# Set the size of the plot
plt.figure(figsize=(12, 6))

# Plot the bar plot with the custom color palette
counts.plot(kind='bar', color=colors)

# Set title and labels
plt.xlabel('State')
plt.ylabel('Count')

# Rotate x-axis labels for better readability and increase font size
plt.xticks(rotation=90, fontsize=8)

# Show plot
plt.show()


# ### Examine Variable 'incident_type'

# In[42]:


#check number of values in each type
counts = df_dis['incident_type'].value_counts(ascending=False)

counts


# In[43]:


#counts for each category
counts = df_dis['incident_type'].value_counts()

counts.plot(kind='bar', color= 'steelblue')
plt.title('Incident Type Frequency')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Incident Type')
plt.ylabel('Count')
plt.show()


# ### Examine Variable 'incident_begin_date'

# In[44]:


df_dis['incident_begin_date'].unique()


# In[45]:


#Convert the 'incident_begin_date' to datetime, then extract the year.
df_dis['year'] = pd.to_datetime(df_dis['incident_begin_date']).dt.year

df_dis.drop(columns = ['incident_begin_date'], inplace = True)


# In[46]:


df_dis = df_dis[df_dis['year'].between(2016, 2018)]


# In[47]:


df_dis.head(5)


# In[48]:


df_dis.shape


# # Merge two dataset

# In[49]:


#merge with feature year
merged_df = pd.merge(df1, df_dis, on='year')


# In[50]:


merged_df.head() #insert state in code above


# In[51]:


merged_df.shape


# In[52]:


#check if there are some duplicate rows
number_of_duplicates = merged_df.duplicated().sum()

print(f"Number of duplicate rows: {number_of_duplicates}")


# In[53]:


merged_df.drop_duplicates(inplace=True)


# In[54]:


#check if there are missing values
merged_df.isnull().sum()


# ## One-hot encoding

# In[55]:


merged_df = pd.get_dummies(merged_df, columns=['incident_type', 'state'])


# In[56]:


merged_df.shape


# ## Correlation Matrix Heatmap

# In[57]:


corr = merged_df. corr()
plt.figure(figsize=(15, 15))
heatmap = sns.heatmap(corr, vmin=-1, vmax=1, annot=True, cmap='BrBG', cbar=True)
plt.show()


# ## Correlation Matrix on correlated pairs

# In[58]:


# Find features that are correlated
correlated_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)): 
        col1 = corr.columns[i]
        col2 = corr.columns[j]
        corr_value = corr.iloc[i, j]
        correlated_pairs.append((col1, col2, corr_value))

# Print out correlated pairs with their correlation coefficient
print("Correlated pairs with their correlation coefficient:")
for col1, col2, corr_value in correlated_pairs:
    print(f"{col1} and {col2} have a correlation of {corr_value:.2f}")


# ## Correlation Matrix with the target variables

# In[59]:


# Find features that are correlated with the target variable 'loan_status'
correlated_with_loan_status = []
for column in merged_df.columns:
    if column != 'loan_status':
        corr_value = merged_df['loan_status'].corr(merged_df[column])
        correlated_with_loan_status.append((column, corr_value))

# Print out correlated pairs with their correlation coefficient
for feature, corr_value in correlated_with_loan_status:
    print(f"{feature} and loan_status have a correlation of {corr_value:.2f}")


# In[60]:


# Calculate correlations
correlation_matrix = merged_df.corr()

# Filter correlations with 'loan_status'
correlation_with_loan_status = correlation_matrix['loan_status'].drop('loan_status')

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_with_loan_status.to_frame(), cmap='BrBG', annot=True, fmt=".2f", vmin=-1, vmax=1, cbar=True)
plt.title("Correlation with loan_status")
plt.xlabel("Features")
plt.ylabel("Loan Status")
plt.show()


# # Predictive Model: Training, testing, and predicting

# In[61]:


X = merged_df.drop('loan_status', axis=1)
y = merged_df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)


# In[62]:


#Random forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=1234)
random_forest_model.fit(X_train, y_train)


# In[63]:


y_pred = random_forest_model.predict(X_test)


# In[64]:


print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred))


# ## Feature Importance

# In[65]:


#Get feature importances
importances = random_forest_model.feature_importances_

#Create a DataFrame to view the features and their importance scores
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

#Sort the DataFrame to see the most important features at the top
features_df.sort_values(by='Importance', ascending=False, inplace=True)

print(features_df)


# In[66]:


#Visualize the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Importance'], color='DarkGreen')
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  
plt.show


# # Final Evaluation

# **Feature Importance score for the Random Forest model:** 
# * 'last_pymnt_amnt' contains a score of 0.54
# * 'total_rec_prncp' contains a score of 0.45 
# * 'year' contains a score of 0.009
# * 'incident_type' contains a score between 0.0001 to 0.0009
# * 'state' contains a score between 0.0001 to 0.0004
# 
# The feature importance score suggests that 'last_pymnt_amnt' and 'total_rec_prncp,' are the most important features for predicting loan defaults, with the natural disaster variables far behind.

# **Research Question:**
# 
# The occurence of natural disasters in the United States from 2012 to 2019 cannot be used as a predictor for loan defaults, as there is no correlation between natural disasters and loan status in the dataset.

# **Alternative Hypothesis (H1):**
# 
# This results in rejecting the Alternative Hypothesis (H1) as the evidence suggests that the occurrence of natural disasters in the United States from 2012 to 2019 is not significant.

# **Null Hypothesis (H0):**
# 
# We accept the Null Hypothesis (H0), as the evidence supports the conclusion that natural disasters in the specified period have no predictive power for loan defaults.
