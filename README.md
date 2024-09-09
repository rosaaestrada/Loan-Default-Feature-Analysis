# Predicting Factors Influencing Loan Default in the U.S.

*This project was completed as my thesis for my Masters degree at National University*

### Data
**Lending Club Loan Data:** Located on [Kaggle.com - Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv)
- Loan data for all loans issued through 2007-2015
- **About dataset:** Lending Club is a peer-to-peer Lending company based in the US. They match people looking to invest money with people looking to borrow money. When investors invest their money through Lending Club, this money is passed onto borrowers, and when borrowers pay their loans back, the capital plus the interest passes on back to the investors.

**US Natural Disasters Declarations:** Located on [Kaggle.com - US Natural Disasters Declarations](https://www.kaggle.com/datasets/headsortails/us-natural-disaster-declarations)
- Natural disaster data from the Federal Emergency Management agency from 1953-2023
- **About dataset:** High-level summary of all federally declared disasters. Note, that the data also includes biological disasters, in particular declarations made in response to the ongoing Covid-19 pandemic.

------------------------------------------------------------------------------------------------------------------------

### Project Overview and Objectives

#### 1️⃣ First Research Question and Hypotheses
🔸 **Research Question:** 

What borrower details, financial attributes, and potential interactions predict loan default for loans issued between 2012 and 2019 using Advanced Machine Learning techniques?

🔸 **Alternative Hypothesis (H1):** 

Suggests that there is a significant between at least one of the borrower details, financial attributes, or potential interactions and loan defaults for loans between 2012 and 2019 using Advanced Machine Learning techniques.

🔸 **Null Hypothesis (H0):** 

Suggests no significant relationship between borrower details, financial attributes, or potential interactions and loan defaults for loans issued between 2012 and 2019 using Advanced Machine Learning techniques.

#### 2️⃣ Second Research Question and Hypotheses
🔸 **Research Question:** 

Could the occurence of natural disasters, as witness in the United States from 2012 to 2019, be used as a predictor for loans defaults?

🔸 **Alternative Hypothesis (H1):** 

Suggests that the occurence of natural disasters in the United States from 2012-2019 is a significant predictor for loan default.

🔸 **Null Hypothesis (H0):** 

Suggests that the occurence ofnatural disasters in the United States from 2012 to 2019 has no predictive power for loan defualts.

------------------------------------------------------------------------------------------------------------------------
**Methodology**

This project employs a structured methodology consisting of several key stages: data cleaning, Exploratory Data Analysis (EDA), feature engineering, and feature selection. Following those steps, predictive modeling is conducted utilizing Logistic Regression, Random Forest, and Decision Tree algorithms. Finally, the project culminates with a comprehensive feature importance analysis, using techniques such as Confusion Matrix and ROC Curve evaluation.

------------------------------------------------------------------------------------------------------------------------
**Kaggle Notebooks:**
- [Kaggle.com: rosaaestrada - Predicting Factors Influencing Loan Default in U.S](https://www.kaggle.com/code/rosaaestrada/predicting-factors-influencing-loan-default-in-u-s)
- [Kaggle.com: rosaaestrada - Natural Disasters as Predictor for Loan Default](https://www.kaggle.com/code/rosaaestrada/natural-disasters-as-predictor-for-loan-default)

------------------------------------------------------------------------------------------------------------------------

### Final Evaluation

**1️⃣ Research Question:** 

**Coefficients in the Logistic Regression**
- 'int_rate' has a positive coefficient of 0.78 (indicating that as the interest rate increases, the likelihood of loan default also increases)
- 'total_rec_prncp' (-1.49) and 'last_pymnt_amnt' (-7.63) have negative coefficients (indicating that higher values of these variables are associated with a lower likelihood of default)

*This supports the research question by providing insights into the relationship between financial attributes and loan defaults.*

**Feature Importance Score for Random Forest and Decision tree Models**

*Random Forest*
- 'last_pymnt_amnt' (0.54) is the most important feature
- 'total_rec_prncp' (0.38) is the next most important feature
- 'int_rate' (0.08) is the least important feature

*Decision Tree*
- 'last_pymnt_amnt' (0.67) is the most important feature
- 'total_rec_prncp' (0.30) is the next most important feature
- 'int_rate' (0.03) is the least important feature

*Overall, these scores suggest that 'last_pymnt_amnt' is the most important feature for predicting loan defaults, followed by 'total_rec_prncp,' and 'int_rate'*

*This supports the research question by identifying that 'last_pymnt_amnt,' a financial attribute, is highly influential in predicting loan default.*

------------------------------------------------------------------------------------------------------------------------
**1️⃣Research Question:** 

Financial attributes such as the last payment amount ('last_pymnt_amnt') and the total received principal ('total_rec_prncp') are significant predictors of loan defaults, with higher values of these attributes associated with a lower likelihood of default, as determined by Advanced Machine Learning techniques applied to loans issued between 2012 and 2019.

🔸 **Alternative Hypothesis (H1):**

We accept the Alternative Hypothesis (H1) as there is evidence suggesting a significant relationship between at least one of the borrower details, financial attributes, or potential interactions and loan defaults for loans issued between 2012 and 2019 using Advanced Machine Learning techniques.

🔸 **Null Hypothesis (H0):**

We reject the Null Hypothesis (H0) as evidence supports the existence of a significant relationship between the identified features and loan defaults.

🔸 **Best Performing Model:**

The Random Forest model where we excluded the variable 'int_rate' from the analysis, appears to be the best for predicting loan defaults, as it achieves the highest overall performance in metrics in precision, recall, F1-score, and accuracy.

------------------------------------------------------------------------------------------------------------------------

### Final Evaluation 2️⃣ Research Question

**Feature Importance score for the Random Forest Model**
- 'last_pymnt_amnt' contains a score of 0.54
- 'total_rec_prncp' contains a score of 0.45
- 'year' contains a score of 0.009
- 'incident_type' contains a score between 0.0001 to 0.0009
- 'state' contains a score between 0.0001 to 0.0004

*The feature importance score suggests that 'last_pymnt_amnt' and 'total_rec_prncp,' are the most important features for predicting loan defaults, with the natural disaster variables far behind.*

**2️⃣Second Research Question:**

The occurence of natural disasters in the United States from 2012 to 2019 cannot be used as a predictor for loan defaults, as there is no correlation between natural disasters and loan status in the dataset.

🔸 **Alternative Hypothesis (H1):**

This results in rejecting the Alternative Hypothesis (H1) as the evidence suggests that the occurrence of natural disasters in the United States from 2012 to 2019 is not significant.

🔸 **Null Hypothesis (H0):**

We accept the Null Hypothesis (H0), as the evidence supports the conclusion that natural disasters in the specified period have no predictive power for loan defaults.
