# Final Evaluation
1️⃣ Research Question

### Coefficients in the Logistic Regression supports the research question by providing insights into the relationship between financial attributes and loan defaults.
- 'int_rate' has a positive coefficient of 0.78 (indicating that as the interest rate increases, the likelihood of loan default also increases)
- 'total_rec_prncp' (-1.49) and 'last_pymnt_amnt' (-7.63) have negative coefficients (indicating that higher values of these variables are associated with a lower likelihood of default)

### Feature Importance Score for Random Forest and Decision Tree Models
- These scores suggest that 'last_pymnt_amnt' (RF=0.54 & DT=0.67) is the most important feature for predicting loan defaults, folled by 'total_rec_prncp' (RF=0.38 & DT=0.30), and 'int_rate'(RF=0.08 & DT=0.03)


Answer to 1️⃣ Research Question:
- Financial attributes such as the last payment amount ('last_pymnt_amnt') and the total received principal ('total_rec_prncp') are significant predictors of loan defaults, with higher values of these attributes associated with a lower likelihood of default, as determined by Advanced Machine Learning techniques applied to loans issued between 2012 and 2019.

🔸 **Best Performing Model:** 
- The Random Forest model where we excluded the variable 'int_rate' from the analysis, appears to be the best for predicting loan defaults, as it achieves the highest overall performance in metrics in precision, recall, F1-score, and accuracy.



