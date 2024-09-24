# Visualizations

## ROC Curve
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/ROC%20curve.png?raw=true" alt="" width="500" height="500">

- The ROC Curve shows that Random Forest (AUC = 0.97) and Decision Tree (AUC = 0.96) models perform significantly better at distibguishing between loan default outcomes compared to Logistic Regression (AUC = 0.87), with all models outperforming the base rate.

## Confusion Matrix: Logistic Regression, Random Forest, and Decision Tree
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Confusion%20Matrix%20Logistic%20Regression.png?raw=true" alt="Confusion Matrix: Logistic Regression" width="400" height="300"><img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Confusion%20Matrix%20Random%20Forest.png?raw=true" alt="Confusion Matrix: Random Forest" width="400" height="300"><img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Confusion%20Matrix%20Decision%20Tree.png?raw=true" alt="Confusion Matrix: Decision Tree" width="400" height="300">

- Logistic Regression struggles slightly with correctly classifying non-default loans (0), with more false positives (4,842), while it performs better in identifying loan defaults (1) with a good balance of true positives (24,799).
- Random Forest shows much stronger perfromance, correctly classifying both non-default loans (25,149) and loan defaults (25,904) with minimal errors, evidenced by the low number of false positives (961) and false negatives (789).
- Decision Tree demonstrates a strong perfromance in distinguishing between non-default loans (0) and loan defaults (1), it has high true positive (25,552) and true negatives (25,055), however, it exhibits a slightly higher rate of false negatives (1,141) compared to false positives (1,055).

## Feature Importance: Random Forest and Decision Tree
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Feautre%20Importance%20Random%20Forest.png?raw=true" alt="Feature Importance: Random Forest" width="600" height="300">
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Feature%20Importance%20Decision%20Tree.png?raw=true" alt="Feature Importance: Decision Tree" width="600" height="300">

- Random Forest and Decision Tree: 'last_pymnt_amnt' is the most influential feature, followed by 'total_rec_prncp' and 'int_rate'.

## Correlation Matrix Heatmap of selected features
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Corr%20Matrix%20of%20selected%20features.png?raw=true" alt="Correlation Matrix Heatmap of selected features" width="500" height="500">

- The heatmap reveals positive correlations between certain features, such as 'last_pymnt_amnt', 'int_rate' and 'total_rec_prncp'
- Features in the light blue shades exhibit little to no correlations

## Correaltion Matrix Heatmap Explanation for Natural Disasters
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Correlation%20Matrix%20ND%20&%20States.png?raw=true" alt="Correaltion Matrix Heatmap Explanation for Natural Disasters" width="500" height="500">

- The heatmap reveals no significant correlation between natural disasters and state
- Natural Disasters do not significantly impact loan status.

## Correlation Matrix Heatmap for Natural Disasters and Loan Status
<img src= "https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Corr%20Matrix%20Heatmap%20ND%20&%20loan_status.png?raw=true" alt= "Correlation Matrix Heatmap for Natural Disasters and Loan Status" width= "500" height= "500">

- The Matrix reveals significant postive correlations with loan variables and significant negative correlations with natural disaster variables.
- Dark brown indicates strong positve corelations while the lighter colors indicate no correlation.


## Feature Importance with Natural Disaster Features
<img src= "https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Feature%20Importance%20with%20ND.png?raw=true" alt= "Feature Importance with Natural Disaster Features" width= "500" height= "500">

- 'last_pymnt_amnt' remains as the most influential feature, followed by 'total_rec_prncp' and 'int_rate'.
- With Natural disaster showing no significance.




