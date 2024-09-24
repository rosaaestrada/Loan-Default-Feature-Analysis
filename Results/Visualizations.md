# Visualizations

## ROC Curve
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/ROC%20curve.png?raw=true" alt="" width="500" height="500">

- The ROC Curve shows that Random Forest (AUC = 0.97) and Decision Tree (AUC = 0.96) models perform significantly better at distibguishing between loan default outcomes compared to Logistic Regression (AUC = 0.87), with all models outperforming the base rate.

## Confusion Matrix: Logistic Regression, Random Forest, and Decision Tree
<img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Confusion%20Matrix%20Logistic%20Regression.png?raw=true" alt="" width="400" height="300"><img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Confusion%20Matrix%20Random%20Forest.png?raw=true" alt="" width="400" height="300"><img src="https://github.com/rosaaestrada/Loan-Default-Prediction/blob/main/Results/Images/Confusion%20Matrix%20Decision%20Tree.png?raw=true" alt="" width="400" height="300">

- Logistic Regression struggles slightly with correctly classifying non-default loans (0), with more false positives (4,842), while it performs better in identifying loan defaults (1) with a good balance of true positives (24,799).
- Random Forest shows much stronger perfromance, correctly classifying both non-default loans (25,149) and loan defaults (25,904) with minimal errors, evidenced by the low number of false positives (961) and false negatives (789).
- Decision Tree demonstrates a strong perfromance in distinguishing between non-default loans (0) and loan defaults (1), it has high true positive (25,552) and true negatives (25,055), however, it exhibits a slightly higher rate of false negatives (1,141) compared to false positives (1,055).




## Feature Importance: Random Forest and Decision Tree
<img src="" alt="" width="500" height="500">
<img src="" alt="" width="500" height="500">

- highlights the most important features for each tree-based model, showing which features have the most predictive power.
- Helps explain why the models perfrom the way they do



## Correlation Matrix Heatmap of selected features
<img src="" alt="" width="500" height="500">

- shows relationships between the key features you've selected for modeling
- provides an understanding of the interactions between the most relevant variables

## Correaltion Matrix Heatmap Explanation for Natural Disasters
<img src="" alt="" width="500" height="500">

- Explains the correlation (or lack thereof) between natural disasters and loan status
- Natural Disasters do not significantly impact loan status.





