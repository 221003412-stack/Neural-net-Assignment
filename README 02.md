Bank Term Deposit Prediction

Overview

This project focuses on building a **Logistic Regression model** to predict whether a bank customer will subscribe to a 
term deposit. The goal is to assist financial institutions in identifying potential customers more effectively, enabling
targeted marketing campaigns and improved conversion rates. Special attention is given to handling class imbalance and
evaluating trade-offs between precision and recall.


Dataset

* File: bank-full.csv
* Total Records: 45,211
* Features:17
* Target Variable: `y` (yes/no → indicates whether the customer subscribed)

The dataset contains a mix of numerical and categorical features related to customer demographics, previous interactions, and campaign details.


Methodology

1. Data Loading and Exploration

* Imported all required libraries
* Loaded the dataset using Pandas with `;` as the delimiter
* Explored data structure, distributions, and class imbalance

2. Data Cleaning

 Identified anomalies such as:

  * `-1` in the `pdays` column
  * `"unknown"` values in categorical features
* Replaced these anomalies with `NaN` to treat them as missing values
* Verified data consistency after cleaning


3. Handling Missing Values

* Filled missing values using:

  * Most frequent values for categorical features
  * `"none"` category for selected columns where appropriate


4. Outlier Treatment

Applied clipping to numerical features such as:

  * `balance`
  * `duration`
  * `campaign`
    This helped reduce the influence of extreme values.

5. Feature Engineering

* Encoded the target variable (`yes` → 1, `no` → 0)
* Applied one-hot encoding to categorical variables

6. Data Preparation

* Split dataset into features (**X**) and target (**y**)
* Performed an **80/20 train-test split**
* Applied **StandardScaler** to normalize feature values

7. Model Training

* Trained a Logistic Regression model with `class_weight='balanced'` to handle class imbalance

8. Model Evaluation

* Evaluated using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * Confusion Matrix

9. Hyperparameter Tuning

* Applied **Grid Search** to optimize model parameters
* Re-evaluated performance after tuning



 Approach

* Prioritized **data cleaning and preprocessing to ensure high-quality input
* Addressed class imbalance using **balanced class weights
* Converted categorical variables using one-hot encoding
* Standardized features to improve model convergence
* Used Grid Search to improve model performance through parameter tuning


Results

 Before Hyperparameter Tuning

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.84  |
| Precision | 0.42  |
| Recall    | 0.81  |
| F1 Score  | 0.55  |

 After Hyperparameter Tuning

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 0.90  |
| Precision | 0.65  |
| Recall    | 0.38  |
| F1 Score  | 0.48  |


 Discussion

The results highlight a clear **trade-off between precision and recall:

Before tuning:

  * High recall (0.81) → model successfully identified most potential subscribers
  * Low precision → many false positives

 After tuning:

  * Higher accuracy and precision → model became more selective
  * Lower recall (0.38) → missed many actual subscribers

This trade-off reflects different business priorities:

* If the goal is to maximize customer acquisition, the pre-tuning model is more suitable due to higher recall
* If the goal is to reduce marketing costs and avoid unnecessary outreach, the tuned model is preferable due to higher precision

 Limitations

* Logistic Regression may not capture complex non-linear relationships in the data
* Performance is sensitive to class imbalance despite using balanced weights
* Feature engineering could be further improved for better predictive power


Future Improvements

* Experiment with advanced models such as Random Forest, XGBoost, or Neural Networks
* Perform feature selection to remove irrelevant variables
* Use techniques like SMOTE for better imbalance handling
* Tune decision thresholds to achieve a better balance between precision and recall
* Explore cross-validation for more robust performance evaluation


 Conclusion

This project demonstrates how Logistic Regression can be effectively applied to predict customer
subscription behavior in banking. While the model performs reasonably well, the results emphasize the
importance of aligning model optimization with business objectives. Balancing precision and recall is crucial,
and the choice of model configuration should depend on whether the priority is maximizing conversions or minimizing costs.
Overall, the project provides a strong foundation for predictive modeling in marketing analytics and highlights key
considerations in real-world machine learning applications.
