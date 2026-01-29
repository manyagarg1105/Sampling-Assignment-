# Sampling Assignment

## Objective
To study the impact of different sampling techniques on imbalanced datasets
and analyze their effect on various machine learning models.

## Dataset
Credit Card Fraud Dataset (imbalanced)

## Steps Involved

1. The credit card fraud dataset was downloaded from the given GitHub link and
   loaded using the Pandas library.

2. Initial data exploration was performed to understand the dataset structure
   and observe the class imbalance.

3. The dataset was divided into features (X) and the target variable (Class).

4. The imbalanced dataset was converted into a balanced dataset using the SMOTE
   technique.

5. Five different samples were created from the balanced dataset using random
   splits.

6. Five sampling techniques were applied to the samples and five machine
   learning models were trained on the resulting datasets.

7. Model performance was evaluated using accuracy as the evaluation metric.

## Sampling Techniques Used
In real-world datasets, class imbalance is a common problem where one class has
significantly more samples than the other. To handle this issue, different
sampling techniques were applied in this assignment.

### 1. Random Over Sampling
Random Over Sampling increases the number of minority class samples by randomly
duplicating existing minority instances. This helps balance the dataset but may
lead to overfitting since the same data points are repeated.

### 2. Random Under Sampling
Random Under Sampling balances the dataset by randomly removing samples from the
majority class. While this reduces dataset size and training time, it may cause
loss of important information.

### 3. SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE generates new synthetic samples for the minority class instead of simply
duplicating existing ones. New samples are created by interpolating between
nearest minority class neighbors, which helps reduce overfitting.

### 4. SMOTE-ENN
SMOTE-ENN is a hybrid technique that first applies SMOTE to oversample the
minority class and then uses Edited Nearest Neighbors (ENN) to remove noisy and
misclassified samples. This improves data quality and model generalization.

### 5. SMOTE-Tomek
SMOTE-Tomek combines SMOTE with Tomek links. After generating synthetic samples,
Tomek links identify and remove overlapping samples from different classes,
resulting in cleaner decision boundaries.

## Models Used
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. Naive Bayes

## Results
Accuracy scores were calculated for each combination of model and sampling technique.
The best sampling method varies depending on the model.

## Conclusion
Sampling techniques significantly affect model performance. SMOTE-based
methods generally provide better balance between accuracy and generalization.
