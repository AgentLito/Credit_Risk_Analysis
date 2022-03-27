# Credit_Risk_Analysis
Credit_Risk_Analysis

Project Overview

This project we are utilizing several models of supervised machine learning on credit loan data to predict credit risk. Credit risk is an unbalanced classification problem, as good loans easily outnumber risky loans. We will use Python, imbalanced-learn, Scikit-learn libraries and several machine learning models to compare the strengths and weaknesses of ML models and determine how well a model classifies and predicts data.

Purpose

The purpose of this analysis is to create a supervised machine learning model that could accurately predict credit risk. To complete this task, we used 6 different methods:

SMOTEENN Sampling
Naive Random Oversampling
SMOTE Oversampling
Balanced Random Forest Classifying
Cluster Centroid Undersampling
SMOTEENN Sampling
Easy Ensemble Classifying

Through each of these methods, we need to split the data into testing and training datasets, compiled accuracy scores, confusion matrices, and classification reports as per results.

Requirements

Use Resampling Models to Predict Credit Risk
Use the SMOTEENN algorithm to Predict Credit Risk
Use Ensemble Classifiers to Predict Credit Risk
A Written Report on the Credit Risk Analysis

Resources

Data Source: LoanStats_2019Q1.csv (file it’s not uploaded to GitHub because its a large file)
Software: Jupyter Notebook
Languages: Python
Libraries: Scikit-learn , imbalanced-learn
Environment: Python 3.7

Results

In this analysis we used six different algorithms of supervised machine learning.

First four algorithms are based on resampling techniques and are designed to deal with class imbalance.
After the data is resampled, Logistic Regression is used to predict the outcome.
Logistic regression predicts binary outcomes .
The last two models are from ensemble learning group.
The concept of ensemble learning is the process of combining multiple models,
Like decision tree algorithms, to help improve the accuracy and robustness, as well as decrease variance of the model,
therefore, increase the overall performance of the model.
1. Naive Random Oversampling and Logistic Regression

In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced.

Accuracy score: 0.65
Precision
For high risk: 0.01
For low risk: 1.00
Recall
For high risk: 0.71
For low risk: 0.60

![Naive_Random_Oversampling](https://user-images.githubusercontent.com/91812090/160291355-2cc56a71-d4aa-443f-b2f6-114ae08940cc.png)

Figure 1: Results for Naive Random Oversampling.

2. SMOTE Oversampling and Logistic Regression

The synthetic minority oversampling technique (SMOTE) is another oversampling approach where the minority class is increased. Unlike other oversampling methods, SMOTE interpolated new instances, that is, for an instance from the minority class, several its closest neighbors are chosen. Based on the values of these neighbors, new values are created.

Accuracy score: 0.65
Precision
For high risk: 0.01
For low risk: 1.00
Recall
For high risk: 0.63
For low risk: 0.68

![SMOTE_Oversampling](https://user-images.githubusercontent.com/91812090/160291360-694b09cf-5638-40f2-91bd-a81b131613ac.png)


Figure 2: Results for SMOTE Oversampling.

3. Cluster Centroids Undersampling and Logistic Regression

Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.

Accuracy score: 0.54
Precision
For high risk: 0.01
For low risk: 1.00
Recall
For high risk: 0.69
For low risk: 0.40

![Undersampling](https://user-images.githubusercontent.com/91812090/160291396-a30fe84a-ca10-4cd2-b1b8-6179d1f5350c.png)


Figure 3: Results for Cluster Centroids Undersampling.

4. SMOTEENN (Combination of Over and Under Sampling) and Logistic Regression

SMOTEENN is an approach to resampling that combines aspects of both oversampling and undersampling - oversample the minority class with SMOTE and clean the resulting data with an undersampling strategy.

Accuracy score: 0.54
Precision
For high risk: 0.01
For low risk: 1.00
Recall
For high risk: 0.72
For low risk: 0.57

![Combination_(Over and Under)_Sampling](https://user-images.githubusercontent.com/91812090/160291423-7fbb1fb0-8011-4e06-9b0d-3bbcbcb54014.png)


Figure 4: Results for SMOTTEENN Model.

5. Balanced Random Forest Classifier

Instead of having a single, complex tree like the ones created by decision trees, a random forest algorithm will sample the data and build several smaller, simpler decision trees. Each tree is simpler because it is built from a random subset of features.

Accuracy score: 0.78
Precision
For high risk: 0.03
For low risk: 1.00
Recall
For high risk: 0.70
For low risk: 0.87

![Balanced_Random_Forest_Classifier](https://user-images.githubusercontent.com/91812090/160291434-4bffaa47-daeb-4f5f-afbf-9a36707c3d42.png)


Figure 5: Results for Balanced Random Forest Classifier Model.

6. Easy Ensemble AdaBoost Classifier

In AdaBoost Classifier, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. The model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. This process is repeated until the error rate is minimized.

Accuracy score: 0.93
Precision
For high risk: 0.09
For low risk: 1.00
Recall
For high risk: 0.92
For low risk: 0.94

![Easy_Ensemble_AdaBoost_Classifier](https://user-images.githubusercontent.com/91812090/160291441-14e52c1b-848a-4f25-915d-189189605700.png)

Figure 6: Results for Easy Ensemble AdaBoost Classifier Model.

Summary

From the results section above we can see how different ML models work on the same data. We will start the interpretation of the results with detail explanation of the outcomes.

Accuracy score tells us what percentage of predictions the model gets it right. However, it is not enough just to see those results, especially with unbalanced data. Equation: accuracy score = number of correct prediction / total number of predictions.

Precision is the measure of how reliable a positive classification is. A low precision is indicative of many false positives. Equation: Precision = TP/ (TP + FP).

Recall is the ability of the classifier to find all the positive samples. A low recall is indicative of many false negatives. Equation: Recall = TP/ (TP + FN).

F1 Score is a weighted average of the true positive rate (recall) and precision, where the best score is 1.0 and the worst is 0.0 (3). Equation: F1 score = 2(Precision * Sensitivity)/(Precision + Sensitivity).

Results summary

First 4 models – Resampling and Logistic regression

From the results above we can see that first four models don’t do well based on the accuracy scores.
Those scores are 0.65, 0.66, 0.54 and 0.54 for Naive Random Oversampling, SMOTE Oversampling, Cluster Centroids Undersampling and SMOTEENN model respectively.
It means the models were accurate roughly a bit more than half of the time.
Precision for all four models is 0.01 for high-risk loans and 1.00 for low-risk loans.
Low precision score for high-risk loans is due to large number of false positives,
It means low risk loans were marked as high-risk loans.
High score for low-risk loans indicate that nearly all low-risk scores were marked correctly,
recall score (0.71 for Naive Random Oversampling and Logistic Regression, for example) indicates that there were quite a few low-risk loans that were market as high risk, when they weren’t.
Actual high-risk loans have slightly better scores on recall (0.60 for Naive Random Oversampling and Logistic Regression, for example) meaning that there weren’t as many false negatives or not too many high-risk loans were marked as low risk loans.
Last 2 models – Ensemble models

These models did better as compared to the above models.
Their accuracy scores are 0.93 and 0.92 for Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier respectively.
Recall scores for each model and with – low and high-risk scores and precision for low risk were high, meaning very good in accuracy.
Precision for high-risk loans in both models are not high.
0.03 and 0.09 for Balanced Random Forest Classifier and Easy Ensemble AdaBoost Classifier respectively,
It indicates that, there are large number of false positives, and it means that large number of low-risk loans are marked as high risk.
Recommendation on the model

As per the above details, first three models(algorithms) didn’t do well on the test, so we can’t recommend using them in the real-word testing without further fine-tuning.

For example, train model on larger dataset, we need to look through the dataset columns that will be used for training the model.
Other two models showed better results, so we can use them carefully, they might be prone to overfitting.

If this occurs and we don’t get desired results when working with new data set, we can do some further pruning (fine-tunning) to avoid the overfitting.
