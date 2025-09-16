# Credit_Risk_Analysis
💳 Credit Risk Analysis
📌 Project Overview

This project applies supervised machine learning models to predict credit risk using imbalanced loan data. Since good loans vastly outnumber risky ones, we compared multiple resampling techniques and ensemble classifiers to evaluate which models best handle class imbalance.

🎯 Purpose

Develop and evaluate ML models that accurately predict credit risk by testing:

Resampling Methods: Naive Random Oversampling, SMOTE, Cluster Centroids Undersampling, SMOTEENN

Ensemble Models: Balanced Random Forest, Easy Ensemble AdaBoost

🛠️ Tools & Resources

Language: Python (v3.7)

Libraries: scikit-learn, imbalanced-learn

Environment: Jupyter Notebook

Dataset: LoanStats_2019Q1.csv (not uploaded due to size)

📊 Results
Resampling + Logistic Regression

Accuracy: ~0.54 – 0.65

Precision (High Risk): 0.01

Recall (High Risk): ~0.63 – 0.72
➡️ Struggled with precision; high false positives for risky loans.

Ensemble Models

Balanced Random Forest: Accuracy 0.78 | Recall (High Risk) 0.70

Easy Ensemble AdaBoost: Accuracy 0.93 | Recall (High Risk) 0.92
➡️ Significantly better performance, though precision for high-risk loans remained low.

✅ Summary & Recommendation

Resampling models underperformed due to class imbalance.

Ensemble methods (especially Easy Ensemble AdaBoost) delivered the highest accuracy and recall, making them the most promising approach for credit risk prediction.

Future work: fine-tune models on larger datasets to reduce false positives and avoid overfitting.

📂 About

Built with Python + Jupyter Notebook

Focused on credit risk prediction, imbalanced classification, and ensemble learning

👉 This rewritten version tells a clear story: what you did, how you did it, and what worked best — without overwhelming detail.

![Naive_Random_Oversampling](https://user-images.githubusercontent.com/91812090/160291355-2cc56a71-d4aa-443f-b2f6-114ae08940cc.png)
![SMOTE_Oversampling](https://user-images.githubusercontent.com/91812090/160291360-694b09cf-5638-40f2-91bd-a81b131613ac.png)
![Undersampling](https://user-images.githubusercontent.com/91812090/160291396-a30fe84a-ca10-4cd2-b1b8-6179d1f5350c.png)
![Combination_(Over and Under)_Sampling](https://user-images.githubusercontent.com/91812090/160291423-7fbb1fb0-8011-4e06-9b0d-3bbcbcb54014.png)
![Balanced_Random_Forest_Classifier](https://user-images.githubusercontent.com/91812090/160291434-4bffaa47-daeb-4f5f-afbf-9a36707c3d42.png)
![Easy_Ensemble_AdaBoost_Classifier](https://user-images.githubusercontent.com/91812090/160291441-14e52c1b-848a-4f25-915d-189189605700.png)



If this occurs and we don’t get desired results when working with new data set, we can do some further pruning (fine-tunning) to avoid the overfitting.
