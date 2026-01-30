# Predictive Modeling
This project implements a classification-based predictive model to determine whether a customer is likely to churn. 
It was completed as part of Level 3 Task 1 during my internship with Codveda Technologies.

## Objectives
- Preprocess data (categorical encoding and feature scaling)
- Train multiple classification models
- Evaluate models using standard performance metrics
- Perform hyperparameter tuning using GridSearchCV
- Select the best-performing model

## Tools & Libraries
- Python
- pandas
- scikit-learn
- matplotlib
  
## Dataset
The project uses the BigML churn dataset:
- churn-bigml-80.csv → Training data
- churn-bigml-20.csv → Test data

The dataset includes both numerical and categorical customer attributes, with Churn as the target variable

## Data Preprocessing
- Categorical variables were handled using One-Hot Encoding
- Feature scaling was performed using StandardScaler
- To prevent data leakage, scaling was fitted only on training data
- Train and test sets were aligned to ensure consistent feature columns

## Models Trained
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Model Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

## Model Performance Summary   
|   Model   |       Accuracy |   Precision   | Recall   |   F1-Score |
|-----------|-----------|------------|---------------|-------------|
| Logistic Regression |   0.86   |    0.51    |    0.25   |  0.34  |
| Decision Tree       |   0.93   |    0.78    |    0.75   |  0.76  |
| Random Forest       |   0.94   |    0.98    |    0.60   |  0.75  |
| Tuned Random Forest |   0.94   |    1.00    |    0.61   |  0.76  |

## Hyperparameter Tuning
GridSearchCV was applied to the Random Forest model using:
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

## Best Parameters Found:
python
{
  max_depth: 20,
  min_samples_leaf: 1,
  min_samples_split: 2,
  n_estimators: 200
}

## Final Conclusion
The tuned Random Forest model achieved the best overall performance, 
particularly with perfect precision, making it highly reliable for identifying churn customers with minimal false positives.

📌 Author

Olanireti Itajobi

Intern, Codveda Technologies
