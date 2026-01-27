# Customer Churn Prediction - Data Science Project

## Project Overview

This project develops predictive models to identify customers at risk of churning using customer behavior and demographic data. The analysis includes exploratory data analysis (EDA), comprehensive feature engineering, multicollinearity handling, and comparison of multiple machine learning models.

## Dataset

**Source:** Customer-Churn-Records.csv
- **Total Records:** 10,000 customers
- **Original Features:** 18 numerical + 3 categorical features
- **Target Variable:** Exited (Churn - Binary: 0/1)
- **Churn Rate:** 20.38% (2,038 churned customers out of 10,000)

## Project Structure

```
├── Customer-Churn-Records.csv              # Raw input data
├── exploratory_analysis.ipynb              # EDA and descriptive statistics
├── feature_eng.ipynb                       # Feature engineering pipeline (11 sections)
├── modeling.ipynb                          # Model training and comparison (9 sections)
├── Customer_Churn_Final_Features.csv       # Engineered features dataset
├── Customer_Churn_Engineered_Features.csv  # Complete feature engineering output
├── Feature_Engineering_Report.txt          # Summary report
└── README.md                               # This file
```

## Notebooks Description

### 1. exploratory_analysis.ipynb
**Purpose:** Initial exploratory data analysis providing statistical baseline

**Sections:**
1. Import libraries and load data
2. Check data types and missing values
3. Numerical column analysis
4. Categorical column analysis
5. Correlation analysis with target
6. Churn distribution analysis
7. Descriptive statistics (skewness, kurtosis)
8. Visualization of key distributions
9. Summary insights

**Key Outputs:** Statistical summaries, correlation matrices, distribution visualizations

---

### 2. feature_eng.ipynb
**Purpose:** Comprehensive feature engineering and preprocessing pipeline

**Sections (11 total):**

1. **Import Libraries** - All necessary ML and data processing libraries
2. **Load & Explore Data** - Dataset overview and structure
3. **Handle Missing Values** - Fill numerical and categorical missing values
4. **Encode Categorical Variables**
   - Label encoding for Gender and Card Type
   - One-hot encoding for Geography
5. **Create New Features (12 engineered features)**
   - **Interaction Features:** Age_Tenure_Interaction, Salary_Balance_Ratio, Credit_Age_Interaction
   - **Polynomial Features:** Age_Squared, CreditScore_Squared, Tenure_Squared
   - **Binned Features:** Age_Group, CreditScore_Category, Tenure_Group
   - **Derived Features:** Has_Balance, Active_Months, Complain_With_Low_Satisfaction
6. **Feature Scaling & Normalization**
   - StandardScaler (mean=0, std=1) for 13 features
   - MinMaxScaler (range 0-1) for 13 features
7. **Feature Selection** - SelectKBest with Mutual Information (k=15)
8. **Feature Importance Summary** - 6 feature categories documented
9. **Correlation & Mutual Information Analysis**
10. **Multicollinearity Handling (VIF Analysis)**
    - Iterative removal of features with VIF > 5
    - Reduced from 15 to 5 final features
11. **Final Clean Dataset Export** - Customer_Churn_Final_Features.csv

**Final Features (5 features after VIF removal):**
1. Complain
2. Complain_With_Low_Satisfaction
3. Age_Squared_StandardScaled
4. NumOfProducts
5. Age_Tenure_Interaction_MinMaxScaled

**Key Outputs:** Final dataset with 5 clean features, VIF analysis report, feature importance rankings

---

### 3. modeling.ipynb
**Purpose:** Train, evaluate, and compare multiple classification models

**Sections (9 total):**

1. **Import Libraries & Load Data** - ML libraries and engineered features
2. **Data Preparation for Modeling**
   - Remove leaky features (Complain, Complain_With_Low_Satisfaction)
   - Final feature set: 3 features
   - Stratified 80-20 train-test split
   - Target distribution: 79.62% retained / 20.38% churned

3. **Logistic Regression Model**
   - Accuracy: 78.35%
   - Precision: 22.22%
   - Recall: 2.45%
   - F1-Score: 4.42%
   - ROC-AUC: 74.14%

4. **Random Forest Model (100 trees)**
   - Accuracy: 81.80%
   - Precision: 57.75%
   - Recall: 40.20%
   - F1-Score: 47.40%
   - ROC-AUC: 78.22%
   - Feature Importance: NumOfProducts (dominant), followed by engineered features

5. **Gradient Boosting Model (100 trees)**
   - Accuracy: 83.85%
   - Precision: 65.23%
   - Recall: 44.61%
   - F1-Score: 52.98%
   - ROC-AUC: 82.66%

6. **Model Comparison & Visualization**
   - 6-subplot performance dashboard
   - ROC curves comparison
   - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

7. **Summary and Recommendations** - Key findings and next steps

8. **Ensemble Method - Voting Classifier (GB + RF)**
   - Soft voting combining best individual models
   - Accuracy: 83.05%
   - Precision: 62.37%
   - Recall: 42.65%
   - F1-Score: 50.66%
   - ROC-AUC: 81.40%

9. **Comprehensive Model Comparison (All Models)**
   - Comparison of 4 models: LR, RF, GB, Ensemble
   - **Best Overall Model:** Gradient Boosting (F1-Score: 0.5298)
   - Comprehensive visualizations with ROC curves

**Key Outputs:** Trained models (lr_model, rf_model, gb_model, voting_clf), performance metrics, visualizations

---

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 78.35% | 22.22% | 2.45% | 4.42% | 74.14% |
| Random Forest | 81.80% | 57.75% | 40.20% | 47.40% | 78.22% |
| Gradient Boosting | **83.85%** | **65.23%** | **44.61%** | **52.98%** | **82.66%** |
| Ensemble (GB+RF) | 83.05% | 62.37% | 42.65% | 50.66% | 81.40% |

**Best Model:** Gradient Boosting with F1-Score of 52.98%

---

## Key Findings

1. **Feature Engineering Impact**
   - Engineered 12 new features from 18 original features
   - Reduced to 5 final features after multicollinearity analysis (VIF ≤ 5)
   - Improved model interpretability and reduced overfitting

2. **Class Imbalance**
   - 20.38% churn rate creates class imbalance challenge
   - Models show varying performance due to minority class prediction difficulty
   - Gradient Boosting handles class imbalance better than other models

3. **Model Comparison**
   - Ensemble and Gradient Boosting significantly outperform Logistic Regression
   - Gradient Boosting achieves best F1-Score (52.98%)
   - Ensemble provides competitive performance with potential robustness benefits

4. **Feature Importance**
   - Top features: NumOfProducts, Age_Squared_StandardScaled, Age_Tenure_Interaction_MinMaxScaled
   - Interaction and polynomial features provide predictive power
   - Original numeric features contribute more than categorical encoded features

5. **Multicollinearity Handling**
   - VIF analysis successfully identified and removed collinear features
   - Final feature set has all VIF scores < 5 (no problematic multicollinearity)
   - Improved model stability and interpretability

---

## Technologies & Libraries

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (preprocessing, ensemble, linear_model, model_selection)
- **Statistical Analysis:** scipy, statsmodels (VIF calculations)
- **Python Version:** 3.13.0

---

## Recommendations & Next Steps

1. **Model Deployment**
   - Deploy Gradient Boosting model to production
   - Monitor performance metrics on new data
   - Implement model versioning and retraining schedule

2. **Class Balancing Improvements**
   - Implement SMOTE (Synthetic Minority Over-sampling Technique)
   - Use weighted loss functions for minority class emphasis
   - Explore different class weight strategies

3. **Feature Engineering Extensions**
   - Develop customer lifetime value (CLV) features
   - Create temporal features from customer activity patterns
   - Add interaction terms between key demographic variables

4. **Model Interpretability**
   - Implement SHAP (SHapley Additive exPlanations) for model explanation
   - Generate feature importance insights for business stakeholders
   - Create decision rules for customer segmentation

5. **Hyperparameter Optimization**
   - Use GridSearchCV or RandomizedSearchCV for hyperparameter tuning
   - Optimize Gradient Boosting parameters (learning_rate, max_depth, n_estimators)
   - Cross-validate with different random seeds for robustness

6. **Ensemble Methods Exploration**
   - Test Stacking ensemble combining diverse models
   - Explore different voting strategies (hard vs. soft)
   - Investigate blending approaches with held-out validation set

---

## How to Run

1. **Exploratory Analysis:**
   ```bash
   jupyter notebook exploratory_analysis.ipynb
   ```

2. **Feature Engineering:**
   ```bash
   jupyter notebook feature_eng.ipynb
   ```

3. **Model Training & Comparison:**
   ```bash
   jupyter notebook modeling.ipynb
   ```

---

## Project Metrics

- **Total Features Generated:** 12 engineered features
- **Final Feature Set:** 5 features (after VIF removal)
- **Models Trained:** 4 (LR, RF, GB, Ensemble)
- **Best Model F1-Score:** 0.5298
- **Best Model Accuracy:** 83.85%
- **Training Data Points:** 8,000
- **Test Data Points:** 2,000

---

## Author Notes

This project demonstrates a complete data science pipeline from raw data to production-ready models, including:
- Proper data preprocessing and feature engineering
- Multicollinearity analysis and handling
- Comprehensive model evaluation and comparison
- Ensemble method implementation
- Professional documentation and reporting

The focus on handling class imbalance and multicollinearity reflects real-world data science challenges and best practices.