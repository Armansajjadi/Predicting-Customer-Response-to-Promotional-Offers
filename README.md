# 📊 Predicting Customer Response to Promotional Offers

## 📌 Project Overview

This project builds a **discrete choice model** to predict whether a customer will accept a promotional offer during a marketing campaign for a financial institution.
The goal is to help the marketing department **target the right customers**, increasing campaign efficiency while reducing costs.

We leverage **data analysis, class balancing techniques, and machine learning models** to deliver actionable insights.

---

## 🗂 Project Workflow

1. **Importing Libraries**

   * `pandas`, `numpy` for data handling
   * `seaborn`, `matplotlib` for visualization
   * `scikit-learn` models and metrics for classification and evaluation
   * `imblearn`’s `SMOTETomek` for class balancing

2. **Loading the Dataset**

   * Reads marketing campaign data from Excel
   * Checks data structure, missing values, and class distribution

3. **Exploratory Data Analysis (EDA)**

   * Visualizes target variable distribution
   * Identifies imbalance between customers who accept vs reject offers

4. **Model Training (Baseline)**

   * Models tested:

     * Logistic Regression
     * K-Nearest Neighbors (KNN)
     * Linear Discriminant Analysis (LDA)
     * Quadratic Discriminant Analysis (QDA)
     * Random Forest
   * Evaluated with **F1-score, Precision, Recall**
   * Plotted confusion matrices for each model

5. **Handling Class Imbalance**

   * Applied **SMOTETomek** to oversample minority and clean data
   * Retrained models on balanced data and compared improvements

6. **Hyperparameter Tuning**

   * Tuned **Random Forest** with `RandomizedSearchCV`
   * Optimized for F1-score using cross-validation
   * Achieved best performance on test set after tuning

---

## 📈 Key Results

* **Class balancing** improved recall and F1-score significantly.
* **Random Forest with tuned parameters** delivered the best overall performance.

---

## 💻 Technologies Used

* **Python 3**
* **Pandas / NumPy** – data processing
* **Seaborn / Matplotlib** – visualization
* **Scikit-learn** – machine learning models and evaluation
* **Imbalanced-learn (SMOTETomek)** – class balancing
