# 📊 Predicting Customer Response to Promotional Offers

## 📌 Project Overview

This project builds and evaluates several models, including a **custom neural network**, to predict whether a customer will accept a promotional offer during a marketing campaign. The goal is to help the marketing department **target the right customers**, increasing campaign efficiency while reducing costs.

We leverage **data analysis, class balancing techniques, and multiple machine learning approaches** to deliver actionable insights.

---

## 🗂 Project Workflow

1.  **Importing Libraries**
    * `pandas`, `numpy` for data handling.
    * `seaborn`, `matplotlib` for visualization.
    * `scikit-learn` for traditional ML models and metrics.
    * `imblearn`’s `SMOTETomek` for class balancing.
    * `PyTorch` for building the neural network.

2.  **Loading the Dataset**
    * Reads marketing campaign data from Excel.
    * Checks data structure, missing values, and class distribution.

3.  **Exploratory Data Analysis (EDA)**
    * Visualizes target variable distribution.
    * Identifies imbalance between customers who accept vs. reject offers.

4.  **Model Training (Baseline)**
    * Models tested:
        * Logistic Regression
        * K-Nearest Neighbors (KNN)
        * Linear Discriminant Analysis (LDA)
        * Quadratic Discriminant Analysis (QDA)
        * Random Forest
    * Evaluated with **F1-score, Precision, and Recall**.

5.  **Handling Class Imbalance**
    * Applied **SMOTETomek** to the training data to oversample the minority class and clean overlapping data points.
    * Retrained models on the balanced data to compare improvements.

6.  **Hyperparameter Tuning (Random Forest)**
    * Tuned **Random Forest** with `RandomizedSearchCV` to find optimal parameters.
    * Optimized for F1-score using cross-validation.

7.  **Custom Neural Network with PyTorch**
    * Defined a custom neural network architecture using PyTorch's `nn.Module`.
    * Implemented a manual training loop to monitor both training and validation loss, helping to identify overfitting.
    * Systematically experimented with hyperparameters, including **model architecture** (depth/width of layers), **learning rate**, and **regularization** (Dropout, Weight Decay) to improve performance.

---

## 📈 Key Results

* **Class balancing** with SMOTETomek was crucial, significantly improving the recall and F1-scores for all models.
* The **tuned Random Forest** delivered strong overall performance on the balanced data.
* The **custom PyTorch neural network** showed improvement over the baseline models but ultimately highlighted the predictive limits of the available features, as performance plateaued after extensive tuning.

---

## 💻 Technologies Used

* **Python 3**
* **Pandas / NumPy** – Data processing
* **Seaborn / Matplotlib** – Visualization
* **Scikit-learn** – Machine learning models and evaluation
* **Imbalanced-learn** – Class balancing with `SMOTETomek`
* **PyTorch** – Deep learning framework for the custom neural network
