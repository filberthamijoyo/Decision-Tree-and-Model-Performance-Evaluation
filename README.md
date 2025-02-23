# Decision Tree and Model Performance Evaluation
This project demonstrates the implementation of **Decision Tree Regression**, **Bagging of Trees**, **Random Forests**, and **Model Evaluation** using Python. The code includes data preprocessing, model training, evaluation, and visualization for classification tasks.

---

## Features

### 1. **Decision Tree Regression**
- **Dataset Handling**: Loads the Penguins dataset and performs data preprocessing.
- **Data Cleaning**: Removes incomplete data points and encodes categorical features.
- **Data Splitting**: Splits the dataset into training and test sets (70% training, 30% test).
- **Data Statistics**: Visualizes feature distributions across classes using histograms.
- **Decision Tree Training**: Trains decision trees with varying maximum depths and minimum node sizes.
- **Accuracy Reporting**: Reports training and test accuracy for each model configuration.
- **Tree Visualization**: Plots the learned decision trees for each configuration.

### 2. **Bagging of Trees**
- **Bagging Implementation**: Uses decision trees as base learners for bagging.
- **Model Training**: Trains bagging models with different numbers of trees and maximum depths.
- **Accuracy Reporting**: Reports training and test accuracy for each configuration.

### 3. **Random Forests**
- **Random Forest Implementation**: Uses decision trees as base learners for random forests.
- **Model Training**: Trains random forest models with different numbers of trees and maximum depths.
- **Accuracy Reporting**: Reports training and test accuracy for each configuration.

### 4. **Bias-Variance Analysis**
- **Bias-Variance Decomposition**: Analyzes the relationship between bias², variance, and the number of trees in random forests.
- **Visualization**: Plots bias² and variance against the number of trees.

---

## Code Structure

### Part 1: Decision Tree Regression and Ensemble Methods
- **Data Preprocessing**: Cleans and encodes the dataset.
- **Data Visualization**: Generates histograms for feature distributions.
- **Decision Tree Training**: Trains and evaluates decision trees with different hyperparameters.
- **Bagging and Random Forests**: Implements bagging and random forests with varying configurations.
- **Bias-Variance Analysis**: Analyzes and visualizes bias² and variance for random forests.

### Part 2: Model Evaluation and Hyperparameter Tuning
- **Train-Validation Split**: Implements a custom function for K-fold cross-validation.
- **Evaluation Metrics**: Implements precision, recall, F1 score, accuracy, and AUROC.
- **Model Training**: Trains logistic regression and SVM models with hyperparameter tuning.
- **Performance Evaluation**: Evaluates models on validation and test sets.

---

## Usage

1. Clone the repository.
2. Ensure the required datasets (`train_validation_data.npy`, `train_validation_label.npy`, `test_data.npy`, `test_label.npy`) are in the correct directory.
3. Run the Python script to see the results.

---

## Dependencies
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `mlxtend`

---

## Results
- **Decision Trees**: Displays training and test accuracy for various configurations.
- **Bagging and Random Forests**: Reports accuracy for different numbers of trees and depths.
- **Bias-Variance Analysis**: Visualizes the relationship between bias², variance, and the number of trees.
- **Model Evaluation**: Provides precision, recall, F1 score, accuracy, and AUROC for logistic regression and SVM.

---

Feel free to contribute or suggest improvements!

