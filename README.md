# KNN Classifier on Diabetes Dataset
This project implements a simple K-Nearest Neighbours (KNN) classifier from scratch in Python to predict the liklihood of diabetes using a well-known dataset.
## Dataset
The model uses the **Pima Indians Diabetes Database** (`diabetes.csv`), a standard dataset used in medical ML tasks. The goal is binary classification: predict whether a patient has diabetes (`Outcome` = 1) or not (`Outcome` = 0).
## Features Used
The following features are cleaned and used in the model:
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`

Zero values in these columns are treated as missing and replaced with the median.
## How It Works
- Implements a cuistom KNN classifier wihthout external ML libraries.
- Calculates Euclidean distance between data points.
- Predicts class based on majority vote of `k` nearest neighbours.
- Evaluates performance for values of `k` between 3 and 10.
## Sample Output
```bash
k value  : 3
accuracy : 76.62338%
...
best k value  : 7
best accuracy : 78.57143%
```
## Structure
- `KNN` class - Custom classifier logic
- `calc_euclid_dist` - Function to compute distance between samples
- `load_and_clean_data` - Handles loading and preprocessing
- `main()` - Orchestrates training/testing and prints accuracy
## Getting Started
1. Clone the repo
`git clone https://github.com/yourusername/K-NN-diabetes.git`
2. Place `diabetes.csv` in the project directory.
3. Run:
`pyton knn_classifier.py`
## Requirements
- `pandas`
- `numpy`

Install with:
```bash
pip install pandas numpy
```
## Notes
- This is a learning/demo project. For production use, prefer scikit-learn's optimised KNN implementation.
- No external libraries like `sklearn` are used intentionally to show the internals of KNN.
## License
MIT License
