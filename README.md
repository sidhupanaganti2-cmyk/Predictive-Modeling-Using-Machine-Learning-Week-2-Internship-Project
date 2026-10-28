# Predictive Modeling Using Machine Learning | Week 2 Internship Project

## Project Overview

This project demonstrates a supervised machine learning workflow using Python and Scikit-learn. Three classification algorithms are trained and evaluated on the Breast Cancer Wisconsin Diagnostic dataset to compare their predictive performance.

The project focuses on model training, testing, evaluation, visualization, and reproducible results.

## Objectives

- Understand supervised machine learning.
- Split data into training and testing sets.
- Train Logistic Regression, Decision Tree, and Random Forest classifiers.
- Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Visualize model performance using confusion matrices and ROC curves.
- Compare algorithms and save the selected model.
- Generate predictions and evaluation reports.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Git and GitHub

## Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset included in Scikit-learn.

The dataset contains numeric features derived from breast mass images. The target represents benign or malignant classifications.

This dataset is used strictly for educational machine learning practice.

## Machine Learning Algorithms

1. **Logistic Regression** — baseline classification model.
2. **Decision Tree** — classification using learned decision rules.
3. **Random Forest** — ensemble classification using multiple decision trees.

## Workflow

1. Load and inspect the dataset.
2. Separate input features and target labels.
3. Split the data into training and testing sets.
4. Train the three classification models.
5. Generate predictions and class probabilities.
6. Evaluate the models using multiple metrics.
7. Generate confusion matrices, ROC curves, and comparison charts.
8. Save the selected model, predictions, and reports.

## Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted positive cases that are correct.
- **Recall:** Proportion of actual positive cases correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Ability to distinguish between positive and negative classes across thresholds.

The positive class is malignant. Model performance should be interpreted using multiple metrics rather than accuracy alone.

## Project Structure

```text
Week 2 Internship/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── best_model.joblib
└── outputs/
    ├── charts/
    └── reports/
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/sidhupanaganti2-cmyk/Predictive-Modeling-Using-Machine-Learning-Week-2-Internship-Project.git
cd sidhupanaganti2-cmyk
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

The program trains and evaluates the models, creates visualizations, and saves the results automatically.

## Outputs

- `outputs/reports/model_comparison.csv` — evaluation metrics for all models.
- `outputs/reports/best_model_predictions.csv` — test-set predictions.
- `outputs/reports/project_summary.txt` — project summary and selected model metrics.
- `outputs/charts/` — confusion matrices, ROC curves, and performance comparisons.
- `models/best_model.joblib` — saved selected model.

## Key Learning Outcomes

- Supervised machine learning
- Classification algorithms
- Training and testing workflows
- Model evaluation and comparison
- Confusion matrix and ROC analysis
- Saving and reusing trained models
- GitHub project documentation

## Limitations and Disclaimer

This is an educational project, not a clinically validated diagnostic system. Model scores depend on the dataset and test split and do not establish real-world medical performance. Clinical use requires rigorous independent validation and qualified expert oversight.

## Author

**PANUGANTI SIDHU**

Data Analytics / Machine Learning Internship | Week 2

## License

For educational and portfolio purposes.
