Customer Churn Prediction
Leveraging Machine Learning to Proactively Identify and Mitigate Customer Attrition.

Table of Contents
1. Project Overview

2. Methodology and Technical Stack

3. Key Findings and Results

4. Repository Contents

5. How to Run the Project

6. Author

1. Project Overview
This project focuses on developing a predictive model to identify customers at high risk of churning from a telecommunications company. By analyzing historical customer data from the Telco Customer Churn dataset, the goal was to build a machine learning system that could predict churn with high accuracy. The solution addresses real-world data challenges, such as class imbalance, and provides actionable insights by identifying the most influential factors contributing to churn.

2. Methodology and Technical Stack
The project followed a robust machine learning pipeline implemented in Python using Jupyter Notebooks. The key steps are outlined below:

Data Exploration and Preprocessing: The raw dataset was analyzed for missing values and inconsistencies. The TotalCharges column was cleaned and converted to a numeric type, and missing values were handled.

Feature Engineering: All categorical features were converted to a numerical format using one-hot encoding with pandas.get_dummies(). This step was essential to prepare the data for the machine learning algorithms.

Handling Class Imbalance: The dataset's target variable (Churn) was heavily imbalanced. This was addressed by applying the SMOTE (Synthetic Minority Oversampling Technique) algorithm on the training data to create a more balanced dataset.

Model Development and Evaluation: Several classification models, including Logistic Regression, Random Forest, and XGBoost, were trained and evaluated. The models were assessed using key metrics such as Accuracy, Precision, and Recall.

Model Optimization: GridSearchCV was used to perform hyperparameter tuning on the Random Forest model, which helped in identifying the optimal parameters and improving its performance.

The primary libraries used in this project are:

Pandas: For data manipulation and analysis.

Numpy: For numerical operations.

Scikit-learn: For machine learning models and utilities.

Matplotlib and Seaborn: For data visualization.

Imbalanced-learn: For handling imbalanced datasets with SMOTE.

XGBoost: For the XGBoost classification model.

3. Key Findings and Results
The final model, a Logistic Regression model trained on SMOTE-resampled data, demonstrated strong predictive capabilities. The key performance metrics for the model are:

Accuracy: 79%

Recall (for Churn): 57%

Precision (for Churn): 61%

The high recall for the churn class is particularly important as it indicates the model's effectiveness in identifying at-risk customers, allowing the business to take proactive measures. Feature importance analysis revealed that Contract, Tenure, and MonthlyCharges were the most significant predictors of customer churn.

4. Repository Contents
The repository is structured as follows:

Customer-Churn-Prediction/
├── Customer-Churn-Prediction.ipynb
├── requirements.txt
├── .gitignore
└── README.md

Customer-Churn-Prediction.ipynb: The main Jupyter Notebook containing all the code for data preprocessing, model training, and evaluation.

requirements.txt: A list of all necessary Python libraries and their versions.

.gitignore: Specifies files to be ignored by Git (e.g., virtual environment folders).

README.md: The project description and instructions you are currently reading.

5. How to Run the Project
To run this project on your local machine, follow these steps:

Clone this repository:

git clone https://github.com/nksk047/Customer-Churn-Prediction.git

Navigate to the project directory:

cd Customer-Churn-Prediction

(Optional but recommended) Create and activate a virtual environment:

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install the required libraries from requirements.txt:

pip install -r requirements.txt

Launch Jupyter Notebook and open Customer-Churn-Prediction.ipynb to explore the code and results.

jupyter notebook

6. Author
Nitin Kumar - GitHub Profile
