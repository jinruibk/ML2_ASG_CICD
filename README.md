-ML2 Assignment – CI/CD Pipeline for Bike Demand Prediction



\-Project Overview



This project implements an automated machine learning pipeline using GitHub Actions

to ensure that only high-quality models are deployed for bike demand prediction.



The system applies a quality gate to verify model performance before acceptance.





\-Project Structure



ML2\_ASG\_CICD/

│

├── src/ # Training scripts (Task 1 \& 2)

├── tests/ # Quality gate test script

│ └── test\_model.py

├── data/ # Evaluation dataset

│ └── day\_2012.csv

├── models/ # Saved trained models

│ └── bike\_demand\_rf\_model.joblib

├── .github/workflows/ # GitHub Actions workflow

│ └── python-app.yml

├── requirements.txt # Dependencies

└── README.md







-Installation



Install required dependencies:



```bash

pip install -r requirements.txt





-Running the Quality Test

To run the quality gate locally:



python tests/test\_model.py

If the model meets the performance threshold, the test will pass.

Otherwise, the test will fail.



-CI/CD Pipeline

The GitHub Actions workflow automatically:

Sets up a Python environment
Installs dependencies
Loads the trained model
Evaluates performance
Applies quality gate checks
Passes or fails the build


Quality Gate Criteria

The deployed model must satisfy the following performance requirements based on baseline results obtained in Task 1:

RMSE ≤ Baseline RMSE × Quality Factor (1.20)
MAE ≤ Baseline MAE × Quality Factor (1.20)
R² ≥ Minimum Threshold (0.80)

These thresholds allow for minor performance fluctuations while preventing significant degradation caused by data drift or modelling issues.

If any of the above conditions are not met, the CI/CD pipeline automatically fails and deployment is blocked.



Author

Name: Teo Jin Rui

Course: Machine Learning 2

Institution: Ngee Ann Polytechnic

