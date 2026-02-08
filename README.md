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

The model must achieve:

RMSE ≤ 95% of baseline RMSE (Task 1)

If this condition is not met, deployment is blocked.



Author

Name: Teo Jin Rui

Course: Machine Learning 2

Institution: Ngee Ann Polytechnic

