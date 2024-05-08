## Bank Customer Churn Prediction Model

### Overview
The goal of the "Bank Customer Churn Prediction Model" project is to predict the likelihood of bank customers discontinuing their services with the bank. This model helps to identify potential churners using various predictive modeling techniques, thereby assisting the bank in implementing effective customer retention strategies.

### Objectives
- Predict which customers are most likely to churn.
- Analyze customer behavior and demographics to identify potential reasons for churn.
- Develop strategies to improve customer retention based on predictive insights.

### Technologies Used
- Python
- Jupyter Notebook, Google Colab
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, LightGBM

### Installation and Usage
1. Clone this repository.
2. Ensure you have Python installed on your system.
3. Install the necessary Python packages:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm
   ```
4. Run the Jupyter Notebooks to view the analysis and predictive models.

### Data
The dataset used in this project is sourced from Kaggle and consists of various features like geography, gender, age, tenure, balance, and more. These features are used to predict whether a customer will exit the bank.

### Features
- **CreditScore**: A score assigned to a customer based on their credit history.
- **Geography**: The customer's location.
- **Gender**: The customer's gender.
- **Age**: The customer's age.
- **Tenure**: Number of years the customer has been with the bank.
- **Balance**: The customer's account balance.
- **NumOfProducts**: Number of products the customer is using.
- **HasCrCard**: Indicates whether the customer has a credit card.
- **IsActiveMember**: Indicates whether the customer is an active member.
- **EstimatedSalary**: The estimated salary of the customer.
- **Exited**: Indicates whether the customer has left the bank.

### Model Performance
The best performing model is a Random Forest Classifier with an accuracy of 93.5%, precision of 91.96%, and a recall of 73.15%.

### Contributing
If you would like to contribute to this project or have any suggestions, please open an issue or a pull request.
