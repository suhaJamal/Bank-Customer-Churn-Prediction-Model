## Bank Customer Churn Prediction Model

## Demo

Experience our "Bank Customer Churn Prediction Model" live! The web app is hosted on Streamlit and provides an interactive interface to explore the predictive model. You can input customer data and receive predictions on the likelihood of churn.

**Explore the Web App:** https://bank-customer-churn-prediction-model.streamlit.app/

This interactive tool allows you to:
- Enter customer details like geography, age, credit score, and more.
- View predictions in real-time as you adjust the inputs.
- Explore different scenarios and see how changes in customer data might affect churn predictions.

The web app serves as a practical demonstration of the model’s capabilities and offers insights into factors influencing customer retention strategies.

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
