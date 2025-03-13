# 🚀 Detecting Fraudulent Credit Card Transactions | Through Advanced Machine Learning Algorithms.  

This project implements a **credit card fraud detection system** using **three different machine learning models**: **Logistic Regression, Random Forest, and Artificial Neural Networks (ANN)**. The goal is to develop a **highly accurate fraud detection system** that helps financial institutions minimize fraudulent transactions.  

![card image](https://github.com/user-attachments/assets/4d1434e1-a2db-466f-9cd3-1ba74c8f7ed3)

---

## 📌 Table of Contents  

- [🔍 Overview](#-overview)  
- [⚙️ Architecture & Workflow](#-architecture--workflow)  
- [🛠️ Tech Stack](#-tech-stack)  
- [📊 Models Implemented](#-models-implemented)  
  - [1️⃣ Logistic Regression](#1️⃣-logistic-regression)  
  - [2️⃣ Random Forest](#2️⃣-random-forest)  
  - [3️⃣ Artificial Neural Network (ANN)](#3️⃣-artificial-neural-network-ann)  
- [📂 Dataset](#-dataset)  
- [🧑‍💻 Preprocessing & Feature Engineering](#-preprocessing--feature-engineering)  
- [📈 Model Evaluation](#-model-evaluation)  
- [🚀 How to Run the Project](#-how-to-run-the-project)  
- [🎯 Key Takeaways](#-key-takeaways)  
- [🔮 Future Enhancements](#-future-enhancements)  
- [📜 License](#-license)  

---

## 🔍 Overview  

Credit card fraud detection is a critical application in the financial sector. This project builds an **end-to-end fraud detection pipeline** that uses a combination of classical machine learning and deep learning models. The system effectively detects fraudulent transactions while minimizing false positives.  

---

## ⚙️ Architecture & Workflow  

![Architecture Diagram](https://github.com/user-attachments/assets/c909acf5-3eaf-408f-a8ec-df7fa8ee32e1)


The project workflow consists of the following stages:  

1. **📥 Data Collection**: A large dataset containing real and fraudulent transactions.  
2. **🛠️ Preprocessing & Feature Engineering**: Cleaning, encoding categorical data, handling missing values, and feature scaling.  
3. **⚖️ Handling Class Imbalance**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset.  
4. **📊 Model Training**:  
   - **Logistic Regression**: Baseline model.  
   - **Random Forest**: Ensemble method improving accuracy.  
   - **ANN**: Deep learning model capturing complex patterns.  
5. **📌 Model Evaluation**: Measured using accuracy, precision, recall, F1-score, and ROC-AUC.  
6. **📉 Visualization**: Confusion matrix and feature importance analysis.  
7. **✅ Final Predictions**: Best-performing model is used to detect fraud.  

---

## 🛠️ Tech Stack  

- **📝 Programming Language**: Python  
- **📚 Libraries Used**:  
  - `pandas`, `numpy`: Data manipulation  
  - `matplotlib`, `seaborn`: Data visualization  
  - `scikit-learn`: Machine learning models  
  - `imbalanced-learn`: SMOTE for class balancing  
  - `tensorflow`, `keras`: Deep learning (ANN)  

---

## 📊 Models Implemented  

### 1️⃣ Logistic Regression  

- **🔹 Purpose**: A simple, interpretable classification model serving as a baseline.  
- **📌 Why Used?**: Helps understand feature importance and provides fast results.  
- **📈 Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, ROC-AUC.  

### 2️⃣ Random Forest  

- **🔹 Purpose**: An ensemble method that combines multiple decision trees.  
- **📌 Why Used?**: Improves accuracy and reduces overfitting.  
- **📊 Feature Importance**: Extracted from decision trees to identify key fraud indicators.  

### 3️⃣ Artificial Neural Network (ANN)  

- **🔹 Purpose**: A deep learning model that captures complex relationships in transaction data.  
- **🛠️ Architecture**:  
  - **3 Hidden Layers** with ReLU activation  
  - **Batch Normalization & Dropout** to prevent overfitting  
  - **Sigmoid Activation** in the output layer for fraud probability prediction  
- **📌 Why Used?**: Detects non-linear patterns and improves fraud detection.  

---

## 📂 Dataset  

- **Dataset Used**: A credit card transactions dataset with **50,000+ records**.  
- **Features**: Includes transaction details such as amount, time, merchant, category, and user information.  
- **Target Variable**: `is_fraud` (1: Fraud, 0: Not Fraud)  

---

## 🧑‍💻 Preprocessing & Feature Engineering  

- **Handling Missing Values**: Replaced or dropped missing values.  
- **Feature Encoding**: One-hot encoding for categorical variables.  
- **Feature Scaling**: Standardization using `StandardScaler`.  
- **Class Imbalance Handling**: Applied **SMOTE** to balance fraud cases.  

---

## 📈 Model Evaluation  

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 94.5%    | 0.71      | 0.43   | 0.54     | 89.3%   |
| Random Forest       | 97.2%    | 0.85      | 0.67   | 0.75     | 94.8%   |
| ANN                 | 99.5%    | 0.99      | 0.88   | 0.93     | 98.2%   |

✅ **Best Performing Model**: ANN achieved the highest accuracy and recall, making it the most effective at detecting fraud.  

---

## 🎯 Key Takeaways
✅ Machine learning can effectively detect fraudulent transactions.    
✅ ANN outperforms traditional models by capturing complex patterns.    
✅ Feature engineering and handling class imbalance significantly improve results.    
✅ Model evaluation with real-world metrics ensures reliability.   

--- 

## 🔮 Future Enhancements
- Deploy the best-performing model using Flask or FastAPI for real-time fraud detection.   
- Implement real-time fraud detection using Apache Kafka & Spark.    
- Optimize ANN hyperparameters using GridSearchCV or Bayesian Optimization.    
- Expand dataset with real-world banking fraud transactions for better generalization.    

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.




