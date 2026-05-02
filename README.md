# Credit Risk Prediction

## 🎯 Project Goal
To develop a machine learning model capable of predicting loan default risk using:
- Home ownership status
- Loan intent(Education, Medical, Personal, etc.)
- Loan Grade(A through G)
- Annual income and loan amount
- Employement length and credit history length
- Interest rate and loan to income ratio

## Workflow and Key steps
- ### Importing Required Libraries
   - Loaded essential Python libraries: pandas, numpy, matplotlib,seaborn and scikit-learn for data handling, visualization, and modeling
- ### Loading the Dataset
   - Dataset : credit risk dataset
   - Used .head(), .shape(), .info(), and .describe() to understand the dataset structure and statistics
- ### Data Exploration and Cleaning
   - Checked for missing values and duplicates
   - Verified data type and converted them where necessary
   - Clipped outliers
   - Ensured dataset consistency and readiness for analysis
- ### Exploratory Data Analysis(EDA)
   - Analysed distribution of borrower age, income,and credit history by loan status
   - Explored loan grade, loan intent, and home ownership patterns
   - Identified key risk indicators using visual analysis
- ### Data Preprocessing
   - Encoded categorical columns using One Hot Encoding and Ordinal Encoding
   - scaled numerical columns using StandardScaler
   - Split dataset into features(X) and target(y)
- ### Handling class Imbalance
   - Applied SMOTE to balance the training data before model training
- ### Model Building
   - Split the dataset into training and testing sets
   - Trained and compared multiple algorithms
     - Logistic Regression
     - Decision Tree Classifier
     - K-Nearest Neighbors(KNN)
     - Random Forest Classifier
- ### Model Evaluation
   - Evaluated models using
     - Accuracy Score
     - Confusion Matrix
     - Classification Report(Precision, Recall, F1-Score)
   - Found that Random Forest outperformed all other algorithms
- ### Model saving
   - Saved the best model using pickle as RF_model.pkl for use in the dashboard

## Technologies Used
- Python 3 
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit

## Future Scope
- Perform hyperparameter tuning with RandomizedSearchCV
- Implement advanced models such as XGBoost and Gradient Boosting
- Retrain the model including numerical features
- Deploy the dashboard publicaly using Streamlit Cloud


