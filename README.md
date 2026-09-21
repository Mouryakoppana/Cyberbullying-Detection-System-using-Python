🛑 Cyberbullying Detection System using Python

📌 Project Overview

This project aims to identify and classify cyberbullying content in social media comments using data science and machine learning techniques.  
The system works on an uncleaned dataset containing user-generated comments along with engagement metrics such as likes, replies, and report counts.

The workflow includes data preprocessing, exploratory data analysis (EDA), visualization, statistical analysis, and machine learning-based text classification.

---

🎯 Objectives

- Detect cyberbullying in online textual data  
- Perform data cleaning on unstructured and incomplete datasets  
- Analyze patterns in user behavior and engagement  
- Visualize key insights using graphs  
- Build and evaluate a machine learning model for classification  

---

📊 Key Insights

- Comments labeled as bullying tend to have higher report counts  
- Engagement metrics such as likes and replies vary across platforms  
- Certain text patterns and keywords are strong indicators of toxic behavior  
- Data visualization helps in understanding distribution and relationships  
- The classification model achieves reliable performance in detecting bullying content  

---

🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

📁 Project Structure

Cyberbullying-Detection/
│

├── analysis.py                         # Main Python script  
├── cyberbullying_uncleaned_dataset.csv # Dataset file  
├── README.md                           # Project documentation  
├── requirements.txt                    # Required libraries  

---

⚙️ Features

✔ Data cleaning and preprocessing  
✔ Handling missing and inconsistent data  
✔ Text normalization for analysis  
✔ Exploratory Data Analysis (EDA)  
✔ Statistical analysis and correlation  
✔ Multiple visualizations for insights  
✔ Machine learning classification model  
✔ Prediction for custom input text  

---

📈 Visualizations Included

- Distribution of bullying vs non-bullying comments  
- Platform-wise comment analysis  
- Histogram of likes and replies  
- Boxplots for detecting outliers  
- Language distribution  
- Report count analysis  
- Correlation heatmap  
- Pairplot for feature relationships  
- Confusion matrix for model evaluation  

---

📊 Statistical Analysis

- Descriptive statistics (mean, median, standard deviation)  
- Correlation between numerical features  
- Identification of outliers using visualization techniques  

---

🤖 Machine Learning

- Model Used: Multinomial Naive Bayes  
- Feature Extraction: TF-IDF Vectorization  
- Target Variable: Bully / Non-bully classification  

Evaluation Metrics

- Accuracy Score  
- Confusion Matrix  
- Precision, Recall, and F1-score  

---

▶️ How to Run

1. Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

2. Run the Python script:

python analysis.py

---

📌 Conclusion

The project demonstrates the application of data science techniques in identifying harmful online behavior.  
By combining data cleaning, visualization, and machine learning, the system effectively classifies cyberbullying content and provides meaningful insights into user interaction patterns.

---

👨‍💻 Author

Koppana Srikar Mourya

---

🚀 Future Improvements

- Implement advanced models such as Logistic Regression, SVM, or Deep Learning  
- Expand dataset with multilingual support  
- Develop a real-time detection system  
- Deploy as a web application using Streamlit  
- Integrate live data sources through APIs
