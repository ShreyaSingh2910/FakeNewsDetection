1) Data Collection ->
Two datasets were used:
Fake.csv - Fake news articles
True.csv - Real news articles

2️) Data Preprocessing-> 
Merged datasets
Cleaned text data
Removed unnecessary columns
Converted labels into numerical format.

3) Feature Extraction -> 
Text data was converted into numerical features using TF-IDF (Term Frequency – Inverse Document Frequency) and N-Grams.

4️) Model Training ->
Two machine learning models were used:
Naive Bayes
Logistic Regression
These models learn patterns in text to classify news articles.

5️) Deployment -> 
The trained model was deployed using Streamlit, allowing users to input a news headline or article and instantly check whether it is Fake or Real.
