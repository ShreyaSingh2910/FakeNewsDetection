import streamlit as st
import pickle

# load model and vectorizer
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("📰 Fake News Detection App")

st.write("Enter a news article below and the model will predict whether it is Fake or Real.")

# user input
news_text = st.text_area("Enter News Text")

if st.button("Predict"):

    if news_text.strip() == "":
        st.write("Please enter some text.")

    else:
        text_vector = vectorizer.transform([news_text])
        prediction = model.predict(text_vector)

        if prediction[0] == 0:
            st.error("🚨 Fake News")
        else:
            st.success("✅ Real News")

        # confidence score
        prob = model.predict_proba(text_vector)
        confidence = max(prob[0]) * 100
        st.write(f"Confidence: {confidence:.2f}%")

        # -------- NEW PART (IMPORTANT WORDS) --------

        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]

        indices = text_vector.nonzero()[1]

        word_importance = []

        for i in indices:
            word_importance.append((feature_names[i], coefficients[i]))

        top_words = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)[:5]

        st.subheader("Top words influencing prediction:")

        for word, weight in top_words:
            st.write(word)
