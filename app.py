import streamlit as st
import joblib

@st.cache_resource
def load_files():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_files()

st.title("Movie Review Sentiment Analysis")
st.write("Type a movie review. The model will say if it is positive or negative.")

review = st.text_area("Your review")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please type a review first.")
    else:
        vec = tfidf.transform([review])
        label = model.predict(vec)[0]
        prob = model.predict_proba(vec).max()
        if label == "positive":
            st.success(f"Positive ({prob:.0%} sure)")
        else:
            st.error(f"Negative ({prob:.0%} sure)")
