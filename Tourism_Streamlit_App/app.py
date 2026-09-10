import streamlit as st

st.set_page_config(page_title="Tourism Experience Analytics", page_icon="🌍", layout="wide")
st.title("🌍 Tourism Experience Analytics")
st.markdown("### Classification, Prediction & Recommendation System")

st.markdown("""
This platform analyzes tourist behavior to:
1. **Predict** the rating a user might give an attraction
2. **Classify** their likely visit mode (Business/Couples/Family/Friends/Solo)
3. **Segment** users into behavioral clusters
4. **Recommend** personalized attractions

Trained on 52,930 transactions across 30 core attractions (with content-based recommendations extending to a 1,698-attraction catalog).

**Use the sidebar to navigate to the Predict page.**
""")