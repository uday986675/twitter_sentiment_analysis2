#!/usr/bin/env python3
"""
app.py

Streamlit web app for Twitter sentiment analysis.
Run with:
  streamlit run app.py
"""

import streamlit as st
import json
from predict import load_model, predict_json

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🐦 Twitter Sentiment Analyzer")
st.markdown("Classify tweets as **Positive**, **Negative**, or **Neutral** using TF-IDF + Logistic Regression.")

# Load model (cached for performance)
@st.cache_resource
def get_model():
    return load_model('models/tfidf_lr_pipeline.joblib')

model = get_model()

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** TF-IDF + Logistic Regression
    
    **Dataset:** Twitter sentiment corpus (74.6k training samples)
    
    **Validation Accuracy:** 97.83%
    
    **Classes:** 3 (Positive, Negative, Neutral)
    """)
    st.divider()
    st.markdown("""
    **Guidelines:**
    - Positive: praise, satisfaction, excitement, approval
    - Negative: frustration, anger, disappointment, sarcasm
    - Neutral: factual statements, announcements, mild complaints
    """)

# Input section
st.markdown("### Enter a Tweet")
input_method = st.radio("Input method:", ["Text Input", "Batch CSV Upload"], horizontal=True)

if input_method == "Text Input":
    tweet_text = st.text_area(
        "Tweet text:",
        placeholder="e.g., I absolutely love this game!",
        height=100
    )
    
    if st.button("🔍 Analyze", use_container_width=True):
        if tweet_text.strip():
            result = predict_json(model, tweet_text)
            
            # Display result with color coding
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment = result['sentiment']
                confidence = result['confidence']
                
                # Color coding
                color_map = {
                    'Positive': '🟢',
                    'Negative': '🔴',
                    'Neutral': '🟡'
                }
                
                st.markdown(f"## {color_map[sentiment]} **{sentiment}**")
                st.metric("Confidence", confidence)
            
            with col2:
                st.json(result)
            
            st.divider()
            st.markdown("**Tweet analyzed:**")
            st.text(tweet_text)
        else:
            st.warning("Please enter a tweet to analyze.")

else:  # Batch CSV Upload
    st.markdown("### Upload CSV with tweets")
    st.info("CSV should have a 'text' column containing tweet text.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        import pandas as pd
        
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'text' not in df.columns:
                st.error("CSV must have a 'text' column.")
            else:
                st.markdown(f"**File preview** ({len(df)} rows):")
                st.dataframe(df.head())
                
                if st.button("🔍 Analyze All", use_container_width=True):
                    # Predict for all rows
                    predictions = []
                    for idx, row in df.iterrows():
                        result = predict_json(model, row['text'])
                        predictions.append(result)
                    
                    # Create output dataframe
                    results_df = df.copy()
                    results_df['sentiment'] = [p['sentiment'] for p in predictions]
                    results_df['confidence'] = [p['confidence'] for p in predictions]
                    
                    # Display stats
                    st.markdown("### Results Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tweets", len(results_df))
                    with col2:
                        pos_count = (results_df['sentiment'] == 'Positive').sum()
                        st.metric("Positive", pos_count)
                    with col3:
                        neg_count = (results_df['sentiment'] == 'Negative').sum()
                        st.metric("Negative", neg_count)
                    
                    col4, col5 = st.columns(2)
                    with col4:
                        neutral_count = (results_df['sentiment'] == 'Neutral').sum()
                        st.metric("Neutral", neutral_count)
                    
                    # Display full results
                    st.markdown("### Full Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download button
                    csv_output = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_output,
                        file_name="sentiment_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

st.divider()
st.markdown("""
---
*Sentiment Analysis System | Built with Streamlit*
""")
