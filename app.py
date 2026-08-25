import streamlit as st
from model import load_model, predict_sentiment

# Page Config
st.set_page_config(
    page_title="Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("Sentiment Analyzer")
st.markdown("Powered by DistilBERT (87% Accuracy)")

# Load model
@st.cache_resource
def get_model():
    return load_model()

model, tokenizer, device = get_model()

# Input Section
st.markdown("---")
st.subheader("Enter Your Text:")
user_input = st.text_area(
    "What's your thought?",
    placeholder="e.g., I absolutely loved this movie! The acting was incredible...",
    height=100,
    label_visibility="collapsed"
)

# Prediction Section
if user_input.strip():
    sentiment, confidence = predict_sentiment(user_input, model, tokenizer, device)
    
    st.markdown("---")
    
    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Sentiment", value=sentiment)
    
    with col2:
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
    
    # Confidence Bar
    st.markdown("### Confidence Score")
    st.progress(confidence)
    
    # Summary
    if confidence > 0.9:
        st.success("Very confident!")
    elif confidence > 0.8:
        st.info("Fairly confident.")
    else:
        st.warning("Moderate confidence. Could be borderline.")

# Sidebar
st.sidebar.markdown("""
### About
- Model: DistilBERT (Fine-tuned)
- Dataset: Sentiment140 (1.58M tweets)
- Accuracy: 87.12%
- Training Time: 11.5 hours

### How It Works
1. Text is tokenized into BERT tokens
2. DistilBERT encodes the tokens
3. Classification head predicts sentiment
4. Confidence score shows certainty
""")

# Footer
st.markdown("---")
st.caption("Model: DistilBERT | Sentiment: Positive/Negative")