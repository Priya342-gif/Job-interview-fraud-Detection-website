"""
Fraud Detection Web Application
================================
Streamlit app for detecting job/interview scams
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack, csr_matrix
import json

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    color: #1f77b4;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    text-align: center;
    color: #666;
    margin-bottom: 2rem;
}
.scam-alert {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
    padding: 1.5rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.legit-alert {
    background-color: #e8f5e9;
    border-left: 5px solid #4caf50;
    padding: 1.5rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.metric-box {
    background-color: #f5f5f5;
    padding: 1rem;
    border-radius: 5px;
    text-align: center;
}
.red-flag {
    background-color: #fff3cd;
    border-left: 4px solid #ff9800;
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-radius: 3px;
}
</style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing objects"""
    try:
        model = joblib.load('fraud_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        scaler = joblib.load('scaler.joblib')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, vectorizer, scaler, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Feature extraction function
def extract_features(text):
    """Extract numeric features from text"""
    if pd.isna(text) or not isinstance(text, str):
        return np.zeros(11)
    
    text = str(text)
    words = text.split()
    
    money_keywords = ['fee', 'pay', 'payment', 'deposit', 'transfer', 'money', 
                     'rupees', 'dollars', 'amount', 'charge', 'cost', 'price']
    
    features = [
        len(text),  # text_length
        len(words),  # word_count
        sum(c.isdigit() for c in text),  # digit_count
        text.count('!'),  # exclamation_count
        text.count('?'),  # question_count
        text.count('₹') + text.count('$') + text.count('£'),  # currency_symbols
        int(bool(re.search(r'http[s]?://', text))),  # has_url
        int(bool(re.search(r'\S+@\S+', text))),  # has_email
        int(bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text))),  # has_phone
        sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,  # uppercase_ratio
        sum(keyword in text.lower() for keyword in money_keywords)  # money_words
    ]
    
    return np.array(features)

# Detect red flags
def detect_red_flags(text):
    """Detect suspicious patterns in text"""
    red_flags = []
    text_lower = text.lower()
    
    # Payment-related red flags
    payment_keywords = ['fee', 'pay', 'payment', 'deposit', 'transfer', 'charge', 'cost']
    if any(kw in text_lower for kw in payment_keywords):
        red_flags.append("⚠️ Payment demand detected")
    
    # Urgency language
    urgency_keywords = ['urgent', 'immediately', 'hurry', 'asap', 'now', 'today', 'limited time']
    if any(kw in text_lower for kw in urgency_keywords):
        red_flags.append("⚠️ Urgency language used")
    
    # Guaranteed/promise language
    guarantee_keywords = ['guaranteed', 'promise', '100%', 'sure', 'definitely']
    if any(kw in text_lower for kw in guarantee_keywords):
        red_flags.append("⚠️ Unrealistic guarantees")
    
    # Personal information requests
    personal_keywords = ['aadhaar', 'aadhar', 'pan', 'credit card', 'debit card', 'bank account', 'password']
    if any(kw in text_lower for kw in personal_keywords):
        red_flags.append("⚠️ Personal information requested")
    
    # Registration fee
    if 'registration' in text_lower and any(kw in text_lower for kw in payment_keywords):
        red_flags.append("⚠️ Registration fee mentioned")
    
    # No interview mentioned
    if 'without interview' in text_lower or 'no interview' in text_lower:
        red_flags.append("⚠️ Job offer without proper interview")
    
    # Suspicious contact methods
    suspicious_contact = ['whatsapp only', 'telegram only', 'gmail', 'yahoo']
    if any(kw in text_lower for kw in suspicious_contact):
        red_flags.append("⚠️ Suspicious contact method")
    
    # Money amounts
    if re.search(r'[₹$£]\s*\d+', text) or re.search(r'\d+\s*rupees', text_lower):
        red_flags.append("⚠️ Specific money amount mentioned")
    
    return red_flags

# Predict function
def predict_fraud(text, model, vectorizer, scaler):
    """Make prediction on input text"""
    # Text vectorization
    text_vec = vectorizer.transform([text])
    
    # Extract features
    features = extract_features(text).reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # Combine
    X_combined = hstack([text_vec, csr_matrix(features_scaled)])
    
    # Predict
    prediction = model.predict(X_combined)[0]
    probability = model.predict_proba(X_combined)[0]
    
    return prediction, probability

# Load model
model, vectorizer, scaler, metadata = load_model_artifacts()

# Header
st.markdown('<p class="main-header">🔍 Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered detection of job and interview scams</p>', unsafe_allow_html=True)

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model Type", metadata['best_model'])
    st.metric("ROC-AUC Score", f"{metadata['roc_auc']:.4f}")
    st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    st.metric("Training Samples", f"{metadata['train_samples']:,}")
    
    st.markdown("---")
    st.header("📖 How to Use")
    st.markdown("""
    1. **Paste text** in the input box
    2. **Click** "Analyze Text"
    3. **Review** the prediction and red flags
    4. **Make** an informed decision
    """)
    
    st.markdown("---")
    st.header("⚠️ Disclaimer")
    st.caption("This tool provides AI-based predictions and should be used as a guide, not as the sole decision-making factor. Always verify job offers through official channels.")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Analyze Text", "🖼️ Image Upload", "📈 Example Cases", "ℹ️ About"])

# TAB 1: TEXT INPUT
with tab1:
    st.header("Analyze Job/Interview Message")
    
    # Text input
    user_input = st.text_area(
        "Paste job posting, interview message, or email here:",
        height=200,
        placeholder="Example: Congratulations! You have been selected for the position. Please pay ₹5000 registration fee to confirm your interview slot."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    if analyze_button:
        if user_input.strip():
            with st.spinner("Analyzing..."):
                # Make prediction
                prediction, probability = predict_fraud(user_input, model, vectorizer, scaler)
                confidence = probability[prediction] * 100
                
                # Detect red flags
                red_flags = detect_red_flags(user_input)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main prediction
                if prediction == 1:  # Scam
                    st.markdown(f"""
                    <div class="scam-alert">
                        <h2 style="color: #f44336; margin: 0;">🚨 FRAUD DETECTED</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                            This message shows signs of being a scam with <strong>{confidence:.1f}% confidence</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Legitimate
                    st.markdown(f"""
                    <div class="legit-alert">
                        <h2 style="color: #4caf50; margin: 0;">✅ APPEARS LEGITIMATE</h2>
                        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                            This message appears to be legitimate with <strong>{confidence:.1f}% confidence</strong>.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Scam Probability", f"{probability[1]*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Legitimate Probability", f"{probability[0]*100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Red flags
                if red_flags:
                    st.markdown("---")
                    st.subheader("🚩 Red Flags Detected")
                    for flag in red_flags:
                        st.markdown(f'<div class="red-flag">{flag}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("---")
                    st.success("✅ No obvious red flags detected in the text.")
                
                # Recommendations
                st.markdown("---")
                st.subheader("💡 Recommendations")
                if prediction == 1:
                    st.warning("""
                    **If this is a suspected scam:**
                    - ❌ Do NOT make any payments
                    - ❌ Do NOT share personal/financial information
                    - ✅ Verify the company through official sources
                    - ✅ Check company website and reviews
                    - ✅ Report to relevant authorities if confirmed scam
                    """)
                else:
                    st.info("""
                    **Even for legitimate-looking messages:**
                    - ✅ Always verify company details
                    - ✅ Check official company website
                    - ✅ Research the recruiter on LinkedIn
                    - ✅ Never pay upfront fees for job applications
                    - ✅ Trust your instincts
                    """)
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# TAB 2: IMAGE UPLOAD (Coming Soon)
with tab2:
    st.header("🖼️ Image Upload Feature")
    st.info("""
    **📸 Image Analysis Feature Available!**
    
    To enable screenshot/image analysis with OCR:
    
    **Step 1: Install Tesseract OCR**
    1. Download from: [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
    2. Download: `tesseract-ocr-w64-setup-5.x.x.exe`
    3. Run installer with default settings
    
    **Step 2: Install Python Package**
    ```
    pip install pytesseract Pillow
    ```
    
    **Step 3: Run Image-Enabled Version**
    ```
    streamlit run app_with_image.py
    ```
    
    ---
    
    **What you'll be able to do:**
    - ✅ Upload WhatsApp screenshots
    - ✅ Upload job posting images
    - ✅ Extract text automatically with OCR
    - ✅ Analyze for fraud detection
    
    **Installation time:** ~5 minutes
    """)
    
    st.markdown("---")
    st.subheader("🎯 Why Image Support?")
    st.markdown("""
    Most scam messages come via:
    - WhatsApp screenshots
    - Email screenshots  
    - Social media images
    
    OCR (Optical Character Recognition) automatically extracts text from images so the AI model can analyze it.
    """)

with tab3:
    st.header("📈 Example Test Cases")
    st.markdown("Try these examples to see how the system works:")
    
    examples = [
        {
            "title": "🚨 Clear Scam Example",
            "text": "Congratulations! You have been selected for the Software Engineer position at TechCorp. Please pay ₹5,000 registration fee immediately to confirm your interview slot. Payment must be made today via UPI to secure your position.",
            "expected": "SCAM"
        },
        {
            "title": "✅ Legitimate Example",
            "text": "Thank you for applying to the Backend Developer position at Google. Your application has been reviewed and we would like to invite you for a technical interview. The interview will be conducted via Google Meet on Monday, October 10th at 10:00 AM. Please confirm your availability.",
            "expected": "LEGITIMATE"
        },
        {
            "title": "🚨 Certification Scam",
            "text": "Great news! After reviewing your profile, we are offering you a Digital Marketing Intern position. To proceed, you need to complete our mandatory certification course for $150. Once paid, you will receive your offer letter and joining details.",
            "expected": "SCAM"
        },
        {
            "title": "⚠️ Suspicious Contact",
            "text": "Hi, we found your resume online and have a perfect job for you. Salary is ₹50,000/month work from home. No interview needed. Contact us on WhatsApp only: 9876543210",
            "expected": "SCAM"
        }
    ]
    
    for i, example in enumerate(examples):
        with st.expander(f"{example['title']} (Expected: {example['expected']})"):
            st.text_area(f"Example {i+1}", example['text'], height=100, key=f"example_{i}", disabled=True)
            if st.button(f"Test Example {i+1}", key=f"btn_{i}"):
                prediction, probability = predict_fraud(example['text'], model, vectorizer, scaler)
                result = "🚨 SCAM" if prediction == 1 else "✅ LEGITIMATE"
                confidence = probability[prediction] * 100
                st.markdown(f"**Result:** {result} (Confidence: {confidence:.1f}%)")

# TAB 4: ABOUT
with tab4:
    st.header("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Purpose")
        st.markdown("""
        This fraud detection system uses machine learning to identify potential job and interview scams. 
        It analyzes text patterns, linguistic features, and suspicious indicators to help job seekers 
        avoid fraudulent opportunities.
        """)
        
        st.subheader("🔬 Technology")
        st.markdown(f"""
        - **Model:** {metadata['best_model']}
        - **Accuracy:** {metadata['accuracy']:.2%}
        - **Features:** Text analysis + {len(metadata['features'])} engineered features
        - **Training Data:** {metadata['train_samples']:,} samples
        """)
    
    with col2:
        st.subheader("⚠️ Common Scam Indicators")
        st.markdown("""
        - Upfront payment requests
        - Urgency pressure ("pay now", "limited slots")
        - No proper interview process
        - Unrealistic salary promises
        - Personal information demands
        - Communication via WhatsApp/Telegram only
        - Generic offer letters without company details
        - Certification/training fee requirements
        """)
        
        st.subheader("✅ Legitimate Job Indicators")
        st.markdown("""
        - Proper interview rounds
        - Official company email domain
        - Verifiable company information
        - No upfront payment requests
        - Clear job description and expectations
        - Professional communication
        - Background verification process
        """)
    
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{metadata['roc_auc']:.4f}")
    with col2:
        st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    with col3:
        st.metric("Precision (Scam)", f"{metadata['precision_scam']:.2%}")
    with col4:
        st.metric("Recall (Scam)", f"{metadata['recall_scam']:.2%}")
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit and Scikit-learn")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Fraud Detection System v1.0</strong></p>
    <p>Protecting job seekers from scams using AI</p>
</div>
""", unsafe_allow_html=True)
