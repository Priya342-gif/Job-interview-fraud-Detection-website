"""
Fraud Detection Web Application - WITH IMAGE SUPPORT
====================================================
Streamlit app with Text + Image (OCR) support
Lighter version without audio (no heavy Whisper dependency)
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack, csr_matrix
import json
from PIL import Image
import io

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
.extracted-text {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 1rem;
    border-radius: 5px;
    font-family: monospace;
    white-space: pre-wrap;
    max-height: 300px;
    overflow-y: auto;
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

# OCR function with better error handling
def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        import pytesseract
        
        # Try to find tesseract executable on Windows
        try:
            pytesseract.get_tesseract_version()
        except:
            # Common Windows installation paths
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            ]
            for path in possible_paths:
                import os
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        # Perform OCR
        text = pytesseract.image_to_string(image, lang='eng')
        
        if not text.strip():
            return None, "⚠️ No text found in image. Make sure the image contains clear, typed text."
        
        return text, None
    except ImportError:
        return None, """
        ❌ **Tesseract OCR not installed.**
        
        **To enable image analysis:**
        1. Install Python package: `pip install pytesseract`
        2. Download & install Tesseract:
           - Windows: https://github.com/UB-Mannheim/tesseract/wiki
           - Choose latest version (e.g., tesseract-ocr-w64-setup-5.x.x.exe)
        3. Restart this app
        """
    except pytesseract.TesseractNotFoundError:
        return None, """
        ❌ **Tesseract executable not found.**
        
        **Please install Tesseract OCR:**
        - Windows: https://github.com/UB-Mannheim/tesseract/wiki
        - Download and run installer
        - Default installation path: `C:\\Program Files\\Tesseract-OCR`
        """
    except Exception as e:
        return None, f"Error during OCR: {str(e)}"

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
        len(text),
        len(words),
        sum(c.isdigit() for c in text),
        text.count('!'),
        text.count('?'),
        text.count('₹') + text.count('$') + text.count('£'),
        int(bool(re.search(r'http[s]?://', text))),
        int(bool(re.search(r'\S+@\S+', text))),
        int(bool(re.search(r'\d{3}[-.]?\d{3}[-.]?\d{4}', text))),
        sum(c.isupper() for c in text) / len(text) if len(text) > 0 else 0,
        sum(keyword in text.lower() for keyword in money_keywords)
    ]
    
    return np.array(features)

# Detect red flags
def detect_red_flags(text):
    """Detect suspicious patterns in text"""
    red_flags = []
    text_lower = text.lower()
    
    payment_keywords = ['fee', 'pay', 'payment', 'deposit', 'transfer', 'charge', 'cost']
    if any(kw in text_lower for kw in payment_keywords):
        red_flags.append("⚠️ Payment demand detected")
    
    urgency_keywords = ['urgent', 'immediately', 'hurry', 'asap', 'now', 'today', 'limited time']
    if any(kw in text_lower for kw in urgency_keywords):
        red_flags.append("⚠️ Urgency language used")
    
    guarantee_keywords = ['guaranteed', 'promise', '100%', 'sure', 'definitely']
    if any(kw in text_lower for kw in guarantee_keywords):
        red_flags.append("⚠️ Unrealistic guarantees")
    
    personal_keywords = ['aadhaar', 'aadhar', 'pan', 'credit card', 'debit card', 'bank account', 'password']
    if any(kw in text_lower for kw in personal_keywords):
        red_flags.append("⚠️ Personal information requested")
    
    if 'registration' in text_lower and any(kw in text_lower for kw in payment_keywords):
        red_flags.append("⚠️ Registration fee mentioned")
    
    if 'without interview' in text_lower or 'no interview' in text_lower:
        red_flags.append("⚠️ Job offer without proper interview")
    
    suspicious_contact = ['whatsapp only', 'telegram only', 'gmail', 'yahoo']
    if any(kw in text_lower for kw in suspicious_contact):
        red_flags.append("⚠️ Suspicious contact method")
    
    if re.search(r'[₹$£]\s*\d+', text) or re.search(r'\d+\s*rupees', text_lower):
        red_flags.append("⚠️ Specific money amount mentioned")
    
    return red_flags

# Predict function
def predict_fraud(text, model, vectorizer, scaler):
    """
    Make prediction with red flag override logic
    If 3+ red flags detected → Force SCAM classification
    """
    text_vec = vectorizer.transform([text])
    features = extract_features(text).reshape(1, -1)
    features_scaled = scaler.transform(features)
    X_combined = hstack([text_vec, csr_matrix(features_scaled)])
    prediction = model.predict(X_combined)[0]
    probability = model.predict_proba(X_combined)[0]
    
    # Get red flags
    red_flags = detect_red_flags(text)
    red_flag_count = len(red_flags)
    
    # OVERRIDE LOGIC: 3+ red flags → Force SCAM
    if red_flag_count >= 3:
        prediction = 1
        adjusted_scam_prob = min(0.95, probability[1] + (red_flag_count * 0.10))
        probability = np.array([1 - adjusted_scam_prob, adjusted_scam_prob])
    elif red_flag_count == 2 and any('Payment demand' in flag for flag in red_flags):
        prediction = 1
        adjusted_scam_prob = min(0.90, probability[1] + 0.25)
        probability = np.array([1 - adjusted_scam_prob, adjusted_scam_prob])
    
    return prediction, probability

# Display results function
def display_results(text, prediction, probability, red_flags, source_type="text"):
    """Display prediction results"""
    confidence = probability[prediction] * 100
    
    st.markdown("---")
    st.subheader(f"📊 Analysis Results ({source_type.upper()})")
    
    if prediction == 1:
        st.markdown(f"""
        <div class="scam-alert">
            <h2 style="color: #f44336; margin: 0;">🚨 FRAUD DETECTED</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                This message shows signs of being a scam with <strong>{confidence:.1f}% confidence</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="legit-alert">
            <h2 style="color: #4caf50; margin: 0;">✅ APPEARS LEGITIMATE</h2>
            <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
                This message appears to be legitimate with <strong>{confidence:.1f}% confidence</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    if red_flags:
        st.markdown("---")
        st.subheader("🚩 Red Flags Detected")
        for flag in red_flags:
            st.markdown(f'<div class="red-flag">{flag}</div>', unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.success("✅ No obvious red flags detected in the text.")
    
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
        - ✅ Never pay upfront fees
        - ✅ Trust your instincts
        """)

# Load model
model, vectorizer, scaler, metadata = load_model_artifacts()

# Header
st.markdown('<p class="main-header">🔍 Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered detection with Text & Image support</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model Type", metadata['best_model'])
    st.metric("ROC-AUC Score", f"{metadata['roc_auc']:.4f}")
    st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    st.metric("Training Samples", f"{metadata['train_samples']:,}")
    
    st.markdown("---")
    st.header("🎯 Supported Inputs")
    st.markdown("""
    - 📝 **Text** - Direct paste
    - 🖼️ **Image** - Screenshot OCR
    """)
    
    st.markdown("---")
    st.header("⚠️ Disclaimer")
    st.caption("AI predictions are guidance only. Always verify through official channels.")

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🖼️ Image Upload", "ℹ️ About"])

# TAB 1: TEXT INPUT
with tab1:
    st.header("🔍 Analyze Text Message")
    
    user_input = st.text_area(
        "Paste job posting, interview message, or email here:",
        height=200,
        placeholder="Example: Congratulations! Please pay ₹5000 to confirm your interview slot."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    if analyze_button:
        if user_input.strip():
            with st.spinner("Analyzing text..."):
                prediction, probability = predict_fraud(user_input, model, vectorizer, scaler)
                red_flags = detect_red_flags(user_input)
                display_results(user_input, prediction, probability, red_flags, "text")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

# TAB 2: IMAGE UPLOAD
with tab2:
    st.header("🖼️ Analyze Screenshot/Image")
    st.markdown("Upload a screenshot of a job posting, WhatsApp message, email, or any text image.")
    
    uploaded_image = st.file_uploader(
        "Choose an image file:",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("**📋 Tips for best OCR results:**")
            st.markdown("""
            - ✅ Use high-resolution images
            - ✅ Ensure text is clearly visible
            - ✅ Crop to text area only
            - ✅ Typed text works better than handwritten
            - ✅ Good lighting and contrast
            """)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_image_button = st.button("🔍 Extract & Analyze", type="primary", use_container_width=True)
        
        if analyze_image_button:
            with st.spinner("Extracting text from image..."):
                extracted_text, error = extract_text_from_image(image)
                
                if error:
                    st.error(error)
                else:
                    st.success("✅ Text extracted successfully!")
                    
                    with st.expander("📄 Extracted Text", expanded=True):
                        st.markdown(f'<div class="extracted-text">{extracted_text}</div>', unsafe_allow_html=True)
                    
                    with st.spinner("Analyzing extracted text..."):
                        prediction, probability = predict_fraud(extracted_text, model, vectorizer, scaler)
                        red_flags = detect_red_flags(extracted_text)
                        display_results(extracted_text, prediction, probability, red_flags, "image")

# TAB 3: ABOUT
with tab3:
    st.header("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Purpose")
        st.markdown("""
        AI-powered fraud detection for job and interview scams. 
        Analyzes text from direct input or images (via OCR).
        """)
        
        st.subheader("🔬 Technology")
        st.markdown(f"""
        - **Model:** {metadata['best_model']}
        - **Accuracy:** {metadata['accuracy']:.2%}
        - **OCR:** Tesseract
        - **Features:** {len(metadata['features'])} engineered features
        """)
    
    with col2:
        st.subheader("⚠️ Scam Indicators")
        st.markdown("""
        - Upfront payments
        - Urgency pressure
        - No proper interview
        - Personal info requests
        - WhatsApp/Telegram only
        - Registration fees
        """)
    
    st.markdown("---")
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{metadata['roc_auc']:.4f}")
    with col2:
        st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    with col3:
        st.metric("Precision", f"{metadata['precision_scam']:.2%}")
    with col4:
        st.metric("Recall", f"{metadata['recall_scam']:.2%}")
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit, Scikit-learn & Tesseract OCR")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p><strong>Fraud Detection System v1.5</strong></p>
    <p>Text • Image • AI-Powered</p>
</div>
""", unsafe_allow_html=True)
