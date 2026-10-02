"""
Fraud Detection Web Application - MULTIMODAL VERSION
====================================================
Streamlit app with Text, Image (OCR), and Audio (Speech-to-Text) support
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
    page_title="Fraud Detection System - Multimodal",
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

# OCR function
def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        import pytesseract
        from PIL import Image
        
        # Convert to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        # Perform OCR
        text = pytesseract.image_to_string(image, lang='eng')
        
        if not text.strip():
            return None, "No text found in image"
        
        return text, None
    except ImportError:
        return None, "❌ Tesseract OCR not installed. Please install: pip install pytesseract"
    except Exception as e:
        return None, f"Error during OCR: {str(e)}"

# Speech-to-Text function
def extract_text_from_audio(audio_bytes):
    """Extract text from audio using Whisper"""
    try:
        import whisper
        import tempfile
        import os
        
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Load Whisper model (tiny for speed, can use 'base' or 'small' for better accuracy)
        model = whisper.load_model("tiny")
        
        # Transcribe
        result = model.transcribe(tmp_path)
        text = result["text"]
        
        # Clean up
        os.unlink(tmp_path)
        
        if not text.strip():
            return None, "No speech detected in audio"
        
        return text, None
    except ImportError:
        return None, "❌ Whisper not installed. Please install: pip install openai-whisper"
    except Exception as e:
        return None, f"Error during transcription: {str(e)}"

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

# Display results function
def display_results(text, prediction, probability, red_flags, source_type="text"):
    """Display prediction results"""
    confidence = probability[prediction] * 100
    
    st.markdown("---")
    st.subheader(f"📊 Analysis Results ({source_type.upper()})")
    
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

# Load model
model, vectorizer, scaler, metadata = load_model_artifacts()

# Header
st.markdown('<p class="main-header">🔍 Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered detection with Text, Image & Audio support</p>', unsafe_allow_html=True)

# Sidebar - Model Info
with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Model Type", metadata['best_model'])
    st.metric("ROC-AUC Score", f"{metadata['roc_auc']:.4f}")
    st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
    st.metric("Training Samples", f"{metadata['train_samples']:,}")
    
    st.markdown("---")
    st.header("🎯 Supported Inputs")
    st.markdown("""
    - 📝 **Text** - Paste directly
    - 🖼️ **Image** - Screenshot OCR
    - 🎤 **Audio** - Voice transcription
    """)
    
    st.markdown("---")
    st.header("📖 How to Use")
    st.markdown("""
    1. Choose input type (Text/Image/Audio)
    2. Upload or paste content
    3. Click "Analyze"
    4. Review prediction & red flags
    """)
    
    st.markdown("---")
    st.header("⚠️ Disclaimer")
    st.caption("AI predictions are guidance only. Always verify through official channels.")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "🖼️ Image Upload", "🎤 Audio Upload", "ℹ️ About"])

# ==========================================
# TAB 1: TEXT INPUT
# ==========================================
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

# ==========================================
# TAB 2: IMAGE UPLOAD (OCR)
# ==========================================
with tab2:
    st.header("🖼️ Analyze Screenshot/Image")
    st.markdown("Upload a screenshot of a job posting, WhatsApp message, email, or any text image.")
    
    uploaded_image = st.file_uploader(
        "Choose an image file:",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_image:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_image_button = st.button("🔍 Extract & Analyze", type="primary", use_container_width=True, key="img_btn")
        
        if analyze_image_button:
            with st.spinner("Extracting text from image..."):
                # Perform OCR
                extracted_text, error = extract_text_from_image(image)
                
                if error:
                    st.error(error)
                    st.info("""
                    **To enable Image OCR:**
                    1. Install Tesseract OCR:
                       - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
                       - Mac: `brew install tesseract`
                       - Linux: `sudo apt-get install tesseract-ocr`
                    2. Install Python package: `pip install pytesseract`
                    3. Restart the app
                    """)
                else:
                    st.success("✅ Text extracted successfully!")
                    
                    # Show extracted text
                    with st.expander("📄 Extracted Text", expanded=True):
                        st.markdown(f'<div class="extracted-text">{extracted_text}</div>', unsafe_allow_html=True)
                    
                    # Analyze extracted text
                    with st.spinner("Analyzing extracted text..."):
                        prediction, probability = predict_fraud(extracted_text, model, vectorizer, scaler)
                        red_flags = detect_red_flags(extracted_text)
                        display_results(extracted_text, prediction, probability, red_flags, "image")

# ==========================================
# TAB 3: AUDIO UPLOAD (Speech-to-Text)
# ==========================================
with tab3:
    st.header("🎤 Analyze Voice Message")
    st.markdown("Upload an audio recording (e.g., voice note from recruiter, phone call recording).")
    
    uploaded_audio = st.file_uploader(
        "Choose an audio file:",
        type=['mp3', 'wav', 'm4a', 'ogg', 'flac'],
        help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
    )
    
    if uploaded_audio:
        # Display audio player
        st.audio(uploaded_audio, format=f'audio/{uploaded_audio.type.split("/")[1]}')
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_audio_button = st.button("🔍 Transcribe & Analyze", type="primary", use_container_width=True, key="audio_btn")
        
        if analyze_audio_button:
            with st.spinner("Transcribing audio... (this may take a minute)"):
                # Read audio bytes
                audio_bytes = uploaded_audio.read()
                
                # Perform speech-to-text
                transcribed_text, error = extract_text_from_audio(audio_bytes)
                
                if error:
                    st.error(error)
                    st.info("""
                    **To enable Audio Transcription:**
                    1. Install Whisper: `pip install openai-whisper`
                    2. Install ffmpeg:
                       - Windows: Download from https://ffmpeg.org/download.html
                       - Mac: `brew install ffmpeg`
                       - Linux: `sudo apt-get install ffmpeg`
                    3. Restart the app
                    
                    **Note:** First run will download Whisper model (~39MB for 'tiny' model)
                    """)
                else:
                    st.success("✅ Audio transcribed successfully!")
                    
                    # Show transcribed text
                    with st.expander("📄 Transcribed Text", expanded=True):
                        st.markdown(f'<div class="extracted-text">{transcribed_text}</div>', unsafe_allow_html=True)
                    
                    # Analyze transcribed text
                    with st.spinner("Analyzing transcribed text..."):
                        prediction, probability = predict_fraud(transcribed_text, model, vectorizer, scaler)
                        red_flags = detect_red_flags(transcribed_text)
                        display_results(transcribed_text, prediction, probability, red_flags, "audio")

# ==========================================
# TAB 4: ABOUT
# ==========================================
with tab4:
    st.header("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Purpose")
        st.markdown("""
        This fraud detection system uses machine learning to identify potential job and interview scams. 
        It analyzes text from multiple sources (direct text, images via OCR, audio via speech-to-text) 
        to help job seekers avoid fraudulent opportunities.
        """)
        
        st.subheader("🔬 Technology")
        st.markdown(f"""
        - **Model:** {metadata['best_model']}
        - **Accuracy:** {metadata['accuracy']:.2%}
        - **OCR:** Tesseract (image text extraction)
        - **Speech-to-Text:** OpenAI Whisper
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
        
        st.subheader("🎯 Input Types Supported")
        st.markdown("""
        - **Text:** Direct paste/typing
        - **Image:** Screenshots (OCR extraction)
        - **Audio:** Voice messages (transcription)
        - **All inputs** analyzed with same ML model
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
    st.caption("Built with ❤️ using Streamlit, Scikit-learn, Tesseract OCR & Whisper AI")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Fraud Detection System v2.0 - Multimodal</strong></p>
    <p>Protecting job seekers from scams using AI • Text • Image • Audio</p>
</div>
""", unsafe_allow_html=True)
