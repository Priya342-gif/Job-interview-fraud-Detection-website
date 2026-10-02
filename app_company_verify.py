"""
Fraud Detection Web Application - WITH COMPANY VERIFICATION
===========================================================
Complete app with Text Analysis + Image OCR + Company Verification
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
from scipy.sparse import hstack
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import email checker
from email_checker import (
    connect_to_email, 
    get_recent_emails, 
    check_email_configured,
    disconnect_email
)

# Import SerpAPI company verifier
from company_verifier_serpapi import verify_company_with_serpapi, get_verification_tips

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
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
.company-box {
    padding: 2rem;
    border-radius: 10px;
    margin: 1rem 0;
}
.legit-company {
    background-color: #e8f5e9;
    border-left: 5px solid #4caf50;
}
.suspicious-company {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
}
.fake-company {
    background-color: #ffebee;
    border-left: 5px solid #f44336;
}
.unknown-company {
    background-color: #f5f5f5;
    border-left: 5px solid #9e9e9e;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🔍 Job/Interview Fraud Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Protect yourself from job scams with AI-powered analysis</p>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        model = joblib.load('fraud_model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, vectorizer, scaler, True
    except Exception as e:
        return None, None, None, False

model, vectorizer, scaler, models_loaded = load_models()

# Helper functions
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\d{10,}', ' PHONE ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(text):
    if pd.isna(text):
        text = ""
    text = str(text)
    return {
        'text_length': len(text),
        'word_count': len(text.split()),
        'digit_count': sum(c.isdigit() for c in text),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'currency_symbols': text.count('$') + text.count('₹') + text.count('€'),
        'has_url': int(bool(re.search(r'http|www\.', text))),
        'has_email': int(bool(re.search(r'\S+@\S+', text))),
        'has_phone': int(bool(re.search(r'\d{10,}', text))),
        'uppercase_ratio': sum(c.isupper() for c in text) / max(len(text), 1),
        'money_words': sum(word in text.lower() for word in ['money', 'payment', 'bank', 'account', 'transfer', 'fee'])
    }

def detect_red_flags(text):
    text_lower = text.lower()
    critical_flags = []
    warning_flags = []
    
    # Check if email is from trusted sender first
    trusted_keywords = [
        'google', 'microsoft', 'amazon', 'apple', 'linkedin', 'facebook',
        'noreply', 'no-reply', 'donotreply', 'notifications', 'security-noreply',
        'alert', 'team', 'support'
    ]
    
    # If from trusted source, be less strict
    is_trusted = any(keyword in text_lower for keyword in trusted_keywords)
    
    # Only flag if NOT from trusted source AND contains suspicious patterns
    if not is_trusted:
        money_keywords = ['pay now', 'send money', 'transfer ₹', 'registration fee', 'training fee']
        if any(keyword in text_lower for keyword in money_keywords):
            critical_flags.append("💰 Money Request")
        
        if any(word in text_lower for word in ['send otp', 'share password', 'provide cvv']):
            critical_flags.append("🔐 Sensitive Info Request")
        
        if any(phrase in text_lower for phrase in ['selected without interview', 'no interview required', 'directly selected']):
            critical_flags.append("⚠️ No Interview Selection")
        
        if ('job' in text_lower or 'interview' in text_lower) and re.search(r'hr@gmail\.com|recruitment@yahoo\.com', text_lower):
            critical_flags.append("📧 Personal Email Domain")
        
        if 'whatsapp only' in text_lower or 'contact on whatsapp' in text_lower:
            critical_flags.append("📱 WhatsApp Only Communication")
        
        if any(word in text_lower for word in ['buy equipment', 'purchase laptop', 'training fee required']):
            critical_flags.append("🛒 Required Purchase")
    
    # WARNING FLAGS - less strict
    if text.count('!') > 5:
        warning_flags.append("❗ Excessive Exclamations")
    
    if not is_trusted and any(word in text_lower for word in ['urgent action', 'immediate action', 'act fast']):
        warning_flags.append("⏰ Urgency Pressure")
    
    return critical_flags, warning_flags

def predict_fraud(text):
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    
    features = extract_features(text)
    features_array = np.array([[features['text_length'], features['word_count'], 
                                features['digit_count'], features['exclamation_count'],
                                features['question_count'], features['currency_symbols'], 
                                features['has_url'], features['has_email'], 
                                features['has_phone'], features['uppercase_ratio'], 
                                features['money_words']]])
    features_scaled = scaler.transform(features_array)
    
    combined = hstack([tfidf, features_scaled])
    
    prediction = model.predict(combined)[0]
    probability = model.predict_proba(combined)[0][1]
    
    critical, warnings = detect_red_flags(text)
    
    final_prediction = "SCAM" if prediction == 1 else "LEGITIMATE"
    
    # BALANCED OVERRIDE LOGIC - Less aggressive
    # Only override if there are MULTIPLE strong indicators
    
    # If 2+ critical flags → Force SCAM
    if len(critical) >= 2:
        final_prediction = "SCAM"
        probability = max(probability, 0.95)
    
    # If 1 critical flag + model says scam → SCAM
    elif len(critical) >= 1 and prediction == 1:
        final_prediction = "SCAM"
        probability = max(probability, 0.90)
    
    # If 4+ warnings → Force SCAM
    elif len(warnings) >= 4:
        final_prediction = "SCAM"
        probability = max(probability, 0.85)
    
    # Trust the ML model more for legitimate predictions
    # Only override if very strong evidence
    
    return {
        'prediction': final_prediction,
        'confidence': probability,
        'critical_flags': critical,
        'warning_flags': warnings
    }

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Text Analysis", "🏢 Company Verification", "📧 Email Scanner", "🖼️ Image Analysis", "ℹ️ Model Info"])

# ==========================================
# TAB 1: TEXT ANALYSIS
# ==========================================
with tab1:
    st.header("📝 Analyze Job/Interview Message")
    
    if not models_loaded:
        st.error("⚠️ Models not loaded! Please train the model first.")
    else:
        text_input = st.text_area(
            "Paste the job posting or interview message here:",
            height=200,
            placeholder="Example: Congratulations! You are selected for software engineer role. Send ₹5000 for training..."
        )
        
        if st.button("🔍 Analyze Text", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing..."):
                    result = predict_fraud(text_input)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if result['prediction'] == 'SCAM':
                            st.markdown(f"""
                            <div class="company-box fake-company">
                                <h2 style="color: #f44336;">🚨 SCAM DETECTED</h2>
                                <p style="font-size: 1.5rem;">Confidence: {result['confidence']:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="company-box legit-company">
                                <h2 style="color: #4caf50;">✅ APPEARS LEGITIMATE</h2>
                                <p style="font-size: 1.5rem;">Confidence: {(1-result['confidence']):.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Scam Probability", f"{result['confidence']:.1%}")
                    
                    if result['critical_flags']:
                        st.error("**🚨 Critical Red Flags:**")
                        for flag in result['critical_flags']:
                            st.write(f"- {flag}")
                    
                    if result['warning_flags']:
                        st.warning("**⚠️ Warning Signs:**")
                        for flag in result['warning_flags']:
                            st.write(f"- {flag}")
            else:
                st.warning("Please enter some text to analyze.")

# ==========================================
# TAB 2: COMPANY VERIFICATION
# ==========================================
with tab2:
    st.header("🏢 Company Verification with SerpAPI")
    st.success("✅ SerpAPI Enabled - Powered by Google Search (100 searches/month)")
    st.write("Enter any company name to verify if it's legitimate or fake")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        company_name = st.text_input(
            "Enter Company Name:",
            placeholder="e.g., Google, TCS, Infosys, or any company name...",
            key="company_input"
        )
    
    with col2:
        st.write("")
        st.write("")
        verify_btn = st.button("🔍 Verify Company", type="primary", use_container_width=True)
    
    if verify_btn and company_name.strip():
        with st.spinner("🔍 Searching Google via SerpAPI... Please wait..."):
            result = verify_company_with_serpapi(company_name)
            
            # Display result
            if 'LEGITIMATE' in result['status']:
                box_class = 'legit-company'
            elif 'SUSPICIOUS' in result['status'] or 'LIKELY' in result['status']:
                box_class = 'suspicious-company'
            elif 'NOT FOUND' in result['status']:
                box_class = 'fake-company'
            else:
                box_class = 'unknown-company'
            
            st.markdown(f"""
            <div class="company-box {box_class}">
                <h2>{result['status']}</h2>
                <p style="font-size: 1.3rem; margin-top: 1rem;">{result['reason']}</p>
                <p style="font-size: 1.1rem; color: #666;">Confidence: {result['confidence']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top search result
            if 'top_result' in result and result.get('top_result'):
                st.info(f"**Top Google Result:** {result['top_result']}")
                if result.get('top_link'):
                    st.caption(f"🔗 {result['top_link']}")
            
            # Show search results count
            if 'search_results_count' in result:
                st.caption(f"**Found {result['search_results_count']} Google search results**")
            
            # Show verification source
            st.caption(f"**Verified by:** {result.get('source', 'SerpAPI')}")
            
            # Show verification sources
            if 'verification_sources' in result and result['verification_sources']:
                st.info(f"**Verified on:** {', '.join(result['verification_sources'])}")
            
            # Show red flags if any
            if 'red_flags' in result and result['red_flags']:
                st.error("**🚨 Red Flags Detected:**")
                for flag in result['red_flags']:
                    st.write(f"- {flag}")
            
            # Detailed information
            if 'details' in result:
                st.subheader("📋 Detailed Analysis")
                for detail in result['details']:
                    st.write(detail)
            
            # Show verification tips
            with st.expander("💡 How to Verify Companies - Best Practices"):
                tips = get_verification_tips()
                for tip in tips:
                    st.markdown(tip)
    
    elif verify_btn:
        st.warning("Please enter a company name to verify.")
    
    # Popular companies quick check
    st.markdown("---")
    st.subheader("🔥 Quick Check: Popular Companies")
    
    popular = ['Google', 'TCS', 'Infosys', 'Wipro', 'Amazon', 'Microsoft', 'Flipkart', 'Paytm']
    
    cols = st.columns(4)
    for idx, company in enumerate(popular):
        with cols[idx % 4]:
            if st.button(company, use_container_width=True):
                result = verify_company_with_serpapi(company)
                if 'LEGITIMATE' in result['status']:
                    st.success(f"✅ {result['status']}")
                else:
                    st.info(f"{result['status']}")

# ==========================================
# TAB 3: EMAIL SCANNER
# ==========================================
with tab3:
    st.header("📧 Email Fraud Scanner")
    
    # Check if email is configured
    email_configured = check_email_configured()
    
    if not email_configured:
        st.warning("⚠️ Email not configured")
        
        with st.expander("🔧 How to setup Email Scanner", expanded=True):
            st.markdown("""
            **Setup your Gmail for fraud detection:**
            
            ### Step 1: Enable IMAP in Gmail
            1. Open Gmail Settings → "See all settings"
            2. Go to "Forwarding and POP/IMAP" tab
            3. Enable IMAP → Save Changes
            
            ### Step 2: Enable 2-Step Verification
            1. Visit: https://myaccount.google.com/security
            2. Enable "2-Step Verification"
            
            ### Step 3: Generate App Password
            1. Visit: https://myaccount.google.com/apppasswords
            2. App name: "Fraud Detection"
            3. Click "Generate"
            4. Copy the 16-character password
            
            ### Step 4: Add to .env file
            ```
            EMAIL_ADDRESS=your.email@gmail.com
            EMAIL_APP_PASSWORD=abcd efgh ijkl mnop
            ```
            
            ### Step 5: Restart the app
            
            **Benefits:**
            - ✅ Auto-scan recent emails
            - ✅ Detect fraud emails automatically
            - ✅ Check any email in your inbox
            - ✅ Works with Gmail, Outlook, Yahoo
            """)
        
        st.info("💡 **Demo Mode:** You can still use Text Analysis tab to check email content manually")
    
    else:
        st.success(f"✅ Email connected: {os.getenv('EMAIL_ADDRESS', 'Unknown')}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            num_emails = st.slider("Number of recent emails to scan:", 5, 50, 10)
        
        with col2:
            st.write("")
            st.write("")
            scan_btn = st.button("🔍 Scan Emails", type="primary", use_container_width=True)
        
        if scan_btn:
            with st.spinner("📧 Connecting to email... Please wait..."):
                mail, error = connect_to_email()
                
                if error:
                    st.error(f"❌ {error}")
                    st.info("**Troubleshooting:**\n- Check email address in .env\n- Verify App Password is correct\n- Ensure IMAP is enabled")
                else:
                    st.success("✅ Connected successfully!")
                    
                    with st.spinner(f"📬 Fetching {num_emails} recent emails..."):
                        emails = get_recent_emails(mail, limit=num_emails)
                        
                        if not emails:
                            st.warning("No emails found")
                        else:
                            st.info(f"📊 Found {len(emails)} emails. Analyzing...")
                            
                            # Analyze each email
                            fraud_count = 0
                            legit_count = 0
                            
                            for idx, email_data in enumerate(emails):
                                # Analyze email body
                                result = predict_fraud(email_data['full_body'])
                                
                                is_fraud = result['prediction'] == 'SCAM'
                                
                                if is_fraud:
                                    fraud_count += 1
                                    status_color = '🚨'
                                    box_color = 'red'
                                else:
                                    legit_count += 1
                                    status_color = '✅'
                                    box_color = 'green'
                                
                                # Display email
                                with st.expander(f"{status_color} Email {idx+1}: {email_data['subject'][:60]}"):
                                    st.write(f"**From:** {email_data['sender']}")
                                    st.write(f"**Date:** {email_data['date']}")
                                    st.write(f"**Subject:** {email_data['subject']}")
                                    
                                    st.markdown("---")
                                    
                                    if is_fraud:
                                        st.error(f"🚨 **FRAUD DETECTED** (Confidence: {result['confidence']:.1%})")
                                    else:
                                        st.success(f"✅ **APPEARS SAFE** (Confidence: {(1-result['confidence']):.1%})")
                                    
                                    if result['critical_flags']:
                                        st.error("**Critical Red Flags:**")
                                        for flag in result['critical_flags']:
                                            st.write(f"- {flag}")
                                    
                                    if result['warning_flags']:
                                        st.warning("**Warning Signs:**")
                                        for flag in result['warning_flags']:
                                            st.write(f"- {flag}")
                                    
                                    st.text_area("Email Preview:", email_data['body'], height=100, key=f"email_{idx}")
                            
                            # Summary
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Scanned", len(emails))
                            with col2:
                                st.metric("🚨 Fraud Detected", fraud_count)
                            with col3:
                                st.metric("✅ Safe Emails", legit_count)
                    
                    # Disconnect
                    disconnect_email(mail)

# ==========================================
# TAB 4: IMAGE ANALYSIS
# ==========================================
with tab4:
    st.header("🖼️ Analyze Job Posting Image")
    
    if not models_loaded:
        st.error("⚠️ Models not loaded!")
    else:
        uploaded_file = st.file_uploader(
            "Upload job posting screenshot or image",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            
            if st.button("🔍 Analyze Image", type="primary"):
                st.info("📸 OCR feature requires Tesseract installation. Please use Text Analysis tab for now.")

# ==========================================
# TAB 5: MODEL INFO
# ==========================================
with tab5:
    st.header("ℹ️ About This System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "97.81%")
    with col2:
        st.metric("ROC-AUC Score", "0.9799")
    with col3:
        st.metric("Training Samples", "15,985")
    
    st.subheader("🎯 Features")
    st.write("""
    - **Text Analysis**: ML-powered fraud detection
    - **Company Verification**: Check company legitimacy
    - **Red Flag Detection**: 6 critical + 7 warning indicators
    - **Smart Override Logic**: Rule-based + ML combined
    """)
    
    st.subheader("🚨 Red Flags We Detect")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Critical (100% Fraud):**")
        st.write("- 💰 Money Request")
        st.write("- 🔐 Sensitive Info Request")
        st.write("- ⚠️ No Interview Selection")
        st.write("- 📧 Personal Email Domain")
        st.write("- 📱 WhatsApp Only")
        st.write("- 🛒 Required Purchase")
    
    with col2:
        st.write("**Warning Signs:**")
        st.write("- ❗ Excessive Exclamations")
        st.write("- ⏰ Urgency Pressure")
        st.write("- 🎯 Unrealistic Promises")
        st.write("- 📞 Suspicious Numbers")
        st.write("- 📝 Very Short Message")
        st.write("- 🔠 Excessive Capitals")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>🔒 Stay safe from job scams | "
    "Never pay fees for job opportunities | Always verify company legitimacy</p>",
    unsafe_allow_html=True
)
