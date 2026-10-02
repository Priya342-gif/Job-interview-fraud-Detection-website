# 🔍 Job/Interview Fraud Detection System

A comprehensive AI-powered system to detect fraudulent job postings and interview scams using Machine Learning, NLP, and real-time company verification.

## 🎯 Features

### 1. **Text Analysis** 📝
- ML-powered fraud detection (97.81% accuracy)
- TF-IDF + engineered features
- Red flag detection system
- Real-time confidence scoring

### 2. **Company Verification** 🏢
- Powered by SerpAPI (Google Search)
- Verifies company legitimacy
- Checks LinkedIn, Glassdoor, Wikipedia
- 100 free searches/month

### 3. **Email Scanner** 📧
- IMAP integration (Gmail/Outlook)
- Auto-scan inbox for fraud emails
- Batch email analysis
- Trusted sender detection

### 4. **Image Analysis** 🖼️
- OCR support (Tesseract)
- Screenshot analysis
- Text extraction from images

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Priya342-gif/Job-interview-fraud-Detection-website.git
cd Job-interview-fraud-Detection-website
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup environment variables**
Create `.env` file in project root:
```env
# SerpAPI (Company Verification)
SERPAPI_KEY=your_serpapi_key_here

# Email Scanner (Optional)
EMAIL_ADDRESS=your.email@gmail.com
EMAIL_APP_PASSWORD=your_app_password_here
IMAP_SERVER=imap.gmail.com
IMAP_PORT=993
```

4. **Run the application**
```bash
streamlit run app_company_verify.py
```

5. **Open in browser**
```
http://localhost:8501
```

## 📊 Model Details

- **Algorithm:** Random Forest
- **Accuracy:** 97.81%
- **ROC-AUC:** 0.9799
- **Precision:** 93.33%
- **Training Samples:** 15,985
- **Features:** TF-IDF (5000) + 11 engineered features

## 🛡️ Red Flag Detection

### Critical Flags (Force SCAM)
- 💰 Money Request
- 🔐 Sensitive Info Request
- ⚠️ No Interview Selection
- 📧 Personal Email Domain
- 📱 WhatsApp Only Communication
- 🛒 Required Purchase

### Warning Signs
- ❗ Excessive Exclamations
- ⏰ Urgency Pressure
- 🎯 Unrealistic Promises

## 📁 Project Structure

```
fraudDetction/
├── app_company_verify.py      # Main Streamlit app
├── company_verifier_serpapi.py # Company verification module
├── email_checker.py            # Email IMAP scanner
├── fraud_detection_colab_final.py # Training notebook
├── train_model.py              # Model training script
├── requirements.txt            # Dependencies
├── .env                        # API keys (not in git)
├── fraud_model.joblib          # Trained model
├── vectorizer.joblib           # TF-IDF vectorizer
├── scaler.joblib               # Feature scaler
└── README.md                   # Documentation
```

## 🔧 Configuration

### Get SerpAPI Key (Free)
1. Visit: https://serpapi.com/
2. Sign up
3. Get API key from dashboard
4. Add to `.env` file

### Setup Gmail Scanner (Optional)
1. Enable IMAP in Gmail settings
2. Enable 2-Step Verification: https://myaccount.google.com/security
3. Generate App Password: https://myaccount.google.com/apppasswords
4. Add credentials to `.env` file

## 📈 Usage

### Text Analysis
```python
# Paste job posting text
# Click "Analyze Text"
# Get instant fraud detection result
```

### Company Verification
```python
# Type company name
# Click "Verify Company"
# See real-time verification results
```

### Email Scanner
```python
# Configure Gmail credentials
# Select number of emails (5-50)
# Click "Scan Emails"
# View fraud detection for each email
```

## 🎓 Academic Project

This project was developed as an academic submission covering:
- **Problem Statement** (2 marks)
- **Dataset Analysis** (2 marks)
- **Preprocessing** (2 marks)
- **Exploratory Data Analysis** (2 marks)
- **Model Comparison** (2 marks)

Total: 10/10 marks

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📝 License

This project is for educational purposes.

## 👩‍💻 Author

**Priya**
- GitHub: [@Priya342-gif](https://github.com/Priya342-gif)

## 🙏 Acknowledgments

- Dataset: Custom collected (15,985 samples)
- SerpAPI for company verification
- Streamlit for web interface
- scikit-learn for ML models

## ⚠️ Disclaimer

This system is for educational and informational purposes. Always verify job offers through multiple sources.

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact: priyahc1906@gmail.com

---

**⭐ Star this repository if you found it helpful!**
