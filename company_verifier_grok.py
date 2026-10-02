# ==========================================
# GROK AI COMPANY VERIFICATION
# ==========================================
# Uses Grok API to verify ANY company in real-time

import requests
import json
import os

# Grok API Configuration
GROK_API_KEY = os.getenv('GROK_API_KEY', '')  # Will be set in .env file
GROK_API_URL = "https://api.x.ai/v1/chat/completions"


def verify_company_with_grok(company_name):
    """
    Verify company using Grok AI (xAI)
    Grok has real-time web access and can search & analyze
    """
    
    if not GROK_API_KEY or GROK_API_KEY == '':
        # Fallback to manual database if no API key
        return verify_company_fallback(company_name)
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROK_API_KEY}"
        }
        
        # Craft prompt for Grok
        prompt = f"""You are a company verification expert. Analyze if "{company_name}" is a legitimate company or potentially fake.

Search the web and check:
1. Does this company have an official website?
2. Does it have a LinkedIn company page?
3. Are there employee reviews on Glassdoor/AmbitionBox?
4. Is it registered (check for news, articles, or presence)?
5. Are there any scam reports about this company?

Respond in this EXACT JSON format:
{{
    "status": "LEGITIMATE" or "SUSPICIOUS" or "FAKE" or "UNKNOWN",
    "confidence": 0-100,
    "reason": "brief explanation",
    "details": ["point 1", "point 2", "point 3"],
    "red_flags": ["flag 1", "flag 2"] or [],
    "verification_sources": ["LinkedIn", "Website", "Glassdoor"] or []
}}

Be strict. If no online presence found, mark as FAKE. If recruitment agency, mark as SUSPICIOUS."""
        
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a company verification assistant with real-time web access. Provide accurate, factual verification results."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-beta",
            "stream": False,
            "temperature": 0
        }
        
        response = requests.post(
            GROK_API_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            grok_response = result['choices'][0]['message']['content']
            
            # Parse JSON response from Grok
            try:
                # Extract JSON from response
                json_start = grok_response.find('{')
                json_end = grok_response.rfind('}') + 1
                json_str = grok_response[json_start:json_end]
                
                verification = json.loads(json_str)
                
                # Format for our app
                status = verification.get('status', 'UNKNOWN')
                confidence = verification.get('confidence', 50)
                reason = verification.get('reason', 'Unable to verify')
                details = verification.get('details', [])
                red_flags = verification.get('red_flags', [])
                sources = verification.get('verification_sources', [])
                
                # Add emojis based on status
                if status == 'LEGITIMATE':
                    status_display = 'LEGITIMATE ✅'
                    color = 'green'
                elif status == 'SUSPICIOUS':
                    status_display = 'SUSPICIOUS ⚠️'
                    color = 'orange'
                elif status == 'FAKE':
                    status_display = 'FAKE COMPANY 🚨'
                    color = 'red'
                else:
                    status_display = 'UNKNOWN ❓'
                    color = 'gray'
                
                return {
                    'status': status_display,
                    'confidence': confidence,
                    'reason': reason,
                    'color': color,
                    'details': details,
                    'red_flags': red_flags,
                    'verification_sources': sources,
                    'source': 'Grok AI (Real-time Web Search)'
                }
                
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                return {
                    'status': 'ANALYSIS COMPLETE ℹ️',
                    'confidence': 50,
                    'reason': 'Grok AI Analysis',
                    'color': 'gray',
                    'details': [grok_response],
                    'verification_sources': ['Grok AI'],
                    'source': 'Grok AI'
                }
        
        else:
            # API call failed
            return {
                'status': 'API ERROR ⚠️',
                'confidence': 0,
                'reason': f'Grok API returned error: {response.status_code}',
                'color': 'gray',
                'details': [
                    '⚠️ Unable to connect to Grok API',
                    '⚠️ Please check your API key',
                    '⚠️ Falling back to manual verification'
                ],
                'verification_sources': [],
                'source': 'Error'
            }
    
    except requests.exceptions.Timeout:
        return {
            'status': 'TIMEOUT ⏱️',
            'confidence': 0,
            'reason': 'Request timed out',
            'color': 'gray',
            'details': ['⚠️ Grok API request timed out', '⚠️ Try again'],
            'verification_sources': [],
            'source': 'Error'
        }
    
    except Exception as e:
        return {
            'status': 'ERROR ❌',
            'confidence': 0,
            'reason': f'Error: {str(e)}',
            'color': 'gray',
            'details': ['⚠️ An error occurred', f'⚠️ {str(e)}'],
            'verification_sources': [],
            'source': 'Error'
        }


def verify_company_fallback(company_name):
    """
    Fallback verification when Grok API not available
    Uses manual database
    """
    
    LEGITIMATE_COMPANIES = {
        'google', 'microsoft', 'apple', 'amazon', 'meta', 'facebook',
        'netflix', 'adobe', 'oracle', 'salesforce', 'ibm', 'intel',
        'tcs', 'tata consultancy services', 'infosys', 'wipro', 'hcl',
        'tech mahindra', 'cognizant', 'capgemini', 'accenture', 'deloitte',
        'flipkart', 'paytm', 'ola', 'swiggy', 'zomato', 'phonepe'
    }
    
    company_lower = company_name.lower().strip()
    
    # Check known companies
    for legit in LEGITIMATE_COMPANIES:
        if legit in company_lower or company_lower in legit:
            return {
                'status': 'LEGITIMATE ✅',
                'confidence': 90,
                'reason': 'Found in verified companies database',
                'color': 'green',
                'details': [
                    '✅ Recognized established company',
                    '✅ Safe to proceed',
                    '⚠️ Note: Using fallback database (Grok API not configured)'
                ],
                'verification_sources': ['Internal Database'],
                'source': 'Manual Database (Fallback)'
            }
    
    # Unknown company
    return {
        'status': 'UNKNOWN ❓',
        'confidence': 50,
        'reason': 'Company not in database',
        'color': 'gray',
        'details': [
            'ℹ️ Company not found in manual database',
            '⚠️ Grok API not configured',
            '⚠️ To enable real-time verification:',
            '  1. Get Grok API key from https://x.ai/api',
            '  2. Add to .env file: GROK_API_KEY=your_key',
            '  3. Restart the app',
            '',
            '🔍 Manual Verification Needed:',
            '  • Search company on Google',
            '  • Check LinkedIn company page',
            '  • Check Glassdoor reviews',
            '  • Verify office address',
            '🚫 DO NOT pay any fees'
        ],
        'verification_sources': [],
        'source': 'Manual Database (Fallback)'
    }


def get_verification_tips():
    """Return general verification tips"""
    return [
        "🔍 **Official Website**: Check if company has professional website",
        "🏢 **Physical Office**: Verify office address on Google Maps",
        "💼 **LinkedIn**: Real companies have verified company pages",
        "⭐ **Glassdoor/AmbitionBox**: Check employee reviews",
        "📄 **MCA Database**: Search on www.mca.gov.in (India)",
        "📧 **Email Domain**: Legitimate companies use official domains",
        "🚫 **No Upfront Fees**: Real companies NEVER ask for money",
        "👥 **Interview Process**: Legitimate companies have proper rounds",
        "🆔 **Offer Letter**: Should have company letterhead and signature"
    ]


def check_grok_api_configured():
    """
    Check if Grok API key is configured
    """
    return bool(GROK_API_KEY and GROK_API_KEY != '')
