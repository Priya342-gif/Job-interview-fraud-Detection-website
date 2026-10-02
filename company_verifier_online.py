# ==========================================
# ONLINE COMPANY VERIFICATION SYSTEM
# ==========================================
# Uses web search to verify ANY company in real-time

import requests
from bs4 import BeautifulSoup
import re
import time

# Known legitimate companies (quick check)
LEGITIMATE_COMPANIES = {
    'google', 'microsoft', 'apple', 'amazon', 'meta', 'facebook',
    'netflix', 'adobe', 'oracle', 'salesforce', 'ibm', 'intel',
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'hcl',
    'tech mahindra', 'cognizant', 'capgemini', 'accenture', 'deloitte',
    'flipkart', 'paytm', 'ola', 'swiggy', 'zomato', 'byju',
    'phonepe', 'razorpay', 'zerodha', 'cred', 'meesho',
    'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'goldman sachs',
}


def search_company_online(company_name):
    """
    Search company using DuckDuckGo (no API key needed)
    Returns search results to verify if company exists
    """
    try:
        # Clean company name
        company_clean = company_name.strip().replace(' ', '+')
        
        # Search on DuckDuckGo (no API key required)
        search_url = f"https://html.duckduckgo.com/html/?q={company_clean}+company"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            # Check if results found
            text = response.text.lower()
            
            # Look for indicators of real company
            indicators = {
                'website': 0,
                'linkedin': 0,
                'wikipedia': 0,
                'news': 0,
                'glassdoor': 0
            }
            
            if 'linkedin.com/company' in text:
                indicators['linkedin'] = 1
            if 'wikipedia.org' in text:
                indicators['wikipedia'] = 1
            if 'glassdoor.co' in text:
                indicators['glassdoor'] = 1
            if any(domain in text for domain in ['.com', '.in', '.co', '.org']):
                indicators['website'] = 1
            if any(news in text for news in ['news', 'article', 'report']):
                indicators['news'] = 1
            
            score = sum(indicators.values())
            
            return {
                'found': True,
                'score': score,
                'indicators': indicators,
                'max_score': 5
            }
        
        return {'found': False, 'score': 0, 'indicators': {}, 'max_score': 5}
    
    except Exception as e:
        # If online search fails, return neutral
        return {'found': False, 'score': 0, 'indicators': {}, 'max_score': 5}


def check_company_pattern(company_name):
    """
    Check if company name has suspicious patterns
    """
    company_lower = company_name.lower().strip()
    
    red_flags = []
    score = 0
    
    # Pattern 1: Too short
    if len(company_lower) < 3:
        red_flags.append("Company name too short")
        score -= 2
    
    # Pattern 2: Generic endings without real name
    generic_patterns = [
        r'^(pvt|ltd|private|limited|inc|corp|llc)\s*(ltd|pvt|limited)?$',
        r'^(solutions|services|consultancy|group|company)\s*(pvt|ltd)?$'
    ]
    
    for pattern in generic_patterns:
        if re.match(pattern, company_lower):
            red_flags.append("Generic name without specific identity")
            score -= 3
    
    # Pattern 3: Numbers in name (unusual for legitimate companies)
    if re.search(r'\d{2,}', company_lower):
        red_flags.append("Contains numbers (unusual)")
        score -= 1
    
    # Pattern 4: Recruitment/staffing keywords
    recruitment_words = ['recruitment', 'staffing', 'manpower', 'placement', 'jobs', 'hiring']
    if any(word in company_lower for word in recruitment_words):
        red_flags.append("Recruitment/Staffing agency (not direct employer)")
        score -= 1
    
    # Pattern 5: Multiple generic words
    generic_words = ['solutions', 'services', 'consultancy', 'technologies', 'systems', 'global', 'international']
    generic_count = sum(1 for word in generic_words if word in company_lower)
    if generic_count >= 2:
        red_flags.append("Multiple generic business terms")
        score -= 1
    
    return {
        'red_flags': red_flags,
        'pattern_score': score
    }


def verify_company_advanced(company_name):
    """
    Advanced company verification combining:
    1. Known companies database
    2. Online search
    3. Pattern analysis
    """
    
    if not company_name or len(company_name.strip()) < 2:
        return {
            'status': 'UNKNOWN',
            'confidence': 0,
            'reason': 'Company name too short',
            'color': 'gray',
            'details': ['Please enter a valid company name']
        }
    
    company_clean = company_name.lower().strip()
    
    # STEP 1: Check known companies (instant)
    for legit in LEGITIMATE_COMPANIES:
        if legit in company_clean or company_clean in legit:
            return {
                'status': 'LEGITIMATE ✅',
                'confidence': 95,
                'reason': 'Recognized as established company',
                'color': 'green',
                'details': [
                    '✅ Found in verified companies database',
                    '✅ Well-known organization',
                    '✅ Safe to proceed with application',
                    f'✅ Company: {company_name}'
                ],
                'verification_sources': ['Internal Database']
            }
    
    # STEP 2: Pattern analysis
    pattern_result = check_company_pattern(company_name)
    pattern_score = pattern_result['pattern_score']
    
    # STEP 3: Online search (real-time)
    print(f"Searching online for: {company_name}...")
    online_result = search_company_online(company_name)
    
    online_score = online_result['score']
    indicators = online_result['indicators']
    
    # Calculate final score
    total_score = online_score + pattern_score
    
    # Build verification sources list
    sources = []
    if indicators.get('linkedin'):
        sources.append('LinkedIn')
    if indicators.get('wikipedia'):
        sources.append('Wikipedia')
    if indicators.get('glassdoor'):
        sources.append('Glassdoor')
    if indicators.get('website'):
        sources.append('Official Website')
    
    # Decision logic
    details = []
    
    # HIGH CONFIDENCE: Online presence + good pattern
    if online_score >= 3 and pattern_score >= -1:
        status = 'LEGITIMATE ✅'
        confidence = min(85 + (online_score * 3), 95)
        reason = 'Strong online presence verified'
        color = 'green'
        details = [
            f'✅ Found on {online_score}/5 major platforms',
            f'✅ Verified on: {", ".join(sources) if sources else "Multiple sources"}',
            '✅ Company appears legitimate',
            '✅ Has professional online presence'
        ]
        if indicators.get('linkedin'):
            details.append('✅ LinkedIn company page exists')
        if indicators.get('glassdoor'):
            details.append('✅ Glassdoor reviews available')
    
    # MEDIUM CONFIDENCE: Some online presence
    elif online_score >= 2 and pattern_score >= -2:
        status = 'LIKELY LEGITIMATE ⚠️'
        confidence = 60 + (online_score * 5)
        reason = 'Some online presence found, verify further'
        color = 'orange'
        details = [
            f'⚠️ Found on {online_score}/5 platforms',
            '⚠️ Company has some online presence',
            '⚠️ Could be small/startup company',
            '⚠️ Recommended: Verify before proceeding'
        ]
        if sources:
            details.append(f'Found on: {", ".join(sources)}')
        details.append('🔍 Check: Official website, LinkedIn, Glassdoor')
    
    # LOW CONFIDENCE: Suspicious pattern
    elif pattern_score <= -3:
        status = 'SUSPICIOUS 🚨'
        confidence = 70
        reason = 'Suspicious company name pattern detected'
        color = 'red'
        details = [
            '🚨 Red flags detected in company name:',
        ]
        details.extend([f'  - {flag}' for flag in pattern_result['red_flags']])
        details.extend([
            '⚠️ Exercise caution',
            '⚠️ Verify company thoroughly',
            '⚠️ Check official website and reviews',
            '🚫 Do NOT pay any fees'
        ])
    
    # NO ONLINE PRESENCE: Potentially fake
    elif online_score == 0:
        status = 'NOT FOUND ❌'
        confidence = 75
        reason = 'No online presence found'
        color = 'red'
        details = [
            '❌ Company not found on major platforms',
            '❌ No LinkedIn company page',
            '❌ No Glassdoor reviews',
            '❌ Limited/No online presence',
            '⚠️ HIGH RISK - Could be fake',
            '🔍 Manual verification required:',
            '  • Search company on Google',
            '  • Check MCA database (www.mca.gov.in)',
            '  • Verify office address on Google Maps',
            '  • Ask for company registration details',
            '🚫 DO NOT share documents or pay fees'
        ]
    
    # UNKNOWN: Insufficient data
    else:
        status = 'UNKNOWN ❓'
        confidence = 50
        reason = 'Insufficient data to verify'
        color = 'gray'
        details = [
            'ℹ️ Limited information available',
            'ℹ️ Could be new/small company',
            'ℹ️ Manual verification needed',
            '🔍 Verify on:',
            '  • Official company website',
            '  • LinkedIn company page',
            '  • Glassdoor reviews',
            '  • MCA database (India)',
            '  • Google Maps (office location)',
            '⚠️ Do NOT pay any fees',
            '⚠️ Verify office address physically'
        ]
    
    return {
        'status': status,
        'confidence': confidence,
        'reason': reason,
        'color': color,
        'details': details,
        'verification_sources': sources if sources else ['Pattern Analysis'],
        'online_score': online_score,
        'pattern_score': pattern_score
    }


def get_verification_tips():
    """Return general verification tips"""
    return [
        "🔍 **Official Website**: Check if company has professional website with .com/.in domain",
        "🏢 **Physical Office**: Verify office address on Google Maps Street View",
        "💼 **LinkedIn**: Real companies have verified LinkedIn company pages with employees",
        "⭐ **Glassdoor/AmbitionBox**: Check employee reviews and ratings",
        "📄 **MCA Database**: Search on www.mca.gov.in for Indian registered companies",
        "📧 **Email Domain**: Legitimate companies use official domains (not @gmail.com)",
        "📞 **Contact Number**: Verify phone number matches company website",
        "🚫 **No Upfront Fees**: Real companies NEVER ask for registration/training fees",
        "👥 **Interview Process**: Legitimate companies have proper technical/HR rounds",
        "🆔 **Offer Letter**: Should have company letterhead, HR signature, office address",
        "📱 **Social Media**: Check official Facebook, Twitter, Instagram pages",
        "📰 **News Articles**: Search for company news on Google News",
        "🏦 **Bank Account**: Verify company bank details (not personal accounts)",
        "⚖️ **Legal Documents**: Ask for GST number, PAN, company registration certificate"
    ]
