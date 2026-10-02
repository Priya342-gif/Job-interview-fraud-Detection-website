# ==========================================
# COMPANY VERIFICATION SYSTEM
# ==========================================
# Simple company checker using known databases

import re

# Known legitimate companies database
LEGITIMATE_COMPANIES = {
    # Tech Giants
    'google', 'microsoft', 'apple', 'amazon', 'meta', 'facebook',
    'netflix', 'adobe', 'oracle', 'salesforce', 'ibm', 'intel',
    
    # Indian IT Companies
    'tcs', 'tata consultancy services', 'infosys', 'wipro', 'hcl',
    'tech mahindra', 'ltts', 'persistent', 'mindtree', 'mphasis',
    'cognizant', 'capgemini', 'accenture', 'deloitte', 'ey',
    'kpmg', 'pwc', 'genpact', 'hexaware', 'coforge',
    
    # Startups & Product
    'flipkart', 'paytm', 'ola', 'swiggy', 'zomato', 'byju',
    'phonepe', 'razorpay', 'zerodha', 'cred', 'meesho',
    'udaan', 'urban company', 'dunzo', 'sharechat', 'dream11',
    
    # Banks & Finance
    'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yes bank',
    'idfc', 'indusind', 'standard chartered', 'hsbc', 'citi',
    
    # MNCs
    'goldman sachs', 'jp morgan', 'morgan stanley', 'barclays',
    'deutsche bank', 'wells fargo', 'bofa', 'bank of america',
    
    # Consulting
    'mckinsey', 'bcg', 'boston consulting', 'bain', 'monitor',
    
    # E-commerce
    'walmart', 'target', 'ebay', 'alibaba', 'jd.com',
    
    # Automotive
    'tesla', 'ford', 'gm', 'bmw', 'mercedes', 'toyota', 'honda',
    'tata motors', 'mahindra', 'maruti', 'hyundai',
    
    # Telecom
    'jio', 'airtel', 'vodafone', 'idea', 'bsnl', 'mtnl',
    'at&t', 'verizon', 't-mobile',
}

# Common fake company patterns
FAKE_PATTERNS = [
    r'\d+',  # Companies with numbers in name (usually fake)
    r'pvt\s*ltd\s*$',  # Ending with just "Pvt Ltd" (suspicious)
    r'limited\s*$',  # Ending with just "Limited"
    r'group\s*$',  # Vague "Group" ending
    r'consultancy\s*$',  # Generic consultancy
    r'solutions\s*$',  # Generic solutions
    r'services\s*$',  # Generic services
]

# Red flag keywords in company names
RED_FLAG_KEYWORDS = [
    'recruitment', 'staffing', 'manpower', 'placement',
    'hr solutions', 'job consultancy', 'career',
    'hiring', 'employment', 'outsourcing'
]


def verify_company(company_name):
    """
    Verify if company is legitimate or potentially fake
    
    Returns:
        dict with 'status', 'confidence', 'reason'
    """
    
    if not company_name or len(company_name.strip()) < 2:
        return {
            'status': 'UNKNOWN',
            'confidence': 0,
            'reason': 'Company name too short',
            'color': 'gray'
        }
    
    # Clean company name
    company_clean = company_name.lower().strip()
    company_clean = re.sub(r'[^\w\s]', ' ', company_clean)
    company_clean = re.sub(r'\s+', ' ', company_clean).strip()
    
    # Check 1: Known legitimate company
    for legit in LEGITIMATE_COMPANIES:
        if legit in company_clean or company_clean in legit:
            return {
                'status': 'LEGITIMATE',
                'confidence': 95,
                'reason': f'Recognized as established company',
                'color': 'green',
                'details': [
                    '✅ Found in verified companies database',
                    '✅ Well-known organization',
                    '✅ Safe to proceed with application'
                ]
            }
    
    # Check 2: Red flag keywords
    red_flags = []
    for keyword in RED_FLAG_KEYWORDS:
        if keyword in company_clean:
            red_flags.append(f"Contains '{keyword}'")
    
    if red_flags:
        return {
            'status': 'SUSPICIOUS',
            'confidence': 70,
            'reason': 'Contains recruitment agency keywords',
            'color': 'orange',
            'details': [
                '⚠️ Appears to be recruitment/staffing agency',
                '⚠️ Not a direct employer',
                '⚠️ Verify legitimacy before sharing documents',
                f"Red flags: {', '.join(red_flags)}"
            ]
        }
    
    # Check 3: Fake patterns
    for pattern in FAKE_PATTERNS:
        if re.search(pattern, company_clean):
            return {
                'status': 'POTENTIALLY FAKE',
                'confidence': 60,
                'reason': 'Generic or suspicious company name pattern',
                'color': 'red',
                'details': [
                    '🚨 Generic company name detected',
                    '🚨 Pattern matches fake companies',
                    '🚨 Research company thoroughly',
                    '🚨 Check official website',
                    '🚨 Verify on LinkedIn/Glassdoor'
                ]
            }
    
    # Check 4: Very short name (suspicious)
    if len(company_clean) < 4:
        return {
            'status': 'SUSPICIOUS',
            'confidence': 50,
            'reason': 'Company name too short/vague',
            'color': 'orange',
            'details': [
                '⚠️ Very short company name',
                '⚠️ Could be fake or incomplete',
                '⚠️ Get full company name',
                '⚠️ Verify on official website'
            ]
        }
    
    # Check 5: Unknown but not suspicious
    return {
        'status': 'UNKNOWN',
        'confidence': 40,
        'reason': 'Company not in database - verify independently',
        'color': 'gray',
        'details': [
            'ℹ️ Not found in known companies database',
            'ℹ️ Could be startup or small company',
            'ℹ️ Verify on:',
            '  - Official company website',
            '  - LinkedIn company page',
            '  - Glassdoor reviews',
            '  - Ministry of Corporate Affairs (MCA)',
            '⚠️ Do NOT pay any fees',
            '⚠️ Verify office address physically'
        ]
    }


def get_verification_tips():
    """Return general verification tips"""
    return [
        "🔍 **Check Official Website**: Legitimate companies have professional websites",
        "🏢 **Verify Office Address**: Use Google Maps to check if office exists",
        "💼 **LinkedIn Company Page**: Real companies have verified LinkedIn pages",
        "⭐ **Glassdoor Reviews**: Check employee reviews and ratings",
        "📄 **MCA Database**: Search on www.mca.gov.in for registered companies",
        "📧 **Email Domain**: Real companies use official domains (not @gmail.com)",
        "📞 **Phone Number**: Verify contact number on official website",
        "🚫 **No Upfront Fees**: Legitimate companies NEVER ask for money",
        "👥 **Interview Process**: Real companies have proper interview rounds",
        "🆔 **Offer Letter**: Should have company letterhead and HR signature"
    ]
