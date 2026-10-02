# ==========================================
# SERPAPI COMPANY VERIFICATION
# ==========================================
# Uses SerpAPI to search Google and verify companies

from serpapi import GoogleSearch
import os
import re

# SerpAPI Key
SERPAPI_KEY = "9e02381b328ee2038ecf236d613488f220b09c839032a4c7aae0c88921f99bc5"


def verify_company_with_serpapi(company_name):
    """
    Verify company using SerpAPI (Google Search)
    Checks: Website, LinkedIn, Glassdoor, News
    """
    
    if not company_name or len(company_name.strip()) < 2:
        return {
            'status': 'INVALID INPUT ❌',
            'confidence': 0,
            'reason': 'Company name too short',
            'color': 'gray',
            'details': ['Please enter a valid company name']
        }
    
    try:
        # Search Google for company
        search_params = {
            "q": f"{company_name} company",
            "api_key": SERPAPI_KEY,
            "num": 10
        }
        
        search = GoogleSearch(search_params)
        results = search.get_dict()
        
        # Analyze results
        organic_results = results.get("organic_results", [])
        
        if not organic_results:
            return {
                'status': 'NOT FOUND ❌',
                'confidence': 80,
                'reason': 'No search results found - likely fake',
                'color': 'red',
                'details': [
                    '❌ No Google search results',
                    '❌ Company does not exist online',
                    '🚨 HIGH RISK - Likely fake company',
                    '⚠️ DO NOT proceed with this company',
                    '⚠️ DO NOT share documents or pay fees'
                ]
            }
        
        # Check indicators
        indicators = {
            'official_website': False,
            'linkedin': False,
            'wikipedia': False,
            'glassdoor': False,
            'news': False,
            'domain_match': False
        }
        
        urls = []
        titles = []
        
        for result in organic_results[:10]:
            link = result.get('link', '').lower()
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            
            urls.append(link)
            titles.append(title)
            
            # Check LinkedIn
            if 'linkedin.com/company' in link:
                indicators['linkedin'] = True
            
            # Check Wikipedia
            if 'wikipedia.org' in link:
                indicators['wikipedia'] = True
            
            # Check Glassdoor
            if 'glassdoor.co' in link or 'glassdoor.com' in link:
                indicators['glassdoor'] = True
            
            # Check news
            if any(news_site in link for news_site in ['news', 'times', 'hindu', 'economic', 'business', 'forbes']):
                indicators['news'] = True
            
            # Check official domain (not gmail/generic)
            if '.com' in link or '.in' in link or '.co' in link:
                if 'gmail' not in link and 'yahoo' not in link:
                    # Check if company name is in domain
                    company_clean = re.sub(r'[^a-z0-9]', '', company_name.lower())
                    if len(company_clean) > 3 and company_clean[:4] in link:
                        indicators['official_website'] = True
                        indicators['domain_match'] = True
        
        # Calculate score
        score = sum(indicators.values())
        max_score = len(indicators)
        
        # Get top result details
        top_result = organic_results[0] if organic_results else {}
        top_link = top_result.get('link', '')
        top_title = top_result.get('title', '')
        
        # Decision logic
        if score >= 4:
            # Strong presence
            status = 'LEGITIMATE ✅'
            confidence = 85 + (score * 2)
            reason = 'Strong online presence verified'
            color = 'green'
            details = [
                f'✅ Found {len(organic_results)} Google results',
                f'✅ Verification score: {score}/{max_score}',
                '✅ Company appears legitimate'
            ]
            
            if indicators['linkedin']:
                details.append('✅ LinkedIn company page exists')
            if indicators['glassdoor']:
                details.append('✅ Glassdoor reviews available')
            if indicators['wikipedia']:
                details.append('✅ Wikipedia page exists')
            if indicators['official_website']:
                details.append(f'✅ Official website: {top_link}')
            if indicators['news']:
                details.append('✅ Found in news articles')
            
            details.append('✅ Safe to proceed with caution')
        
        elif score >= 2:
            # Moderate presence
            status = 'LIKELY LEGITIMATE ⚠️'
            confidence = 60 + (score * 5)
            reason = 'Some online presence found'
            color = 'orange'
            details = [
                f'⚠️ Found {len(organic_results)} Google results',
                f'⚠️ Verification score: {score}/{max_score}',
                '⚠️ Company has some online presence',
                '⚠️ Could be small/startup company'
            ]
            
            if indicators['linkedin']:
                details.append('✅ LinkedIn page found')
            if indicators['glassdoor']:
                details.append('✅ Glassdoor reviews found')
            
            details.extend([
                '🔍 Recommended: Verify further',
                '🔍 Check official website',
                '🔍 Verify office address',
                '⚠️ Do NOT pay any fees'
            ])
        
        else:
            # Weak presence
            status = 'SUSPICIOUS 🚨'
            confidence = 65
            reason = 'Limited online presence - exercise caution'
            color = 'red'
            details = [
                f'🚨 Only {len(organic_results)} Google results found',
                f'🚨 Verification score: {score}/{max_score}',
                '🚨 Very limited online presence',
                '🚨 No LinkedIn or Glassdoor presence',
                '⚠️ HIGH RISK - Verify thoroughly',
                '⚠️ Could be fake or very new company',
                '🔍 Manual verification required:',
                '  • Call and verify office address',
                '  • Check company registration (MCA)',
                '  • Ask for registration documents',
                '🚫 DO NOT share documents',
                '🚫 DO NOT pay any fees'
            ]
        
        # Build verification sources
        sources = []
        if indicators['linkedin']:
            sources.append('LinkedIn')
        if indicators['glassdoor']:
            sources.append('Glassdoor')
        if indicators['wikipedia']:
            sources.append('Wikipedia')
        if indicators['official_website']:
            sources.append('Official Website')
        if indicators['news']:
            sources.append('News Articles')
        
        if not sources:
            sources = ['Google Search']
        
        return {
            'status': status,
            'confidence': min(confidence, 95),
            'reason': reason,
            'color': color,
            'details': details,
            'verification_sources': sources,
            'search_results_count': len(organic_results),
            'top_result': top_title,
            'top_link': top_link,
            'source': 'SerpAPI (Google Search)'
        }
    
    except Exception as e:
        # API error or limit exceeded
        return {
            'status': 'API ERROR ⚠️',
            'confidence': 0,
            'reason': f'SerpAPI error: {str(e)}',
            'color': 'gray',
            'details': [
                '⚠️ Unable to verify via SerpAPI',
                f'⚠️ Error: {str(e)}',
                '⚠️ Possible reasons:',
                '  • API quota exceeded (100 searches/month)',
                '  • Network connection issue',
                '  • Invalid API key',
                '',
                '💡 Fallback to manual verification:',
                '  • Search company on Google directly',
                '  • Check LinkedIn company page',
                '  • Check Glassdoor reviews',
                '  • Verify on MCA database (India)'
            ],
            'verification_sources': [],
            'source': 'Error'
        }


def get_verification_tips():
    """Return general verification tips"""
    return [
        "🔍 **Official Website**: Legitimate companies have professional websites",
        "🏢 **Physical Office**: Verify address on Google Maps Street View",
        "💼 **LinkedIn**: Real companies have verified company pages with employees",
        "⭐ **Glassdoor/AmbitionBox**: Check employee reviews and company ratings",
        "📄 **MCA Database**: Search on www.mca.gov.in for Indian registered companies",
        "📧 **Email Domain**: Real companies use official domains (not @gmail.com)",
        "📞 **Phone Number**: Verify contact matches company website",
        "🚫 **No Upfront Fees**: Legitimate companies NEVER ask for registration/training fees",
        "👥 **Interview Process**: Real companies have proper technical/HR interview rounds",
        "🆔 **Offer Letter**: Should have company letterhead, HR signature, and office address",
        "📱 **Social Media**: Check official Facebook, Twitter, Instagram pages",
        "📰 **Google News**: Search for company in news articles",
        "🏦 **Bank Details**: Verify company bank account (not personal accounts)",
        "⚖️ **Registration**: Ask for GST number, PAN, and company registration certificate"
    ]
