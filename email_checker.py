# ==========================================
# EMAIL FRAUD CHECKER (IMAP)
# ==========================================
# Connects to Gmail/Outlook and checks emails for fraud

import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Email configuration
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS', '')
EMAIL_PASSWORD = os.getenv('EMAIL_APP_PASSWORD', '')
IMAP_SERVER = os.getenv('IMAP_SERVER', 'imap.gmail.com')
IMAP_PORT = int(os.getenv('IMAP_PORT', 993))


def connect_to_email():
    """
    Connect to email account via IMAP
    Returns: (mail_connection, error_message)
    """
    try:
        # Connect to IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        
        # Login
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        
        return mail, None
    
    except imaplib.IMAP4.error as e:
        return None, f"Login failed: {str(e)}\nCheck your email/password"
    except Exception as e:
        return None, f"Connection error: {str(e)}"


def get_recent_emails(mail, folder='INBOX', limit=10):
    """
    Fetch recent emails from specified folder
    """
    try:
        # Select mailbox
        mail.select(folder)
        
        # Search for all emails
        status, messages = mail.search(None, 'ALL')
        
        if status != 'OK':
            return []
        
        # Get message IDs
        message_ids = messages[0].split()
        
        # Get most recent emails (last N)
        recent_ids = message_ids[-limit:] if len(message_ids) > limit else message_ids
        
        emails = []
        
        for msg_id in reversed(recent_ids):  # Most recent first
            # Fetch email
            status, msg_data = mail.fetch(msg_id, '(RFC822)')
            
            if status != 'OK':
                continue
            
            # Parse email
            raw_email = msg_data[0][1]
            email_message = email.message_from_bytes(raw_email)
            
            # Extract details
            subject = decode_email_header(email_message.get('Subject', ''))
            sender = email_message.get('From', '')
            date = email_message.get('Date', '')
            
            # Get email body
            body = get_email_body(email_message)
            
            emails.append({
                'id': msg_id.decode(),
                'subject': subject,
                'sender': sender,
                'date': date,
                'body': body[:500],  # First 500 chars
                'full_body': body
            })
        
        return emails
    
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []


def decode_email_header(header):
    """
    Decode email header (subject, etc.)
    """
    if not header:
        return ""
    
    decoded = decode_header(header)
    result = ""
    
    for part, encoding in decoded:
        if isinstance(part, bytes):
            try:
                result += part.decode(encoding or 'utf-8')
            except:
                result += part.decode('utf-8', errors='ignore')
        else:
            result += str(part)
    
    return result


def get_email_body(email_message):
    """
    Extract email body text
    """
    body = ""
    
    if email_message.is_multipart():
        # Multiple parts (HTML + text)
        for part in email_message.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition', ''))
            
            # Skip attachments
            if 'attachment' in content_disposition:
                continue
            
            # Get text content
            if content_type == 'text/plain':
                try:
                    body += part.get_payload(decode=True).decode()
                except:
                    pass
    else:
        # Single part
        try:
            body = email_message.get_payload(decode=True).decode()
        except:
            body = str(email_message.get_payload())
    
    return body.strip()


def check_email_configured():
    """
    Check if email credentials are configured
    """
    return bool(EMAIL_ADDRESS and EMAIL_PASSWORD and 
                EMAIL_ADDRESS != 'your.email@gmail.com' and 
                EMAIL_PASSWORD != 'your_app_password_here')


def get_email_folders(mail):
    """
    Get list of available email folders
    """
    try:
        status, folders = mail.list()
        if status == 'OK':
            folder_list = []
            for folder in folders:
                # Decode folder name
                folder_name = folder.decode().split('"')[-2]
                folder_list.append(folder_name)
            return folder_list
        return ['INBOX']
    except:
        return ['INBOX']


def mark_email_as_spam(mail, email_id):
    """
    Move email to Spam/Junk folder
    (Note: This requires appropriate permissions)
    """
    try:
        # Copy to Spam
        mail.copy(email_id, '[Gmail]/Spam')
        # Delete from Inbox
        mail.store(email_id, '+FLAGS', '\\Deleted')
        mail.expunge()
        return True
    except Exception as e:
        print(f"Error marking as spam: {e}")
        return False


def disconnect_email(mail):
    """
    Close email connection
    """
    try:
        mail.close()
        mail.logout()
    except:
        pass
