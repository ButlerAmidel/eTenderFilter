import smtplib
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

class EmailSender:
    def __init__(self, smtpServer: str = "smtp.gmail.com", smtpPort: int = 587):
        load_dotenv()
        
        self.smtpServer = smtpServer
        self.smtpPort = smtpPort
        
        # Get email credentials from environment variables
        self.senderEmail = os.getenv('SENDER_EMAIL')
        self.senderPassword = os.getenv('SENDER_PASSWORD')
        
        if not self.senderEmail or not self.senderPassword:
            raise ValueError("Email credentials not found. Set SENDER_EMAIL and SENDER_PASSWORD environment variables.")
    

    
    def createEmailTable(self, tenders: pd.DataFrame) -> str:
        """Create HTML table from tender data"""
        try:
            if tenders.empty:
                return "<p>No tenders found for this client.</p>"
            
            # Create HTML table
            htmlTable = """
            <table border="1" style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">
                <thead>
                    <tr style="background-color: #f2f2f2;">
                        <th style="padding: 8px; text-align: left;">Tender ID</th>
                        <th style="padding: 8px; text-align: left;">Description</th>
                        <th style="padding: 8px; text-align: left;">Department</th>
                        <th style="padding: 8px; text-align: left;">Province</th>
                        <th style="padding: 8px; text-align: left;">Category</th>
                        <th style="padding: 8px; text-align: left;">Closing Date</th>
                        <th style="padding: 8px; text-align: left;">Briefing Session</th>
                        <th style="padding: 8px; text-align: left;">Compulsory Briefing</th>
                        <th style="padding: 8px; text-align: left;">Link</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for _, tender in tenders.iterrows():
                # Truncate description if too long (200 characters)
                description = str(tender.get('TENDER_DESCRIPTION', ''))
                if len(description) > 200:
                    description = description[:197] + "..."
                
                # Format closing date
                closingDate = str(tender.get('CLOSING_DATE', ''))
                
                # Create link
                link = str(tender.get('LINK', ''))
                linkHtml = f'<a href="{link}" target="_blank">View Tender</a>' if link else 'N/A'
                
                # Format briefing session information
                briefingSession = str(tender.get('IS_THERE_A_BRIEFING_SESSION', 'N/A'))
                compulsoryBriefing = str(tender.get('COMPULSORY_BRIEFING', 'N/A'))
                
                htmlTable += f"""
                    <tr>
                        <td style="padding: 8px; text-align: left;">{tender.get('TENDER_ID', 'N/A')}</td>
                        <td style="padding: 8px; text-align: left;">{description}</td>
                        <td style="padding: 8px; text-align: left;">{tender.get('DEPARTMENT', 'N/A')}</td>
                        <td style="padding: 8px; text-align: left;">{tender.get('PROVINCE', 'N/A')}</td>
                        <td style="padding: 8px; text-align: left;">{tender.get('CATEGORY', 'N/A')}</td>
                        <td style="padding: 8px; text-align: left;">{closingDate}</td>
                        <td style="padding: 8px; text-align: left;">{briefingSession}</td>
                        <td style="padding: 8px; text-align: left;">{compulsoryBriefing}</td>
                        <td style="padding: 8px; text-align: left;">{linkHtml}</td>
                    </tr>
                """
            
            htmlTable += """
                </tbody>
            </table>
            """
            
            return htmlTable
        
        except Exception as e:
            print(f"Error creating email table: {str(e)}")
            return "<p>Error creating tender table.</p>"
    
    def createEmailContent(self, clientName: str, tenders: pd.DataFrame) -> str:
        """Create complete email content"""
        try:
            tenderCount = len(tenders) if not tenders.empty else 0
            currentDate = datetime.now().strftime('%Y-%m-%d')
            
            emailContent = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .footer {{ background-color: #f2f2f2; padding: 10px; text-align: center; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Tender Alert - {clientName}</h1>
                    <p>Date: {currentDate}</p>
                </div>
                
                <div class="content">
                    <h2>Hello,</h2>
                    <p>We found <strong>{tenderCount}</strong> tender(s) that match your criteria.</p>
                    
                    {self.createEmailTable(tenders)}
                    
                    <p><strong>Note:</strong> Please click on the links above to view the full tender details and submission requirements.</p>
                </div>
                
                <div class="footer">
                    <p>This is an automated tender alert from Amidel Tender System.</p>
                    <p>Generated on: {currentDate}</p>
                </div>
            </body>
            </html>
            """
            
            return emailContent
        
        except Exception as e:
            print(f"Error creating email content: {str(e)}")
            return f"<p>Error creating email content for {clientName}.</p>"
    
    def sendTenderEmail(self, clientName: str, tenders: pd.DataFrame, emailList: List[str]) -> bool:
        """Send tender email to specified recipients"""
        try:
            if not emailList:
                print(f"No email recipients specified for {clientName}")
                return False
            
            # Create email content
            emailContent = self.createEmailContent(clientName, tenders)
            
            # Send individual emails to each recipient
            with smtplib.SMTP(self.smtpServer, self.smtpPort) as server:
                server.starttls()
                server.login(self.senderEmail, self.senderPassword)
                
                for recipient in emailList:
                    # Create individual message for each recipient
                    msg = MIMEMultipart('alternative')
                    msg['Subject'] = f"Tender Alert - {clientName} - {datetime.now().strftime('%Y-%m-%d')}"
                    msg['From'] = self.senderEmail
                    msg['To'] = recipient  # Single recipient
                    
                    # Attach HTML content
                    htmlPart = MIMEText(emailContent, 'html')
                    msg.attach(htmlPart)
                    
                    # Send to this specific recipient
                    server.sendmail(self.senderEmail, recipient, msg.as_string())
                    print(f"Email sent to {recipient} for {clientName}")
            
            return True
        
        except Exception as e:
            print(f"Error sending email for {clientName}: {str(e)}")
            return False
    
    def sendBulkEmails(self, filteredResults: Dict[str, pd.DataFrame], clientConfigs: Dict[str, Any]) -> Dict[str, bool]:
        """Send emails to all clients with filtered results"""
        try:
            emailResults = {}
            
            for clientName, clientTenders in filteredResults.items():
                if not clientTenders.empty:
                    # Get client configuration
                    clientConfig = None
                    for client in clientConfigs.get('clients', []):
                        if client.get('name') == clientName:
                            clientConfig = client
                            break
                    
                    if clientConfig and clientConfig.get('enabled', False):
                        emailList = clientConfig.get('distribution_emails', [])
                        success = self.sendTenderEmail(clientName, clientTenders, emailList)
                        emailResults[clientName] = success
                        
                        if success:
                            print(f"Email sent successfully for {clientName} with {len(clientTenders)} tenders")
                        else:
                            print(f"Failed to send email for {clientName}")
                    else:
                        print(f"Client {clientName} not found or disabled")
                        emailResults[clientName] = False
                else:
                    print(f"No tenders to send for {clientName}")
                    emailResults[clientName] = True  # Not an error, just no tenders
            
            return emailResults
        
        except Exception as e:
            print(f"Error in bulk email sending: {str(e)}")
            return {}
    
    def getEmailStats(self, emailResults: Dict[str, bool]) -> Dict[str, Any]:
        """Get statistics about email sending results"""
        try:
            stats = {
                'totalClients': len(emailResults),
                'successfulEmails': 0,
                'failedEmails': 0,
                'successRate': 0.0
            }
            
            for clientName, success in emailResults.items():
                if success:
                    stats['successfulEmails'] += 1
                else:
                    stats['failedEmails'] += 1
            
            if stats['totalClients'] > 0:
                stats['successRate'] = (stats['successfulEmails'] / stats['totalClients']) * 100
            
            return stats
        
        except Exception as e:
            print(f"Error getting email stats: {str(e)}")
            return {} 