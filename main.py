import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ConfigManager import ConfigManager
from TenderProcessor import TenderProcessor
from AiFilter import AIFilter
from EmailSender import EmailSender

# TenderAutomation: Main orchestrator class that coordinates the entire daily tender processing workflow.
# Handles file management, AI filtering, email sending, and generates processing summaries.
class TenderAutomation:
    def __init__(self):
        """Initialize the tender automation system"""
        try:
            # Initialize components
            self.configManager = ConfigManager()
            self.tenderProcessor = TenderProcessor()
            self.aiFilter = AIFilter()
            self.emailSender = EmailSender()
            
            print("Tender Automation System initialized successfully")
        
        except Exception as e:
            print(f"Error initializing Tender Automation: {str(e)}")
            raise
    
    def runDailyProcess(self) -> Dict[str, Any]:
        """Run the complete daily tender processing workflow"""
        try:
            print(f"\n{'='*50}")
            print(f"Starting Daily Tender Processing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # Step 1: Complete cleanup for fresh start
            print("\n1. Performing complete cleanup for fresh start...")
            self.aiFilter.clearAllData()
            
            # Step 2: Get latest tender file
            print("\n2. Finding latest tender file...")
            latestFile = self.tenderProcessor.getLatestTenderFile()
            if not latestFile:
                raise FileNotFoundError("No tender files found in data folder")
            
            print(f"Found file: {latestFile}")
            
            # Step 3: Read and prepare tender data
            print("\n3. Reading and preparing tender data...")
            tenderData = self.tenderProcessor.readExcelFile(latestFile)
            preparedData = self.tenderProcessor.prepareForFiltering(tenderData)
            
            # Step 4: Get client configurations
            print("\n4. Loading client configurations...")
            clientConfigs = self.configManager.config
            enabledClients = self.configManager.getEnabledClients()
            print(f"Found {len(enabledClients)} enabled clients")
            
            # Step 5: AI filtering
            print("\n5. Running AI filtering...")
            filteredResults = self.aiFilter.batchFilter(preparedData, enabledClients)
            
            # Step 6: Save filtered results (overwrites if same name)
            print("\n6. Saving filtered results...")
            outputPath = f"FilteredTenders/filtered_tenders_{datetime.now().strftime('%Y%m%d')}.xlsx"
            self.tenderProcessor.saveFilteredResults(filteredResults, outputPath)
            
            # Step 7: Send emails
            print("\n7. Sending email notifications...")
            emailResults = self.emailSender.sendBulkEmails(filteredResults, clientConfigs)
            
            # Step 8: Generate summary
            print("\n8. Generating summary...")
            summary = self.generateSummary(
                tenderData, filteredResults, emailResults, latestFile, outputPath
            )
            
            print(f"\n{'='*50}")
            print("Daily Processing Complete!")
            print(f"{'='*50}")
            
            return summary
        
        except Exception as e:
            print(f"Error in daily process: {str(e)}")
            return {'error': str(e)}
    
    def generateSummary(self, tenderData, filteredResults, emailResults, inputFile, outputFile) -> Dict[str, Any]:
        """Generate summary of the processing results"""
        try:
            # Get tender statistics
            tenderStats = self.tenderProcessor.getTenderStats(tenderData)
            
            # Get filtering statistics
            filteringStats = self.aiFilter.getFilteringStats(filteredResults)
            
            # Get email statistics
            emailStats = self.emailSender.getEmailStats(emailResults)
            
            summary = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'inputFile': inputFile,
                'outputFile': outputFile,
                'tenderStats': tenderStats,
                'filteringStats': filteringStats,
                'emailStats': emailStats,
                'clientResults': {}
            }
            
            # Add per-client results
            for clientName, clientTenders in filteredResults.items():
                summary['clientResults'][clientName] = {
                    'tenderCount': len(clientTenders) if not clientTenders.empty else 0,
                    'emailSent': emailResults.get(clientName, False)
                }
            
            return summary
        
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return {'error': str(e)}
    
    def getSystemStatus(self) -> Dict[str, Any]:
        """Get current system status"""
        try:
            status = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'configValid': self.configManager.validateConfig(),
                'enabledClients': len(self.configManager.getEnabledClients()),
                'latestTenderFile': self.tenderProcessor.getLatestTenderFile(),
                'openaiConfigured': bool(os.getenv('OPENAI_API_KEY')),
                'emailConfigured': bool(os.getenv('SENDER_EMAIL') and os.getenv('SENDER_PASSWORD'))
            }
            return status
        
        except Exception as e:
            return {'error': str(e)}

def main():
    """Main function to run the tender automation"""
    try:
        # Initialize the system
        automation = TenderAutomation()
        
        # Run daily process
        summary = automation.runDailyProcess()
        
        # Print summary
        if 'error' not in summary:
            print("\nProcessing Summary:")
            print(f"Total Tenders Processed: {summary['tenderStats']['totalTenders']}")
            print(f"Total Matches Found: {summary['filteringStats']['totalMatches']}")
            print(f"Emails Sent Successfully: {summary['emailStats']['successfulEmails']}")
            print(f"Output File: {summary['outputFile']}")
        else:
            print(f"Processing failed: {summary['error']}")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main() 