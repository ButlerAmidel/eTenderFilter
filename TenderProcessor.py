import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# TenderProcessor: Handles Excel file operations for tender data including reading, validation, preparation, and saving.
# Manages file discovery, data cleaning, and statistics generation for tender processing workflows.
class TenderProcessor:
    def __init__(self, dataFolderPath: str = "data"):
        self.dataFolderPath = dataFolderPath
        self.requiredColumns = [
            'TENDER_DESCRIPTION', 'PROVINCE', 'CATEGORY', 
            'TENDER_ID', 'PUBLICATION_DATE', 'CLOSING_DATE', 'LINK',
            'DEPARTMENT', 'IS_THERE_A_BRIEFING_SESSION', 'COMPULSORY_BRIEFING'
        ]
    
    def getLatestTenderFile(self) -> Optional[str]:
        """Get the most recent tender Excel file from the data folder"""
        try:
            dataPath = Path(self.dataFolderPath)
            if not dataPath.exists():
                raise FileNotFoundError(f"Data folder {self.dataFolderPath} not found")
            
            # Find all Excel files with date pattern
            excelFiles = list(dataPath.glob("tenders_*.xlsx"))
            if not excelFiles:
                return None
            
            # Sort by modification time and get the latest
            latestFile = max(excelFiles, key=lambda x: x.stat().st_mtime)
            return str(latestFile)
        
        except Exception as e:
            print(f"Error finding latest tender file: {str(e)}")
            return None
    
    def readExcelFile(self, filePath: str) -> pd.DataFrame:
        """Read and validate Excel file"""
        try:
            df = pd.read_excel(filePath)
            
            # Check if required columns exist
            missingColumns = [col for col in self.requiredColumns if col not in df.columns]
            if missingColumns:
                raise ValueError(f"Missing required columns: {missingColumns}")
            
            print(f"Successfully loaded {len(df)} tenders from {filePath}")
            return df
        
        except Exception as e:
            raise Exception(f"Error reading Excel file {filePath}: {str(e)}")
    
    def prepareForFiltering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for AI filtering"""
        try:
            # Create a copy to avoid modifying original
            filteredDf = df.copy()
            
            # Ensure TENDER_DESCRIPTION is string type
            filteredDf['TENDER_DESCRIPTION'] = filteredDf['TENDER_DESCRIPTION'].astype(str)
            
            # Remove rows with empty descriptions
            filteredDf = filteredDf[filteredDf['TENDER_DESCRIPTION'].str.strip() != '']
            
            # Add processing timestamp
            filteredDf['PROCESSING_DATE'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Ensure all required columns are present and clean
            for col in self.requiredColumns:
                if col not in filteredDf.columns:
                    filteredDf[col] = ''
                else:
                    filteredDf[col] = filteredDf[col].fillna('')
            
            print(f"Prepared {len(filteredDf)} tenders for filtering")
            return filteredDf
        
        except Exception as e:
            raise Exception(f"Error preparing data for filtering: {str(e)}")
    
    def saveFilteredResults(self, filteredResults: Dict[str, pd.DataFrame], outputPath: str) -> None:
        """Save filtered results to Excel with client matching information"""
        try:
            # Combine all filtered results into one DataFrame
            combinedData = []
            
            for clientName, clientTenders in filteredResults.items():
                if not clientTenders.empty:
                    # Add client match column
                    clientTendersCopy = clientTenders.copy()
                    clientTendersCopy['CLIENT_MATCH'] = clientName
                    combinedData.append(clientTendersCopy)
            
            if combinedData:
                finalDf = pd.concat(combinedData, ignore_index=True)
                
                # Sort by client name for better organization
                finalDf = finalDf.sort_values(['CLIENT_MATCH', 'TENDER_ID'])
                
                # Save to Excel
                finalDf.to_excel(outputPath, index=False)
                print(f"Saved {len(finalDf)} filtered tenders to {outputPath}")
            else:
                print("No filtered tenders to save")
        
        except Exception as e:
            raise Exception(f"Error saving filtered results: {str(e)}")
    
    def getTenderStats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about the tender data"""
        try:
            stats = {
                'totalTenders': len(df),
                'provinces': df['PROVINCE'].value_counts().to_dict(),
                'categories': df['CATEGORY'].value_counts().to_dict(),
                'dateRange': {
                    'earliest': df['PUBLICATION_DATE'].min() if 'PUBLICATION_DATE' in df.columns else None,
                    'latest': df['PUBLICATION_DATE'].max() if 'PUBLICATION_DATE' in df.columns else None
                }
            }
            return stats
        
        except Exception as e:
            print(f"Error getting tender stats: {str(e)}")
            return {} 