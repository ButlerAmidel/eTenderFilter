# Data Folder

This folder contains the Excel files that the Tender Automation System reads.

## File Naming Convention
- Files should be named: `tenders_YYYYMMDD.xlsx`
- Example: `tenders_20241201.xlsx`

## How to Add Files
1. Upload your Excel file to this folder
2. Name it according to the convention above
3. Commit and push to GitHub
4. The system will automatically detect the latest file

## File Structure
Your Excel file should contain these columns:
- TENDER_DESCRIPTION
- PROVINCE
- CATEGORY
- TENDER_ID
- PUBLICATION_DATE
- CLOSING_DATE
- LINK
- DEPARTMENT
- IS_THERE_A_BRIEFING_SESSION
- COMPULSORY_BRIEFING

## Notes
- The system will automatically find the most recent file
- Only one file is processed at a time
- Old files are kept for reference 