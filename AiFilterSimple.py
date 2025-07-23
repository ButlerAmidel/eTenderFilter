import openai
import pandas as pd
from typing import Dict, List, Any, Tuple
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime

class SimpleAIFilter:
    def __init__(self, openaiApiKey: str = None, confidenceThreshold: float = 0.5):
        load_dotenv()
        
        # Store API key for lazy initialization
        self.apiKey = openaiApiKey or os.getenv('OPENAI_API_KEY')
        if not self.apiKey:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it to constructor.")
        
        self.confidenceThreshold = confidenceThreshold
        
        # Initialize OpenAI client only when needed
        self.client = None
        
        # System prompt for tender matching
        self.system_prompt = """You are a tender matching assistant. Your job is to analyze tender descriptions and determine if they match specific keywords.

IMPORTANT: Be STRICT and CONSERVATIVE in your matching. Only match if there is a clear, direct, and strong relationship.

Consider:
- Semantic meaning (not just exact word matches)
- Industry context and terminology
- Technical and professional services
- Client's core business focus

STRICT FILTERING RULES:
- Reject matches that are too vague, tangential, or loosely related
- Only match if the tender clearly and directly relates to the client's core business
- Be very conservative
- If in doubt, reject the match
- Consider the specificity and relevance of the tender to the client's expertise

Respond with a JSON object containing:
{
    "matches": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of why it matches or doesn't match",
    "relevant_keywords": ["list of keywords that were relevant"]
}"""

    def _initialize_client(self):
        """Initialize OpenAI client only when needed"""
        if self.client is None:
            try:
                self.client = openai.OpenAI(api_key=self.apiKey)
                print("OpenAI client initialized successfully")
            except Exception as e:
                print(f"Error initializing OpenAI client: {str(e)}")
                raise e

    def _ai_enhanced_matching(self, tender_content: str, keywords: List[str]) -> Dict[str, Any]:
        """Use AI to enhance matching beyond simple similarity"""
        try:
            # Initialize client if not already
            self._initialize_client()
            
            # Create the prompt
            prompt_text = f"""
{self.system_prompt}

Tender Content:
{tender_content}

Keywords to match against:
{', '.join(keywords)}

Analyze if this tender matches the keywords and respond with JSON:
"""
            
            # Get AI response using direct OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Tender Content:\n{tender_content}\n\nKeywords to match against:\n{', '.join(keywords)}\n\nAnalyze if this tender matches the keywords and respond with JSON:"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse JSON response
            try:
                result = json.loads(response.choices[0].message.content)
                return result
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "matches": False,
                    "confidence": 0.0,
                    "reasoning": "Error parsing AI response",
                    "relevant_keywords": []
                }
                
        except Exception as e:
            print(f"Error in AI enhanced matching: {str(e)}")
            return {
                "matches": False,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "relevant_keywords": []
            }

    def filterTendersForClient(self, tenders: pd.DataFrame, clientConfig: Dict[str, Any]) -> pd.DataFrame:
        """Filter tenders for a specific client using simple AI matching"""
        try:
            clientName = clientConfig['name']
            keywords = clientConfig['keywords']
            
            print(f"Filtering tenders for client: {clientName}")
            print(f"Using keywords: {keywords}")
            print(f"Processing {len(tenders)} tenders...")
            
            filteredTenders = []
            
            for index, tender in tenders.iterrows():
                # Create tender content for analysis
                tender_content = f"""
Tender ID: {tender.get('TENDER_ID', 'N/A')}
Description: {tender.get('TENDER_DESCRIPTION', 'N/A')}
Category: {tender.get('CATEGORY', 'N/A')}
Location: {tender.get('LOCATION', 'N/A')}
Closing Date: {tender.get('CLOSING_DATE', 'N/A')}
                """.strip()
                
                # Use AI to enhance matching
                ai_result = self._ai_enhanced_matching(tender_content, keywords)
                
                # Check if meets threshold and AI explicitly matches
                if ai_result['matches'] and ai_result['confidence'] >= self.confidenceThreshold:
                    # Add enhanced information to tender data
                    tenderCopy = tender.copy()
                    tenderCopy['AI_CONFIDENCE'] = ai_result['confidence']
                    tenderCopy['BEST_MATCH_KEYWORD'] = ", ".join(ai_result.get('relevant_keywords', []))
                    tenderCopy['AI_REASONING'] = ai_result.get('reasoning', '')
                    filteredTenders.append(tenderCopy)
                    
                    print(f"Match found: {tender['TENDER_ID']} - AI Confidence: {ai_result['confidence']:.3f}")
            
            if filteredTenders:
                resultDf = pd.DataFrame(filteredTenders)
                
                # Final deduplication by TENDER_ID to ensure no duplicates in final results
                if not resultDf.empty and 'TENDER_ID' in resultDf.columns:
                    original_count = len(resultDf)
                    resultDf = resultDf.drop_duplicates(subset=['TENDER_ID'], keep='first')
                    final_count = len(resultDf)
                    if original_count != final_count:
                        print(f"Removed {original_count - final_count} duplicate TENDER_IDs from final results")
                
                print(f"Found {len(resultDf)} unique matches for {clientName}")
                return resultDf
            else:
                print(f"No matches found for {clientName}")
                return pd.DataFrame()
        
        except Exception as e:
            print(f"Error filtering tenders for {clientName}: {str(e)}")
            return pd.DataFrame()

    def batchFilter(self, tenders: pd.DataFrame, clients: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Filter tenders for all clients using simple AI matching"""
        try:
            filteredResults = {}
            
            for client in clients:
                if client.get('enabled', False):
                    clientName = client['name']
                    print(f"\nProcessing client: {clientName}")
                    
                    clientResults = self.filterTendersForClient(tenders, client)
                    if not clientResults.empty:
                        filteredResults[clientName] = clientResults
                    else:
                        filteredResults[clientName] = pd.DataFrame()
            
            return filteredResults
        
        except Exception as e:
            print(f"Error in batch filtering: {str(e)}")
            return {}

    def getFilteringStats(self, filteredResults: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get statistics about filtering results"""
        try:
            stats = {
                'totalClients': len(filteredResults),
                'clientsWithMatches': 0,
                'totalMatches': 0,
                'clientBreakdown': {},
                'averageScores': {}
            }
            
            for clientName, clientTenders in filteredResults.items():
                matchCount = len(clientTenders) if not clientTenders.empty else 0
                stats['clientBreakdown'][clientName] = matchCount
                stats['totalMatches'] += matchCount
                
                if matchCount > 0:
                    stats['clientsWithMatches'] += 1
                    # Calculate average scores
                    if 'AI_CONFIDENCE' in clientTenders.columns:
                        avg_score = clientTenders['AI_CONFIDENCE'].mean()
                        stats['averageScores'][clientName] = round(avg_score, 3)
            
            return stats
        
        except Exception as e:
            print(f"Error getting filtering stats: {str(e)}")
            return {} 