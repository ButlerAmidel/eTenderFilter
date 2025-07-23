import openai
import pandas as pd
from typing import Dict, List, Any, Tuple
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class AIFilter:
    def __init__(self, openaiApiKey: str = None, confidenceThreshold: float = 0.3):
        load_dotenv()
        
        # Store API key for lazy initialization
        self.apiKey = openaiApiKey or os.getenv('OPENAI_API_KEY')
        if not self.apiKey:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass it to constructor.")
        
        self.confidenceThreshold = confidenceThreshold
        
        # Initialize basic components (no OpenAI clients yet)
        self.embeddings = None  # Will be initialized when needed
        self.vectorstore = None
        self.llm = None  # Will be initialized when needed
        self.tender_documents = []
        self.current_data_hash = None
        
        # Vector store directory (created only when needed)
        self.vectorstore_dir = Path("vectorstore")
        
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

    def _create_tender_documents(self, tenders: pd.DataFrame) -> List[Document]:
        """Convert tender DataFrame to LangChain documents"""
        documents = []
        
        for index, tender in tenders.iterrows():
            # Create document content using original Excel structure
            content = f"""
Tender ID: {tender.get('TENDER_ID', 'N/A')}
Description: {tender.get('TENDER_DESCRIPTION', 'N/A')}
Category: {tender.get('CATEGORY', 'N/A')}
Location: {tender.get('LOCATION', 'N/A')}
Closing Date: {tender.get('CLOSING_DATE', 'N/A')}
            """.strip()
            
            # Create metadata
            metadata = {
                'tender_id': str(tender.get('TENDER_ID', '')),
                'category': str(tender.get('CATEGORY', '')),
                'location': str(tender.get('LOCATION', '')),
                'closing_date': str(tender.get('CLOSING_DATE', '')),
                'row_index': index
            }
            
            # Create LangChain document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents

    def _initialize_embeddings(self):
        """Initialize embeddings only when needed"""
        if self.embeddings is None:
            try:
                self.embeddings = OpenAIEmbeddings(api_key=self.apiKey)
                print("Embeddings initialized successfully")
            except Exception as e:
                print(f"Error initializing embeddings: {str(e)}")
                raise e
    
    def _initialize_llm(self):
        """Initialize LLM only when needed"""
        if self.llm is None:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",  # Better reasoning, lower cost than 3.5-turbo
                    # model="gpt-3.5-turbo",  # Original model
                    temperature=0.3,
                    api_key=self.apiKey
                )
                print("LLM initialized successfully")
            except Exception as e:
                print(f"Error initializing LLM: {str(e)}")
                raise e
    
    def _get_data_hash(self, tenders: pd.DataFrame) -> str:
        """Generate a hash of the tender data to check if we need to rebuild vector store"""
        import hashlib
        data_str = tenders.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()

    def _build_vectorstore(self, tenders: pd.DataFrame) -> bool:
        """Build or rebuild the vector store from tender data"""
        try:
            print("Building vector store from tender data...")
            
            # Initialize embeddings if needed
            self._initialize_embeddings()
            
            # Convert tenders to documents
            self.tender_documents = self._create_tender_documents(tenders)
            print(f"Created {len(self.tender_documents)} tender documents")
            
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=self.tender_documents,
                embedding=self.embeddings,
                persist_directory=str(self.vectorstore_dir)
            )
            
            # Vector store is automatically persisted in ChromaDB 0.4.x
            # No need to call persist() manually
            
            print("Vector store built and persisted successfully")
            return True
            
        except Exception as e:
            print(f"Error building vector store: {str(e)}")
            return False

    def _load_existing_vectorstore(self) -> bool:
        """Load existing vector store if available"""
        try:
            if self.vectorstore_dir.exists() and any(self.vectorstore_dir.iterdir()):
                self.vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_dir),
                    embedding_function=self.embeddings
                )
                print("Loaded existing vector store")
                return True
        except Exception as e:
            print(f"Error loading existing vector store: {str(e)}")
        
        return False

    def _enhanced_similarity_search(self, keywords: List[str], k: int = 20) -> List[Tuple[Document, float]]:
        """Perform enhanced similarity search using LangChain"""
        try:
            # Create a query from keywords
            query = " ".join(keywords)
            
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Filter results by similarity score threshold (0.3 = 70% similarity)
            filtered_results = [(doc, score) for doc, score in results if score >= 0.3]
            
            # Sort by similarity score (highest first)
            filtered_results.sort(key=lambda x: x[1], reverse=True)
            
            print(f"Vector search returned {len(results)} results, {len(filtered_results)} above threshold")
            
            return filtered_results
            
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def _ai_enhanced_matching(self, tender_content: str, keywords: List[str]) -> Dict[str, Any]:
        """Use AI to enhance matching beyond simple similarity"""
        try:
            # Initialize LLM if not already
            self._initialize_llm()
            
            # Create the prompt directly
            prompt_text = f"""
{self.system_prompt}

Tender Content:
{tender_content}

Keywords to match against:
{', '.join(keywords)}

Analyze if this tender matches the keywords and respond with JSON:
"""
            
            # Get AI response
            response = self.llm.invoke(prompt_text)
            
            # Parse JSON response
            import json
            try:
                result = json.loads(response.content)
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
        """Filter tenders for a specific client using LangChain RAG"""
        try:
            clientName = clientConfig['name']
            keywords = clientConfig['keywords']
            
            print(f"Filtering tenders for client: {clientName}")
            print(f"Using keywords: {keywords}")
            print(f"Processing {len(tenders)} tenders...")
            
            # Debug: Show sample tender data
            self.debugTenderData(tenders)
            
            # Check if we need to rebuild vector store
            data_hash = self._get_data_hash(tenders)
            if data_hash != self.current_data_hash:
                print("Data changed, rebuilding vector store...")
                if not self._build_vectorstore(tenders):
                    return pd.DataFrame()
                self.current_data_hash = data_hash
            elif self.vectorstore is None:
                if not self._load_existing_vectorstore():
                    if not self._build_vectorstore(tenders):
                        return pd.DataFrame()
                    self.current_data_hash = data_hash
            
            # Perform enhanced similarity search
            search_results = self._enhanced_similarity_search(keywords, k=min(20, len(tenders)))
            print(f"Vector search returned {len(search_results)} initial results")
            
            # Deduplicate search results by row_index to prevent processing same tender multiple times
            seen_indices = set()
            unique_results = []
            
            for doc, similarity_score in search_results:
                row_index = doc.metadata.get('row_index')
                if row_index is not None and row_index not in seen_indices:
                    seen_indices.add(row_index)
                    unique_results.append((doc, similarity_score))
            
            print(f"Found {len(search_results)} total results, {len(unique_results)} unique tenders after deduplication")
            
            if len(unique_results) == 0:
                print(f"⚠️ No vector search results found for client {clientName} with keywords: {keywords}")
                print("🔄 Trying simple keyword matching as fallback...")
                
                # Simple keyword matching fallback
                simple_matches = []
                for index, tender in tenders.iterrows():
                    description = tender.get('TENDER_DESCRIPTION', '').lower()
                    category = tender.get('CATEGORY', '').lower()
                    content = f"{description} {category}"
                    
                    # Check if any keyword matches
                    for keyword in keywords:
                        if keyword.lower() in content:
                            simple_matches.append((index, tender, 0.8))  # High confidence for exact matches
                            break
                
                print(f"Simple keyword matching found {len(simple_matches)} matches")
                
                if simple_matches:
                    filteredTenders = []
                    for index, tender, confidence in simple_matches:
                        tenderCopy = tender.copy()
                        tenderCopy['SIMILARITY_SCORE'] = 0.8
                        tenderCopy['AI_CONFIDENCE'] = confidence
                        tenderCopy['COMBINED_SCORE'] = confidence
                        tenderCopy['BEST_MATCH_KEYWORD'] = "Simple keyword match"
                        tenderCopy['AI_REASONING'] = "Matched via simple keyword search"
                        filteredTenders.append(tenderCopy)
                    
                    if filteredTenders:
                        resultDf = pd.DataFrame(filteredTenders)
                        print(f"Simple matching found {len(resultDf)} matches for {clientName}")
                        return resultDf
                
                return pd.DataFrame()
            
            filteredTenders = []
            
            for doc, similarity_score in unique_results:
                # Get original tender data
                row_index = doc.metadata.get('row_index')
                if row_index is not None and row_index < len(tenders):
                    tender = tenders.iloc[row_index]
                    
                    # Use AI to enhance matching
                    ai_result = self._ai_enhanced_matching(doc.page_content, keywords)
                    
                    # Combine similarity score with AI confidence (weighted towards AI confidence)
                    combined_score = (similarity_score * 0.3 + ai_result['confidence'] * 0.7)
                    
                    print(f"Tender {tender.get('TENDER_ID', 'N/A')}: Similarity={similarity_score:.3f}, AI_Confidence={ai_result['confidence']:.3f}, Combined={combined_score:.3f}, AI_Matches={ai_result['matches']}")
                    
                    # Check if meets threshold and AI explicitly matches
                    if combined_score >= self.confidenceThreshold and ai_result['matches'] and ai_result['confidence'] >= 0.6:
                        # Add enhanced information to tender data
                        tenderCopy = tender.copy()
                        tenderCopy['SIMILARITY_SCORE'] = similarity_score
                        tenderCopy['AI_CONFIDENCE'] = ai_result['confidence']
                        tenderCopy['COMBINED_SCORE'] = combined_score
                        tenderCopy['BEST_MATCH_KEYWORD'] = ", ".join(ai_result.get('relevant_keywords', []))
                        tenderCopy['AI_REASONING'] = ai_result.get('reasoning', '')
                        filteredTenders.append(tenderCopy)
                        
                        print(f"Match found: {tender['TENDER_ID']} - Combined Score: {combined_score:.3f} - AI Confidence: {ai_result['confidence']:.3f}")
            
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
        """Filter tenders for all clients using LangChain RAG"""
        try:
            filteredResults = {}
            
            # Always create fresh vector store for each processing run
            print("Creating fresh vector store for processing...")
            self.clearVectorStore()  # Clear any existing vectorstore
            if not self._build_vectorstore(tenders):
                print("Failed to build vector store")
                return {}
            
            print("Vector store created successfully")
            
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
                    if 'COMBINED_SCORE' in clientTenders.columns:
                        avg_score = clientTenders['COMBINED_SCORE'].mean()
                        stats['averageScores'][clientName] = round(avg_score, 3)
            
            return stats
        
        except Exception as e:
            print(f"Error getting filtering stats: {str(e)}")
            return {}

    def clearVectorStore(self):
        """Clear the vector store cache and all in-memory data"""
        try:
            # Clear vector store directory completely
            if self.vectorstore_dir.exists():
                import shutil
                shutil.rmtree(self.vectorstore_dir)
                print("Vector store directory completely cleared")
            
            # Recreate empty directory
            self.vectorstore_dir.mkdir(exist_ok=True)
            
            # Clear all in-memory data
            self.vectorstore = None
            self.tender_documents = []
            self.current_data_hash = None
            
            # Force garbage collection to free memory
            import gc
            gc.collect()
            
            print("Complete memory cleanup - fresh vector store will be built on next processing")
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")
    
    def clearAllData(self):
        """Complete cleanup of all data and memory"""
        try:
            print("Performing complete data cleanup...")
            
            # Clear vector store
            self.clearVectorStore()
            
            # Clear any cached embeddings or models
            if hasattr(self, 'embeddings'):
                # Force re-initialization of embeddings
                del self.embeddings
                self.embeddings = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Clear LLM cache
            if hasattr(self, 'llm'):
                # Force re-initialization of LLM
                del self.llm
                self.llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key=os.getenv('OPENAI_API_KEY')
                )
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("Complete cleanup finished - all data and memory cleared")
        except Exception as e:
            print(f"Error in complete cleanup: {str(e)}")
    
    def getVectorStoreInfo(self) -> Dict[str, Any]:
        """Get information about the current vector store"""
        try:
            info = {
                'exists': False,
                'document_count': 0,
                'directory_size': 0
            }
            
            if self.vectorstore_dir.exists():
                info['exists'] = True
                
                # Get directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(self.vectorstore_dir):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)
                info['directory_size'] = total_size
                
                # Try to get document count
                if self.vectorstore is not None:
                    try:
                        collection = self.vectorstore._collection
                        info['document_count'] = collection.count()
                    except:
                        info['document_count'] = 'Unknown'
            
            return info
        except Exception as e:
            return {'error': str(e)}
    
    def debugTenderData(self, tenders: pd.DataFrame, max_samples: int = 3) -> None:
        """Debug method to show sample tender data"""
        print(f"\n🔍 DEBUG: Sample tender data ({len(tenders)} total tenders):")
        for i in range(min(max_samples, len(tenders))):
            tender = tenders.iloc[i]
            print(f"\nTender {i+1}:")
            print(f"  ID: {tender.get('TENDER_ID', 'N/A')}")
            print(f"  Description: {tender.get('TENDER_DESCRIPTION', 'N/A')[:100]}...")
            print(f"  Category: {tender.get('CATEGORY', 'N/A')}")
            print(f"  Province: {tender.get('PROVINCE', 'N/A')}")
        print("\n" + "="*50) 