import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import glob

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

class TenderQueryBot:
    def __init__(self, openaiApiKey: str = None):
        load_dotenv()
        
        # Store API key for lazy initialization
        self.apiKey = openaiApiKey or os.getenv('OPENAI_API_KEY')
        if not self.apiKey:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize components (no OpenAI clients yet)
        self.embeddings = None  # Will be initialized when needed
        self.llm = None  # Will be initialized when needed
        self.vectorStore = None
        self.qaChain = None
        self.currentDataFile = None
    
    def _initialize_embeddings(self):
        """Initialize embeddings only when needed"""
        if self.embeddings is None:
            try:
                self.embeddings = OpenAIEmbeddings(api_key=self.apiKey)
                print("TenderQueryBot embeddings initialized successfully")
            except Exception as e:
                print(f"Error initializing TenderQueryBot embeddings: {str(e)}")
                raise e
    
    def _initialize_llm(self):
        """Initialize LLM only when needed"""
        if self.llm is None:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",  # Better reasoning, lower cost
                    # model="gpt-3.5-turbo",  # Original model
                    temperature=0.5,
                    api_key=self.apiKey
                )
                print("TenderQueryBot LLM initialized successfully")
            except Exception as e:
                print(f"Error initializing TenderQueryBot LLM: {str(e)}")
                raise e
    
    def loadFilteredData(self) -> bool:
        """Load the latest filtered tender data for querying"""
        try:
            # Find the latest filtered tender file
            filteredFiles = glob.glob("FilteredTenders/filtered_tenders_*.xlsx")
            if not filteredFiles:
                print("No filtered tender files found")
                return False
            
            # Get the most recent file
            latestFile = max(filteredFiles, key=lambda x: os.path.getmtime(x))
            
            # Check if we already have this file loaded and vector store is ready
            if (latestFile == self.currentDataFile and 
                self.vectorStore is not None and 
                self.qaChain is not None):
                return True  # Already loaded and ready
            
            print(f"Loading filtered data from: {latestFile}")
            
            # Load the Excel file
            df = pd.read_excel(latestFile)
            
            if df.empty:
                print("No data found in filtered file")
                return False
            
            # Create documents from DataFrame
            documents = self.createDocumentsFromDataFrame(df)
            
            # Initialize embeddings if needed
            self._initialize_embeddings()
            
            # Create vector store
            self.vectorStore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name="tender_data"
            )
            
            # Create QA chain
            self.qaChain = self.createQaChain()
            
            self.currentDataFile = latestFile
            print(f"Successfully loaded {len(documents)} tender documents")
            return True
        
        except Exception as e:
            print(f"Error loading filtered data: {str(e)}")
            return False
    
    def createDocumentsFromDataFrame(self, df: pd.DataFrame) -> List:
        """Create documents from DataFrame for vector storage"""
        try:
            documents = []
            
            for index, row in df.iterrows():
                # Create a comprehensive text representation of each tender
                tenderText = f"""
                Tender ID: {row.get('TENDER_ID', 'N/A')}
                Description: {row.get('TENDER_DESCRIPTION', 'N/A')}
                Province: {row.get('PROVINCE', 'N/A')}
                Category: {row.get('CATEGORY', 'N/A')}
                Publication Date: {row.get('PUBLICATION_DATE', 'N/A')}
                Closing Date: {row.get('CLOSING_DATE', 'N/A')}
                Client Match: {row.get('CLIENT_MATCH', 'N/A')}
                Similarity Score: {row.get('SIMILARITY_SCORE', 'N/A')}
                AI Confidence: {row.get('AI_CONFIDENCE', 'N/A')}
                Combined Score: {row.get('COMBINED_SCORE', 'N/A')}
                Best Match Keyword: {row.get('BEST_MATCH_KEYWORD', 'N/A')}
                AI Reasoning: {row.get('AI_REASONING', 'N/A')}
                Link: {row.get('LINK', 'N/A')}
                """
                
                # Create metadata
                metadata = {
                    'tender_id': str(row.get('TENDER_ID', '')),
                    'province': str(row.get('PROVINCE', '')),
                    'category': str(row.get('CATEGORY', '')),
                    'client_match': str(row.get('CLIENT_MATCH', '')),
                    'similarity_score': str(row.get('SIMILARITY_SCORE', '')),
                    'index': index
                }
                
                # Create document
                from langchain.schema import Document
                doc = Document(
                    page_content=tenderText.strip(),
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
        
        except Exception as e:
            print(f"Error creating documents: {str(e)}")
            return []
    
    def createQaChain(self) -> RetrievalQA:
        """Create the QA chain for querying"""
        try:
            # Initialize LLM if needed
            self._initialize_llm()
            
            # Create prompt template with better guidance
            promptTemplate = PromptTemplate(
                input_variables=["context", "question"],
                template="""
                You are a helpful assistant that answers questions about filtered tender data. 
                
                IMPORTANT INSTRUCTIONS:
                - You have access to filtered tender data that has been matched to specific clients
                - Each tender has: Tender ID, Description, Province, Category, Publication Date, Closing Date, Client Match, Similarity Score, Best Match Keyword, and Link
                - For questions about counts or statistics, provide accurate numbers based on the context
                - For questions asking to list tenders, provide a comprehensive list from the context
                - If asked about "all tenders", include all tenders visible in the context
                - Be specific and detailed in your responses
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:"""
            )
            
            # Create QA chain with more context (set k=50 for consistency)
            qaChain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorStore.as_retriever(search_kwargs={"k": 50}),
                chain_type_kwargs={"prompt": promptTemplate}
            )
            
            return qaChain
        
        except Exception as e:
            print(f"Error creating QA chain: {str(e)}")
            return None
    
    def queryTenders(self, question: str) -> dict:
        """Query the tender data with a natural language question and return both context and response"""
        try:
            # Check if we have data loaded and ready
            if (self.vectorStore is None or 
                self.qaChain is None or 
                self.currentDataFile is None):
                # Only load if not already loaded
                if not self.loadFilteredData():
                    return {"context": None, "response": "Sorry, I don't have any tender data to query. Please run the tender processing first."}
            
            if not self.qaChain:
                return {"context": None, "response": "Sorry, the query system is not properly initialized."}
            
            # For questions about statistics or counts, provide direct answers
            if any(keyword in question.lower() for keyword in ['how many', 'count', 'total', 'statistics', 'summary']):
                stats = self.getTenderStats()
                if 'error' not in stats:
                    # Add statistics to the question context
                    enhanced_question = f"""
                    Question: {question}
                    
                    Available Statistics:
                    - Total Tenders: {stats.get('totalTenders', 0)}
                    - Clients: {', '.join([f'{k} ({v})' for k, v in stats.get('clients', {}).items()])}
                    - Provinces: {', '.join([f'{k} ({v})' for k, v in stats.get('provinces', {}).items()])}
                    - Categories: {', '.join([f'{k} ({v})' for k, v in stats.get('categories', {}).items()])}
                    """
                else:
                    enhanced_question = question
            else:
                enhanced_question = question
            
            # Retrieve context documents (increase k to 50)
            retriever = self.vectorStore.as_retriever(search_kwargs={"k": 50})
            context_docs = retriever.get_relevant_documents(enhanced_question)
            context_text = '\n---\n'.join([doc.page_content for doc in context_docs])
            
            # Get answer
            response = self.qaChain.invoke({"query": enhanced_question})
            llm_answer = response.get('result', 'Sorry, I could not find an answer to your question.')
            
            return {"context": context_text, "response": llm_answer}
        
        except Exception as e:
            print(f"Error querying tenders: {str(e)}")
            return {"context": None, "response": f"Sorry, I encountered an error while processing your question: {str(e)}"}
    
    def clearAllData(self):
        """Clear all loaded data and force reload"""
        try:
            print("Clearing chatbot data and forcing reload...")
            
            # Clear current data
            self.vectorStore = None
            self.qaChain = None
            self.currentDataFile = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            print("Chatbot data cleared - will reload on next query")
            return True
        except Exception as e:
            print(f"Error clearing chatbot data: {str(e)}")
            return False
    
    def getTenderStats(self) -> Dict[str, Any]:
        """Get statistics about the loaded tender data"""
        try:
            if not self.loadFilteredData():
                return {"error": "No data loaded"}
            
            # Load the data to get statistics
            filteredFiles = glob.glob("FilteredTenders/filtered_tenders_*.xlsx")
            if not filteredFiles:
                return {"error": "No filtered files found"}
            
            latestFile = max(filteredFiles, key=lambda x: os.path.getmtime(x))
            df = pd.read_excel(latestFile)
            
            stats = {
                'totalTenders': len(df),
                'clients': df['CLIENT_MATCH'].value_counts().to_dict() if 'CLIENT_MATCH' in df.columns else {},
                'provinces': df['PROVINCE'].value_counts().to_dict() if 'PROVINCE' in df.columns else {},
                'categories': df['CATEGORY'].value_counts().to_dict() if 'CATEGORY' in df.columns else {},
                'dataFile': latestFile,
                'lastUpdated': datetime.fromtimestamp(os.path.getmtime(latestFile)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return stats
        
        except Exception as e:
            print(f"Error getting tender stats: {str(e)}")
            return {"error": str(e)}
    
    def searchTenders(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for specific tenders based on query"""
        try:
            if not self.loadFilteredData() or not self.vectorStore:
                return []
            
            # Search the vector store
            results = self.vectorStore.similarity_search(query, k=limit)
            
            # Format results
            formattedResults = []
            for doc in results:
                formattedResults.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
            
            return formattedResults
        
        except Exception as e:
            print(f"Error searching tenders: {str(e)}")
            return []
    
    def getClientSummary(self, clientName: str) -> Dict[str, Any]:
        """Get summary for a specific client"""
        try:
            if not self.loadFilteredData():
                return {"error": "No data loaded"}
            
            # Load the data
            filteredFiles = glob.glob("filtered_tenders_*.xlsx")
            if not filteredFiles:
                return {"error": "No filtered files found"}
            
            latestFile = max(filteredFiles, key=lambda x: os.path.getmtime(x))
            df = pd.read_excel(latestFile)
            
            # Filter for specific client
            clientData = df[df['CLIENT_MATCH'] == clientName]
            
            if clientData.empty:
                return {"error": f"No data found for client: {clientName}"}
            
            summary = {
                'clientName': clientName,
                'totalTenders': len(clientData),
                'provinces': clientData['PROVINCE'].value_counts().to_dict(),
                'categories': clientData['CATEGORY'].value_counts().to_dict(),
                'averageSimilarityScore': clientData['SIMILARITY_SCORE'].mean() if 'SIMILARITY_SCORE' in clientData.columns else 0,
                'topKeywords': clientData['BEST_MATCH_KEYWORD'].value_counts().head(5).to_dict() if 'BEST_MATCH_KEYWORD' in clientData.columns else {}
            }
            
            return summary
        
        except Exception as e:
            print(f"Error getting client summary: {str(e)}")
            return {"error": str(e)} 