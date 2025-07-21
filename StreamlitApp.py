import streamlit as st
import pandas as pd
import os
import sys
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import json

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ConfigManager import ConfigManager
from TenderProcessor import TenderProcessor
from AiFilter import AIFilter
from EmailSender import EmailSender
from TenderQueryBot import TenderQueryBot

# StreamlitApp: Web-based user interface for the tender automation system using Streamlit.
# Provides interactive pages for querying tenders, managing client configurations, and monitoring system status.
class StreamlitApp:
    def __init__(self):
        """Initialize the Streamlit application"""
        try:
            # Setup secrets from Streamlit Cloud
            self._setupSecrets()
            
            self.configManager = ConfigManager()
            self.tenderProcessor = TenderProcessor()
            self.aiFilter = AIFilter()
            self.emailSender = EmailSender()
            
            # Initialize queryBot only once using session state
            if "queryBot" not in st.session_state:
                st.session_state.queryBot = TenderQueryBot()
            
            self.queryBot = st.session_state.queryBot
            
            # Set page config
            st.set_page_config(
                page_title="Tender Automation System",
                page_icon="📋",
                layout="wide",
                initial_sidebar_state="expanded"
            )
        
        except Exception as e:
            st.error(f"Error initializing app: {str(e)}")
    
    def _setupSecrets(self):
        """Setup environment variables from Streamlit secrets if available"""
        try:
            if hasattr(st, 'secrets'):
                # Load secrets into environment variables for compatibility
                if 'OPENAI_API_KEY' in st.secrets:
                    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                if 'SENDER_EMAIL' in st.secrets:
                    os.environ['SENDER_EMAIL'] = st.secrets['SENDER_EMAIL']
                if 'SENDER_PASSWORD' in st.secrets:
                    os.environ['SENDER_PASSWORD'] = st.secrets['SENDER_PASSWORD']
        except Exception as e:
            print(f"Error setting up secrets: {str(e)}")
    

    
    def run(self):
        """Run the Streamlit application"""
        try:
            # Sidebar navigation
            st.sidebar.title("📋 Tender Automation")
            
            page = st.sidebar.radio(
                "Navigation",
                ["Query Bot", "Configuration", "System Status"]
            )
            
            # Page routing
            if page == "Query Bot":
                self.showQueryBot()
            elif page == "Configuration":
                self.showConfiguration()
            elif page == "System Status":
                self.showSystemStatus()
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")
    
    def showQueryBot(self):
        """Show the tender query bot interface with dashboard info"""
        st.title("Tender Query Bot")
        
        # Dashboard section at the top
        st.subheader("System Overview")
        
        # Get latest tender file
        latestFile = self.tenderProcessor.getLatestTenderFile()
        
        # System status cards with smaller text
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Total Clients:** {len(self.configManager.getEnabledClients())}")
        
        with col2:
            if latestFile:
                # Extract just the filename from the full path
                filename = latestFile.split('/')[-1] if '/' in latestFile else latestFile.split('\\')[-1]
                st.markdown(f"**Latest File:** {filename}")
            else:
                st.markdown("**Latest File:** Not Found")
        
        # Client tender summary
        st.subheader("Client Tender Summary")
        
        # Try to load the latest filtered tender data
        try:
            
            # Find the latest filtered tender file
            filteredFiles = glob.glob("FilteredTenders/filtered_tenders_*.xlsx")
            if filteredFiles:
                latestFilteredFile = max(filteredFiles, key=lambda x: os.path.getmtime(x))
                df = pd.read_excel(latestFilteredFile)
                
                # Get tender counts per client
                if 'CLIENT_MATCH' in df.columns:
                    clientCounts = df['CLIENT_MATCH'].value_counts()
                    
                    # Display in a more compact format
                    st.markdown("**Client Tender Counts:**")
                    client_text = []
                    for client, count in clientCounts.items():
                        client_text.append(f"**{client}:** {count}")
                    
                    # Display in 2 columns for better layout
                    col1, col2 = st.columns(2)
                    with col1:
                        for i in range(0, len(client_text), 2):
                            if i < len(client_text):
                                st.markdown(client_text[i])
                    with col2:
                        for i in range(1, len(client_text), 2):
                            if i < len(client_text):
                                st.markdown(client_text[i])
                    
                    # Show total filtered tenders
                    st.markdown(f"**Total Filtered Tenders:** {len(df)}")
                    
                    # Show processing date
                    if 'PROCESSING_DATE' in df.columns:
                        processingDate = df['PROCESSING_DATE'].iloc[0] if len(df) > 0 else "Unknown"
                        st.markdown(f"*Last processed: {processingDate}*")
                else:
                    st.info("No client matching data found in filtered tenders")
            else:
                st.info("No filtered tender files found. Run the tender processing first.")
                
        except Exception as e:
            st.warning(f"Could not load tender summary: {str(e)}")
        
        st.divider()
        
        # Chatbot section
        st.markdown("## 💬 Ask your question below:")
        
        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Type your question here..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response (context + LLM answer)
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    result = self.queryBot.queryTenders(prompt)
                    context = result.get("context")
                    llm_answer = result.get("response")
                    if context:
                        with st.expander("🔎 Context retrieved for this answer", expanded=False):
                            context_chunks = context.split('\n---\n') if context else []
                            for chunk in context_chunks:
                                lines = chunk.strip().split('\n')
                                if lines:
                                    # Try to bold the first line (Tender ID or Client Match)
                                    first_line = lines[0]
                                    rest = '\n'.join(lines[1:])
                                    st.markdown(f"<div style='border:1px solid #ddd; border-radius:6px; padding:8px; margin-bottom:8px; background:#fafbfc;'><b>{first_line}</b><br>{rest.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                    st.markdown(llm_answer)
            
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": llm_answer})
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    def showConfiguration(self):
        """Show configuration management interface"""
        st.title("⚙️ Configuration Management")
        
        # Client management
        st.subheader("Client Management")
        
        # Add new client
        with st.expander("➕ Add New Client"):
            with st.form("add_client"):
                clientName = st.text_input("Client Name")
                keywords = st.text_area("Keywords (one per line)")
                emails = st.text_area("Email Addresses (one per line)")
                enabled = st.checkbox("Enabled", value=True)
                
                if st.form_submit_button("Add Client"):
                    if clientName and keywords and emails:
                        try:
                            keywordList = [k.strip() for k in keywords.split('\n') if k.strip()]
                            emailList = [e.strip() for e in emails.split('\n') if e.strip()]
                            
                            clientData = {
                                'name': clientName,
                                'keywords': keywordList,
                                'distribution_emails': emailList,
                                'enabled': enabled
                            }
                            
                            self.configManager.addClient(clientData)
                            st.success(f"Client '{clientName}' added successfully!")
                            st.rerun()
                        
                        except Exception as e:
                            st.error(f"Error adding client: {str(e)}")
                    else:
                        st.error("Please fill in all required fields")
        
        # View and edit existing clients
        st.subheader("Existing Clients")
        if "edit_client" not in st.session_state:
            st.session_state.edit_client = None
        if "delete_client" not in st.session_state:
            st.session_state.delete_client = None
        
        for client in self.configManager.config.get('clients', []):
            is_editing = st.session_state.edit_client == client['name']
            is_deleting = st.session_state.delete_client == client['name']
            with st.expander(f"📋 {client['name']} ({'✅ Enabled' if client.get('enabled') else '❌ Disabled'})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if is_editing:
                        with st.form(f"edit_client_form_{client['name']}"):
                            new_name = st.text_input("Client Name", value=client['name'])
                            new_keywords = st.text_area("Keywords (one per line)", value='\n'.join(client['keywords']))
                            new_emails = st.text_area("Email Addresses (one per line)", value='\n'.join(client['distribution_emails']))
                            new_enabled = st.checkbox("Enabled", value=client.get('enabled', True))
                            save = st.form_submit_button("Save")
                            cancel = st.form_submit_button("Cancel")
                        if save:
                            try:
                                keywordList = [k.strip() for k in new_keywords.split('\n') if k.strip()]
                                emailList = [e.strip() for e in new_emails.split('\n') if e.strip()]
                                updated_client = {
                                    'name': new_name,
                                    'keywords': keywordList,
                                    'distribution_emails': emailList,
                                    'enabled': new_enabled
                                }
                                # Remove old client and add updated one
                                self.configManager.removeClient(client['name'])
                                self.configManager.addClient(updated_client)
                                st.session_state.edit_client = None
                                st.success(f"Client '{new_name}' updated successfully!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating client: {str(e)}")
                        elif cancel:
                            st.session_state.edit_client = None
                            st.rerun()
                    else:
                        st.write(f"**Keywords:** {', '.join(client['keywords'])}")
                        st.write(f"**Emails:** {', '.join(client['distribution_emails'])}")
                with col2:
                    if not is_editing and not is_deleting:
                        if st.button("✏️ Edit", key=f"edit_{client['name']}"):
                            st.session_state.edit_client = client['name']
                            st.rerun()
                        if st.button("🗑️ Delete", key=f"delete_{client['name']}"):
                            st.session_state.delete_client = client['name']
                            st.rerun()
                    elif is_deleting:
                        st.warning(f"Are you sure you want to delete '{client['name']}'?")
                        confirm, cancel = st.columns(2)
                        with confirm:
                            if st.button("Yes, delete", key=f"confirm_delete_{client['name']}"):
                                try:
                                    self.configManager.removeClient(client['name'])
                                    st.session_state.delete_client = None
                                    st.success(f"Client '{client['name']}' deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting client: {str(e)}")
                        with cancel:
                            if st.button("Cancel", key=f"cancel_delete_{client['name']}"):
                                st.session_state.delete_client = None
                                st.rerun()
        
        # Configuration validation
        st.subheader("Configuration Status")
        if self.configManager.validateConfig():
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration has errors")
    
    def showSystemStatus(self):
        """Show system status and diagnostics"""
        st.title("🔍 System Status")
        
        # Environment variables
        st.subheader("Environment Variables")
        
        envVars = {
            "OpenAI API Key": os.getenv('OPENAI_API_KEY', 'Not Set'),
            "Sender Email": os.getenv('SENDER_EMAIL', 'Not Set'),
            "Sender Password": "***" if os.getenv('SENDER_PASSWORD') else 'Not Set'
        }
        
        for var, value in envVars.items():
            if value != 'Not Set':
                st.success(f"✅ {var}: {value[:20]}..." if len(str(value)) > 20 else f"✅ {var}: {value}")
            else:
                st.error(f"❌ {var}: {value}")
        
        # File system status
        st.subheader("File System Status")
        
        dataFolder = Path("../eTenders/data")
        if dataFolder.exists():
            excelFiles = list(dataFolder.glob("tenders_*.xlsx"))
            st.success(f"✅ Data folder found with {len(excelFiles)} Excel files")
            
            if excelFiles:
                latestFile = max(excelFiles, key=lambda x: x.stat().st_mtime)
                st.info(f"📁 Latest file: {latestFile.name}")
        else:
            st.error("❌ Data folder not found")
        
        # Configuration status
        st.subheader("Configuration Status")
        
        try:
            configValid = self.configManager.validateConfig()
            if configValid:
                st.success("✅ Client configuration is valid")
                enabledClients = self.configManager.getEnabledClients()
                st.info(f"📊 {len(enabledClients)} clients enabled")
            else:
                st.error("❌ Client configuration has errors")
        except Exception as e:
            st.error(f"❌ Configuration error: {str(e)}")
        

        

        
        # AI Filter status
        st.subheader("AI Filter Status")
        
        try:
            # Check vector store
            vectorstore_dir = Path("vectorstore")
            if vectorstore_dir.exists() and any(vectorstore_dir.iterdir()):
                st.success("✅ Vector store ready")
                
                # Show vector store info
                try:
                    from langchain_community.vectorstores import Chroma
                    from langchain_openai import OpenAIEmbeddings
                    
                    vectorstore = Chroma(
                        persist_directory=str(vectorstore_dir),
                        embedding_function=OpenAIEmbeddings()
                    )
                    
                    # Get collection info
                    collection = vectorstore._collection
                    count = collection.count()
                    st.info(f"📊 Vector store contains {count} tender embeddings")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not read vector store details: {str(e)}")
            else:
                st.info("💾 No vector store found - will build on first processing run")
            
            # Check OpenAI
            if os.getenv('OPENAI_API_KEY'):
                st.success("✅ OpenAI API configured")
            else:
                st.error("❌ OpenAI API key not configured")
                
        except Exception as e:
            st.error(f"❌ AI Filter error: {str(e)}")

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main() 