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
            st.write("🚀 Starting StreamlitApp initialization...")
            
            # Setup secrets from Streamlit Cloud
            st.write("🔐 Setting up secrets...")
            self._setupSecrets()
            st.write("✅ Secrets setup complete")
            
            st.write("⚙️ Initializing ConfigManager...")
            self.configManager = ConfigManager()
            st.write("✅ ConfigManager initialized")
            
            st.write("📁 Initializing TenderProcessor...")
            self.tenderProcessor = TenderProcessor()
            st.write("✅ TenderProcessor initialized")
            
            st.write("🤖 Initializing AIFilter...")
            self.aiFilter = AIFilter()
            st.write("✅ AIFilter initialized")
            
            st.write("📧 Initializing EmailSender...")
            self.emailSender = EmailSender()
            st.write("✅ EmailSender initialized")
            
            # Initialize queryBot only once using session state
            if "queryBot" not in st.session_state:
                try:
                    st.write("🔍 Initializing TenderQueryBot...")
                    st.session_state.queryBot = TenderQueryBot()
                    st.write("✅ TenderQueryBot initialized successfully")
                except Exception as e:
                    st.write(f"❌ Error initializing TenderQueryBot: {str(e)}")
                    # Don't raise the error - just log it and continue
                    # The query bot will be initialized on first use
                    st.warning("⚠️ QueryBot initialization failed - will retry on first use")
                    st.session_state.queryBot = None
            
            self.queryBot = st.session_state.queryBot
            if self.queryBot:
                st.write("✅ QueryBot assigned to self")
            else:
                st.write("⚠️ QueryBot will be initialized on first use")
            
            # Set page config
            st.write("📄 Setting page config...")
            st.set_page_config(
                page_title="Tender Automation System",
                page_icon="📋",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            st.write("✅ Page config set")
            
            st.write("🎉 StreamlitApp initialization complete!")
        
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
                ["Dashboard", "Configuration", "System Status"]
            )
            
            # Page routing
            if page == "Dashboard":
                self.showDashboard()
            elif page == "Configuration":
                self.showConfiguration()
            elif page == "System Status":
                self.showSystemStatus()
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")
    
    def showDashboard(self):
        """Show the dashboard interface with query bot and daily process"""
        st.title("Dashboard")
        
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
        
        # Daily Process Section
        st.subheader("🔄 Daily Process")
        
        # Check if processing is already running
        if "processing_status" not in st.session_state:
            st.session_state.processing_status = "idle"
        
        if st.session_state.processing_status == "running":
            st.warning("⚠️ Processing is currently running...")
        else:
            # Get latest tender file info
            latestFile = self.tenderProcessor.getLatestTenderFile()
            
            if latestFile:
                st.success(f"✅ Found tender file: {latestFile.split('/')[-1]}")
                
                # Show file info
                try:
                    df = self.tenderProcessor.readExcelFile(latestFile)
                    st.info(f"📊 File contains {len(df)} tenders")
                    
                    # Run daily process button
                    if st.button("🚀 Run Daily Process", type="primary", use_container_width=True):
                        self._runDailyProcess()
                        
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
            else:
                st.error("❌ No tender files found in data folder")
        
        # Show processing results if available
        if "processing_results" in st.session_state:
            st.subheader("📊 Processing Results")
            results = st.session_state.processing_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Clients", results.get('totalClients', 0))
            with col2:
                st.metric("Clients with Matches", results.get('clientsWithMatches', 0))
            with col3:
                st.metric("Total Tenders", results.get('totalTenders', 0))
            
            # Download button
            if "filtered_data" in st.session_state:
                st.download_button(
                    "📥 Download Filtered Results",
                    st.session_state.filtered_data,
                    "filtered_tenders.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        st.divider()
        
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
    
    def _runDailyProcess(self):
        """Run the daily tender processing workflow"""
        try:
            # Check if aiFilter is available
            if not hasattr(self, 'aiFilter') or self.aiFilter is None:
                st.error("❌ AIFilter is not initialized. Please refresh the page and try again.")
                return
            
            st.session_state.processing_status = "running"
            
            # Create progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Get latest tender file
            status_text.text("📁 Finding latest tender file...")
            progress_bar.progress(10)
            
            latestFile = self.tenderProcessor.getLatestTenderFile()
            if not latestFile:
                st.error("No tender file found")
                st.session_state.processing_status = "idle"
                return
            
            # Step 2: Read and prepare data
            status_text.text("📊 Reading and preparing tender data...")
            progress_bar.progress(20)
            
            tenderData = self.tenderProcessor.readExcelFile(latestFile)
            preparedData = self.tenderProcessor.prepareForFiltering(tenderData)
            
            # Step 3: Get client configurations
            status_text.text("⚙️ Loading client configurations...")
            progress_bar.progress(30)
            
            enabledClients = self.configManager.getEnabledClients()
            
            # Step 4: Run AI filtering
            status_text.text("🤖 Running AI filtering...")
            progress_bar.progress(50)
            
            filteredResults = self.aiFilter.batchFilter(preparedData, enabledClients)
            
            # Step 5: Generate statistics
            status_text.text("📈 Generating statistics...")
            progress_bar.progress(80)
            
            stats = self.aiFilter.getFilteringStats(filteredResults)
            
            # Step 6: Prepare results for download
            status_text.text("💾 Preparing results...")
            progress_bar.progress(90)
            
            # Combine all filtered results into one DataFrame
            combinedData = []
            for clientName, clientTenders in filteredResults.items():
                if not clientTenders.empty:
                    clientTendersCopy = clientTenders.copy()
                    clientTendersCopy['CLIENT_MATCH'] = clientName
                    combinedData.append(clientTendersCopy)
            
            if combinedData:
                finalDf = pd.concat(combinedData, ignore_index=True)
                finalDf = finalDf.sort_values(['CLIENT_MATCH', 'TENDER_ID'])
                
                # Convert to Excel bytes for download
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    finalDf.to_excel(writer, index=False, sheet_name='Filtered Tenders')
                output.seek(0)
                
                # Store results in session state
                st.session_state.filtered_data = output.getvalue()
                st.session_state.processing_results = stats
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.success(f"✅ Successfully processed {len(finalDf)} tenders for {len([k for k, v in filteredResults.items() if not v.empty])} clients!")
                
            else:
                st.warning("⚠️ No tenders matched the filtering criteria")
                st.session_state.processing_results = stats
            
            # Clear status
            st.session_state.processing_status = "idle"
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.session_state.processing_status = "idle"
    
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
        
        dataFolder = Path("data")
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
                    
                    # Add button to clear vectorstore if there's an error
                    if st.button("🗑️ Clear Vector Store (Fix Compatibility Issue)"):
                        try:
                            import shutil
                            if vectorstore_dir.exists():
                                shutil.rmtree(vectorstore_dir)
                                st.success("✅ Vector store cleared successfully!")
                                st.rerun()
                        except Exception as clear_error:
                            st.error(f"❌ Error clearing vector store: {str(clear_error)}")
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