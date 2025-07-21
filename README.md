# Tender Automation System

A comprehensive AI-powered tender processing and filtering system that automatically extracts, filters, and distributes relevant tenders to clients using semantic similarity and email automation.

## 🚀 Features

- **Advanced AI-Powered Filtering**: Uses LangChain + OpenAI for enhanced semantic similarity matching with context understanding
- **Vector Store Optimization**: Chroma vector database for fast and efficient tender retrieval
- **AI-Enhanced Matching**: Combines vector similarity with GPT-3.5 reasoning for better accuracy
- **Automated Email Distribution**: Sends filtered tenders to clients via email with duplicate prevention
- **Interactive Web Interface**: Streamlit-based dashboard with persistent chat history
- **LangChain RAG**: Advanced querying capabilities for tender data
- **Configuration Management**: Easy client and keyword management
- **Real-time Processing**: Daily automated tender processing workflow
- **Persistent Storage**: Vector store caching for improved performance

## 📁 Project Structure

```
eTenderFilter/
├── main.py                    # Main orchestration
├── ConfigManager.py           # Configuration handling
├── TenderProcessor.py         # Excel reading and data prep
├── AiFilter.py               # LangChain-based semantic filtering with AI enhancement
├── EmailSender.py            # Email automation
├── StreamlitApp.py           # Streamlit UI with bot
├── TenderQueryBot.py         # LangChain RAG setup
├── ClientConfig.json         # Client configurations
├── data/                     # Input Excel files
├── FilteredTenders/          # Output filtered files
├── requirements.txt          # Dependencies
└── README.md                # This file
```

## 🛠️ Setup Instructions

### 1. Install Dependencies

```bash
cd eTenderFilter
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the eTenderFilter directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Email Configuration (Gmail)
SENDER_EMAIL=your_email@gmail.com
SENDER_PASSWORD=your_app_password_here
```

**Note**: For Gmail, you'll need to use an App Password instead of your regular password.

### 3. Client Configuration

The system uses `ClientConfig.json` to define clients and their filtering criteria. Each client has:

- **name**: Client identifier
- **enabled**: Whether the client is active
- **keywords**: List of keywords for semantic matching
- **distribution_emails**: List of email addresses to receive tenders

Example:
```json
{
  "name": "IT",
  "enabled": true,
  "keywords": ["Microsoft Licenses", "Laptops", "Cybersecurity"],
  "distribution_emails": ["client@example.com"]
}
```

## 🚀 Usage

### Running the Complete System

1. **Command Line Processing**:
   ```bash
   python main.py
   ```

2. **Web Interface**:
   ```bash
   streamlit run StreamlitApp.py
   ```

### Daily Workflow

1. **Tender Extraction**: The eTenders scraper creates Excel files in `../eTenders/data/`
2. **AI Filtering**: System reads the latest Excel file and filters using semantic similarity
3. **Email Distribution**: Filtered tenders are sent to respective clients (with duplicate prevention)
4. **Data Storage**: Filtered results are saved with client matching information
5. **Query Interface**: Users can ask questions about the filtered tender data

### Web Interface Features

- **Dashboard**: System status and quick actions
- **Process Tenders**: Manual processing and configuration
- **Query Bot**: Natural language queries about tender data
- **Configuration**: Add/remove clients and manage settings
- **System Status**: Environment and configuration diagnostics

## 🤖 Query Bot Examples

The query bot supports natural language questions like:

- "How many IT tenders are there?"
- "Show me tenders from North West province"
- "What's the average closing date for construction tenders?"
- "Which tenders have the highest budget?"

## 🧠 LangChain AI Filter Features

### Enhanced Matching
- **Context Understanding**: AI understands industry terminology and related concepts
- **Semantic Flexibility**: Matches "construction" with "building projects", "infrastructure development"
- **Reasoning**: Provides explanations for why tenders match or don't match

### Performance Optimizations
- **Vector Store Caching**: Embeddings are stored locally in Chroma database
- **One-time Processing**: Tenders are embedded once, then reused for all clients
- **Batch Processing**: Efficient handling of multiple clients simultaneously

### Example Matching Scenarios
- Keyword: "IT services" → Matches: "software development", "digital solutions", "technology consulting"
- Keyword: "construction" → Matches: "building projects", "civil works", "infrastructure development"
- Keyword: "healthcare" → Matches: "medical equipment", "hospital services", "healthcare facilities"

## 📧 Email System

### Email Format
Emails include:
- Tender ID and description
- Province and category
- Closing date
- Direct link to tender
- Formatted as HTML table

### Duplicate Prevention
The system includes intelligent email tracking to prevent duplicate emails:

- **Tender Tracking**: Each tender is tracked by `TENDER_ID` and `PUBLICATION_DATE`
- **Client-Specific**: Tracking is maintained per client
- **Automatic Prevention**: Duplicate tenders are automatically filtered out
- **Manual Reset**: Email tracking can be cleared via the web interface

### Email Tracking File
- **Location**: `email_tracking.json`
- **Format**: JSON file tracking which tenders have been emailed to each client
- **Management**: Can be cleared via Streamlit interface or manually deleted

## 🔧 Configuration

### Adding a New Client

1. Use the web interface Configuration page, or
2. Edit `ClientConfig.json` directly:

```json
{
  "name": "New Client",
  "enabled": true,
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "distribution_emails": ["email1@example.com", "email2@example.com"]
}
```

### AI Filtering Settings

- **Confidence Threshold**: Minimum combined score (default: 0.5)
- **Embedding Model**: OpenAI text-embedding-3-small
- **Vector Store**: Chroma for fast similarity search
- **AI Enhancement**: GPT-3.5 for context-aware matching
- **Combined Scoring**: Vector similarity + AI confidence

## 📊 Output Files

- **Filtered Tenders**: `FilteredTenders/filtered_tenders_YYYYMMDD.xlsx`
  - Contains all original columns plus:
    - `CLIENT_MATCH`: Which client(s) matched
    - `SIMILARITY_SCORE`: Vector similarity score
    - `AI_CONFIDENCE`: GPT-3.5 confidence score
    - `COMBINED_SCORE`: Average of similarity and AI confidence
    - `BEST_MATCH_KEYWORD`: Most relevant keywords
    - `AI_REASONING`: AI explanation for the match
    - `PROCESSING_DATE`: When processed

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure `OPENAI_API_KEY` is set in `.env`
   - Check API key validity and credits

2. **Email Sending Failed**:
   - Verify Gmail credentials in `.env`
   - Use App Password for Gmail
   - Check SMTP settings

3. **No Tender Files Found**:
   - Ensure eTenders scraper has run
   - Check file path: `../eTenders/data/`

4. **Configuration Errors**:
   - Validate `ClientConfig.json` format
   - Check required fields for each client

### Debug Mode

Run with verbose logging:
```bash
python main.py --debug
```

## 🔄 Automation

For daily automation, set up a cron job or Windows Task Scheduler:

```bash
# Daily at 9 AM
0 9 * * * cd /path/to/eTenderFilter && python main.py
```

## 📈 Monitoring

The system provides:
- Processing statistics
- Email delivery reports
- Client matching summaries
- System health status

## 🤝 Contributing

### Development Rules
1. **Naming Conventions**: Follow Python naming conventions (PascalCase for classes, camelCase for methods/variables)
2. **Type Hints**: Add type hints to all function parameters and return values
3. **Documentation**: Add docstrings to all functions and classes
4. **Class Summaries**: Always add class summaries before each class definition
5. **Error Handling**: Use try-catch blocks for all file operations and API calls
6. **Code Style**: Follow PEP 8 guidelines and use Black formatter
7. **Validation**: Validate input data before processing
8. **Environment Variables**: Use environment variables for sensitive configuration
9. **Comments**: Add comments explaining complex AI/ML logic
10. **Testing**: Test edge cases and handle missing/corrupted data gracefully

### File Organization
- Keep related functionality in the same module
- Use clear import statements at the top of files
- Maintain consistent file structure across the project
- Add proper error handling for file operations

### AI/ML Specific Guidelines
- Document AI model parameters and configurations
- Explain vector search and embedding logic
- Add confidence thresholds for AI filtering
- Handle API rate limits and errors gracefully

## 📄 License

This project is proprietary to Amidel.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section
2. Review system status in the web interface
3. Check logs for detailed error messages 