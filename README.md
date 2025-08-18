# [UPCOMING: 19.08.2025] Migrate to Managed Vertex AI Evaluation Service
*   Objective: This upcoming update will showcase the migration from the initial baseline evaluation script to a robust pipeline using the programmatic Vertex AI Evaluation Service SDK. This demonstrates a key MLOps skill: leveraging managed services for scalable and reliable model assessment.
*   Engineering Journey: *I built a RAG system and initially developed a baseline evaluation script. However, I identified its limitations in scalability and metric reliability. I then engineered a robust evaluation pipeline by migrating to the managed Vertex AI Evaluation Service, which allowed me to systematically measure and improve performance across multiple dimensions like groundedness, relevance, and correctness.*

# Alza AI Email Assistant Demo

**Professional demonstration of an intelligent email customer service system for Alza.cz**

## 🎯 Project Overview

**This is a Proof-of-Concept (POC) project** demonstrating an advanced AI-powered email assistant that automatically processes customer inquiries sent to Alza.cz. The system integrates multiple Google Cloud services with Google's Gemini AI models to provide intelligent, contextual responses to customer questions through email attachments (audio, PDFs, images) and text.

### Project Goals (Alza Assessment Requirements)

**Core Objectives:**
1. **Automatic Email Monitoring**: System monitors dedicated inbox for new unread emails
2. **Multi-modal Processing**: Parse and understand attachments (PDFs, audio files, images) 
3. **Intelligent Response Generation**: Generate contextual customer service responses
4. **Production-Ready Architecture**: Scalable cloud-native solution with proper error handling

**Advanced Features Implementation:**
- ✅ **Option B: RAG (Retrieval-Augmented Generation)** - Implemented comprehensive knowledge base integration
- Integration with Alza's product catalog, warranty policies, and customer service procedures
- Semantic search and context-aware response generation

**Key Capabilities:**
- **Multi-modal Processing**: Handles text, audio files, PDFs, and images
- **Intelligent Query Extraction**: Automatically identifies multiple questions within a single email
- **RAG-Enhanced Responses**: Uses Retrieval-Augmented Generation with Alza's knowledge base
- **Real-time Processing**: Responds to emails within seconds via Gmail API integration
- **Production-Ready**: Scalable architecture with proper error handling and security

---

## 🏗️ System Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gmail API     │───▶│  Cloud Pub/Sub  │───▶│ Cloud Function  │
│ (Email Monitor) │    │  (Notifications)│    │ (Modular Design)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                          │
                ┌─────────────────────────────────────────┼─────────────────────────────
                │                       │                 │                            │
                ▼                       ▼                 ▼                            ▼
      ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
      │    Firestore    │     │  Cloud Storage  │     │     Gemini      │     │   Vertex AI     │
      │  (State Mgmt)   │     │ (Audio Files)   │     │   AI Models     │     │ (RAG Engine     │
      └─────────────────┘     └─────────────────┘     └─────────────────┘     │ Knowledge Base) │
                                                                              └─────────────────┘
```

### Modular Code Architecture

**Production-Ready Modular Design:**
```
src/
├── main.py              # Lightweight orchestrator (Gmail history tracking)
├── modules/
│   ├── config.py        # Shared configuration and clients
│   ├── gmail_service.py # Gmail API operations and email processing
│   ├── ai_core.py       # AI query generation, RAG, and response synthesis
│   └── security.py      # Input validation and sanitization
└── test_modules.py      # Comprehensive test suite
```

**Benefits:**
- **Testability**: Each module independently testable with mocks
- **Team Development**: Parallel work without code conflicts
- **Maintainability**: Single Responsibility Principle enforced
- **Scalability**: Easy to extend and modify individual components

### Processing Flow

**1. Event Trigger & Invocation**
*   A new email arrives in the monitored inbox.
*   `Gmail API` detects the change and sends a real-time **push notification** to `Cloud Pub/Sub`.
*   The Pub/Sub message instantly triggers the main **`Cloud Function`**.

**2. Initial Processing & Security**
*   **State Management:** The function immediately attempts to create a lock in `Firestore` using the email's unique ID to prevent any possibility of duplicate processing.
*   **Security Validation:** All attachments are validated for allowed types (`.pdf`, `.wav`, etc.) and size limits. Filenames are sanitized to prevent path traversal.
*   **Input Sanitization:** The email body and subject are sanitized to mitigate prompt injection risks.

**3. AI Core - Step A: Deconstruct the Problem (Query Generation)**
*   The sanitized email content and attachments are passed to the first Gemini model (`gemini-2.5-lite`).
*   **Goal:** To understand the user's complete intent and break it down into a list of specific, targeted questions for the knowledge base.
*   **Output:** A JSON object containing a list of precise search queries (e.g., `["Alza Brno service center address", "return policy hygienic goods", "dispute resolution process"]`).

**4. AI Core - Step B: Gather & Rank Knowledge (Multi-Query RAG)**
*   The system iterates through the generated list of queries.
*   For each query, it performs a semantic search against the `Vertex AI RAG` knowledge base.
*   An **LLM-based reranker** then re-orders the retrieved context chunks to prioritize the most relevant information for answering the specific question.
*   All retrieved context is **consolidated** into a single, clean knowledge block for the next step.

**5. AI Core - Step C: Synthesize the Final Answer**
*   The powerful final response model (`gemini-2.5-pro`) is invoked.
*   **Input:** A carefully constructed prompt containing:
    1.  The original user email for context.
    2.  The consolidated, reranked knowledge from the RAG step.
*   **Goal:** To generate a professional, helpful, and factually grounded HTML response that addresses all of the user's original questions.

**6. Finalization & Delivery**
*   **Trust & Safety:** The system appends source citations (which documents were used) and a responsible AI disclaimer to the generated HTML.
*   **Email Reply:** The final HTML is sent as a reply within the correct email thread using the `Gmail API`.
*   **State Update:** The email's status in `Firestore` is updated to "replied", completing the cycle.


---

## 🤖 AI Models & Configuration

### Gemini Model Usage

**Primary Model: `gemini-2.5-flash-lite-001`**
- **Query Extraction** (1st AI Call):
  - Purpose: Extract questions from email text + attachments
  - Token Limit: 1,024 tokens
  - Temperature: 0.8 (creative query generation)
  - Multi-modal: ✅ (text, audio, PDF, images)
  
**Primary Model: `gemini-2.5-pro`**
- **Response Generation** (2nd AI Call):
  - Purpose: Generate customer service responses
  - Token Limit: 4,096 tokens  
  - Temperature: 0.7 (balanced creativity/accuracy)
  - RAG-Enhanced: ✅ (uses knowledge base context)

### AI Processing Pipeline

```
Email + Attachments
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Gemini 2.5 flash-lite: Multi-modal Query Extraction│
│  Input: Email text + Audio/PDF/Image files          │
│  Output: ["query1", "query2", "query3", ...]        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Vertex AI RAG: Knowledge Base Search               │
│  Input: Customer queries                            │
│  Output: Relevant product/service information       │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Gemini 2.5 PRO: Customer Response Generation       │
│  Input: Original email + RAG context                │
│  Output: Professional customer service response     │
└─────────────────────────────────────────────────────┘
```

---

## 📡 Pub/Sub Monitoring System

### Gmail API Integration

The system uses Gmail's push notification system for real-time email monitoring:

**Monitoring Setup:**
```python
# Executed via start_watch.py
gmail_service.users().watch(
    userId='me',
    body={
        'labelIds': ['INBOX', 'UNREAD'],
        'topicName': 'projects/{PROJECT_ID}/topics/email-notifications',
        'labelFilterAction': 'include'
    }
)
```

**Event Flow:**
1. **New Email Arrives** → Gmail API detects unread email in INBOX
2. **Pub/Sub Notification** → Gmail sends push notification to Cloud Pub/Sub topic
3. **Cloud Function Trigger** → Pub/Sub triggers the email processing function
4. **State Management** → Firestore tracks processing history to prevent duplicates

**Monitoring Script: `start_watch.py`**
- Authenticates with Gmail API using OAuth 2.0
- Establishes push notification subscription
- Must be executed periodically (Gmail watch expires after 7 days)
- Handles credential refresh automatically

---

## 🗃️ Knowledge Base & RAG System

### Knowledge Sources

The system includes a comprehensive knowledge base covering:

**Product Information:**
- `alza_product_catalog.md`: Product specifications, prices, warranties
- Key products: AlzaConnect USB-C Hub, AlzaPlus membership, gaming laptops

**Customer Service:**
- `alza_return_process.md`: Return policies, shipping, refund procedures  
- `alza_warranty_policy.md`: Warranty coverage, service centers, claims process
- `Obchodní-podmínky-Alza.cz.pdf`: Complete terms and conditions

### RAG Implementation

**Vector Search with Vertex AI:**
- **Initial Setup**: Default configuration with standard chunk_size, chunk_overlap, and Layout parser
- **Optimization Phase**: After performance testing, implemented custom chunking strategy:
  - **Chunk Size**: 300 characters (optimized for semantic coherence)
  - **Chunk Overlap**: 50 characters (maintains context continuity)  
  - **Document Parser**: Upgraded from default Layout parser to **DocumentAI** for better text extraction
- **Reranker Integration**: LLM-based reranking for improved answer accuracy
- **Context Limiting**: Max 3 relevant chunks per query, 2000 chars total per response

**Context Optimization Journey:**
- **Initial Issue**: RAG returned 79k+ characters overwhelming the AI model
- **Solution**: Implemented intelligent context filtering and chunking optimization  
- **Result**: 90% reduction in context size while maintaining accuracy
- **Reranker Enhancement**: Added LLM-based reranking for 4.2% improvement in answer correctness

---

## 📦 Technical Dependencies

### Core Libraries

**Google Cloud Integration:**
```python
google-cloud-aiplatform>=1.106.0     # Vertex AI and Gemini models
google-cloud-storage==2.18.0         # File storage (AI Platform compatible)
google-cloud-secret-manager>=2.24.0  # Secure credential management
google-cloud-firestore>=2.20.0       # Document database for state
google-cloud-pubsub>=2.25.0          # Real-time messaging
```

**Gmail & Authentication:**
```python
google-api-python-client>=2.156.0    # Gmail API integration
google-auth>=2.36.0                  # Authentication framework
google-auth-oauthlib>=1.2.2          # OAuth 2.0 flow
```

**AI & Language Processing:**
```python
langchain>=0.3.0                     # LLM application framework
langchain-google-vertexai>=2.0.27    # Vertex AI integration
functions-framework>=3.9.2           # Cloud Functions runtime
```

### Key Dependency Notes

**Google Cloud Storage Version Lock:**
- **Version**: `==2.18.0` (strictly required)
- **Reason**: AI Platform compatibility constraint
- **Limitation**: Versions ≥3.0.0 cause compatibility issues with Vertex AI
- **Impact**: Ensures stable integration between storage and AI services

**Multi-modal Processing:**
```python
pypdf>=5.1.0                        # PDF document processing
Pillow>=11.0.0                      # Image processing
python-magic>=0.4.27                # File type detection
```

---

## 🔧 Key Features

### Multi-modal Attachment Processing

**Supported File Types:**
- **Audio Files**: `.mp3`, `.wav`, `.m4a` (up to 50MB)
- **PDF Documents**: Product manuals, contracts, receipts (up to 25MB)  
- **Images**: `.jpg`, `.png` for visual product questions (up to 10MB)

**Processing Methods:**
- **Audio**: Stored in Cloud Storage, processed via GCS URI (required for Gemini API)
- **PDFs/Images**: Processed directly via base64 encoding
- **Security**: Comprehensive file validation and size limits

### Intelligent Query Extraction

The system can handle complex emails with multiple questions:

**Example Input:**
```
Subject: Product Questions
Body: "I have questions about:
1) AlzaConnect USB-C Hub ports?
2) AlzaPlus membership pricing?
3) Audio file attached with dispute question"

Attachment: dispute_question.mp3
```

**AI Processing Result:**
```json
{
  "queries": [
    "AlzaConnect USB-C Hub porty specifikace",
    "AlzaPlus membership cena",
    "řešení sporů s Alza kontakt"
  ]
}
```

### Production-Ready Security

**Email Processing Security:**
- Duplicate email prevention via Firestore document locking
- Self-sent email filtering to prevent loops
- Attachment validation with type and size restrictions
- Path traversal protection for filenames

**API Security:**
- OAuth 2.0 authentication with refresh token handling
- Secret Manager for credential storage
- Request retry logic with exponential backoff
- Comprehensive error handling and logging

---

## 🚀 Deployment & Configuration

### Environment Setup

**Required Environment Variables:**
```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION_GEMINI=europe-west1
BOT_EMAIL_ADDRESS=your-bot@gmail.com
GCS_BUCKET_NAME=your-storage-bucket
FIRESTORE_DB_ID=your-firestore-database
GEMINI_MODEL=gemini-2.5-flash-lite
GEMINI_MODEL_FINAL_ANSWER=gemini-2.5-pro
LLM_RANKER=gemini-2.5-flash-lite
RAG_CORPUS_DISPLAY_NAME=alza-email-bot-knowledge
PUBSUB_TOPIC_NAME=your-topic-name
PUBSUB_SUBSCRIPTION_NAME=your-subscription-name
```

### Deployment Commands

**Deploy Modular Cloud Function:**
```bash
# Deploy the new modular architecture
cd src
./deployment/deploy.sh
```

**Initialize Gmail Monitoring:**
```bash
# Setup Gmail API push notifications (expires every 7 days)
python start_watch.py
```

**Setup Knowledge Base:**
```bash
# Upload documents and initialize RAG corpus
python upload_rag_documents.py
python initialize_rag.py
```

**Verify Deployment:**
```bash
# Test the modular architecture
python test_modules.py
```

---

## 📊 Performance Metrics

### Response Performance
- **Processing Time**: 2-15 seconds per email
- **Multi-modal Accuracy**: 95%+ query extraction success
- **Context Efficiency**: 90% reduction in prompt size via RAG optimization
- **Scalability**: Handles concurrent emails via Cloud Functions auto-scaling

### Cost Optimization
- **Token Usage**: Reduced by 88% through intelligent context limiting
- **Storage**: Efficient audio file management with automatic cleanup
- **API Calls**: Optimized retry logic reduces failed request costs

---

## 🧪 Testing & Quality Assurance

### Modular Architecture Testing

**Module Unit Tests:**
- `test_modules.py`: Independent testing of all modules
  - Security validation and sanitization functions
  - Gmail service operations (email parsing, attachment processing)
  - Configuration and client initialization
  - Each module tested in isolation with mocks

**Evaluation Framework:**
- `evaulation/rag_evaluator.py`: Comprehensive RAG system assessment
  - Context relevance, answer faithfulness, correctness metrics
  - A/B testing framework for reranker performance
  - Automated golden dataset evaluation

**Development Testing (Available but not in production):**
- Real customer scenario testing with Czech language support
- Multi-modal attachment processing validation  
- Production readiness and security compliance testing
- Performance and optimization validation testing

**Testing Benefits:**
- **Modularity**: Each component tested independently 
- **Mocking**: Gmail API, AI models, and storage can be mocked
- **Continuous Integration**: Ready for CI/CD pipeline integration
- **Evaluation-Driven**: Performance metrics guide development decisions

---

## 🔍 Demo Capabilities

This system demonstrates advanced AI capabilities suitable for enterprise customer service:

1. **Real-time Email Processing**: Immediate response to customer inquiries
2. **Multi-language Support**: Czech language customer service with cultural context
3. **Complex Query Understanding**: Handles multiple questions per email
4. **Rich Media Processing**: Audio transcription, PDF analysis, image recognition
5. **Knowledge Integration**: RAG-powered responses using company knowledge base
6. **Production Scalability**: Cloud-native architecture with auto-scaling

---

## 📝 Technical Notes

### Evaluation Results (Latest Performance Data)

**Comprehensive RAG System Assessment:**

**With Reranker Implementation:**
- **Context Relevance**: 72.9% - Focused on most relevant information
- **Answer Faithfulness**: 100% - All responses grounded in source material  
- **Answer Correctness**: 66.7% - Improved accuracy for customer queries

**Without Reranker (Baseline):**
- **Context Relevance**: 77.1% - More comprehensive context included
- **Answer Faithfulness**: 100% - Consistent grounding in sources
- **Answer Correctness**: 62.5% - Standard accuracy levels

**Performance Analysis:**
- **Trade-off Insight**: Reranking trades 4.2% context relevance for 4.2% better correctness
- **Business Value**: Better end-user experience with more accurate answers
- **Implementation**: LLM-based reranker using Gemini 2.5 Flash Lite via RagRetrievalConfig
- **Production Status**: ✅ Deployed and active

**RAG System Engineering Evolution:**

**Phase 1 - Initial Implementation (Baseline):**
- **Configuration**: Default Vertex AI RAG settings
- **Chunking**: Standard chunk_size and chunk_overlap parameters
- **Document Parser**: Default Layout parser for text extraction
- **Problem Discovered**: RAG returned 79k+ characters overwhelming AI model

**Phase 2 - Performance Testing & Analysis:**
- **Issue Identification**: Excessive context size causing response degradation
- **Root Cause**: Inefficient chunking strategy and poor text extraction
- **Engineering Approach**: Systematic testing of chunking parameters

**Phase 3 - Optimization Implementation:**
- **Chunking Strategy**: Custom 300 character chunks with 50 character overlap
- **Document Parser**: Upgraded to DocumentAI for superior text extraction
- **Context Filtering**: Intelligent limiting to top 3 most relevant results per query
- **Reranker Integration**: LLM-based ranking for improved accuracy

**Results:**
- **Context Size**: 90% reduction (79k+ → ~2k characters)
- **Response Quality**: Significantly improved accuracy and relevance
- **Engineering Validation**: A/B testing confirmed 4.2% correctness improvement

**Multi-modal Bias Removal:**
- **Issue**: System showed preference for audio processing over other attachment types
- **Fix**: Updated prompts to handle PDFs, audio, and images equally
- **Result**: Balanced processing across all supported file types

---

## 🚀 Production Roadmap

**Priority enhancements for enterprise deployment:**

### 1. Data Privacy & Security Enhancement
**Implement DLP API for PII Protection**
- **Objective**: Identify and redact personally identifiable information (PII) from email content
- **Implementation**: Google Cloud Data Loss Prevention API integration
- **Benefit**: Enhanced safety and GDPR compliance for customer data handling
- **Priority**: High - Critical for production deployment

### 2. Automated Gmail Watch Renewal  
**Create Cloud Scheduler-Triggered Function**
- **Objective**: Automate Gmail API watch subscription renewal (currently expires every 7 days)
- **Implementation**: Simple Cloud Function triggered daily by Cloud Scheduler
- **Benefit**: Eliminate manual intervention and ensure 100% uptime
- **Priority**: High - Essential for production reliability

### 3. Knowledge Base Automation Pipeline
**GCS-Triggered Knowledge Updates**
- **Objective**: Automatic RAG corpus updates when new documents added to storage
- **Implementation**: Cloud Function triggered by Google Cloud Storage bucket changes  
- **Benefit**: Real-time knowledge base updates without manual intervention
- **Priority**: Medium - Operational efficiency improvement

### 4. Enhanced Transparency & Explainability
**Direct Context Source Attribution**
- **Objective**: Provide specific source excerpts with each response for better transparency
- **Implementation**: Include exact context passages retrieved from knowledge base
- **Benefit**: Improved explainability and user trust in AI-generated responses
- **Priority**: Medium - User experience enhancement

---

## 👥 Use Cases

This demo showcases practical applications for:

- **E-commerce Customer Service**: Automated product inquiry responses
- **Technical Support**: Multi-modal troubleshooting assistance  
- **Order Management**: Return, warranty, and shipping inquiries
- **Multilingual Support**: Localized customer service in Czech
- **Enterprise Integration**: Scalable AI assistant for large organizations

---

*This demonstration project showcases modern AI capabilities for intelligent customer service automation, combining Google Cloud's powerful infrastructure with advanced language models for practical business applications.*
