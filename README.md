# Context System (BAP Memory System)

A sophisticated AI memory and context management system designed for intelligent chat applications. This system implements a multi-tiered memory architecture with hybrid retrieval capabilities, combining vector similarity search, full-text search, and temporal relevance to provide contextually aware AI responses.

## Features

- **Multi-Tiered Memory Architecture**: Short-term and long-term memory with episodic, semantic, and file-based storage
- **Hybrid Retrieval**: Combines vector embeddings (pgvector), BM25 full-text search, and recency scoring for optimal context retrieval
- **PostgreSQL Integration**: Robust data persistence with pgvector extension for efficient vector operations
- **Flask API**: RESTful API for chat interactions and memory management
- **CLI Interface**: Command-line chat interface for testing and development
- **Background Jobs**: Automated episodization and instancization processes for memory lifecycle management
- **Groq LLM Integration**: Powered by Groq's fast inference for generating responses

## Architecture Overview

### Memory Hierarchy

```
┌─────────────────────────────────────────────┐
│          SHORT-TERM MEMORY                  │
│  - Working Memory (Charter, Notes, Context) │
│  - Cache Entries (L1, L2, L3)              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│          LONG-TERM MEMORY                   │
│  ┌───────────────────────────────────────┐  │
│  │ Episodic (Conversations & Summaries)  │  │
│  │  - Super Chat Messages                │  │
│  │  - Deep Dive Conversations            │  │
│  │  - Episodes (< 30 days)               │  │
│  │  - Instances (> 30 days)              │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │ Semantic Memory                       │  │
│  │  - User Persona                       │  │
│  │  - Knowledge & Entities               │  │
│  │  - Processes & Skills                 │  │
│  └───────────────────────────────────────┘  │
│  ┌───────────────────────────────────────┐  │
│  │ File Memory (User Files)              │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Project Structure & File Functionality

### Core Application Files

#### `app.py`

**Functionality**: Main Flask web application entry point.

- Defines the `/chat` POST endpoint for handling chat interactions
- Receives user messages, builds context using the retrieval system, calls the LLM, and returns responses
- Manages the core chat flow: store message → build context → generate response → store response
- Runs a development server when executed directly

#### `cli_chat.py`

**Functionality**: Command-line interface for interactive chat testing.

- Provides a terminal-based chat interface for development and testing
- Demonstrates the full context retrieval process with detailed output
- Shows retrieved episodes, scores, and metadata for each interaction
- Uses a hardcoded test user ID for simplicity

#### `requirements.txt`

**Functionality**: Python package dependencies specification.

- Lists all required packages: Flask, psycopg2-binary, sentence-transformers, rank-bm25, groq, numpy, pgvector
- Used by pip to install project dependencies

### Memory & Retrieval System

#### `context_builder.py`

**Functionality**: Orchestrates context construction for LLM calls.

- Initializes the HybridRetriever for a specific user and optional deepdive conversation
- Performs search using the user's query to find relevant past episodes
- Formats retrieved episodes into a context structure suitable for LLM consumption
- Returns a list of context messages with metadata for the chat system

#### `hybrid_retriever.py`

**Functionality**: Implements advanced hybrid search combining multiple retrieval methods.

- Loads episodes from database and builds BM25 index for full-text search
- Performs three-way hybrid scoring: vector similarity (60%), BM25 text matching (30%), recency (10%)
- Uses pgvector for efficient vector similarity search in PostgreSQL
- Ranks results by combined score and returns top-k most relevant episodes
- Supports both super chat and deep dive conversation contexts

#### `embeddings.py`

**Functionality**: Manages text-to-vector embeddings for semantic search.

- Uses sentence-transformers library with a pre-trained model
- Provides encode() method to convert text into 1536-dimensional vectors
- Compatible with OpenAI ada-002 embedding dimensions
- Handles batch encoding for efficient processing

#### `bm25_index.py`

**Functionality**: Implements BM25 full-text search indexing and querying.

- Builds an in-memory BM25 index from episode text content
- Tokenizes and indexes text using BM25 scoring algorithm
- Provides search functionality to find episodes by keyword relevance
- Returns normalized BM25 scores for hybrid ranking

### Database & Persistence

#### `db.py`

**Functionality**: Database connection management and utility functions.

- Provides get_conn() context manager for PostgreSQL connections
- Handles connection pooling and error management
- Centralizes database configuration and access patterns

#### `db_setup.py`

**Functionality**: Database initialization and schema setup.

- Creates required PostgreSQL extensions (pgvector, uuid-ossp, pg_trgm)
- Sets up database tables according to the memory system schema
- Initializes indexes and constraints for optimal performance
- Should be run once during initial setup

### Chat & Message Handling

#### `chat_service.py`

**Functionality**: Handles chat message storage and retrieval operations.

- Provides add_super_chat_message() to store user and assistant messages
- Manages message persistence in the super_chat_messages table
- Tracks message metadata like tokens used and model information
- Supports both super chat and deep dive conversation types

### Memory Lifecycle Management

#### `episodization.py`

**Functionality**: Processes raw chat messages into structured episodes.

- Runs periodically (every 6 hours) to group messages into episodes
- Creates episode records from recent messages that haven't been processed
- Extracts date ranges, message counts, and source information
- Prepares episodes for vector embedding and search indexing

#### `instancization.py`

**Functionality**: Archives old episodes into long-term instances.

- Moves episodes older than 30 days to the instances table
- Compresses and archives episode data for long-term storage
- Maintains links between original episodes and their archived instances
- Optimizes storage by removing redundant data from active episodes

#### `jobs/run_episodization.py`

**Functionality**: Background job runner for episodization process.

- Simple wrapper script to execute the episodization logic
- Can be scheduled to run periodically (e.g., via cron)
- Imports and calls the main episodization function

#### `jobs/run_instancization.py`

**Functionality**: Background job runner for instancization process.

- Wrapper script for the instancization archiving process
- Executes the logic to move old episodes to long-term storage
- Designed for automated execution in production environments

### LLM Integration

#### `llm.py`

**Functionality**: Interface to Groq's language model API.

- Initializes Groq client with API key from environment variables
- Provides call_llm() function to generate responses from message context
- Supports configurable model selection (defaults to openai/gpt-oss-120b)
- Handles API communication and response parsing

### Configuration & Initialization

#### `__init__.py`

**Functionality**: Python package initialization file.

- Marks the directory as a Python package
- Can contain package-level imports and initialization code
- Currently minimal but allows for future package configuration

#### `.env` (environment file)

**Functionality**: Environment variable configuration.

- Stores sensitive configuration like API keys and database URLs
- Contains GROQ_API_KEY and DATABASE_URL settings
- Should be added to .gitignore to prevent credential exposure

## Prerequisites

- Python 3.8+
- PostgreSQL 15+ with pgvector extension
- Groq API key

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:

   ```bash
   cd context_system
   ```

2. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL database**:

   - Install PostgreSQL 15+ and pgvector extension
   - Create a database named `bap_database` (or update connection settings)
   - Run the database setup script:
     ```bash
     python db_setup.py
     ```

4. **Configure environment variables**:
   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   DATABASE_URL=postgresql://username:password@localhost:5432/bap_database
   ```

## Database Setup

The system uses PostgreSQL with the following extensions:

- `pgvector`: For vector similarity search
- `uuid-ossp`: For UUID generation
- `pg_trgm`: For trigram similarity matching

Run the schema creation from `MEMORY_SYSTEM_SCHEMA.md` or use the provided setup scripts.

## Usage

### API Usage

Start the Flask server:

```bash
python app.py
```

Send a POST request to `/chat`:

```json
{
  "user_id": "11111111-1111-1111-1111-111111111111",
  "message": "Hello, how are you?",
  "deepdive_id": null
}
```

Response:

```json
{
  "response": "I'm doing well, thank you! How can I help you today?"
}
```

### CLI Usage

Run the command-line chat interface:

```bash
python cli_chat.py
```

This will start an interactive chat session that demonstrates context retrieval and response generation.

### Background Jobs

Run memory lifecycle management jobs:

**Episodization** (process messages into episodes):

```bash
python jobs/run_episodization.py
```

**Instancization** (archive old episodes):

```bash
python jobs/run_instancization.py
```

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key for LLM calls
- `DATABASE_URL`: PostgreSQL connection string

### Database Configuration

Vector dimensions are set to 1536 (OpenAI ada-002 compatible). Adjust in `embeddings.py` if using different models.

## API Endpoints

- `POST /chat`: Main chat endpoint
  - Body: `{"user_id": "uuid", "message": "string", "deepdive_id": "uuid"}`
  - Returns: `{"response": "string"}`

## Development

### Adding New Features

1. Implement new memory types in the database schema
2. Update retrieval logic in `hybrid_retriever.py`
3. Add API endpoints in `app.py`
4. Create background jobs in `jobs/` directory

### Testing

Run the CLI interface to test basic functionality:

```bash
python cli_chat.py
```

The system will show retrieved context and generated responses.

## Performance Considerations

- Vector search uses IVFFlat indexing with optimized probes
- BM25 provides efficient full-text search
- Recency scoring ensures temporal relevance
- Background jobs manage memory lifecycle automatically

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Built using pgvector for vector operations
- Powered by Groq for fast LLM inference
- Inspired by cognitive architectures and memory systems research
