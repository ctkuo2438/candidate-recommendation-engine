# Candidate Recommendation Engine

Live Demo: [https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine](https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine)

## Features
- **Semantic Similarity Matching**: Uses SentenceTransformers (all-MiniLM-L6-v2) for deep text understanding beyond keyword matching
- **Multiple Input Methods**: Support for PDF/TXT file uploads and direct text input
- **Intelligent Scoring**: Cosine similarity scoring with smart chunking for long documents
- **AI-Generated Summaries**: LLM-powered explanations using Groq's Llama3-8b model
- **Top-K Ranking**: Configurable number of top candidates (1-10)

## Tech Stack
- **Backend**: Python 3.10+
- **ML/NLP**: 
  - SentenceTransformers (all-MiniLM-L6-v2)
  - scikit-learn (cosine similarity)
  - LangChain (text chunking)
- **LLM Integration**: Groq API (llama-3.1-8b-instant)
- **Frontend**: Gradio
- **File Processing**: pdfplumber
- **Deployment**: Hugging Face Spaces

## Prerequisites
- Python 3.10 or higher
- (Optional) Groq API key for AI summaries - [Get one here](https://console.groq.com)

## Quick Start
### Option 1: Use the Live Demo
[https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine](https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine)

### Option 2: Run Locally
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/candidate-recommendation-engine.git
   cd candidate-recommendation-engine
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Or if you prefer to use uv:
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Create virtual environment and install dependencies
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```
3. **Set up environment variables** (optional, for AI summaries):
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```
4. **Run the application**:
   ```bash
   python app.py
   uv run app.py # If using uv
    ```
5. **Access the application**:
   Open your browser and go to `http://localhost:7860`

## How to Use
1. **Enter Job Description**: 
   - Paste the complete job description in the text area
2. **Add Candidate Resumes**:
   - **Option A**: Upload PDF or TXT files (multiple files supported)
   - **Option B**: Paste resume text directly (separate multiple resumes with `---`)
3. **Configure Settings**:
   - Select how many top candidates to display (1-10)
   - Click "Find Best Candidates"
4. **Review Results**:
   - View ranked candidates with similarity scores (0.0-1.0)
   - Read AI-generated summaries explaining the match (if Groq API key is configured)
   - Higher scores indicate better matches

## Configuration
### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for AI-generated summaries (optional)