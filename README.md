Here's the complete markdown code to paste into VSCode:

```markdown
# Candidate Recommendation Engine

A smart AI-powered system that matches job descriptions with candidate resumes using semantic similarity analysis and generates intelligent summaries explaining why each candidate is a great fit.

## 🚀 Live Demo
**Try it now**: [https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine](https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine)

## ✨ Features
- **🧠 Semantic Similarity Matching**: Uses SentenceTransformers (all-MiniLM-L6-v2) for deep text understanding beyond keyword matching
- **📄 Multiple Input Methods**: Support for PDF/TXT file uploads and direct text input
- **📊 Intelligent Scoring**: Cosine similarity scoring with smart chunking for long documents
- **🤖 AI-Generated Summaries**: LLM-powered explanations using Groq's Llama3-8b model (optional)
- **🏆 Top-K Ranking**: Configurable number of top candidates (1-10)
- **⚡ Fast Processing**: Efficient embedding generation and similarity computation

## 🛠️ Tech Stack
- **Backend**: Python 3.10+
- **ML/NLP**: 
  - SentenceTransformers (all-MiniLM-L6-v2)
  - scikit-learn (cosine similarity)
  - LangChain (text chunking)
- **LLM Integration**: Groq API (Llama3-8b-8192)
- **Frontend**: Gradio
- **File Processing**: pdfplumber
- **Deployment**: Hugging Face Spaces

## 📋 Prerequisites
- Python 3.10 or higher
- (Optional) Groq API key for AI summaries - [Get one here](https://console.groq.com)

## 🏃‍♂️ Quick Start

### Option 1: Use the Live Demo
Visit [https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine](https://huggingface.co/spaces/ctk2438/candidate-recommendation-engine)

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

3. **Set up environment variables** (optional, for AI summaries):
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```
   
5. **Access the application**:
   Open your browser and go to `http://localhost:8000`

## 📖 How to Use

1. **Enter Job Description**: 
   - Paste the complete job description in the text area
   - Include required skills, qualifications, and responsibilities for best results

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

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for AI-generated summaries (optional)

### Customization
- Adjust the embedding model in `main.py` (default: all-MiniLM-L6-v2)
- Modify chunk size and overlap in `CandidateRecommendationEngine.__init__`
- Change the number of top chunks used for summaries

## 📁 Project Structure
```
candidate-recommendation-engine/
├── app.py              # Gradio interface
├── main.py             # Core recommendation engine
├── utils.py            # File processing utilities
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments
- [Hugging Face](https://huggingface.co/) for hosting and SentenceTransformers
- [Groq](https://groq.com/) for LLM API
- [Gradio](https://gradio.app/) for the UI framework