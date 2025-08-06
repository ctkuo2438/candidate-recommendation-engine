import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from utils import extract_text_from_file, extract_candidate_name
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")

class CandidateRecommendationEngine:
    def __init__(self):
        # sentence embedding model, all-MiniLM-L6-v2 has 256 token limit
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.chunker = SentenceTransformersTokenTextSplitter(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            tokens_per_chunk=200,
            chunk_overlap=20
        )
        # llm for AI summaries
        self.groq_available = False
        try:
            if os.getenv("GROQ_API_KEY"):
                self.llm = ChatGroq(temperature=0.5, model="llama3-8b-8192")
                self.groq_available = True
                print("GROQ API initialized successfully")
        except Exception as e:
            print(f"GROQ initialization failed: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)
    
    def calculate_similarity(self, job_embedding: np.ndarray, resume_embeddings: np.ndarray) -> np.ndarray:
        job_embedding = job_embedding.reshape(1, -1) # (384,) -> (1, 384)
        similarities = cosine_similarity(job_embedding, resume_embeddings)[0] # get the similarity scores
        return similarities
    
    def chunk(self, resume_text: str, job_embedding: np.ndarray) -> Tuple[float, str]:
        """
        Process a resume using chunking and return best similarity score and representative text
        
        Args:
            resume_text: Full resume text, not chunked, not embedded
            job_embedding: Job description embedding
        
        Returns:
            Tuple: (best_similarity_score, representative_text_for_ai_summary)
        """
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        tokens = tokenizer.tokenize(resume_text)
        token_count = len(tokens)
        
        if token_count <= 250:  # no chunking needed
            resume_embedding = self.generate_embeddings([resume_text])
            similarity = self.calculate_similarity(job_embedding, resume_embedding)[0] # get score
            return similarity, resume_text
        
        # resume is too long, use chunking
        chunk_texts = self.chunker.split_text(resume_text)

        if not chunk_texts:
            # if chunking fails, use the first 2000 characters
            resume_embedding = self.generate_embeddings([resume_text[:2000]])  # Truncate as fallback
            similarity = self.calculate_similarity(job_embedding, resume_embedding)[0]
            return similarity, resume_text[:2000]
        
        # Generate embeddings for each chunk
        chunk_embeddings = self.generate_embeddings(chunk_texts)

        # Calculate similarity for each chunk
        chunk_similarities = self.calculate_similarity(job_embedding, chunk_embeddings)

        # use maximum similarity across chunks
        final_similarity = np.max(chunk_similarities)

        # for the llm summary, use the top chunks, get top 3 chunks by similarity
        num_top_chunks = min(3, len(chunk_texts))
        top_chunk_indices = np.argsort(chunk_similarities)[-num_top_chunks:][::-1]

        representative_chunks = []
        for idx in top_chunk_indices:
            chunk_text = chunk_texts[idx]
            representative_chunks.append(f"[Chunk {idx+1}] {chunk_text}")
        representative_text = "\n\n".join(representative_chunks)

        # return the best similarity score and representative text for llm summary
        return final_similarity, representative_text

    def generate_llm_summary(self, job_description: str, resume_text: str, candidate_name: str) -> str:
        if not self.groq_available:
            return "LLM summary requires GROQ API key"
        try:
            # prompt for llm to generate a summary
            prompt_template = PromptTemplate(
                input_variables=["job_description", "resume_text", "candidate_name"],
                template="""
                Based on the job description and {candidate_name}'s resume, 
                explain in 2-3 sentences why this candidate is a great fit for the role. 
                Focus on specific matching skills, experiences, and qualifications.
                
                Job Description:
                {job_description}
                
                Candidate Resume:
                {resume_text}
                
                Response format: Start with "{candidate_name} is an excellent fit because..." and highlight the most relevant qualifications.
                """
            )
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            summary = chain.run(
                job_description=job_description,
                resume_text=resume_text,
                candidate_name=candidate_name
            )
            return summary.strip()
        except Exception as e:
            return f"Error generating LLM summary: {str(e)}"

    def process_candidates(self, job_description: str, candidate_files: List, candidate_texts: str, top_k: int = 5) -> str:
        """process candidates and return recommendations"""
        try:
            if not job_description.strip():
                return "Please provide a job description."
            
            candidates = [] # collect candidates from files and text

            # process uploaded files when user uploads pdf files
            if candidate_files:
                files = candidate_files if isinstance(candidate_files, list) else [candidate_files]
                for file in files:
                    if file and hasattr(file, 'name'):
                        text = extract_text_from_file(file)
                        if text and text.strip() and not text.startswith("Error"):
                            candidates.append({
                                'text': text,
                                'source': f"File: {file.name}"
                            })
            
            # process text input (split by ---) when user inputs text
            if candidate_texts and candidate_texts.strip():
                for i, text in enumerate(candidate_texts.strip().split('\n---\n')):
                    if text.strip():
                        candidates.append({
                            'text': text.strip(),
                            'source': f"Text Input #{i+1}"
                        })

            if not candidates:
                return "Please provide candidate resumes either as files or text input."
            
            # generate job description embedding
            job_embedding = self.generate_embeddings([job_description])[0] # (1, 384) -> (384,)
            
            # process candidates and calculate similarities
            results = []
            for candidate in candidates:
                similarity, representative_text = self.chunk(candidate['text'], job_embedding)
                candidate_name = extract_candidate_name(candidate['text'])
                results.append({
                    'name': candidate_name,
                    'similarity': similarity,
                    'representative_text': representative_text,
                    'source': candidate['source'] # file name or text input identifier
                })
            
            # sort by similarity and get top k, reverse order for highest similarity first
            results.sort(key=lambda x: x['similarity'], reverse=True)
            top_candidates = results[:min(top_k, len(results))]

            # format output
            output = ["**TOP CANDIDATE RECOMMENDATIONS**\n"]
            for i, candidate in enumerate(top_candidates, 1): # enumerate starts from 1
                output.append(f"**{i}. {candidate['name']}**") # name of candidate
                output.append(f"**Similarity Score:** {candidate['similarity']:.3f}") # similarity score
                
                # generate summary
                if self.groq_available: # use llm for summary
                    summary = self.generate_llm_summary(
                        job_description,
                        candidate['representative_text'],
                        candidate['name']
                    )
                else: # if no llm available, use semantic similarity analysis
                    score = candidate['similarity']
                    strength = 'Excellent' if score > 0.7 else 'Good' if score > 0.5 else 'Moderate'
                    summary = f"{strength} match based on semantic similarity analysis."
                
                output.append(f"**Why this candidate is a great fit:** {summary}")
                output.append("")
            return '\n'.join(output)
            
        except Exception as e:
            return f"Error processing candidates: {str(e)}"

engine = CandidateRecommendationEngine()

def recommend_candidates(job_description, candidate_files, candidate_texts, top_k):
    return engine.process_candidates(job_description, candidate_files, candidate_texts, top_k)
