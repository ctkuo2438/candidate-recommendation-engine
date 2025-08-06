# Create Gradio interface
import gradio as gr
from main import recommend_candidates

def create_interface():
    with gr.Blocks(title="Candidate Recommendation Engine") as demo:
        gr.Markdown("""
        # Candidate Recommendation Engine
        
        **Upload job description and candidate resumes to find the best matches using AI-powered similarity analysis.**
        
        ### How to use:
        1. Enter the job description in the text area below
        2. Upload candidate resume files (PDF, TXT) OR paste resume texts
        3. Select how many top candidates you want to see
        4. Click "Find Best Candidates" to get recommendations
        
        ### Features:
        - ✅ Semantic similarity matching using SentenceTransformers
        - ✅ Support for multiple file formats (PDF, TXT)
        - ✅ Cosine similarity scoring
        - ✅ Top-K candidate ranking
        - ✅ AI-generated summaries (llama3-8b-8192)
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                job_description = gr.Textbox(
                    label="Job Description",
                    placeholder="Enter the job description here...",
                    lines=8,
                    max_lines=15
                )
                
                with gr.Row():
                    candidate_files = gr.File(
                        label="Upload Resume Files",
                        file_count="multiple",
                        file_types=[".pdf", ".txt"]
                    )
                candidate_texts = gr.Textbox(
                    label="Or Paste Resume Texts (separate multiple resumes with '---')",
                    placeholder="Paste resume text here...\n---\nSecond resume here...\n---\nThird resume here...",
                    lines=6,
                    max_lines=10
                )
                
                with gr.Row():
                    top_k = gr.Slider(
                        label="Number of Top Candidates",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    
                    submit_btn = gr.Button("Find Best Candidates", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                results = gr.Textbox(
                    label="Recommendation Results",
                    lines=25,
                    max_lines=30,
                    show_copy_button=True
                )
        
        # set up the interface
        submit_btn.click(
            fn=recommend_candidates,
            inputs=[job_description, candidate_files, candidate_texts, top_k],
            outputs=results
        )
        
        gr.Markdown("""
        ### Tips:
        - For best results, ensure job descriptions and resumes are detailed
        - The system uses semantic similarity, so it understands context beyond keyword matching
        - Higher similarity scores (closer to 1.0) indicate better matches
        - Set GROQ_API_KEY environment variable to enable AI-generated summaries
        """)
    
    return demo

demo = create_interface()
demo.launch()
