import pdfplumber
import io
import re

def extract_text_from_file(file) -> str:
    try:
        if file is None:
            return ""
        # Handle file path (string) or file object
        if hasattr(file, 'read'):
            # It's a file object
            file_content = file.read()
            file_name = file.name.lower()
        else:
            # It's a file path (string)
            with open(file, 'rb') as f:
                file_content = f.read()
            file_name = file.lower()
        
        if file_name.endswith('.pdf'):
            return _extract_from_pdf(file_content)
        elif file_name.endswith('.txt'):
            return file_content.decode('utf-8')
        else:
            return file_content.decode('utf-8')
    except Exception as e:
        return f"Error reading file: {str(e)}"

def _extract_from_pdf(file_content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(file_content)
        text = ""
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            return "No text could be extracted from this PDF file."
        return text.strip()
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_candidate_name(resume_text: str) -> str:
    lines = resume_text.split('\n')
    # keywords that indicate it's NOT a name line
    skip_keywords = ['email', 'phone', 'tel', '@', 'linkedin', 'github', 
                     'portfolio', 'address', 'http', 'www', '|', 'resume', 
                     'cv', 'curriculum']
    
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue

        if any(keyword in line.lower() for keyword in skip_keywords):
            continue
            
        # This handles cases like "John Smith | San Jose"
        parts = line.split('|')[0].strip()
        name_for_validation = parts
        if '(' in name_for_validation and ')' in name_for_validation:
            # Extract name without parentheses content
            name_for_validation = re.sub(r'\([^)]*\)', '', parts).strip()
        
        words = name_for_validation.split()
        
        # check if it looks like a name (2-4 words)
        if 2 <= len(words) <= 4:
            # check if words are mostly alphabetic (allowing for some special chars)
            valid_words = 0
            for word in words:
                # remove common punctuation and check
                cleaned = word.replace('.', '').replace(',', '').replace('-', '').replace("'", '')
                if cleaned and cleaned.isalpha():
                    valid_words += 1
            # if most words are valid, consider it a name
            if valid_words >= len(words) * 0.6:
                # return the original line part
                return parts
    
    return "Unknown Name"