import re
from PyPDF2 import PdfReader

def join_split_words(text):
    # Join words split across lines (pattern: word fragment + hyphen + newline + word fragment)
    joined_text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    # Join words split across lines without hyphen (common in PDFs)
    joined_text = re.sub(r'(\w+)\n(\w+)', r'\1\2', joined_text)
    return joined_text

def remove_newlines_and_extra_spaces(text):
    # Replace newlines with spaces, then remove extra spaces
    return ' '.join(text.replace('\n', ' ').split())

def extract_pdf_content(pdf_path):
    reader = PdfReader(pdf_path)
    content = ""
    
    for page in reader.pages:
        text = page.extract_text()
        
        # Join split words
        text = join_split_words(text)
        
        # Remove newlines and extra spaces
        text = remove_newlines_and_extra_spaces(text)
        
        # Remove page numbers (assuming they're at the start or end of a line)
        text = re.sub(r'^\d+|\d+$', '', text).strip()
        
        content += text + " "
    
    # Final cleaning of the entire content
    content = remove_newlines_and_extra_spaces(content)
    
    return content.strip()

def split_into_chapters(content):
    chapters = re.split(r'\s+([IVX]+)\s+', content)
    result = []
    for i in range(1, len(chapters), 2):
        chapter_title = chapters[i]
        chapter_content = chapters[i+1] if i+1 < len(chapters) else ""
        result.append(f"{chapter_title}\n{chapter_content.strip()}")
    return result

# Example usage
pdf_path = 'utils/ElDebateSobreLaInevitabilidadDelEstadoNotasCriticas.pdf'
extracted_content = extract_pdf_content(pdf_path)
chapters = split_into_chapters(extracted_content)

# Print or save each chapter
for i, chapter in enumerate(chapters, 1):
    print(f"Chapter {i}:")
    print(chapter)
    print("\n" + "="*50 + "\n")

# Optionally, save the content to a text file
with open('extracted_content.txt', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(chapters))