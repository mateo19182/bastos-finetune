import os
import PyPDF2

def pdf_to_txt(pdf_folder, txt_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)

    # Iterate through all files in the PDF folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            txt_path = os.path.join(txt_folder, filename[:-4] + '.txt')

            # Open the PDF file
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Extract text from each page
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text()

            # Write the extracted text to a TXT file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

            print(f"Converted {filename} to TXT")

# Example usage
pdf_folder = 'bastos/data/mises.org'
txt_folder = 'bastos/data/txt.org'
pdf_to_txt(pdf_folder, txt_folder)