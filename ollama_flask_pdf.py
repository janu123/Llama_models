from flask import Flask, request, jsonify
import fitz  # PyMuPDF
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from langchain_community.llms import Ollama
import json
import logging
import io

app = Flask(__name__)

# Initialize the LLM (Ollama model)
llm = Ollama(model="llama3")

# AWS configuration
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
AWS_REGION_NAME = ""
AWS_BUCKET_NAME = ""
AWS_PDF_DIRECTORY = ""

# Initialize S3 client
s3 = boto3.client(
    's3',
    region_name=AWS_REGION_NAME,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def download_pdf_from_s3(file_name):
    file_key = f"{AWS_PDF_DIRECTORY}/{file_name}"
    try:
        # Download PDF to a bytes buffer
        pdf_buffer = io.BytesIO()
        s3.download_fileobj(AWS_BUCKET_NAME, file_key, pdf_buffer)
        pdf_buffer.seek(0)  # Reset buffer position to the beginning
        return pdf_buffer
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Credentials error: {e}")
        raise e
    except Exception as e:
        logging.error(f"Error downloading file from S3: {e}")
        raise e

def extract_text_from_pdf(pdf_buffer):
    # Open the PDF from the byte stream
    pdf_document = fitz.open(stream=pdf_buffer, filetype='pdf')
    
    # Initialize an empty string to store extracted text
    extracted_text = ""
    
    # Loop through each page in the PDF
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        # Extract text from the page
        extracted_text += page.get_text()
    
    # Close the PDF document
    pdf_document.close()
    
    return extracted_text

def process_text_with_llama(extracted_text):
    # Prompt to extract document type and amount-related information
    prompt = "Extract document type and amount-related information from the text below. Provide the details in JSON format with 'Document Type' and 'Amounts'. Ensure the amounts include any numerical values related to money, such as total amount, overdue amount, and taxable amount."
    # Combine prompt and extracted text
    combined_input = f"{prompt}: {extracted_text}"

    # Invoke the Llama3 model
    llm_response = llm.invoke(combined_input)
    print("LLM Output: ", llm_response)

    # Assuming the response is a JSON formatted string
    extracted_data = json.loads(llm_response)

    # Convert extracted_data to a dictionary
    extracted_dict = {
        "Document Type": extracted_data.get("Document Type", "Not Provided"),
        "Amounts": extracted_data.get("Amounts", "Not Provided")
    }
    
    return extracted_dict

@app.route('/extract_info', methods=['POST'])
def submit_form():
    data = request.json
    if not data or 'file_name' not in data:
        return jsonify({'error': 'Invalid JSON'}), 400

    file_name = data['file_name']

    try:
        # Download the PDF from S3 as a byte stream
        pdf_buffer = download_pdf_from_s3(file_name)

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_buffer)
        print("Extracted Text: ", extracted_text)

        # Process text with the Llama model
        extracted_info = process_text_with_llama(extracted_text)

        return jsonify(extracted_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)

