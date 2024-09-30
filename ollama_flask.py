from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR
from langchain_community.llms import Ollama
import logging
import json

app = Flask(__name__)

llm = Ollama(model="llama3")

def extract_text_from_image_base64(image_base64):
    # Decode base64 to image bytes
    image_bytes = base64.b64decode(image_base64)
    
    # Open image using PIL and convert to numpy array
    image = Image.open(BytesIO(image_bytes))
    image_np = np.array(image)
    
    # Initialize PaddleOCR with English language support
    ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv3', use_space_char=True)
    
    # Perform OCR on the image
    result = ocr.ocr(image_np, cls=True)
    
    # Extract and join recognized text
    extracted_text = " ".join([word[-1][0] for line in result for word in line])
    logging.debug(f"Extracted text: {extracted_text}")
    
    return extracted_text

@app.route('/extract_info', methods=['POST'])
def submit_form():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'Invalid JSON'}), 400

    try:
        # Extract text from the base64 image
        extracted_text = extract_text_from_image_base64(data['image'])
        print("Extracted Text: ", extracted_text)

        # Hard-coded prompt
        prompt = "parse key details properly from the string below by analysing the surrounding words especially document type, full name and date of birth and give me the output as a json. i dont want any other details.add spaces between words in both key and value pairs.i dont need any other details while extracting but only the specific words. i just need the full name and not other details"
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
            "Full Name": extracted_data.get("Full Name", "Not Provided"),
            "Date of Birth": extracted_data.get("Date of Birth", "Not Provided")
        }
        
        return jsonify(extracted_dict)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
