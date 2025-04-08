from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
from spellchecker import SpellChecker
import pytesseract
from PIL import Image
from text_enhance import text_corrector_llm, text_enhancer
from enhance import image_enhance

app = Flask(__name__)


# Define the upload folder and ensure it exists
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize spell checker
spell = SpellChecker()

def correct_spelling(text):
    words = text.split()
    corrected_words = [spell.correction(word) if spell.correction(word) else word for word in words]
    return ' '.join(corrected_words)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_text')
def upload_form():
    return render_template('upload.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image with OCR
        img = Image.open(filepath)
        extracted_text = pytesseract.image_to_string(img)
        corrected_text = correct_spelling(extracted_text)
        
        return render_template('result.html', image_url=filepath, extracted_text=corrected_text)

@app.route('/text')
def text():
    return render_template('text_enhancer.html')

@app.route('/text_redirect')
def text_redirect():
    return render_template('text_redirect.html')

@app.route('/download')
def download():
    # Replace 'file_path' with the path to your hardcoded file
    file_path = 'static/uploads/enhanced_img.png'
    # Send the file for download
    return send_file(file_path, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400

    if file:
        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Generate URL to access the uploaded image
        uploaded_image_url = url_for('static', filename='uploads/' + file.filename)

        return jsonify({'uploaded_image_url': uploaded_image_url})

@app.route('/enhance', methods=['POST'])
def enhance():
    data = request.get_json()
    input_image_path = data.get('image_path')

    # Convert the URL to an absolute file path
    input_image_path = input_image_path.replace('/static/', 'static/')

    # Enhance the image
    print(input_image_path)
    enhanced_image_path = image_enhance(input_image_path)
    if enhanced_image_path == -1:
        return jsonify({'error': 'Could not enhance the image'}), 500

    # Return the URL of the enhanced image
    enhanced_image_url = url_for('static', filename='uploads/' + os.path.basename(enhanced_image_path))
    
    return jsonify({'enhanced_image_url': enhanced_image_url})


@app.route('/encrypt', methods=['GET'])
def encrypt_file():
    return render_template('encrypt.html')

@app.route('/encrypt_redirect')
def encrypt_redirect(): 
    return render_template('encrypt_redirect.html')

# Route for text correction
@app.route('/correct_text', methods=['POST'])
def correct_text():
    data = request.get_json()
    text = data.get('text', '')
    if text is not "":
        corrected_text = text_corrector_llm(text)
        return jsonify({'corrected_text': corrected_text})

# Route for text enhancement
@app.route('/enhance_text', methods=['POST'])
def enhance_text():
    data = request.get_json()
    text = data.get('text', '')
    enhanced_text = text_enhancer(text)
    return jsonify({'enhanced_text': enhanced_text})


if __name__ == '__main__':
    app.run(debug=True)
