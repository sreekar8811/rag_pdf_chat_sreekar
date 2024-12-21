from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
from src.pipeline.rag_pipeline import PDFRagPipeline

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize pipeline
api_key = os.getenv('OPENAI_API_KEY')
pipeline = PDFRagPipeline(openai_api_key=api_key)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process PDF
    try:
        pipeline.process_pdf(filepath)
        return jsonify({'message': f'File {filename} processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({'error': 'Query parameter is missing'}), 400

    try:
        response = pipeline.query(data['query'])
        return jsonify({'response': response}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
