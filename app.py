from flask import Flask, request, render_template
import os
from utils.pdf_parser import extract_text_from_pdf
from utils.preprocessor import clean_text
from models.scorer import get_final_score

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_file = request.files['resume']
    jd_text = request.form['job_description']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)

    results = get_final_score(resume_clean, jd_clean)

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)