from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load the pre-trained model for semantic search
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to find the best matching answer for a question
def find_answer_from_pdf(pdf_text, question):
    sentences = pdf_text.split("\n")
    embeddings = model.encode(sentences, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Calculate cosine similarities
    scores = util.pytorch_cos_sim(question_embedding, embeddings)[0]

    # Find the highest scoring sentence
    best_idx = scores.argmax().item()
    best_answer = sentences[best_idx]
    return best_answer

@app.route('/query-pdf', methods=['POST'])
def query_pdf():
    if 'pdf' not in request.files or 'question' not in request.form:
        return jsonify({"error": "PDF file and question are required."}), 400

    pdf_file = request.files['pdf']
    question = request.form['question']

    try:
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_file)

        # Find the best answer to the question
        answer = find_answer_from_pdf(pdf_text, question)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
