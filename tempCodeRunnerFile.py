from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import pytesseract
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# --- Data Loading and Preprocessing ---
try:
    with open("discourse_data.json", "r", encoding="utf-8") as f:
        discourse_data = json.load(f)
    logging.info("Successfully loaded discourse_data.json")
except FileNotFoundError:
    logging.error("discourse_data.json not found. Please ensure it's in the same directory.")
    discourse_data = []
except json.JSONDecodeError:
    logging.error("Error decoding discourse_data.json. Check JSON format.")
    discourse_data = []

corpus = [entry["content"] for entry in discourse_data]
urls = [entry["url"] for entry in discourse_data]
titles = [entry["title"] for entry in discourse_data]

# Initialize and fit TF-IDF Vectorizer once
vectorizer = TfidfVectorizer()
if corpus:
    X_corpus = vectorizer.fit_transform(corpus)
    logging.info("TF-IDF vectorizer fitted successfully.")
else:
    X_corpus = None
    logging.warning("Corpus is empty. TF-IDF vectorizer not fitted.")

# --- Helper Functions ---
def extract_text_from_image(base64_str):
    """Extracts text from a base64 encoded image using Tesseract OCR."""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        text = pytesseract.image_to_string(image)
        logging.info(f"Extracted text from image: {text[:50]}...")
        return text
    except Exception as e:
        logging.error(f"Error during OCR image processing: {e}")
        return ""

def find_best_answer(question):
    """
    Finds the best matching answer from the discourse data based on cosine similarity
    of TF-IDF vectors.
    """
    if not corpus or X_corpus is None:
        logging.warning("Cannot find best answer: Corpus is empty.")
        return "No discourse data available to find an answer.", []

    try:
        q_vec = vectorizer.transform([question])
        sim_scores = cosine_similarity(q_vec, X_corpus)[0]
        top_idx = sim_scores.argmax()
        answer = corpus[top_idx]
        link = {"url": urls[top_idx], "text": titles[top_idx]}
        logging.info(f"Found best answer for question: '{question[:50]}...' with similarity score: {sim_scores[top_idx]:.2f}")
        return answer, [link]
    except Exception as e:
        logging.error(f"Error finding best answer: {e}")
        return "An internal error occurred while processing your request.", []

# --- API Routes ---
@app.route("/", methods=["GET"])
def home():
    return "Hello from Flask! The server is running."

@app.route("/api/", methods=["POST"])
@app.route("/api", methods=["POST"])
def api():
    logging.info("API endpoint /api/ was hit!")
    try:
        data = request.get_json()
        if not data:
            logging.error("No JSON data received in request.")
            return jsonify({"error": "No data provided"}), 400

        question = data.get("question", "")
        image_data = data.get("image")

        full_question_text = question.strip()
        if image_data:
            logging.info("Image data received, attempting OCR...")
            extracted_text = extract_text_from_image(image_data)
            if extracted_text:
                full_question_text = (full_question_text + " " + extracted_text).strip()
                logging.info(f"Question augmented with OCR text. New question: {full_question_text[:100]}...")
            else:
                logging.warning("No text extracted from image or error occurred during OCR.")

        if not full_question_text:
            logging.warning("Received an empty question after image processing.")
            return jsonify({"answer": "Please provide a question or an image with text.", "links": []}), 200

        answer, links = find_best_answer(full_question_text)
        response_data = {"answer": answer, "links": links}
        logging.info(f"API response prepared: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logging.exception("An unhandled error occurred in the /api/ endpoint.")
        return jsonify({"error": "An internal server error occurred.", "details": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0')
