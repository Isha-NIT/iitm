from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64, io, os, json, logging
from PIL import Image
import pytesseract
from dotenv import load_dotenv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ──────────────────────────────────────────────────────────────
# Load environment
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
# Corrected line: Changed "gemini-pro" to "gemini-1.5-flash"
gemini_model = genai.GenerativeModel("gemini-1.5-flash") 

# ──────────────────────────────────────────────────────────────
# Flask setup
app = Flask(__name__, template_folder='templates')
CORS(app)
logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────────────────────
# Load corpus (discourse_data.json)
try:
    with open("discourse_data.json", "r", encoding="utf-8") as f:
        discourse_data = json.load(f)
except Exception as e:
    print("⚠️ Error loading discourse data:", e)
    discourse_data = []

corpus = [entry["content"] for entry in discourse_data]
urls = [entry["url"] for entry in discourse_data]
titles = [entry["title"] for entry in discourse_data]
vectorizer = TfidfVectorizer()
X_corpus = vectorizer.fit_transform(corpus) if corpus else None

# ──────────────────────────────────────────────────────────────
# Extract text from base64 image
def extract_text_from_image(base64_str):
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        return pytesseract.image_to_string(image)
    except Exception as e:
        print("❌ OCR Error:", e)
        return ""

# ──────────────────────────────────────────────────────────────
# TF-IDF corpus match
def find_best_corpus_answer(question):
    if not corpus or X_corpus is None:
        return None, []
    try:
        q_vec = vectorizer.transform([question])
        sim_scores = cosine_similarity(q_vec, X_corpus)[0]
        top_idx = sim_scores.argmax()
        # You might want to adjust this threshold based on your data
        if sim_scores[top_idx] < 0.3: 
            return None, []
        return corpus[top_idx], [{"url": urls[top_idx], "text": titles[top_idx]}]
    except Exception as e:
        print("❌ TF-IDF error:", e)
        return None, []

# ──────────────────────────────────────────────────────────────
# Gemini fallback
def get_gemini_answer(prompt):
    try:
        print(f"📤 Sending prompt to Gemini: {prompt[:80]}...")
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("❌ Gemini error:", e)
        return "Sorry, Gemini couldn't respond."

# ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/", methods=["POST"])
def api():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        image_data = data.get("image")

        if image_data:
            extracted = extract_text_from_image(image_data)
            question = f"{question} {extracted}".strip()

        if not question:
            return jsonify({"answer": "Please provide a question or image.", "links": []})

        answer, links = find_best_corpus_answer(question)
        if not answer:
            answer = get_gemini_answer(question)
            links = []

        return jsonify({"answer": answer, "links": links})

    except Exception as e:
        print("❌ API error:", e)
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"✅ Loaded Gemini API Key (last 6): {GEMINI_API_KEY[-6:]}")
    app.run(debug=True)
