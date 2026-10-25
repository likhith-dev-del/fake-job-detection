import os
import re
import string
import json
import logging
import sqlite3
import joblib
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK packages are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# Preprocessing utilities
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

# Flask application configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'fake_job_detector_secret_key_1234'
DATABASE = 'history.db'

# Load model, vectorizer, and coefficients
MODEL_PATH = "model/fake_job_model.pkl"
VECTORIZER_PATH = "model/tfidf_vectorizer.pkl"
COEFFICIENTS_PATH = "model/coefficients.json"

model = None
vectorizer = None
coefficients = {}

def load_ml_assets():
    global model, vectorizer, coefficients
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Machine Learning model and vectorizer loaded successfully.")
        else:
            logger.error("Model or vectorizer file missing. Please run model training script first.")
        
        if os.path.exists(COEFFICIENTS_PATH):
            with open(COEFFICIENTS_PATH, "r") as f:
                coefficients = json.load(f)
            logger.info("Feature coefficients loaded successfully.")
        else:
            logger.warning("Coefficients file missing. Keyword analysis might be limited.")
    except Exception as e:
        logger.exception(f"Error loading machine learning assets: {e}")

# Initialize Database
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        with get_db() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    company TEXT,
                    description TEXT NOT NULL,
                    requirements TEXT,
                    benefits TEXT,
                    prediction INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    risk_score INTEGER NOT NULL,
                    suspicious_keywords TEXT,
                    recommendations TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        logger.info("SQLite database initialized successfully.")
    except Exception as e:
        logger.exception(f"Error initializing SQLite database: {e}")

# Helper: Compute Risk Score and Explanations
def analyze_prediction(title, company, description, requirements, benefits, prediction_proba, prediction_class):
    # Base confidence score (probability of selected class)
    prob_fake = prediction_proba[1]
    confidence = max(prediction_proba) * 100
    
    # Preprocess texts
    combined_input = f"{title} {company} {description} {requirements} {benefits}"
    cleaned = clean_text(combined_input)
    words = cleaned.split()
    
    # Identify matching suspicious keywords (positive coefficients from model coefficients)
    flagged_keywords = []
    suspicious_score = 0
    
    for word in set(words):
        if word in coefficients:
            coef_val = coefficients[word]
            if coef_val > 1.0: # Significant indicator of fraud
                flagged_keywords.append({"word": word, "weight": round(coef_val, 2)})
                # Weight matches count towards suspicious scoring
                suspicious_score += coef_val
                
    # Sort flagged keywords by weight descending
    flagged_keywords = sorted(flagged_keywords, key=lambda x: x["weight"], reverse=True)
    top_flagged = [item["word"] for item in flagged_keywords[:10]]
    
    # Calculate Risk Score (0-100)
    # Combination of ML probability of fake and manual heuristics
    base_risk = prob_fake * 100
    
    # Add risk flags
    flags_triggered = []
    
    # Heuristic 1: Missing company profile (+15 risk)
    if not company or len(company.strip()) < 10:
        base_risk += 15
        flags_triggered.append("Missing or extremely short company profile.")
        
    # Heuristic 2: Too many suspicious words (+10 risk)
    if len(flagged_keywords) >= 5:
        base_risk += 15
        flags_triggered.append("High density of high-risk keywords detected.")
        
    # Heuristic 3: Over-promising benefits or requirements (+10 risk if empty requirements but high benefits)
    if (not requirements or len(requirements.strip()) < 10) and len(benefits.strip()) > 30:
        base_risk += 10
        flags_triggered.append("No requirements listed but includes attractive benefits.")
        
    risk_score = min(max(int(base_risk), 0), 100)
    
    # Category definition
    if risk_score <= 35:
        risk_level = "Low Risk"
    elif risk_score <= 70:
        risk_level = "Medium Risk"
    else:
        risk_level = "High Risk"
        
    # Recommendations
    recommendations = []
    if prediction_class == 1 or risk_score > 35:
        recommendations.append("Do NOT pay any upfront registration, training, or material fees.")
        recommendations.append("Verify the company profile independently on LinkedIn, Google, or Glassdoor.")
        recommendations.append("Check if the email domain matches the official company website (avoid Gmail/Yahoo recruiters).")
        recommendations.append("Be wary of immediate job offers without a proper technical or HR interview process.")
        recommendations.append("Avoid jobs offering high payout for very basic tasks (e.g., general data entry with no experience).")
    else:
        recommendations.append("Standard application recommended. Ensure you customize your resume for the role.")
        recommendations.append("Research the company's recent work and reviews before attending an interview.")
        recommendations.append("Keep a record of your application details and communication.")
        
    return {
        "confidence": confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "suspicious_keywords": top_flagged,
        "flags_triggered": flags_triggered,
        "recommendations": recommendations
    }

# Routes
@app.route("/")
@app.route("/dashboard")
def dashboard():
    try:
        # Load stats
        with get_db() as conn:
            total_scans = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            fake_count = conn.execute("SELECT COUNT(*) FROM history WHERE prediction = 1").fetchone()[0]
            real_count = conn.execute("SELECT COUNT(*) FROM history WHERE prediction = 0").fetchone()[0]
            avg_risk = conn.execute("SELECT AVG(risk_score) FROM history").fetchone()[0] or 0
            
            # Fetch recent history
            history = conn.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 10").fetchall()
            
        stats = {
            "total_scans": total_scans,
            "fake_count": fake_count,
            "real_count": real_count,
            "avg_risk": round(avg_risk, 1)
        }
        
        return render_template("index.html", stats=stats, history=history)
    except Exception as e:
        logger.exception(f"Error rendering dashboard: {e}")
        return "Internal Server Error", 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        title = request.form.get("title", "").strip()
        company = request.form.get("company_profile", "").strip()
        description = request.form.get("description", "").strip()
        requirements = request.form.get("requirements", "").strip()
        benefits = request.form.get("benefits", "").strip()
        
        # Validation
        if not title or not description:
            return render_template("index.html", error="Job Title and Job Description are required fields.")
            
        if model is None or vectorizer is None:
            return render_template("index.html", error="Machine learning model is not initialized. Please train the model first.")

        # Preprocessing & Vectorization
        text = f"{title} {company} {description} {requirements} {benefits}"
        text_cleaned = clean_text(text)
        vector = vectorizer.transform([text_cleaned])
        
        # Inference
        prediction = int(model.predict(vector)[0])
        probability = model.predict_proba(vector)[0].tolist()

        print("Prediction:", prediction)
        print("Probability:", probability)
        print("Classes:", model.classes_)
        
        # Explanation and Risk analysis
        analysis = analyze_prediction(
            title, company, description, requirements, benefits, probability, prediction
        )
        
        # Save to database
        with get_db() as conn:
            conn.execute('''
                INSERT INTO history (title, company, description, requirements, benefits, prediction, confidence, risk_score, suspicious_keywords, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                title, company, description, requirements, benefits, prediction,
                analysis["confidence"], analysis["risk_score"],
                ", ".join(analysis["suspicious_keywords"]),
                " | ".join(analysis["recommendations"])
            ))
            conn.commit()
            
        # Re-fetch stats and history
        with get_db() as conn:
            total_scans = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            fake_count = conn.execute("SELECT COUNT(*) FROM history WHERE prediction = 1").fetchone()[0]
            real_count = conn.execute("SELECT COUNT(*) FROM history WHERE prediction = 0").fetchone()[0]
            avg_risk = conn.execute("SELECT AVG(risk_score) FROM history").fetchone()[0] or 0
            history = conn.execute("SELECT * FROM history ORDER BY timestamp DESC LIMIT 10").fetchall()
            
        stats = {
            "total_scans": total_scans,
            "fake_count": fake_count,
            "real_count": real_count,
            "avg_risk": round(avg_risk, 1)
        }
        
        result = {
            "title": title,
            "company": company,
            "prediction": prediction,
            "confidence": round(analysis["confidence"], 2),
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "suspicious_keywords": analysis["suspicious_keywords"],
            "flags": analysis["flags_triggered"],
            "recommendations": analysis["recommendations"]
        }
        
        return render_template("index.html", stats=stats, result=result, history=history)
        
    except Exception as e:
        logger.exception(f"Error handling predict request: {e}")
        return render_template("index.html", error="An error occurred during prediction.")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json() or {}
        title = data.get("title", "").strip()
        company = data.get("company_profile", "").strip()
        description = data.get("description", "").strip()
        requirements = data.get("requirements", "").strip()
        benefits = data.get("benefits", "").strip()
        
        if not title or not description:
            return jsonify({"error": "Title and description are required fields."}), 400
            
        if model is None or vectorizer is None:
            return jsonify({"error": "ML model is not loaded."}), 500
            
        text = f"{title} {company} {description} {requirements} {benefits}"
        text_cleaned = clean_text(text)
        vector = vectorizer.transform([text_cleaned])
        
        prediction = int(model.predict(vector)[0])
        probability = model.predict_proba(vector)[0].tolist()
        
        analysis = analyze_prediction(
            title, company, description, requirements, benefits, probability, prediction
        )
        
        # Save to database
        with get_db() as conn:
            conn.execute('''
                INSERT INTO history (title, company, description, requirements, benefits, prediction, confidence, risk_score, suspicious_keywords, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                title, company, description, requirements, benefits, prediction,
                analysis["confidence"], analysis["risk_score"],
                ", ".join(analysis["suspicious_keywords"]),
                " | ".join(analysis["recommendations"])
            ))
            conn.commit()
            
        return jsonify({
            "status": "success",
            "prediction": "Fake" if prediction == 1 else "Real",
            "prediction_code": prediction,
            "confidence": round(analysis["confidence"], 2),
            "risk_score": analysis["risk_score"],
            "risk_level": analysis["risk_level"],
            "suspicious_keywords": analysis["suspicious_keywords"],
            "flags_triggered": analysis["flags_triggered"],
            "recommendations": analysis["recommendations"]
        })
        
    except Exception as e:
        logger.exception(f"API Prediction failed: {e}")
        return jsonify({"error": "An internal error occurred during prediction."}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM history ORDER BY timestamp DESC").fetchall()
            
        history_list = []
        for r in rows:
            history_list.append({
                "id": r["id"],
                "title": r["title"],
                "company": r["company"],
                "prediction": "Fake" if r["prediction"] == 1 else "Real",
                "confidence": r["confidence"],
                "risk_score": r["risk_score"],
                "suspicious_keywords": r["suspicious_keywords"].split(", ") if r["suspicious_keywords"] else [],
                "recommendations": r["recommendations"].split(" | ") if r["recommendations"] else [],
                "timestamp": r["timestamp"]
            })
            
        return jsonify({"status": "success", "history": history_list})
    except Exception as e:
        logger.exception(f"Error fetching API history: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/export", methods=["GET"])
def export_history():
    try:
        with get_db() as conn:
            df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
            
        # Map predictions to human readable strings
        df['prediction_label'] = df['prediction'].map({0: 'Real', 1: 'Fake'})
        
        export_path = "prediction_history_export.csv"
        df.to_csv(export_path, index=False)
        
        return send_file(
            export_path,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"fake_job_prediction_history_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        )
    except Exception as e:
        logger.exception(f"Error exporting history: {e}")
        return "Internal Server Error during export", 500

# Error Handlers
@app.errorhandler(404)
def not_found_error(error):
    inline_html = "<html><body style='font-family:sans-serif; text-align:center; padding:50px;'><h1>404 Not Found</h1><p>The requested page could not be found.</p><a href='/'>Back to Dashboard</a></body></html>"
    return inline_html, 404

@app.errorhandler(500)
def internal_error(error):
    inline_html = "<html><body style='font-family:sans-serif; text-align:center; padding:50px;'><h1>500 Internal Server Error</h1><p>An internal server error occurred.</p><a href='/'>Back to Dashboard</a></body></html>"
    return inline_html, 500

# Initialization before running
init_db()
load_ml_assets()

if __name__ == "__main__":
    app.run(debug=True)