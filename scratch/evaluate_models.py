import os
import re
import string
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Set seeds for reproducibility
np.random.seed(42)

# Make sure assets folder exists
assets_dir = r"c:\Users\LIKHITH\OneDrive\Desktop\fake-job-detection\static\assets"
os.makedirs(assets_dir, exist_ok=True)

# Download NLTK resources if not present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)
    # Remove emails
    text = re.sub(r"\S+@\S+", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Lemmatize and remove stop words
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

def main():
    print("Step 1: Loading dataset...")
    df = pd.read_csv("dataset/fake_job_postings.csv")
    print(f"Original dataset shape: {df.shape}")

    # Handle missing values in text fields
    text_cols = ["title", "company_profile", "description", "requirements", "benefits"]
    for col in text_cols:
        df[col] = df[col].fillna("")

    # Combine text fields to capture full context
    df["combined_text"] = (
        df["title"] + " " +
        df["company_profile"] + " " +
        df["description"] + " " +
        df["requirements"] + " " +
        df["benefits"]
    )

    print("Step 2: Cleaning text data (this may take a minute)...")
    df["clean_text"] = df["combined_text"].apply(clean_text)

    # Drop duplicates based on cleaned text and target label
    df.drop_duplicates(subset=["clean_text", "fraudulent"], inplace=True)
    print(f"Dataset shape after dropping duplicates: {df.shape}")

    X = df["clean_text"]
    y = df["fraudulent"]

    print("Step 3: Vectorizing using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vect = tfidf.fit_transform(X)
    print(f"TF-IDF shape: {X_vect.shape}")

    # Split into train/test sets with stratification to preserve class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Train fraudulent class ratio: {np.mean(y_train):.4f}")
    print(f"Test fraudulent class ratio: {np.mean(y_test):.4f}")

    # Model Dictionary for comparisons
    models = {}
    
    print("\nStep 4: Training Baseline Models...")
    
    # 1. Logistic Regression (Balanced)
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    models["Logistic Regression"] = lr_model

    # 2. Naive Bayes (Multinomial)
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    models["Naive Bayes"] = nb_model

    # 3. Random Forest (Balanced) - Base
    print("Training Base Random Forest...")
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    
    # Let's perform Randomized Search for Hyperparameter Tuning on Random Forest
    print("\nStep 5: Tuning Random Forest via RandomizedSearchCV (5-fold Cross-Validation)...")
    param_dist = {
        'n_estimators': [100, 150],
        'max_depth': [20, 30, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    rf_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_dist,
        n_iter=4,
        cv=cv,
        scoring='f1',
        random_state=42,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    
    rf_best = rf_search.best_estimator_
    models["Random Forest (Tuned)"] = rf_best
    print(f"Best RF Parameters: {rf_search.best_params_}")

    print("\nStep 6: Evaluating Models...")
    comparison_data = []

    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Get probability of the positive class (fraudulent)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        precision_fake = report["1"]["precision"]
        recall_fake = report["1"]["recall"]
        f1_fake = report["1"]["f1-score"]
        
        print(f"\n{name} Performance:")
        print(classification_report(y_test, y_pred))
        
        comparison_data.append({
            "Model": name,
            "Accuracy": acc,
            "Precision (Fake)": precision_fake,
            "Recall (Fake)": recall_fake,
            "F1-Score (Fake)": f1_fake
        })
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

    # Complete ROC Plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(assets_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.4f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(assets_dir, "pr_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Model comparison dataframe and plot
    comp_df = pd.DataFrame(comparison_data)
    print("\n--- Model Comparison Table ---")
    print(comp_df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    melted_df = pd.melt(comp_df, id_vars="Model", value_vars=["Accuracy", "Precision (Fake)", "Recall (Fake)", "F1-Score (Fake)"])
    sns.barplot(x="Model", y="value", hue="variable", data=melted_df, palette="muted")
    plt.ylim([0.4, 1.02])
    plt.title("Model Metrics Comparison")
    plt.ylabel("Score")
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(assets_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Generate Confusion Matrix for Tuned RF Model
    best_rf = models["Random Forest (Tuned)"]
    y_pred_rf = best_rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_rf)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix - Random Forest (Tuned)')
    plt.savefig(os.path.join(assets_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save final Random Forest model and Vectorizer
    print("\nSaving final Random Forest model & vectorizer...")
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_rf, "model/fake_job_model.pkl")
    joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

    # Generate coefficients map from Logistic Regression for explainability highlights
    # Positive coefficients indicate terms highly predictive of FAKE jobs.
    print("Generating coefficients.json from Logistic Regression for feature explainability highlights...")
    feature_names = tfidf.get_feature_names_out()
    coefs = lr_model.coef_[0]
    coef_dict = {word: float(coef) for word, coef in zip(feature_names, coefs)}
    
    with open("model/coefficients.json", "w") as f:
        json.dump(coef_dict, f)

    print("\nModel Training and Performance Asset Generation Completed Successfully!")

if __name__ == "__main__":
    main()
