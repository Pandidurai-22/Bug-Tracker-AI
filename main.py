"""
Fast CPU-friendly AI Service for Bug Tracker
Uses lightweight models: sentence-transformers + scikit-learn
No GPU required - runs fast on laptops!
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle
import os
from typing import List, Optional
import time

app = FastAPI(title="Bug Tracker AI Service", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded once at startup)
embedding_model = None
severity_classifier = None
tag_classifier = None
vectorizer = None

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, CPU-friendly
SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]
TAG_CATEGORIES = ["UI", "Backend", "Database", "Authentication", "Performance", "Security", "API", "Other"]

# Request/Response models
class BugRequest(BaseModel):
    text: str
    task: str = "predict_severity"

class ComprehensiveAnalysisRequest(BaseModel):
    text: str

class SimilarBugRequest(BaseModel):
    text: str
    limit: int = 5

class SimilarBugResponse(BaseModel):
    bug_id: int
    similarity_score: float
    title: str

class ComprehensiveAnalysisResponse(BaseModel):
    severity: str
    priority: str
    tags: List[str]
    summary: str
    embedding: List[float]

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class SimilarBugsResponse(BaseModel):
    similar_bugs: List[SimilarBugResponse]

@app.on_event("startup")
async def startup_event():
    """Load models on startup - happens once"""
    global embedding_model, severity_classifier, tag_classifier, vectorizer
    
    print("🚀 Loading AI models (CPU-friendly)...")
    start_time = time.time()
    
    # Load embedding model (for duplicate detection)
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        embedding_model = None
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    
    # Train lightweight classifiers (using rule-based + keyword features)
    # In production, you'd train these on real bug data
    train_severity_classifier()
    train_tag_classifier()
    
    elapsed = time.time() - start_time
    print(f"✅ All models loaded in {elapsed:.2f}s - Ready for inference!")
    print("💡 Models optimized for CPU - no GPU required!")

def train_severity_classifier():
    """Train severity classifier using keyword-based features"""
    global severity_classifier, vectorizer
    
    # Training data (keyword-based - you can expand this)
    training_texts = [
        # Critical
        "application crash data loss system down",
        "security breach unauthorized access",
        "database corruption all data lost",
        "complete system failure",
        # High
        "login not working authentication failed",
        "payment processing error",
        "api endpoint returning 500 error",
        "major feature broken",
        # Medium
        "button not clicking properly",
        "slow response time",
        "minor display issue",
        "form validation error",
        # Low
        "typo in error message",
        "color scheme suggestion",
        "minor UI improvement",
        "cosmetic issue"
    ]
    
    training_labels = [
        "Critical", "Critical", "Critical", "Critical",
        "High", "High", "High", "High",
        "Medium", "Medium", "Medium", "Medium",
        "Low", "Low", "Low", "Low"
    ]
    
    # Expand training data
    expanded_texts = training_texts * 3
    expanded_labels = training_labels * 3
    
    try:
        X = vectorizer.fit_transform(expanded_texts)
        severity_classifier = LogisticRegression(max_iter=1000, random_state=42)
        severity_classifier.fit(X, expanded_labels)
        print("✅ Severity classifier trained")
    except Exception as e:
        print(f"⚠️ Severity classifier training failed: {e}")
        severity_classifier = None

def train_tag_classifier():
    """Train tag classifier"""
    global tag_classifier
    
    training_texts = [
        # UI
        "button color issue", "layout problem", "responsive design",
        "frontend component", "user interface", "css styling",
        # Backend
        "server error", "api endpoint", "business logic",
        "backend service", "server-side", "application logic",
        # Database
        "database query", "sql error", "data retrieval",
        "database connection", "data persistence", "query optimization",
        # Authentication
        "login failed", "authentication error", "session expired",
        "password reset", "user credentials", "access control",
        # Performance
        "slow loading", "response time", "performance issue",
        "optimization needed", "bottleneck", "timeout error",
        # Security
        "security vulnerability", "xss attack", "sql injection",
        "unauthorized access", "data breach", "encryption",
        # API
        "rest api", "endpoint error", "http request",
        "api integration", "webhook", "service call",
        # Other
        "general issue", "miscellaneous", "other problem"
    ]
    
    training_labels = [
        "UI", "UI", "UI", "UI", "UI", "UI",
        "Backend", "Backend", "Backend", "Backend", "Backend", "Backend",
        "Database", "Database", "Database", "Database", "Database", "Database",
        "Authentication", "Authentication", "Authentication", "Authentication", "Authentication", "Authentication",
        "Performance", "Performance", "Performance", "Performance", "Performance", "Performance",
        "Security", "Security", "Security", "Security", "Security", "Security",
        "API", "API", "API", "API", "API", "API",
        "Other", "Other", "Other"
    ]
    
    try:
        X = vectorizer.transform(training_texts)
        tag_classifier = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        tag_classifier.fit(X, training_labels)
        print("✅ Tag classifier trained")
    except Exception as e:
        print(f"⚠️ Tag classifier training failed: {e}")
        tag_classifier = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "embedding": embedding_model is not None,
            "severity": severity_classifier is not None,
            "tags": tag_classifier is not None
        },
        "cpu_optimized": True
    }

@app.post("/analyze", response_model=dict)
async def analyze_bug(req: BugRequest):
    """
    Single task analysis - fast CPU-friendly predictions
    Tasks: predict_severity, predict_priority, categorize, summarize
    """
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Bug description too short")
    
    try:
        if req.task == "predict_severity":
            result = predict_severity(req.text)
            return {"result": result}
        
        elif req.task == "predict_priority":
            # Priority is similar to severity but slightly different logic
            severity = predict_severity(req.text)
            # Map severity to priority
            priority_map = {
                "Critical": "High",
                "High": "High",
                "Medium": "Medium",
                "Low": "Low"
            }
            result = priority_map.get(severity, "Medium")
            return {"result": result}
        
        elif req.task == "categorize":
            tags = predict_tags(req.text)
            result = tags[0] if tags else "Other"
            return {"result": result}
        
        elif req.task == "summarize":
            # Simple summarization - take first sentence or first 100 chars
            sentences = req.text.split('.')
            summary = sentences[0].strip() if sentences else req.text[:100]
            if len(summary) < 10:
                summary = req.text[:100] + "..."
            return {"result": summary}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def predict_severity(text: str) -> str:
    """Predict bug severity using TF-IDF + Logistic Regression"""
    if severity_classifier is None:
        # Fallback to rule-based
        return rule_based_severity(text)
    
    try:
        X = vectorizer.transform([text.lower()])
        prediction = severity_classifier.predict(X)[0]
        return prediction
    except:
        return rule_based_severity(text)

def rule_based_severity(text: str) -> str:
    """Rule-based fallback for severity prediction"""
    text_lower = text.lower()
    
    critical_keywords = ["crash", "data loss", "security breach", "corruption", "system down", "complete failure"]
    high_keywords = ["not working", "error", "failed", "broken", "critical", "urgent"]
    low_keywords = ["suggestion", "enhancement", "improvement", "cosmetic", "typo", "minor"]
    
    if any(kw in text_lower for kw in critical_keywords):
        return "Critical"
    elif any(kw in text_lower for kw in high_keywords):
        return "High"
    elif any(kw in text_lower for kw in low_keywords):
        return "Low"
    else:
        return "Medium"

def predict_tags(text: str, top_n: int = 3) -> List[str]:
    """Predict bug tags/categories"""
    if tag_classifier is None:
        return rule_based_tags(text)
    
    try:
        X = vectorizer.transform([text.lower()])
        probabilities = tag_classifier.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        tags = [TAG_CATEGORIES[i] for i in top_indices if probabilities[i] > 0.1]
        return tags if tags else ["Other"]
    except:
        return rule_based_tags(text)

def rule_based_tags(text: str) -> List[str]:
    """Rule-based fallback for tag prediction"""
    text_lower = text.lower()
    tags = []
    
    if any(kw in text_lower for kw in ["ui", "button", "layout", "design", "frontend", "css", "html"]):
        tags.append("UI")
    if any(kw in text_lower for kw in ["backend", "server", "api", "endpoint", "service"]):
        tags.append("Backend")
    if any(kw in text_lower for kw in ["database", "sql", "query", "db", "data"]):
        tags.append("Database")
    if any(kw in text_lower for kw in ["login", "auth", "password", "session", "credential"]):
        tags.append("Authentication")
    if any(kw in text_lower for kw in ["slow", "performance", "timeout", "optimize"]):
        tags.append("Performance")
    if any(kw in text_lower for kw in ["security", "vulnerability", "breach", "xss", "sql injection"]):
        tags.append("Security")
    
    return tags[:3] if tags else ["Other"]

@app.post("/analyze/comprehensive", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_analysis(req: ComprehensiveAnalysisRequest):
    """
    Comprehensive analysis: severity, priority, tags, summary, embedding
    All in one call - optimized for speed
    """
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Bug description too short")
    
    try:
        # Parallel predictions (all CPU-friendly)
        severity = predict_severity(req.text)
        
        # Priority based on severity
        priority_map = {"Critical": "High", "High": "High", "Medium": "Medium", "Low": "Low"}
        priority = priority_map.get(severity, "Medium")
        
        # Tags
        tags = predict_tags(req.text, top_n=3)
        
        # Summary (simple)
        sentences = req.text.split('.')
        summary = sentences[0].strip() if sentences else req.text[:150]
        if len(summary) < 10:
            summary = req.text[:150] + "..."
        
        # Embedding (for duplicate detection)
        embedding = []
        if embedding_model:
            try:
                emb = embedding_model.encode(req.text, convert_to_numpy=True)
                embedding = emb.tolist()
            except:
                embedding = []
        
        return ComprehensiveAnalysisResponse(
            severity=severity,
            priority=priority,
            tags=tags,
            summary=summary,
            embedding=embedding
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

@app.post("/embedding", response_model=EmbeddingResponse)
async def generate_embedding(req: ComprehensiveAnalysisRequest):
    """
    Generate embedding vector for duplicate bug detection
    Uses sentence-transformers (CPU-friendly)
    """
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        embedding = embedding_model.encode(req.text, convert_to_numpy=True)
        return EmbeddingResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

@app.post("/similar", response_model=SimilarBugsResponse)
async def find_similar_bugs(req: SimilarBugRequest):
    """
    Find similar bugs using cosine similarity
    Note: This endpoint expects bug_ids and embeddings from backend
    For now, returns structure - backend should handle similarity search
    """
    # This is a placeholder - actual similarity search should be done in backend
    # with bug embeddings stored in database
    return SimilarBugsResponse(similar_bugs=[])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
