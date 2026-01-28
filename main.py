
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from typing import List, Optional, Tuple
import time
import os

app = FastAPI(title="Bug Tracker AI Service", version="2.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (lazy-loaded on first request)
embedding_model = None
severity_pipeline = None
tag_pipeline = None
_models_loading = False
_models_loaded = False

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEVERITY_LABELS = ["Low", "Medium", "High", "Critical"]
TAG_CATEGORIES = ["UI", "Backend", "Database", "Authentication", "Performance", "Security", "API", "AI", "Other"]

# Model versioning (for production tracking)
MODEL_VERSION = "severity-v1.0"
TAG_MODEL_VERSION = "tags-v1.0"

# Confidence threshold (below this, prediction is uncertain)
CONFIDENCE_THRESHOLD = 0.6

# Request/Response models
class BugRequest(BaseModel):
    text: str
    task: str = "predict_severity"

class ComprehensiveAnalysisRequest(BaseModel):
    text: str

class ComprehensiveAnalysisResponse(BaseModel):
    severity: str
    priority: str
    tags: List[str]
    summary: str
    embedding: List[float]
    confidence: Optional[float] = None
    modelVersion: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embedding: List[float]

def ensure_models_loaded():
    """Lazy-load models on first request - prevents blocking startup"""
    global embedding_model, severity_pipeline, tag_pipeline, _models_loading, _models_loaded
    
    # Already loaded
    if _models_loaded:
        return
    
    # Another request is already loading models, wait for it
    if _models_loading:
        # Wait up to 5 minutes for models to load
        wait_start = time.time()
        while _models_loading and (time.time() - wait_start) < 300:
            time.sleep(0.5)
        if _models_loaded:
            return
    
    # Start loading models
    _models_loading = True
    try:
        print("🚀 Loading AI models (lazy initialization)...")
        print("⏳ This may take 2-5 minutes on first request...")
        start_time = time.time()
        
        # Load embedding model (for duplicate detection)
        try:
            print("   📦 Loading embedding model...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"   ✅ Loaded embedding model: {EMBEDDING_MODEL_NAME}")
        except Exception as e:
            print(f"   ❌ Failed to load embedding model: {e}")
            embedding_model = None
        
        # Train ML models with proper sentence-based data
        print("   🧠 Training severity classifier...")
        train_severity_model()
        
        print("   🏷️  Training tag classifier...")
        train_tag_model()
        
        elapsed = time.time() - start_time
        print(f"✅ All models loaded in {elapsed:.2f}s - Ready for inference!")
        _models_loaded = True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise
    finally:
        _models_loading = False

def get_training_data():

    # Severity training data (real bug descriptions)
    severity_training = [
        # Critical severity
        ("Application crashes when user submits login form causing data loss", "Critical"),
        ("Database corruption detected all user data is lost", "Critical"),
        ("Security breach unauthorized access to admin panel", "Critical"),
        ("Complete system failure server is down", "Critical"),
        ("Payment processing fails and charges users incorrectly", "Critical"),
        ("User accounts are being deleted automatically", "Critical"),
        
        # High severity
        ("Login page not working users cannot access the application", "High"),
        ("API endpoint returning 500 error for all requests", "High"),
        ("Payment gateway integration is broken", "High"),
        ("Major feature not working as expected", "High"),
        ("User authentication is failing for all users", "High"),
        ("Critical workflow is broken and blocking users", "High"),
        
        # Medium severity
        ("Button not clicking properly on mobile devices", "Medium"),
        ("Page loads slowly takes more than 5 seconds", "Medium"),
        ("Form validation error message is unclear", "Medium"),
        ("Minor display issue on dashboard", "Medium"),
        ("Search functionality returns incorrect results sometimes", "Medium"),
        ("Navigation menu has alignment issues", "Medium"),
        
        # Low severity
        ("Typo in error message text", "Low"),
        ("Color scheme suggestion for better UX", "Low"),
        ("Minor UI improvement request", "Low"),
        ("Cosmetic issue with button styling", "Low"),
        ("Text alignment could be improved", "Low"),
        ("Suggestion for better error messages", "Low"),
    ]
    
    # Tag training data (real bug descriptions)
    tag_training = [
        # UI tags
        ("Button color issue on homepage", ["UI"]),
        ("Layout problem on mobile view", ["UI"]),
        ("Responsive design breaks on tablet", ["UI"]),
        ("Frontend component not rendering correctly", ["UI"]),
        ("CSS styling issue with navigation", ["UI"]),
        ("User interface element is misaligned", ["UI"]),
        ("Login page ui is not correct", ["UI", "Authentication"]),
        ("Login page design is broken", ["UI", "Authentication"]),
        ("Login form layout issue", ["UI", "Authentication"]),
        ("Login button not visible", ["UI", "Authentication"]),
        ("Login page styling problem", ["UI", "Authentication"]),
        ("UI element not displaying properly", ["UI"]),
        ("Page layout is broken", ["UI"]),
        ("Frontend page not loading correctly", ["UI"]),
        ("User interface is not working", ["UI"]),
        ("Page design issue", ["UI"]),
        ("Visual bug on page", ["UI"]),
        ("Styling problem with component", ["UI"]),
        
        # Backend tags
        ("Server error when processing requests", ["Backend"]),
        ("API endpoint not responding correctly", ["Backend", "API"]),
        ("Business logic error in calculation", ["Backend"]),
        ("Backend service is timing out", ["Backend"]),
        ("Server-side validation is failing", ["Backend"]),
        ("Application logic has a bug", ["Backend"]),
        ("Backend api is not calling", ["Backend", "API"]),
        ("Backend service not responding", ["Backend"]),
        ("Backend endpoint failing", ["Backend", "API"]),
        ("Server not processing requests", ["Backend"]),
        ("Backend logic error", ["Backend"]),
        ("Backend service down", ["Backend"]),
        ("Backend code has bug", ["Backend"]),
        ("Server-side issue", ["Backend"]),
        
        # Database tags
        ("Database query is taking too long", ["Database", "Performance"]),
        ("SQL error when fetching user data", ["Database"]),
        ("Data retrieval is slow", ["Database", "Performance"]),
        ("Database connection is failing", ["Database"]),
        ("Data persistence issue", ["Database"]),
        ("Query optimization needed", ["Database", "Performance"]),
        
        # Authentication tags
        ("Login failed for valid users", ["Authentication"]),
        ("Authentication error when accessing dashboard", ["Authentication"]),
        ("Session expired too quickly", ["Authentication"]),
        ("Password reset functionality broken", ["Authentication"]),
        ("User credentials not working", ["Authentication"]),
        ("Access control issue", ["Authentication", "Security"]),
        ("Login not working", ["Authentication"]),
        ("Cannot login to system", ["Authentication"]),
        ("Login page not working", ["Authentication", "UI"]),
        ("Authentication system broken", ["Authentication"]),
        ("User login failing", ["Authentication"]),
        ("Login credentials invalid", ["Authentication"]),
        ("Sign in not working", ["Authentication"]),
        ("Login form submission error", ["Authentication"]),
        
        # Performance tags
        ("Slow loading time on dashboard", ["Performance"]),
        ("Response time is too high", ["Performance"]),
        ("Performance issue with data processing", ["Performance"]),
        ("Optimization needed for faster queries", ["Performance"]),
        ("Bottleneck in API calls", ["Performance", "API"]),
        ("Timeout error on slow connections", ["Performance"]),
        
        # Security tags
        ("Security vulnerability in authentication", ["Security", "Authentication"]),
        ("XSS attack possible in user input", ["Security"]),
        ("SQL injection vulnerability detected", ["Security", "Database"]),
        ("Unauthorized access to admin functions", ["Security"]),
        ("Data breach risk identified", ["Security"]),
        ("Encryption issue with sensitive data", ["Security"]),
        
        # API tags
        ("REST API endpoint returning errors", ["API"]),
        ("HTTP request failing", ["API"]),
        ("API integration broken", ["API"]),
        ("Webhook not receiving data", ["API"]),
        ("Service call timeout", ["API", "Performance"]),
        ("API call not working", ["API"]),
        ("API request failing", ["API"]),
        ("API endpoint error", ["API", "Backend"]),
        ("API service not responding", ["API", "Backend"]),
        ("API call timeout", ["API", "Performance"]),
        ("HTTP request error", ["API"]),
        ("API integration issue", ["API"]),
        
        # AI tags
        ("AI model prediction is incorrect", ["AI"]),
        ("Machine learning model not working", ["AI"]),
        ("AI service returning wrong results", ["AI"]),
        ("ML model accuracy is low", ["AI"]),
        ("AI prediction failed", ["AI"]),
        ("Neural network error", ["AI"]),
        ("AI analysis is slow", ["AI", "Performance"]),
        ("ML training process crashed", ["AI"]),
        ("AI embedding generation failed", ["AI"]),
        ("Machine learning pipeline broken", ["AI"]),
        ("AI recommendation system not working", ["AI"]),
        ("Natural language processing error", ["AI"]),
        ("AI classification is wrong", ["AI"]),
        ("Deep learning model timeout", ["AI", "Performance"]),
        
        # Other
        ("General issue needs investigation", ["Other"]),
        ("Miscellaneous bug report", ["Other"]),
    ]
    
    return severity_training, tag_training

def train_severity_model():
   
    global severity_pipeline
    
    severity_training, _ = get_training_data()
    
    # Extract texts and labels
    texts = [item[0] for item in severity_training]
    labels = [item[1] for item in severity_training]
    
    # Expand training data with variations (data augmentation)
    expanded_texts = texts * 2  # Double the dataset
    expanded_labels = labels * 2
    
    try:
        # Create ML pipeline: TF-IDF → Logistic Regression
        severity_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),  # Unigrams and bigrams
                stop_words='english',
                max_features=5000,
                min_df=1,
                max_df=0.95
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs'
            ))
        ])
        
        # Train the model
        severity_pipeline.fit(expanded_texts, expanded_labels)
        print(f" Severity classifier trained (v{MODEL_VERSION}) - {len(expanded_texts)} samples")
    except Exception as e:
        print(f" Severity classifier training failed: {e}")
        severity_pipeline = None

def train_tag_model():
   
    global tag_pipeline
    
    _, tag_training = get_training_data()
    
    # Extract texts and labels
    texts = [item[0] for item in tag_training]
    # Convert multi-label to binary matrix
    labels = []
    for item in tag_training:
        label_vector = [1 if tag in item[1] else 0 for tag in TAG_CATEGORIES]
        labels.append(label_vector)
    
    # Expand training data
    expanded_texts = texts * 2
    expanded_labels = labels * 2
    
    try:
        # Create ML pipeline for multi-label classification
        tag_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words='english',
                max_features=5000,
                min_df=1,
                max_df=0.95
            )),
            ("clf", OneVsRestClassifier(
                LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
            ))
        ])
        
        # Train the model
        tag_pipeline.fit(expanded_texts, expanded_labels)
        print(f" Tag classifier trained (v{TAG_MODEL_VERSION}) - {len(expanded_texts)} samples")
    except Exception as e:
        print(f" Tag classifier training failed: {e}")
        tag_pipeline = None

@app.get("/health")
async def health_check():
    """Health check endpoint - responds immediately even during model loading"""
    # Always return healthy status - models can load in background
    models_loaded = {
        "embedding": embedding_model is not None,
        "severity": severity_pipeline is not None,
        "tags": tag_pipeline is not None
    }
    
    # Determine overall status
    status = "healthy" if all(models_loaded.values()) else "loading"
    
    return {
        "status": status,
        "models_loaded": models_loaded,
        "model_versions": {
            "severity": MODEL_VERSION,
            "tags": TAG_MODEL_VERSION
        },
        "cpu_optimized": True,
        "production_ready": all(models_loaded.values())
    }

def predict_severity_with_confidence(text: str) -> Tuple[str, float]:

    if severity_pipeline is None:
        raise ValueError("Severity model not loaded")
    
    try:
        # Get prediction and probabilities
        prediction = severity_pipeline.predict([text])[0]
        probabilities = severity_pipeline.predict_proba([text])[0]
        confidence = float(max(probabilities))
        
        return prediction, confidence
    except Exception as e:
        raise ValueError(f"Severity prediction failed: {e}")

def predict_tags_with_confidence(text: str, top_n: int = 3) -> Tuple[List[str], float]:

    if tag_pipeline is None:
        raise ValueError("Tag model not loaded")
    
    try:
        # Get probabilities for all tags
        probabilities = tag_pipeline.predict_proba([text])[0]
        
        # Create list of (tag, probability) pairs
        tag_probs = [(TAG_CATEGORIES[i], float(probabilities[i])) for i in range(len(TAG_CATEGORIES))]
        
        # Sort by probability (descending)
        tag_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter tags with probability > 0.15 (increased threshold for better accuracy)
        # and take top N
        tags = []
        confidences = []
        
        for tag, prob in tag_probs:
            if prob > 0.15 and len(tags) < top_n:  # Higher threshold, limit to top_n
                # Avoid duplicate or conflicting tags
                if tag not in tags:
                    tags.append(tag)
                    confidences.append(prob)
        
        # If no tags meet threshold, use top tag anyway (but with lower confidence)
        if not tags and len(tag_probs) > 0:
            tags = [tag_probs[0][0]]
            confidences = [tag_probs[0][1]]
        
        # Calculate average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        # Ensure at least one tag
        return tags if tags else ["Other"], avg_confidence
    except Exception as e:
        raise ValueError(f"Tag prediction failed: {e}")

@app.post("/analyze", response_model=dict)
async def analyze_bug(req: BugRequest):
    # Lazy-load models on first request
    ensure_models_loaded()
    
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Bug description too short")
    
    try:
        if req.task == "predict_severity":
            severity, confidence = predict_severity_with_confidence(req.text)
            return {
                "result": severity,
                "confidence": confidence,
                "modelVersion": MODEL_VERSION,
                "needsReview": confidence < CONFIDENCE_THRESHOLD
            }
        
        elif req.task == "predict_priority":
            severity, confidence = predict_severity_with_confidence(req.text)
            # Map severity to priority
            priority_map = {
                "Critical": "High",
                "High": "High",
                "Medium": "Medium",
                "Low": "Low"
            }
            priority = priority_map.get(severity, "Medium")
            return {
                "result": priority,
                "confidence": confidence,
                "modelVersion": MODEL_VERSION
            }
        
        elif req.task == "categorize":
            tags, confidence = predict_tags_with_confidence(req.text, top_n=1)
            return {
                "result": tags[0] if tags else "Other",
                "confidence": confidence,
                "modelVersion": TAG_MODEL_VERSION
            }
        
        elif req.task == "summarize":
            # Simple summarization - take first sentence or first 100 chars
            sentences = req.text.split('.')
            summary = sentences[0].strip() if sentences else req.text[:100]
            if len(summary) < 10:
                summary = req.text[:100] + "..."
            return {"result": summary}
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")
    
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/comprehensive", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_analysis(req: ComprehensiveAnalysisRequest):
    # Lazy-load models on first request
    ensure_models_loaded()
    
    if not req.text or len(req.text.strip()) < 3:
        raise HTTPException(status_code=400, detail="Bug description too short")
    
    try:
        # Get ML predictions with confidence
        severity, severity_conf = predict_severity_with_confidence(req.text)
        tags, tag_conf = predict_tags_with_confidence(req.text, top_n=3)
        
        # Priority based on severity
        priority_map = {"Critical": "High", "High": "High", "Medium": "Medium", "Low": "Low"}
        priority = priority_map.get(severity, "Medium")
        
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
        
        # Average confidence across predictions
        avg_confidence = (severity_conf + tag_conf) / 2
        
        return ComprehensiveAnalysisResponse(
            severity=severity,
            priority=priority,
            tags=tags,
            summary=summary,
            embedding=embedding,
            confidence=avg_confidence,
            modelVersion=MODEL_VERSION
        )
    
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

@app.post("/embedding", response_model=EmbeddingResponse)
async def generate_embedding(req: ComprehensiveAnalysisRequest):
    # Lazy-load models on first request
    ensure_models_loaded()
    
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    
    try:
        embedding = embedding_model.encode(req.text, convert_to_numpy=True)
        return EmbeddingResponse(embedding=embedding.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
