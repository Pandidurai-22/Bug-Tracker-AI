# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional
import re

app = FastAPI(title="Bug Tracker AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face Models
T5_MODEL_NAME = "google/flan-t5-small"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Load models
print("Loading Hugging Face models...")
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_NAME)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_NAME)
t5_model.eval()

# Load embedding model for similarity search (using SentenceTransformer)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Models loaded successfully!")

class BugRequest(BaseModel):
    text: str
    task: str = "summarize"

class ComprehensiveAnalysisRequest(BaseModel):
    text: str

class EmbeddingRequest(BaseModel):
    text: str

def generate_embedding(text: str) -> List[float]:
    """Generate embedding vector for text similarity search"""
    # SentenceTransformer handles tokenization and encoding automatically
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from bug description using pattern matching"""
    entities = {
        "components": [],
        "error_types": [],
        "file_paths": [],
        "urls": []
    }
    
    # Extract file paths
    file_pattern = r'[\w/\\]+\.(java|js|py|ts|tsx|jsx|html|css|json|xml|yml|yaml|properties|sql|md|txt)'
    file_paths = re.findall(file_pattern, text, re.IGNORECASE)
    entities["file_paths"] = list(set(file_paths))
    
    # Extract URLs
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    entities["urls"] = list(set(urls))
    
    # Extract common error types
    error_keywords = {
        "NullPointerException": ["null pointer", "nullpointer", "npe"],
        "TimeoutException": ["timeout", "timed out"],
        "SQLException": ["sql error", "database error", "query failed"],
        "IOException": ["io error", "file not found", "cannot read"],
        "AuthenticationError": ["authentication failed", "login error", "unauthorized"],
        "ValidationError": ["validation failed", "invalid input", "bad request"]
    }
    
    text_lower = text.lower()
    for error_type, keywords in error_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            entities["error_types"].append(error_type)
    
    # Extract component names (simple heuristic)
    component_keywords = ["api", "database", "frontend", "backend", "ui", "auth", "payment", "email"]
    for component in component_keywords:
        if component.lower() in text_lower:
            entities["components"].append(component.capitalize())
    
    return entities

@app.on_event("startup")
async def startup_event():
    """Warm up models on startup"""
    print("Warming up models...")
    with torch.no_grad():
        # Warm up T5 model
        inputs = t5_tokenizer(
            "summarize the following bug report:\nwarm up",
            return_tensors="pt"
        )
        t5_model.generate(**inputs, max_new_tokens=10)
        
        # Warm up embedding model
        embedding_model.encode("warm up")
    print("Models warmed up!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "bug-tracker-ai"}

@app.post("/analyze")
async def analyze_bug(req: BugRequest):
    """Single task analysis endpoint"""
    try:
        # ----- Prompt selection -----
        if req.task == "summarize":
            prompt = f"Summarize this software bug in one sentence:\n{req.text}"
        elif req.task == "predict_priority":
            prompt = f"Classify the priority of this bug as Critical, High, Medium, or Low:\n{req.text}"
        elif req.task == "predict_severity":
            prompt = f"Classify the severity of this bug as Critical, Major, Normal, Minor, or Enhancement:\n{req.text}"
        elif req.task == "categorize":
            prompt = f"classify category of bug:\n{req.text}"
        else:
            prompt = req.text

        # ----- Tokenize + Generate -----
        with torch.no_grad():
            inputs = t5_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=256
            )

            outputs = t5_model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=4,
                do_sample=False,
                early_stopping=True
            )

        result = t5_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # ----- Fallback for bad outputs -----
        if not result or result in [".", ","]:
            result = req.text[:100]

        # ----- Normalize outputs -----
        if req.task == "predict_priority":
            text = result.lower()
            if "critical" in text:
                result = "CRITICAL"
            elif "high" in text:
                result = "HIGH"
            elif "low" in text:
                result = "LOW"
            else:
                result = "MEDIUM"

        elif req.task == "predict_severity":
            text = result.lower()
            if "critical" in text:
                result = "CRITICAL"
            elif "major" in text:
                result = "MAJOR"
            elif "minor" in text:
                result = "MINOR"
            elif "enhancement" in text:
                result = "ENHANCEMENT"
            else:
                result = "NORMAL"

        elif req.task == "categorize":
            categories = ["Authentication", "UI", "Database", "Performance", "Security", "API", "Integration"]
            result = min(categories, key=lambda x: abs(len(x) - len(result)))

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/comprehensive")
async def comprehensive_analysis(req: ComprehensiveAnalysisRequest):
    """Comprehensive AI analysis endpoint - returns all analyses at once"""
    try:
        text = req.text
        
        # Run all analyses in parallel where possible
        with torch.no_grad():
            # Priority prediction
            priority_prompt = f"Classify the priority of this bug as Critical, High, Medium, or Low:\n{text}"
            priority_inputs = t5_tokenizer(priority_prompt, return_tensors="pt", truncation=True, max_length=256)
            priority_outputs = t5_model.generate(**priority_inputs, max_new_tokens=20, num_beams=2, early_stopping=True)
            priority_result = t5_tokenizer.decode(priority_outputs[0], skip_special_tokens=True).strip().lower()
            
            # Severity prediction
            severity_prompt = f"Classify the severity of this bug as Critical, Major, Normal, Minor, or Enhancement:\n{text}"
            severity_inputs = t5_tokenizer(severity_prompt, return_tensors="pt", truncation=True, max_length=256)
            severity_outputs = t5_model.generate(**severity_inputs, max_new_tokens=20, num_beams=2, early_stopping=True)
            severity_result = t5_tokenizer.decode(severity_outputs[0], skip_special_tokens=True).strip().lower()
            
            # Summarization
            summary_prompt = f"Summarize this software bug in one sentence:\n{text}"
            summary_inputs = t5_tokenizer(summary_prompt, return_tensors="pt", truncation=True, max_length=256)
            summary_outputs = t5_model.generate(**summary_inputs, max_new_tokens=50, num_beams=3, early_stopping=True)
            summary_result = t5_tokenizer.decode(summary_outputs[0], skip_special_tokens=True).strip()
            
            # Generate embedding
            embedding = generate_embedding(text)
        
        # Normalize priority
        if "critical" in priority_result:
            priority = "CRITICAL"
        elif "high" in priority_result:
            priority = "HIGH"
        elif "low" in priority_result:
            priority = "LOW"
        else:
            priority = "MEDIUM"
        
        # Normalize severity
        if "critical" in severity_result:
            severity = "CRITICAL"
        elif "major" in severity_result:
            severity = "MAJOR"
        elif "minor" in severity_result:
            severity = "MINOR"
        elif "enhancement" in severity_result:
            severity = "ENHANCEMENT"
        else:
            severity = "NORMAL"
        
        # Extract entities
        entities = extract_entities(text)
        
        # Generate solution suggestions
        suggestions = generate_solution_suggestions(text)
        
        return {
            "priority": priority,
            "severity": severity,
            "summary": summary_result if summary_result else text[:100],
            "embedding": embedding,
            "entities": entities,
            "suggestions": suggestions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embedding")
async def get_embedding(req: EmbeddingRequest):
    """Generate embedding vector for similarity search"""
    try:
        embedding = generate_embedding(req.text)
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_solution_suggestions(text: str) -> List[str]:
    """Generate solution suggestions based on bug description"""
    suggestions = []
    text_lower = text.lower()
    
    # Pattern-based suggestions
    if "null pointer" in text_lower or "npe" in text_lower:
        suggestions.append("Check for null references before accessing object properties or methods.")
        suggestions.append("Add null checks in the code where the error occurs.")
        suggestions.append("Review the stack trace to identify the exact line causing the NullPointerException.")
    
    if "timeout" in text_lower or "timed out" in text_lower:
        suggestions.append("Increase the timeout threshold in the configuration.")
        suggestions.append("Optimize the database queries that might be causing the delay.")
        suggestions.append("Check network connectivity and server response times.")
    
    if "authentication" in text_lower or "login" in text_lower or "unauthorized" in text_lower:
        suggestions.append("Verify user credentials and authentication tokens.")
        suggestions.append("Check if the user has proper permissions for the requested resource.")
        suggestions.append("Review authentication middleware and session management.")
    
    if "database" in text_lower or "sql" in text_lower:
        suggestions.append("Check database connection and query syntax.")
        suggestions.append("Verify database schema matches the expected structure.")
        suggestions.append("Review database logs for detailed error information.")
    
    if "memory" in text_lower or "out of memory" in text_lower:
        suggestions.append("Check for memory leaks in the application.")
        suggestions.append("Increase JVM heap size if applicable.")
        suggestions.append("Review object creation and garbage collection patterns.")
    
    # Default suggestions if no specific ones match
    if not suggestions:
        suggestions.append("Check the application logs for more detailed error information.")
        suggestions.append("Verify that all required dependencies are properly installed and configured.")
        suggestions.append("Review recent code changes that might have introduced this issue.")
        suggestions.append("Test the functionality in a different environment to isolate the problem.")
    
    return suggestions[:5]  # Return top 5 suggestions
