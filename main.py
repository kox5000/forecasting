"""
============================================================================
AI SERVICE - CV/Document Analysis Microservice
============================================================================

A production-grade Python microservice for:
- PDF/Image file upload and processing
- OCR (Optical Character Recognition)
- NLP analysis (entity extraction, skill detection, etc.)
- Structured JSON response

This service is COMPLETELY INDEPENDENT and communicates ONLY via REST API.
NO shared code with Laravel - pure HTTP communication.

Framework: FastAPI
Author: Senior Backend Architect
Date: 2026-05-06
============================================================================
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import logging
import uuid
from datetime import datetime
import re

try:
    import pytesseract 
    from PIL import Image
    import PyPDF2  
    import io
except ImportError:
    pass


try:
    import spacy  # NLP entity extraction
except ImportError:
    spacy = None


import uvicorn
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SERVICE_NAME = "AI Document Analysis Service"
SERVICE_VERSION = "1.0.0"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
ALLOWED_FORMATS = ['.pdf', '.jpg', '.jpeg', '.png']


nlp = None
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✓ spaCy NLP model loaded")
except:
    logger.warning("⚠ spaCy model not found")



class AnalysisResult(BaseModel):
    """Response model for document analysis"""
    status: str = Field(..., description="Status of analysis")
    file_id: str = Field(..., description="Unique file identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    text_extracted: str = Field(..., description="Extracted text from document")
    entities: Dict[str, List[str]] = Field(..., description="Extracted entities")
    skills: List[str] = Field(..., description="Detected skills (if CV)")
    metadata: Dict[str, Any] = Field(..., description="File metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Analysis timestamp")
    model_version: str = Field(..., description="AI model version")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    nlp_available: bool
    timestamp: datetime



class DocumentProcessor:
    """Handles file upload and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_bytes: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            logger.info(f"✓ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            logger.error(f"✗ PDF extraction error: {str(e)}")
            raise ValueError(f"Failed to extract PDF: {str(e)}")
    
    @staticmethod
    def extract_text_from_image(file_bytes: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image)
            logger.info(f"✓ Extracted {len(text)} characters from image via OCR")
            return text
        except Exception as e:
            logger.error(f"✗ OCR extraction error: {str(e)}")
            raise ValueError(f"Failed to extract image: {str(e)}")
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension"""
        return Path(filename).suffix.lower()



class NLPAnalyzer:
    """Performs NLP analysis on extracted text"""
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract named entities (PERSON, ORG, GPE, etc)"""
        if not nlp or not text:
            return {}
        
        try:
            doc = nlp(text[:500000]) 
            
            entities = {}
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in entities:
                    entities[entity_type] = []
                if ent.text.lower() not in [e.lower() for e in entities[entity_type]]:
                    entities[entity_type].append(ent.text)
            
            logger.info(f"✓ Extracted {len(entities)} entity types")
            return entities
        except Exception as e:
            logger.error(f"✗ Entity extraction error: {str(e)}")
            return {}
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses using regex"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = list(set(re.findall(pattern, text)))
        logger.info(f"✓ Found {len(emails)} emails")
        return emails
    
    @staticmethod
    def extract_phones(text: str) -> List[str]:
        """Extract phone numbers using regex"""
        pattern = r'(\+?1?\d{9,15}|\(\d{3}\)\s?\d{3}-?\d{4})'
        phones = list(set(re.findall(pattern, text)))
        logger.info(f"✓ Found {len(phones)} phone numbers")
        return phones
    
    @staticmethod
    def extract_skills(text: str) -> List[str]:
        """Extract technical skills from CV/resume"""
        # Common technical skills
        skills_list = [
            'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'swift',
            'react', 'angular', 'vue', 'nodejs', 'django', 'flask', 'spring',
            'fastapi', 'aws', 'azure', 'docker', 'kubernetes', 'jenkins',
            'git', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
            'machine learning', 'deep learning', 'nlp', 'ai', 'data science',
            'tableau', 'power bi', 'excel', 'rest api', 'graphql', 'microservices',
            'agile', 'scrum', 'devops', 'linux', 'unix', 'windows',
            'html', 'css', 'sass', 'webpack', 'CI/CD', 'TDD',
            'xgboost', 'tensorflow', 'pytorch', 'sklearn', 'pandas', 'numpy',
            'laravel', 'symfony', 'next.js', 'typescript', 'golang'
        ]
        
        text_lower = text.lower()
        found_skills = []
        for skill in skills_list:
            if skill in text_lower and skill not in found_skills:
                found_skills.append(skill)
        
        logger.info(f"✓ Found {len(found_skills)} skills")
        return found_skills

# ============================================
# FASTAPI APPLICATION
# ============================================

app = FastAPI(
    title="AI Document Analysis Service",
    description="Microservice for CV/PDF/Image analysis with OCR and NLP",
    version=SERVICE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# CORS MIDDLEWARE
# ============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific Laravel domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        nlp_available=nlp is not None,
        timestamp=datetime.utcnow()
    )

@app.post("/analyze-cv", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_cv(file: UploadFile = File(...)):
    """
    Analyze CV/Resume document
    
    Accepts: PDF, JPG, PNG images
    Returns: Extracted text, entities, skills, contact info
    
    Args:
        file: Upload file (CV/Resume)
    
    Returns:
        AnalysisResult with extracted information
    """
    import time
    start_time = time.time()
    
    file_id = str(uuid.uuid4())
    
    try:
        # Validate file
        logger.info(f"CV analysis request: {file.filename}")
        
        file_ext = DocumentProcessor.get_file_extension(file.filename)
        if file_ext not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"File format not allowed. Allowed: {ALLOWED_FORMATS}"
            )
        
        # Read file
        file_bytes = await file.read()
        file_size = len(file_bytes)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Extract text based on file type
        if file_ext == '.pdf':
            extracted_text = DocumentProcessor.extract_text_from_pdf(file_bytes)
        else:  # image
            extracted_text = DocumentProcessor.extract_text_from_image(file_bytes)
        
        # Perform NLP analysis
        entities = NLPAnalyzer.extract_entities(extracted_text)
        entities['EMAIL'] = NLPAnalyzer.extract_emails(extracted_text)
        entities['PHONE'] = NLPAnalyzer.extract_phones(extracted_text)
        
        skills = NLPAnalyzer.extract_skills(extracted_text)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✓ CV analysis completed: {file_id}")
        
        return AnalysisResult(
            status="completed",
            file_id=file_id,
            filename=file.filename,
            file_size=file_size,
            text_extracted=extracted_text[:1000],  # Truncate for API response
            entities=entities,
            skills=skills,
            metadata={
                "content_type": file.content_type,
                "upload_time": datetime.utcnow().isoformat(),
                "file_extension": file_ext
            },
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow(),
            model_version=SERVICE_VERSION
        )
    
    except HTTPException as e:
        logger.error(f"Validation error: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"✗ CV analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.post("/analyze-document", response_model=AnalysisResult, tags=["Analysis"])
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyze any document (generic)
    Same as analyze-cv but for general documents
    """
    return await analyze_cv(file)

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with service information"""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "analyze_cv": "POST /analyze-cv",
            "analyze_document": "POST /analyze-document",
            "docs": "GET /docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# ERROR HANDLERS
# ============================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP Error {exc.status_code}: {exc.detail}")
    return {
        "status": "error",
        "message": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "status": "error",
        "message": "Internal server error",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# STARTUP/SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"🚀 {SERVICE_NAME} v{SERVICE_VERSION} starting...")
    logger.info(f"Upload directory: {UPLOAD_DIR.absolute()}")
    logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024)}MB")

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info(f"🛑 {SERVICE_NAME} shutting down...")

# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    # Production: gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
    # Development: uvicorn main:app --reload --host 127.0.0.1 --port 8001
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info",
        reload=True
    )
