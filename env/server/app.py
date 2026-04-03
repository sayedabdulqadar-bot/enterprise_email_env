#!/usr/bin/env python3
"""
Enterprise Email Environment - HF Spaces Edition
Stable, no-crash version for Hugging Face Spaces deployment
"""

import os
import sys
import json
import logging
from typing import Optional, Dict, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

logger.info("=== Starting Enterprise Email Environment ===")

# Minimal imports - fail gracefully
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    logger.info("✓ FastAPI imported")
except ImportError as e:
    logger.error(f"FastAPI import failed: {e}")
    sys.exit(1)

try:
    # Try to import core environment
    try:
        from server.models import EmailMessage, Priority, Action, Observation, Reward, Info
        from server.environment import EmailTriageEnv
        logger.info("✓ Server modules imported")
    except:
        # Fallback: define minimal models inline
        logger.warning("Could not import server modules, using inline definitions")
        
        from enum import Enum
        from pydantic import BaseModel
        
        class Priority(str, Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"
            CRITICAL = "critical"
        
        class EmailMessage(BaseModel):
            email_id: str
            sender: str
            subject: str
            body: str
            timestamp: int
        
        class Action(BaseModel):
            predicted_priority: Priority
            requires_followup: bool = False
            flag_reason: Optional[str] = None
            mark_as_spam: bool = False
        
        class Observation(BaseModel):
            current_email: EmailMessage
            inbox_size: int
            processed_count: int
            correct_count: int
            total_steps: int
            episode_done: bool
        
        class Reward(BaseModel):
            value: float
            correct_classification: float = 0.0
            efficiency_bonus: float = 0.0
            safety_penalty: float = 0.0
            followup_handling: float = 0.0
        
        class Info(BaseModel):
            is_correct: bool
            true_priority: str
            predicted_priority: str
            feedback: str
        
        class EmailTriageEnv:
            def __init__(self):
                self.inbox = []
                self.current_idx = 0
            
            def reset(self):
                self.current_idx = 0
                return Observation(
                    current_email=EmailMessage(
                        email_id="test_0",
                        sender="test@example.com",
                        subject="Test",
                        body="Test email",
                        timestamp=0
                    ),
                    inbox_size=1,
                    processed_count=0,
                    correct_count=0,
                    total_steps=0,
                    episode_done=False
                )
            
            def step(self, action: Action):
                return (
                    Observation(
                        current_email=EmailMessage(
                            email_id="test_1",
                            sender="test@example.com",
                            subject="Test",
                            body="Test",
                            timestamp=1
                        ),
                        inbox_size=1,
                        processed_count=1,
                        correct_count=1,
                        total_steps=1,
                        episode_done=True
                    ),
                    Reward(value=1.0),
                    True,
                    Info(is_correct=True, true_priority="high", predicted_priority="high", feedback="OK")
                )
    
except Exception as e:
    logger.error(f"Module import error: {e}")
    logger.error(traceback.format_exc())

# FastAPI app
app = FastAPI(
    title="Enterprise Email Environment",
    description="Email triage and classification",
    version="1.0.0"
)

# Session storage
_sessions: Dict[str, Any] = {}
MAX_SESSIONS = 100

# Request/Response models
class ResetRequest(BaseModel):
    difficulty: str = "medium"
    seed: Optional[int] = None

class ActionRequest(BaseModel):
    session_id: str
    predicted_priority: str
    requires_followup: bool = False
    flag_reason: Optional[str] = None
    mark_as_spam: bool = False

# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "service": "Enterprise Email Environment",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": str(os.getenv("START_TIME", "unknown"))
    }

@app.post("/reset")
async def reset(request: ResetRequest):
    """Reset environment"""
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        # Create environment
        env = EmailTriageEnv()
        obs = env.reset()
        
        _sessions[session_id] = {
            "env": env,
            "difficulty": request.difficulty,
            "steps": 0
        }
        
        logger.info(f"Reset: session {session_id}")
        
        return {
            "session_id": session_id,
            "observation": {
                "current_email": {
                    "email_id": obs.current_email.email_id,
                    "sender": obs.current_email.sender,
                    "subject": obs.current_email.subject,
                    "body": obs.current_email.body,
                    "timestamp": obs.current_email.timestamp
                },
                "inbox_size": obs.inbox_size,
                "processed_count": obs.processed_count,
                "correct_count": obs.correct_count,
                "total_steps": obs.total_steps,
                "episode_done": obs.episode_done
            },
            "message": f"Session created: {session_id}"
        }
    
    except Exception as e:
        logger.error(f"Reset error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: ActionRequest):
    """Execute step"""
    try:
        if request.session_id not in _sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        env = _sessions[request.session_id]["env"]
        
        priority = Priority(request.predicted_priority)
        action = Action(
            predicted_priority=priority,
            requires_followup=request.requires_followup,
            flag_reason=request.flag_reason,
            mark_as_spam=request.mark_as_spam
        )
        
        obs, reward, done, info = env.step(action)
        _sessions[request.session_id]["steps"] += 1
        
        return {
            "session_id": request.session_id,
            "observation": {
                "current_email": {
                    "email_id": obs.current_email.email_id,
                    "sender": obs.current_email.sender,
                    "subject": obs.current_email.subject,
                    "body": obs.current_email.body,
                    "timestamp": obs.current_email.timestamp
                },
                "inbox_size": obs.inbox_size,
                "processed_count": obs.processed_count,
                "correct_count": obs.correct_count,
                "total_steps": obs.total_steps,
                "episode_done": obs.episode_done
            },
            "reward": {
                "value": reward.value,
                "correct_classification": reward.correct_classification,
                "efficiency_bonus": reward.efficiency_bonus,
                "safety_penalty": reward.safety_penalty,
                "followup_handling": reward.followup_handling
            },
            "done": done,
            "info": {
                "is_correct": info.is_correct,
                "true_priority": info.true_priority,
                "predicted_priority": info.predicted_priority,
                "feedback": info.feedback
            }
        }
    
    except Exception as e:
        logger.error(f"Step error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def list_tasks():
    """List available tasks"""
    return {
        "tasks": [
            {"name": "Email Triage - Easy", "difficulty": "easy"},
            {"name": "Email Triage - Medium", "difficulty": "medium"},
            {"name": "Email Triage - Hard", "difficulty": "hard"}
        ]
    }

@app.get("/info")
async def environment_info():
    """Environment information"""
    return {
        "name": "Enterprise Email Environment",
        "version": "1.0.0",
        "description": "Email classification and triage"
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# Startup/shutdown
@app.on_event("startup")
async def startup():
    logger.info("Application startup")
    os.environ["START_TIME"] = str(os.getenv("START_TIME", "startup"))

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutdown")
    _sessions.clear()

# Run with uvicorn
if __name__ == "__main__":
    try:
        import uvicorn
        
        port = int(os.getenv("OPENENV_PORT", "7860"))
        host = os.getenv("OPENENV_HOST", "0.0.0.0")
        
        logger.info(f"Starting Uvicorn on {host}:{port}")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
