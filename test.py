import asyncio
import json
import os
import uuid
import websockets
import time
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from openai import OpenAI
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import aiofiles
import wave
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech Sentiment Analysis Service")

# Configuration
ALIBABA_ACCESS_KEY_ID = os.getenv("ALIBABA_ACCESS_KEY_ID")
ALIBABA_ACCESS_KEY_SECRET = os.getenv("ALIBABA_ACCESS_KEY_SECRET")
ALIBABA_REGION = "ap-southeast-1"
ALIBABA_APPKEY = os.getenv("ALIBABA_APPKEY")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
SENTIMENT_CHECK_INTERVAL = 5  # Check sentiment every 5 seconds
TEMP_AUDIO_DIR = "temp_audio"

# Create temp directory if it doesn't exist
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Initialize OpenAI client for Qwen
qwen_client = OpenAI(
    api_key=QWEN_API_KEY, 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Active transcription sessions
active_sessions = {}

# Function to get Alibaba access token
def get_alibaba_token():
    client = AcsClient(
        ALIBABA_ACCESS_KEY_ID,
        ALIBABA_ACCESS_KEY_SECRET,
        ALIBABA_REGION
    )
    
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
    request.set_version('2019-07-17')
    request.set_action_name('CreateToken')
    
    try:
        response = client.do_action_with_exception(request)
        response_json = json.loads(response)
        return response_json.get("Token", {}).get("Id")
    except Exception as e:
        logger.error(f"Failed to get Alibaba token: {e}")
        raise HTTPException(status_code=500, detail="Failed to get access token")

# Function to analyze sentiment using LLM
async def analyze_sentiment(text):
    if not text or len(text.strip()) < 10:  # Ignore very short texts
        return {"sentiment": "neutral", "score": 0.5}
    
    try:
        completion = qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis assistant. Analyze the sentiment of the following text and respond with a JSON object containing 'sentiment' (positive, neutral, or negative) and 'score' (0 to 1 where 0 is very negative and 1 is very positive)."},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(completion.choices[0].message.content)
        return result
    except Exception as e:
        logger.error(f"Failed to analyze sentiment: {e}")
        return {"sentiment": "neutral", "score": 0.5}

# Function to send alert
async def send_alert(session_id, text, sentiment_result):
    # This is a placeholder - in a real application, you might send an email, SMS, or push notification
    logger.warning(f"ALERT for session {session_id}: Negative sentiment detected! Score: {sentiment_result['score']}")
    logger.warning(f"Text: {text}")
    
    # Update session data with alert
    if session_id in active_sessions:
        active_sessions[session_id]['alerts'].append({
            'timestamp': time.time(),
            'text': text,
            'sentiment': sentiment_result
        })
    
    # In a real application, you would implement your alert mechanism here
    # For example:
    # await send_email(os.getenv("ALERT_EMAIL"), "Negative Sentiment Alert", f"Negative sentiment detected: {text}")
    # or
    # await send_sms(os.getenv("ALERT_PHONE"), f"Negative sentiment detected: {text[:100]}...")

class SpeechTranscriptionService:
    def __init__(self, session_id):
        self.session_id = session_id
        self.current_transcription = ""
        self.last_sentiment_check_time = time.time()
        self.websocket_url = "wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1"
        self.appkey = ALIBABA_APPKEY
        self.is_running = True
        
    async def start_transcription(self, audio_path=None, websocket_client=None):
        token = get_alibaba_token()
        ws_url = f"{self.websocket_url}?token={token}"
        
        # Update session status
        active_sessions[self.session_id]['status'] = 'connecting'
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Generate unique IDs for the session
                message_id = str(uuid.uuid4()).replace('-', '')
                task_id = str(uuid.uuid4()).replace('-', '')
                
                # Send StartTranscription command
                start_cmd = {
                    "header": {
                        "message_id": message_id,
                        "task_id": task_id,
                        "namespace": "SpeechTranscriber",
                        "name": "StartTranscription",
                        "appkey": self.appkey
                    },
                    "payload": {
                        "format": "pcm",
                        "sample_rate": 16000,
                        "enable_intermediate_result": True,
                        "enable_punctuation_prediction": True,
                        "enable_inverse_text_normalization": True
                    }
                }
                
                await websocket.send(json.dumps(start_cmd))
                
                # Receive TranscriptionStarted event
                response = await websocket.recv()
                logger.info(f"Session {self.session_id} - Received: {response}")
                
                # Update session status
                active_sessions[self.session_id]['status'] = 'transcribing'
                
                # Start sending audio data if provided
                if audio_path:
                    send_task = asyncio.create_task(self.send_audio_file(websocket, audio_path))
                
                # Process receiving transcription results
                receive_task = asyncio.create_task(self.receive_transcription_results(websocket, websocket_client))
                
                # Wait for tasks to complete
                if audio_path:
                    await send_task
                    
                    # Send stop command when audio is fully sent
                    stop_cmd = {
                        "header": {
                            "message_id": message_id,
                            "task_id": task_id,
                            "namespace": "SpeechTranscriber",
                            "name": "StopTranscription",
                            "appkey": self.appkey
                        }
                    }
                    await websocket.send(json.dumps(stop_cmd))
                
                # For streaming mode, we keep the connection open
                if not audio_path and websocket_client:
                    while self.is_running:
                        await asyncio.sleep(1)
                
                await receive_task
                
        except Exception as e:
            logger.error(f"Session {self.session_id} - Error in transcription: {e}")
            active_sessions[self.session_id]['status'] = 'error'
            active_sessions[self.session_id]['error'] = str(e)
            if websocket_client:
                try:
                    await websocket_client.send_json({
                        'event': 'error',
                        'data': {'message': str(e)}
                    })
                except:
                    pass
                    
    async def send_audio_file(self, websocket, audio_path):
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                chunk_size = 1024 * 8  # 8KB chunks
                data = wav_file.readframes(chunk_size)
                
                while data and self.is_running:
                    await websocket.send(data)
                    data = wav_file.readframes(chunk_size)
                    # Small delay to prevent overwhelming the server
                    await asyncio.sleep(0.02)
                    
        except Exception as e:
            logger.error(f"Session {self.session_id} - Error sending audio file: {e}")
            raise
            
    async def send_audio_chunk(self, websocket, audio_chunk):
        try:
            if self.is_running:
                await websocket.send(audio_chunk)
                return True
            return False
        except Exception as e:
            logger.error(f"Session {self.session_id} - Error sending audio chunk: {e}")
            return False
            
    async def receive_transcription_results(self, websocket, websocket_client=None):
        while self.is_running:
            try:
                response = await websocket.recv()
                response_json = json.loads(response)
                
                event_name = response_json.get("header", {}).get("name")
                
                if event_name == "TranscriptionResultChanged":
                    result = response_json.get("payload", {}).get("result", "")
                    self.current_transcription += result + " "
                    
                    # Update session data
                    active_sessions[self.session_id]['transcription'] = self.current_transcription
                    active_sessions[self.session_id]['last_update'] = time.time()
                    
                    # Send to client if connected
                    if websocket_client:
                        await websocket_client.send_json({
                            'event': 'transcription_update',
                            'data': {
                                'text': self.current_transcription,
                                'partial': True
                            }
                        })
                    
                    logger.info(f"Session {self.session_id} - Current transcription: {self.current_transcription}")
                    
                elif event_name == "SentenceEnd":
                    result = response_json.get("payload", {}).get("result", "")
                    self.current_transcription += result + " "
                    
                    # Update session data
                    active_sessions[self.session_id]['transcription'] = self.current_transcription
                    active_sessions[self.session_id]['last_update'] = time.time()
                    
                    # Send to client if connected
                    if websocket_client:
                        await websocket_client.send_json({
                            'event': 'transcription_update',
                            'data': {
                                'text': self.current_transcription,
                                'partial': False
                            }
                        })
                    
                    logger.info(f"Session {self.session_id} - Sentence ended: {result}")
                
                elif event_name == "TranscriptionCompleted":
                    logger.info(f"Session {self.session_id} - Transcription completed")
                    active_sessions[self.session_id]['status'] = 'completed'
                    
                    # Send to client if connected
                    if websocket_client:
                        await websocket_client.send_json({
                            'event': 'transcription_completed',
                            'data': {
                                'text': self.current_transcription
                            }
                        })
                    
                    # Final sentiment check
                    sentiment_result = await analyze_sentiment(self.current_transcription)
                    active_sessions[self.session_id]['sentiment'] = sentiment_result
                    
                    if websocket_client:
                        await websocket_client.send_json({
                            'event': 'sentiment_update',
                            'data': sentiment_result
                        })
                    
                    if sentiment_result.get("sentiment") == "negative" and sentiment_result.get("score", 0.5) < 0.3:
                        await send_alert(self.session_id, self.current_transcription, sentiment_result)
                    
                    break
                
                # Check sentiment periodically
                current_time = time.time()
                if current_time - self.last_sentiment_check_time >= SENTIMENT_CHECK_INTERVAL:
                    self.last_sentiment_check_time = current_time
                    
                    # Only check if we have enough text
                    if len(self.current_transcription.strip()) > 10:
                        sentiment_result = await analyze_sentiment(self.current_transcription)
                        active_sessions[self.session_id]['sentiment'] = sentiment_result
                        
                        if websocket_client:
                            await websocket_client.send_json({
                                'event': 'sentiment_update',
                                'data': sentiment_result
                            })
                        
                        if sentiment_result.get("sentiment") == "negative" and sentiment_result.get("score", 0.5) < 0.3:
                            await send_alert(self.session_id, self.current_transcription, sentiment_result)
                        
                        logger.info(f"Session {self.session_id} - Sentiment: {sentiment_result}")
                    
            except Exception as e:
                logger.error(f"Session {self.session_id} - Error in receiving transcription: {e}")
                active_sessions[self.session_id]['status'] = 'error'
                active_sessions[self.session_id]['error'] = str(e)
                if websocket_client:
                    try:
                        await websocket_client.send_json({
                            'event': 'error',
                            'data': {'message': str(e)}
                        })
                    except:
                        pass
                break
                
    def stop(self):
        self.is_running = False

# FastAPI endpoint models
class TranscriptionFileRequest(BaseModel):
    language: str = "en"
    sentiment_threshold: float = 0.3
    check_interval: int = 5

class TranscriptionStreamRequest(BaseModel):
    language: str = "en"
    sentiment_threshold: float = 0.3
    check_interval: int = 5

class SessionResponse(BaseModel):
    session_id: str
    status: str
    created_at: float

# HTML for the demo page
HTML_DEMO = """
<!DOCTYPE html>
<html>
<head>
    <title>Speech Sentiment Analysis Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        #transcription { white-space: pre-wrap; border: 1px solid #ddd; padding: 10px; min-height: 200px; }
        #sentiment { margin-top: 10px; padding: 10px; border: 1px solid #ddd; }
        .negative { background-color: #ffdddd; }
        .neutral { background-color: #f8f8f8; }
        .positive { background-color: #ddffdd; }
        #controls { margin: 20px 0; }
        button { padding: 10px; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>Speech Sentiment Analysis Demo</h1>
    
    <div id="controls">
        <button id="startRecording">Start Recording</button>
        <button id="stopRecording" disabled>Stop Recording</button>
        <input type="file" id="audioFile" accept="audio/*" />
        <button id="uploadAudio">Upload Audio</button>
    </div>
    
    <h2>Transcription:</h2>
    <div id="transcription"></div>
    
    <h2>Sentiment:</h2>
    <div id="sentiment" class="neutral">Waiting for analysis...</div>
    
    <script>
        let websocket;
        let mediaRecorder;
        let audioChunks = [];
        let sessionId;
        
        // Setup WebSocket connection
        function connectWebSocket() {
            return new Promise((resolve, reject) => {
                fetch('/create-session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        language: 'en',
                        sentiment_threshold: 0.3,
                        check_interval: 5
                    })
                })
                .then(response => response.json())
                .then(data => {
                    sessionId = data.session_id;
                    
                    websocket = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);
                    
                    websocket.onopen = function(e) {
                        console.log('WebSocket connection established');
                        resolve();
                    };
                    
                    websocket.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        
                        if (data.event === 'transcription_update') {
                            document.getElementById('transcription').textContent = data.data.text;
                        }
                        else if (data.event === 'sentiment_update') {
                            const sentimentDiv = document.getElementById('sentiment');
                            sentimentDiv.textContent = `Sentiment: ${data.data.sentiment}, Score: ${data.data.score}`;
                            sentimentDiv.className = data.data.sentiment;
                        }
                        else if (data.event === 'error') {
                            console.error('Error:', data.data.message);
                            alert('Error: ' + data.data.message);
                        }
                    };
                    
                    websocket.onclose = function(event) {
                        console.log('WebSocket connection closed');
                    };
                    
                    websocket.onerror = function(error) {
                        console.error('WebSocket error:', error);
                        reject(error);
                    };
                })
                .catch(error => {
                    console.error('Error creating session:', error);
                    reject(error);
                });
            });
        }
        
        // Start recording
        document.getElementById('startRecording').addEventListener('click', async () => {
            try {
                await connectWebSocket();
                
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = function(event) {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        
                        // Convert to ArrayBuffer and send over WebSocket
                        event.data.arrayBuffer().then(buffer => {
                            if (websocket && websocket.readyState === WebSocket.OPEN) {
                                websocket.send(buffer);
                            }
                        });
                    }
                };
                
                mediaRecorder.onstop = function() {
                    // Clean up stream tracks
                    stream.getTracks().forEach(track => track.stop());
                };
                
                // Start recording
                mediaRecorder.start(1000); // Capture in 1-second chunks
                
                document.getElementById('startRecording').disabled = true;
                document.getElementById('stopRecording').disabled = false;
            } catch (error) {
                console.error('Error starting recording:', error);
                alert('Error starting recording: ' + error.message);
            }
        });
        
        // Stop recording
        document.getElementById('stopRecording').addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
            
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({ command: 'stop' }));
            }
            
            document.getElementById('startRecording').disabled = false;
            document.getElementById('stopRecording').disabled = true;
        });
        
        // Upload audio file
        document.getElementById('uploadAudio').addEventListener('click', async () => {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an audio file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio_file', file);
            formData.append('language', 'en');
            
            try {
                const response = await fetch('/transcribe-file', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                sessionId = data.session_id;
                
                // Poll for updates
                pollSessionUpdates();
            } catch (error) {
                console.error('Error uploading file:', error);
                alert('Error uploading file: ' + error.message);
            }
        });
        
        // Poll for updates on file upload
        function pollSessionUpdates() {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`/session/${sessionId}`);
                    const data = await response.json();
                    
                    document.getElementById('transcription').textContent = data.transcription || '';
                    
                    if (data.sentiment) {
                        const sentimentDiv = document.getElementById('sentiment');
                        sentimentDiv.textContent = `Sentiment: ${data.sentiment.sentiment}, Score: ${data.sentiment.score}`;
                        sentimentDiv.className = data.sentiment.sentiment;
                    }
                    
                    if (data.status === 'completed' || data.status === 'error') {
                        clearInterval(interval);
                    }
                } catch (error) {
                    console.error('Error polling for updates:', error);
                    clearInterval(interval);
                }
            }, 1000);
        }
    </script>
</body>
</html>
"""

# FastAPI endpoints
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return HTML_DEMO

@app.post("/create-session", response_model=SessionResponse)
async def create_session(request: TranscriptionStreamRequest):
    """Create a new streaming transcription session"""
    session_id = str(uuid.uuid4())
    
    active_sessions[session_id] = {
        'created_at': time.time(),
        'status': 'created',
        'transcription': '',
        'sentiment': None,
        'alerts': [],
        'settings': {
            'language': request.language,
            'sentiment_threshold': request.sentiment_threshold,
            'check_interval': request.check_interval
        }
    }
    
    return {
        'session_id': session_id,
        'status': 'created',
        'created_at': active_sessions[session_id]['created_at']
    }

@app.post("/transcribe-file")
async def transcribe_file(
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
    sentiment_threshold: float = Form(0.3),
    check_interval: int = Form(5)
):
    """Transcribe an uploaded audio file and analyze sentiment"""
    session_id = str(uuid.uuid4())
    
    # Save the uploaded file temporarily
    temp_path = os.path.join(TEMP_AUDIO_DIR, f"{session_id}.wav")
    async with aiofiles.open(temp_path, 'wb') as f:
        content = await audio_file.read()
        await f.write(content)
    
    # Create a new session
    active_sessions[session_id] = {
        'created_at': time.time(),
        'status': 'processing',
        'transcription': '',
        'sentiment': None,
        'alerts': [],
        'settings': {
            'language': language,