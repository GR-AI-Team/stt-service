import asyncio
import json
import os
import uuid
import websockets
import time
from fastapi import FastAPI, WebSocket, UploadFile, File, Form, HTTPException, BackgroundTasks, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import logging
from openai import OpenAI
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import aiofiles
import wave
from dotenv import load_dotenv


load_dotenv()
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech Sentiment Analysis Service")

# Configuration - Load from environment variables in production
ALIBABA_ACCESS_KEY_ID = os.getenv("ALIBABA_ACCESS_KEY_ID")
ALIBABA_ACCESS_KEY_SECRET = os.getenv("ALIBABA_ACCESS_KEY_SECRET")
ALIBABA_REGION = os.getenv("ALIBABA_REGION", "ap-southeast-1")
ALIBABA_APPKEY = os.getenv("ALIBABA_APPKEY")
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
SENTIMENT_CHECK_INTERVAL = 5  # Check sentiment every 5 seconds
SENTIMENT_THRESHOLD = 0.3  # Threshold for negative sentiment alerts (0-1)
TEMP_AUDIO_DIR = "temp_audio"

# Create temp directory if it doesn't exist
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# Initialize OpenAI client for Qwen LLM
qwen_client = OpenAI(
    api_key=QWEN_API_KEY, 
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# Store active transcription sessions
active_sessions = {}

# Function to get Alibaba access token
def get_alibaba_token():
    """Obtain an access token for Alibaba Speech Recognition service"""
    client = AcsClient(
        ALIBABA_ACCESS_KEY_ID,
        ALIBABA_ACCESS_KEY_SECRET,
        ALIBABA_REGION
    )
    
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain(f'nlsmeta.{ALIBABA_REGION}.aliyuncs.com')
    request.set_version('2019-07-17')
    request.set_action_name('CreateToken')
    
    try:
        response = client.do_action_with_exception(request)
        response_json = json.loads(response)
        return response_json.get("Token", {}).get("Id")
    except Exception as e:
        logger.error(f"Failed to get Alibaba token: {e}")
        raise HTTPException(status_code=500, detail="Failed to get access token")

# Function to analyze sentiment using Qwen LLM
async def analyze_sentiment(text):
    """Analyze sentiment of text using the Qwen LLM"""
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

# Function to send alert when negative sentiment is detected
async def send_alert(session_id, text, sentiment_result):
    """Send an alert when negative sentiment is detected"""
    logger.warning(f"ALERT for session {session_id}: Negative sentiment detected! Score: {sentiment_result['score']}")
    logger.warning(f"Text: {text}")
    
    # Update session data with alert
    if session_id in active_sessions:
        active_sessions[session_id]['alerts'].append({
            'timestamp': time.time(),
            'text': text,
            'sentiment': sentiment_result
        })
    
    # In a real application, implement your alert mechanism here:
    # Examples:
    # 1. Send email notification
    # await send_email("alerts@example.com", "Negative Sentiment Alert", f"Negative sentiment detected: {text}")
    # 
    # 2. Send SMS alert
    # await send_sms("+1234567890", f"Negative sentiment detected: {text[:100]}...")
    #
    # 3. Post to webhook/API
    # await requests.post("https://api.example.com/alerts", json={"text": text, "sentiment": sentiment_result})

class SpeechTranscriptionService:
    """Main service to handle speech transcription and sentiment analysis"""
    def __init__(self, session_id):
        self.session_id = session_id
        self.current_transcription = ""
        self.last_sentiment_check_time = time.time()
        self.websocket_url = f"wss://nls-gateway-{ALIBABA_REGION}.aliyuncs.com/ws/v1"
        self.appkey = ALIBABA_APPKEY
        self.is_running = True
        
    async def start_transcription(self, audio_path=None, websocket_client=None):
        """Start the transcription process"""
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
                logger.info(f"Session {self.session_id} - Started transcription")
                
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
        """Send audio file data to the transcription service"""
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
        """Send a single audio chunk to the transcription service"""
        try:
            if self.is_running:
                await websocket.send(audio_chunk)
                return True
            return False
        except Exception as e:
            logger.error(f"Session {self.session_id} - Error sending audio chunk: {e}")
            return False
            
    async def receive_transcription_results(self, websocket, websocket_client=None):
        """Receive and process transcription results from Alibaba service"""
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
                    
                    logger.info(f"Session {self.session_id} - Partial transcription updated")
                    
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
                    
                    logger.info(f"Session {self.session_id} - Sentence completed: {result}")
                
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
                    
                    if sentiment_result.get("sentiment") == "negative" and sentiment_result.get("score", 0.5) < SENTIMENT_THRESHOLD:
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
                        
                        if sentiment_result.get("sentiment") == "negative" and sentiment_result.get("score", 0.5) < SENTIMENT_THRESHOLD:
                            await send_alert(self.session_id, self.current_transcription, sentiment_result)
                        
                        logger.info(f"Session {self.session_id} - Sentiment analyzed: {sentiment_result['sentiment']}")
                    
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
        """Stop the transcription service"""
        self.is_running = False

# FastAPI endpoint models
class TranscriptionRequest(BaseModel):
    """Request model for transcription settings"""
    language: str = "en"
    sentiment_threshold: float = 0.3
    check_interval: int = 5

class SessionResponse(BaseModel):
    """Response model for session information"""
    session_id: str
    status: str
    created_at: float

# HTML for the demo page
HTML_DEMO = """
<!DOCTYPE html>
<html>
<head>
    <title>Speech Sentiment Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
        #transcription { white-space: pre-wrap; border: 1px solid #ddd; padding: 10px; min-height: 100px; margin-top: 10px; }
        #sentiment { margin-top: 10px; padding: 10px; border: 1px solid #ddd; }
        .negative { background-color: #ffdddd; }
        .neutral { background-color: #f8f8f8; }
        .positive { background-color: #ddffdd; }
        #controls { margin: 20px 0; }
        button { padding: 10px; margin-right: 10px; }
    </style>
</head>
<body>
    <h1>Speech Sentiment Analysis</h1>
    
    <div class="container">
        <h2>Upload Audio File</h2>
        <input type="file" id="audioFile" accept="audio/*" />
        <button id="uploadAudio">Upload & Analyze</button>
        
        <div id="uploadStatus"></div>
    </div>
    
    <div class="container" style="margin-top: 20px;">
        <h2>Recording</h2>
        <button id="startRecording">Start Recording</button>
        <button id="stopRecording" disabled>Stop Recording</button>
    </div>
    
    <div class="container" style="margin-top: 20px;">
        <h2>Results</h2>
        <div>
            <h3>Transcription:</h3>
            <div id="transcription">Waiting for transcription...</div>
        </div>
        
        <div>
            <h3>Sentiment Analysis:</h3>
            <div id="sentiment" class="neutral">Waiting for analysis...</div>
        </div>
        
        <div id="alerts" style="margin-top: 20px;">
            <h3>Alerts:</h3>
            <div id="alertsList"></div>
        </div>
    </div>
    
    <script>
        let websocket;
        let mediaRecorder;
        let audioChunks = [];
        let sessionId;
        
        // Upload audio file
        document.getElementById('uploadAudio').addEventListener('click', async () => {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an audio file first');
                return;
            }
            
            document.getElementById('uploadStatus').innerHTML = 'Uploading and processing...';
            
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
                
                document.getElementById('uploadStatus').innerHTML = 'Processing audio...';
                
                // Poll for updates
                pollSessionUpdates();
            } catch (error) {
                console.error('Error uploading file:', error);
                document.getElementById('uploadStatus').innerHTML = 'Error: ' + error.message;
            }
        });
        
        // Start recording
        document.getElementById('startRecording').addEventListener('click', async () => {
            try {
                // Create a new session
                const response = await fetch('/create-session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        language: 'en',
                        sentiment_threshold: 0.3,
                        check_interval: 5
                    })
                });
                
                const data = await response.json();
                sessionId = data.session_id;
                
                // Connect WebSocket
                websocket = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);
                
                websocket.onopen = async function(e) {
                    console.log('WebSocket connection established');
                    
                    // Get audio stream
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
                        stream.getTracks().forEach(track => track.stop());
                    };
                    
                    // Start recording
                    mediaRecorder.start(1000); // Capture in 1-second chunks
                    
                    document.getElementById('startRecording').disabled = true;
                    document.getElementById('stopRecording').disabled = false;
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
                        
                        // If negative sentiment, add to alerts
                        if (data.data.sentiment === 'negative' && data.data.score < 0.3) {
                            const alertItem = document.createElement('div');
                            alertItem.className = 'negative';
                            alertItem.style.padding = '10px';
                            alertItem.style.marginBottom = '10px';
                            alertItem.innerHTML = `<strong>Alert!</strong> Negative sentiment detected (${data.data.score})<br>Text: ${document.getElementById('transcription').textContent}`;
                            document.getElementById('alertsList').appendChild(alertItem);
                        }
                    }
                    else if (data.event === 'error') {
                        console.error('Error:', data.data.message);
                        alert('Error: ' + data.data.message);
                    }
                    else if (data.event === 'transcription_completed') {
                        document.getElementById('transcription').textContent = data.data.text;
                        document.getElementById('stopRecording').disabled = true;
                        document.getElementById('startRecording').disabled = false;
                    }
                };
                
                websocket.onclose = function(event) {
                    console.log('WebSocket connection closed');
                    document.getElementById('stopRecording').disabled = true;
                    document.getElementById('startRecording').disabled = false;
                };
                
                websocket.onerror = function(error) {
                    console.error('WebSocket error:', error);
                    alert('WebSocket error occurred');
                };
                
            } catch (error) {
                console.error('Error starting recording:', error);
                alert('Error: ' + error.message);
            }
        });
        
        // Stop recording
        document.getElementById('stopRecording').addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({ action: 'stop' }));
                }
                
                document.getElementById('stopRecording').disabled = true;
            }
        });
        
        // Poll for session updates (for file upload mode)
        async function pollSessionUpdates() {
            if (!sessionId) return;
            
            try {
                const response = await fetch(`/session/${sessionId}`);
                const data = await response.json();
                
                if (data.transcription) {
                    document.getElementById('transcription').textContent = data.transcription;
                }
                
                if (data.sentiment) {
                    const sentimentDiv = document.getElementById('sentiment');
                    sentimentDiv.textContent = `Sentiment: ${data.sentiment.sentiment}, Score: ${data.sentiment.score}`;
                    sentimentDiv.className = data.sentiment.sentiment;
                }
                
                if (data.alerts && data.alerts.length > 0) {
                    const alertsList = document.getElementById('alertsList');
                    alertsList.innerHTML = '';
                    
                    data.alerts.forEach(alert => {
                        const alertItem = document.createElement('div');
                        alertItem.className = 'negative';
                        alertItem.style.padding = '10px';
                        alertItem.style.marginBottom = '10px';
                        alertItem.innerHTML = `<strong>Alert!</strong> Negative sentiment detected (${alert.sentiment.score})<br>Text: ${alert.text}`;
                        alertsList.appendChild(alertItem);
                    });
                }
                
                if (data.status !== 'completed' && data.status !== 'error') {
                    setTimeout(pollSessionUpdates, 2000);
                } else {
                    document.getElementById('uploadStatus').innerHTML = 'Processing completed';
                }
                
            } catch (error) {
                console.error('Error polling updates:', error);
                document.getElementById('uploadStatus').innerHTML = 'Error polling updates: ' + error.message;
            }
        }
    </script>
</body>
</html>
"""

# FastAPI Routes
@app.get("/", response_class=HTMLResponse)
async def get_demo_page():
    """Serve the demo page"""
    return HTML_DEMO

@app.post("/create-session", response_model=SessionResponse)
async def create_session(request: TranscriptionRequest):
    """Create a new transcription session"""
    session_id = str(uuid.uuid4())
    
    # Initialize session data
    active_sessions[session_id] = {
        'status': 'created',
        'created_at': time.time(),
        'transcription': '',
        'sentiment': None,
        'alerts': [],
        'settings': {
            'language': request.language,
            'sentiment_threshold': request.sentiment_threshold,
            'check_interval': request.check_interval
        },
        'last_update': time.time()
    }
    
    return {
        'session_id': session_id,
        'status': 'created',
        'created_at': active_sessions[session_id]['created_at']
    }

@app.post("/transcribe-file")
async def transcribe_file(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
    language: str = Form("en"),
    sentiment_threshold: float = Form(0.3),
    check_interval: int = Form(5)
):
    """Transcribe an uploaded audio file and analyze sentiment"""
    # Create session
    session_id = str(uuid.uuid4())
    
    # Save file temporarily
    file_path = os.path.join(TEMP_AUDIO_DIR, f"{session_id}.wav")
    
    try:
        # Initialize session data
        active_sessions[session_id] = {
            'status': 'created',
            'created_at': time.time(),
            'transcription': '',
            'sentiment': None,
            'alerts': [],
            'settings': {
                'language': language,
                'sentiment_threshold': sentiment_threshold,
                'check_interval': check_interval
            },
            'file_path': file_path,
            'last_update': time.time()
        }
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await audio_file.read()
            await out_file.write(content)
        
        # Start transcription in background
        service = SpeechTranscriptionService(session_id)
        background_tasks.add_task(service.start_transcription, file_path)
        
        return {
            'session_id': session_id,
            'status': 'processing'
        }
    except Exception as e:
        logger.error(f"Error processing file upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get current session status and results"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return active_sessions[session_id]

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its resources"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Stop the session if it's running
    if 'service' in active_sessions[session_id]:
        active_sessions[session_id]['service'].stop()
    
    # Delete temporary file if it exists
    if 'file_path' in active_sessions[session_id]:
        file_path = active_sessions[session_id]['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove from active sessions
    session_data = active_sessions.pop(session_id)
    
    return {"status": "deleted", "session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time audio streaming and results"""
    
    if session_id not in active_sessions:
        # Create new session if it doesn't exist
        active_sessions[session_id] = {
            'status': 'created',
            'created_at': time.time(),
            'transcription': '',
            'sentiment': None,
            'alerts': [],
            'settings': {
                'language': 'en',
                'sentiment_threshold': SENTIMENT_THRESHOLD,
                'check_interval': SENTIMENT_CHECK_INTERVAL
            },
            'last_update': time.time()
        }
    
    await websocket.accept()
    
    # Start the service
    service = SpeechTranscriptionService(session_id)
    active_sessions[session_id]['service'] = service
    
    # Start the transcription task
    transcription_task = asyncio.create_task(service.start_transcription(websocket_client=websocket))
    
    try:
        # Connect to Alibaba's WebSocket service
        token = get_alibaba_token()
        ws_url = f"{service.websocket_url}?token={token}"
        
        async with websockets.connect(ws_url) as alibaba_ws:
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
                    "appkey": service.appkey
                },
                "payload": {
                    "format": "pcm",
                    "sample_rate": 16000,
                    "enable_intermediate_result": True,
                    "enable_punctuation_prediction": True,
                    "enable_inverse_text_normalization": True
                }
            }
            
            await alibaba_ws.send(json.dumps(start_cmd))
            
            # Receive TranscriptionStarted event
            response = await alibaba_ws.recv()
            logger.info(f"WebSocket session {session_id} - Started transcription")
            
            # Handle incoming audio data from client and response data from Alibaba
            while service.is_running:
                try:
                    # Use wait_for to implement a timeout
                    client_message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=0.1  # Short timeout to allow checking server messages
                    )
                    
                    # Check if it's a text message (command)
                    if "text" in client_message:
                        message = json.loads(client_message["text"])
                        
                        if message.get("action") == "stop":
                            # Send stop command to Alibaba
                            stop_cmd = {
                                "header": {
                                    "message_id": message_id,
                                    "task_id": task_id,
                                    "namespace": "SpeechTranscriber",
                                    "name": "StopTranscription",
                                    "appkey": service.appkey
                                }
                            }
                            await alibaba_ws.send(json.dumps(stop_cmd))
                            logger.info(f"WebSocket session {session_id} - Stopping transcription")
                    
                    # If it's binary data (audio), forward to Alibaba
                    elif "bytes" in client_message:
                        await alibaba_ws.send(client_message["bytes"])
                
                except asyncio.TimeoutError:
                    # Timeout is expected, check for messages from Alibaba
                    pass
                except WebSocketDisconnect:
                    logger.info(f"WebSocket session {session_id} - Client disconnected")
                    break
                
                # Check for messages from Alibaba
                try:
                    # Non-blocking check for messages from Alibaba
                    alibaba_response = await asyncio.wait_for(
                        alibaba_ws.recv(),
                        timeout=0.1
                    )
                    
                    # Process the response from Alibaba
                    response_json = json.loads(alibaba_response)
                    event_name = response_json.get("header", {}).get("name")
                    
                    # Handle events from Alibaba service
                    if event_name == "TranscriptionResultChanged":
                        result = response_json.get("payload", {}).get("result", "")
                        service.current_transcription += result + " "
                        
                        # Update session data
                        active_sessions[session_id]['transcription'] = service.current_transcription
                        active_sessions[session_id]['last_update'] = time.time()
                        
                        # Send to client
                        await websocket.send_json({
                            'event': 'transcription_update',
                            'data': {
                                'text': service.current_transcription,
                                'partial': True
                            }
                        })
                    
                    elif event_name == "SentenceEnd":
                        result = response_json.get("payload", {}).get("result", "")
                        service.current_transcription += result + " "
                        
                        # Update session data
                        active_sessions[session_id]['transcription'] = service.current_transcription
                        active_sessions[session_id]['last_update'] = time.time()
                        
                        # Send to client
                        await websocket.send_json({
                            'event': 'transcription_update',
                            'data': {
                                'text': service.current_transcription,
                                'partial': False
                            }
                        })
                        
                        # Check sentiment for completed sentence
                        sentiment_result = await analyze_sentiment(service.current_transcription)
                        active_sessions[session_id]['sentiment'] = sentiment_result
                        
                        await websocket.send_json({
                            'event': 'sentiment_update',
                            'data': sentiment_result
                        })
                        
                        if sentiment_result.get("sentiment") == "negative" and sentiment_result.get("score", 0.5) < SENTIMENT_THRESHOLD:
                            await send_alert(session_id, service.current_transcription, sentiment_result)
                    
                    elif event_name == "TranscriptionCompleted":
                        logger.info(f"WebSocket session {session_id} - Transcription completed")
                        active_sessions[session_id]['status'] = 'completed'
                        
                        # Send to client
                        await websocket.send_json({
                            'event': 'transcription_completed',
                            'data': {
                                'text': service.current_transcription
                            }
                        })
                        
                        # Final sentiment check
                        sentiment_result = await analyze_sentiment(service.current_transcription)
                        active_sessions[session_id]['sentiment'] = sentiment_result
                        
                        await websocket.send_json({
                            'event': 'sentiment_update',
                            'data': sentiment_result
                        })
                        
                        break
                
                except asyncio.TimeoutError:
                    # No message from Alibaba, continue
                    pass
                except Exception as e:
                    logger.error(f"WebSocket session {session_id} - Error processing Alibaba response: {e}")
                    # Send error to client
                    await websocket.send_json({
                        'event': 'error',
                        'data': {'message': str(e)}
                    })
                    break
            
            # Send final stop command if needed
            try:
                stop_cmd = {
                    "header": {
                        "message_id": message_id,
                        "task_id": task_id,
                        "namespace": "SpeechTranscriber",
                        "name": "StopTranscription",
                        "appkey": service.appkey
                    }
                }
                await alibaba_ws.send(json.dumps(stop_cmd))
            except:
                pass
    
    except Exception as e:
        logger.error(f"WebSocket session {session_id} - Connection error: {e}")
        try:
            await websocket.send_json({
                'event': 'error',
                'data': {'message': str(e)}
            })
        except:
            pass
    finally:
        # Clean up
        service.stop()
        # Cancel the transcription task if it's still running
        if not transcription_task.done():
            transcription_task.cancel()

# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    # Check if required environment variables are set
    if not ALIBABA_ACCESS_KEY_ID or not ALIBABA_ACCESS_KEY_SECRET or not ALIBABA_APPKEY:
        logger.error("Missing required environment variables. Please set ALIBABA_ACCESS_KEY_ID, ALIBABA_ACCESS_KEY_SECRET, and ALIBABA_APPKEY")
        exit(1)
    
    if not QWEN_API_KEY:
        logger.error("Missing required environment variable DASHSCOPE_API_KEY for Qwen LLM")
        exit(1)
    
    # Start the service
    uvicorn.run(app, host="0.0.0.0", port=8000)