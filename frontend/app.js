/**
 * ASL Recognition System - Frontend JavaScript
 * 
 * Features:
 * - Real-time webcam capture and ASL recognition
 * - Live token display with deduplication
 * - Automatic pause detection (2 second threshold)
 * - Sentence generation after pause
 * - Split-screen UI with clean design
 */

const API_BASE = '';  // Same origin
const PAUSE_THRESHOLD = 2000;  // 2 seconds
const CAPTURE_INTERVAL = 300;  // 300ms between frames

// DOM Elements
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const clearBtn = document.getElementById('clearBtn');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const gestureOverlay = document.getElementById('gestureOverlay');
const currentGesture = document.getElementById('currentGesture');
const tokenDisplay = document.getElementById('tokenDisplay');
const tokenCounter = document.getElementById('tokenCounter');
const sentenceDisplay = document.getElementById('sentenceDisplay');
const sentenceText = document.getElementById('sentenceText');
const processing = document.getElementById('processing');
const pauseIndicator = document.getElementById('pauseIndicator');
const pauseBar = document.getElementById('pauseBar');
const errorDisplay = document.getElementById('errorDisplay');
const speakBtn = document.getElementById('speakBtn');
const manualInput = document.getElementById('manualInput');
const manualBtn = document.getElementById('manualBtn');

// State
let stream = null;
let captureTimer = null;
let pauseTimer = null;
let tokens = [];
let lastToken = null;
let lastTokenTime = 0;
let startTime = null;

// Initialize
init();

function init() {
  setupEventListeners();
  updateTokenDisplay();
  updateSentenceDisplay();
}

function setupEventListeners() {
  startBtn.addEventListener('click', startCamera);
  stopBtn.addEventListener('click', stopCamera);
  clearBtn.addEventListener('click', clearAll);
  speakBtn.addEventListener('click', speakSentence);
  manualBtn.addEventListener('click', processManualInput);
  manualInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') processManualInput();
  });
}

// Camera Management
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 }, 
      audio: false 
    });
    
    video.srcObject = stream;
    await video.play();
    
    // Setup overlay canvas
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    
    updateStatus('active', 'Camera Active');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    
    // Start capture loop
    startTime = Date.now();
    captureTimer = setInterval(captureFrame, CAPTURE_INTERVAL);
    
  } catch (error) {
    showError('Camera access denied: ' + error.message);
  }
}

function stopCamera() {
  if (captureTimer) {
    clearInterval(captureTimer);
    captureTimer = null;
  }
  
  if (pauseTimer) {
    clearTimeout(pauseTimer);
    pauseTimer = null;
  }
  
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  
  video.srcObject = null;
  
  updateStatus('', 'Camera Off');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  
  hideGesture();
  hidePauseIndicator();
}

// Frame Capture and Processing
async function captureFrame() {
  if (!stream || !video.videoWidth) return;
  
  try {
    // Capture frame to canvas
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    ctx.drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    
    // Send to API
    const response = await fetch(`${API_BASE}/interpret/frame`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataURL })
    });
    
    if (!response.ok) throw new Error('API request failed');
    
    const data = await response.json();
    
    // Update status
    if (data.hand_detected) {
      updateStatus('hand', 'Hand Detected');
    } else {
      updateStatus('active', 'Camera Active');
    }
    
    // Process token
    if (data.token && data.token !== lastToken) {
      addToken(data.token);
      showGesture(data.token);
      resetPauseTimer();
    } else if (!data.token && tokens.length > 0) {
      // No gesture detected, start pause timer if we have tokens
      startPauseTimer();
    }
    
  } catch (error) {
    console.warn('Frame processing error:', error);
    // Don't show error for network issues during live capture
  }
}

// Token Management
function addToken(token) {
  lastToken = token;
  lastTokenTime = Date.now();
  
  // Add to tokens array
  tokens.push(token);
  updateTokenDisplay();
  
  console.log('Added token:', token, '| Total:', tokens.length);
}

function updateTokenDisplay() {
  if (tokens.length === 0) {
    tokenDisplay.innerHTML = '<div class="token-placeholder">Tokens will appear here as you sign...</div>';
    tokenCounter.textContent = '0 tokens';
  } else {
    const tokenChips = tokens.map(token => 
      `<span class="token-chip">${token}</span>`
    ).join('');
    
    tokenDisplay.innerHTML = tokenChips;
    tokenCounter.textContent = `${tokens.length} token${tokens.length !== 1 ? 's' : ''}`;
  }
}

// Pause Detection
function startPauseTimer() {
  if (pauseTimer) return; // Already started
  
  showPauseIndicator();
  
  let elapsed = 0;
  const interval = 50; // Update every 50ms
  
  pauseTimer = setInterval(() => {
    elapsed += interval;
    const progress = (elapsed / PAUSE_THRESHOLD) * 100;
    
    pauseBar.style.width = `${Math.min(progress, 100)}%`;
    
    if (elapsed >= PAUSE_THRESHOLD) {
      clearTimeout(pauseTimer);
      pauseTimer = null;
      hidePauseIndicator();
      
      if (tokens.length > 0) {
        generateSentence();
      }
    }
  }, interval);
}

function resetPauseTimer() {
  if (pauseTimer) {
    clearTimeout(pauseTimer);
    pauseTimer = null;
    hidePauseIndicator();
  }
}

// Sentence Generation
async function generateSentence() {
  if (tokens.length === 0) return;
  
  showProcessing(true);
  hideError();
  
  try {
    const response = await fetch(`${API_BASE}/interpret/finalize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        tokens: tokens,
        start_time_ms: startTime
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Processing failed');
    }
    
    const data = await response.json();
    showSentence(data.sentence);  // This will just show the letters joined together
    
  } catch (error) {
    showError('Processing failed: ' + error.message);
  } finally {
    showProcessing(false);
  }
}

// UI Updates
function updateStatus(type, text) {
  statusDot.className = `status-dot ${type}`;
  statusText.textContent = text;
}

function showGesture(gesture) {
  currentGesture.textContent = gesture;
  gestureOverlay.style.display = 'block';
  
  // Auto-hide after 1.5 seconds
  setTimeout(() => {
    gestureOverlay.style.display = 'none';
  }, 1500);
}

function hideGesture() {
  gestureOverlay.style.display = 'none';
}

function showPauseIndicator() {
  pauseIndicator.style.display = 'block';
  pauseBar.style.width = '0%';
}

function hidePauseIndicator() {
  pauseIndicator.style.display = 'none';
}

function showProcessing(show) {
  processing.style.display = show ? 'flex' : 'none';
}

function updateSentenceDisplay() {
  const placeholder = sentenceDisplay.querySelector('.sentence-placeholder');
  if (placeholder) {
    placeholder.style.display = 'block';
  }
  sentenceText.style.display = 'none';
  speakBtn.style.display = 'none';
}

function showSentence(letters) {
  const placeholder = sentenceDisplay.querySelector('.sentence-placeholder');
  if (placeholder) {
    placeholder.style.display = 'none';
  }
  
  sentenceText.textContent = letters;
  sentenceText.style.display = 'block';
  speakBtn.style.display = 'inline-block';
}

function showError(message) {
  errorDisplay.textContent = message;
  errorDisplay.style.display = 'block';
  
  // Auto-hide after 5 seconds
  setTimeout(() => {
    errorDisplay.style.display = 'none';
  }, 5000);
}

function hideError() {
  errorDisplay.style.display = 'none';
}

// Actions
function clearAll() {
  tokens = [];
  lastToken = null;
  lastTokenTime = 0;
  
  updateTokenDisplay();
  updateSentenceDisplay();
  hideError();
  hidePauseIndicator();
  
  if (pauseTimer) {
    clearTimeout(pauseTimer);
    pauseTimer = null;
  }
}

function speakSentence() {
  const text = sentenceText.textContent;
  if (text && 'speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    speechSynthesis.speak(utterance);
  }
}

async function processManualInput() {
  const input = manualInput.value.trim();
  if (!input) return;
  
  const inputTokens = input.toUpperCase().split(/\s+/);
  
  // Clear current tokens and add manual ones
  tokens = inputTokens;
  updateTokenDisplay();
  
  // Generate sentence
  await generateSentence();
  
  manualInput.value = '';
}
