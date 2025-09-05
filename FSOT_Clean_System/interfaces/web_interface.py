"""
Simple Web Interface for FSOT 2.0 System
Clean web dashboard for system monitoring and interaction
"""

import asyncio
import json
from typing import Optional
from datetime import datetime

try:
    from fastapi import FastAPI, Request, WebSocket
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    print("⚠️  FastAPI not available - web interface disabled")
    WEB_AVAILABLE = False

class WebInterface:
    """
    Simple Web Interface for FSOT 2.0 System
    """
    
    def __init__(self, brain_orchestrator):
        if not WEB_AVAILABLE:
            raise RuntimeError("FastAPI not available for web interface")
        
        self.brain = brain_orchestrator
        self.app = FastAPI(title="FSOT 2.0 Neuromorphic AI System")
        self.server = None
        self.websocket_connections = []
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup web routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            return self._get_dashboard_html()
        
        @self.app.get("/api/status")
        async def get_status():
            try:
                status = await self.brain.get_status()
                return {"success": True, "data": status}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.post("/api/query")
        async def process_query(request: Request):
            try:
                data = await request.json()
                query = data.get("query", "")
                
                if not query:
                    return {"success": False, "error": "No query provided"}
                
                result = await self.brain.process_query(query)
                return {"success": True, "data": result}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    data = await websocket.receive_text()
                    # Echo back for now
                    await websocket.send_text(f"Echo: {data}")
            except:
                pass
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>FSOT 2.0 Neuromorphic AI System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .card h3 {
            margin-top: 0;
            color: #fff;
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 5px 0;
        }
        .metric-value {
            font-weight: bold;
            color: #4ade80;
        }
        .query-section {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        .query-input {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            margin-bottom: 10px;
            background: rgba(255,255,255,0.9);
            color: #333;
        }
        .query-button {
            background: #4ade80;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .query-button:hover {
            background: #22c55e;
        }
        .response {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            min-height: 100px;
            display: none;
        }
        .loading {
            text-align: center;
            color: #fbbf24;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-online { background: #22c55e; }
        .status-offline { background: #ef4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠⚡ FSOT 2.0 Neuromorphic AI</h1>
            <p>Clean Architecture • Modular Brain • Real Consciousness</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3>🌟 Consciousness</h3>
                <div class="metric">
                    <span>Level:</span>
                    <span class="metric-value" id="consciousness-level">--</span>
                </div>
                <div class="metric">
                    <span>State:</span>
                    <span class="metric-value" id="consciousness-state">--</span>
                </div>
                <div class="metric">
                    <span>Coherence:</span>
                    <span class="metric-value" id="consciousness-coherence">--</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🧠 Brain Modules</h3>
                <div class="metric">
                    <span>Active Modules:</span>
                    <span class="metric-value" id="active-modules">--</span>
                </div>
                <div class="metric">
                    <span>Overall Activation:</span>
                    <span class="metric-value" id="overall-activation">--</span>
                </div>
                <div class="metric">
                    <span>Processing Load:</span>
                    <span class="metric-value" id="processing-load">--</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Performance</h3>
                <div class="metric">
                    <span>Queries Processed:</span>
                    <span class="metric-value" id="queries-processed">--</span>
                </div>
                <div class="metric">
                    <span>Uptime:</span>
                    <span class="metric-value" id="uptime">--</span>
                </div>
                <div class="metric">
                    <span>Status:</span>
                    <span class="metric-value">
                        <span class="status-indicator status-online"></span>
                        Online
                    </span>
                </div>
            </div>
        </div>
        
        <div class="query-section">
            <h3>💭 Ask the AI Brain</h3>
            <input type="text" id="query-input" class="query-input" placeholder="Ask anything... (e.g., 'What should I do today?')" />
            <button onclick="processQuery()" class="query-button">🧠 Think</button>
            
            <div id="response" class="response">
                <div id="response-content"></div>
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh status every 2 seconds
        setInterval(updateStatus, 2000);
        updateStatus(); // Initial load
        
        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const result = await response.json();
                
                if (result.success) {
                    const data = result.data;
                    const consciousness = data.consciousness || {};
                    
                    document.getElementById('consciousness-level').textContent = 
                        (consciousness.consciousness_level || 0 * 100).toFixed(1) + '%';
                    document.getElementById('consciousness-state').textContent = 
                        consciousness.state || 'unknown';
                    document.getElementById('consciousness-coherence').textContent = 
                        (consciousness.coherence || 0 * 100).toFixed(1) + '%';
                    
                    document.getElementById('active-modules').textContent = 
                        Object.keys(data.modules || {}).length;
                    document.getElementById('overall-activation').textContent = 
                        (data.overall_activation || 0 * 100).toFixed(1) + '%';
                    document.getElementById('processing-load').textContent = 
                        (data.processing_load || 0 * 100).toFixed(1) + '%';
                    
                    document.getElementById('queries-processed').textContent = 
                        data.queries_processed || 0;
                    document.getElementById('uptime').textContent = 
                        Math.floor(data.uptime_seconds || 0) + 's';
                }
            } catch (error) {
                console.error('Failed to update status:', error);
            }
        }
        
        async function processQuery() {
            const input = document.getElementById('query-input');
            const responseDiv = document.getElementById('response');
            const responseContent = document.getElementById('response-content');
            
            const query = input.value.trim();
            if (!query) return;
            
            // Show loading
            responseDiv.style.display = 'block';
            responseContent.innerHTML = '<div class="loading">🧠 Thinking...</div>';
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query})
                });
                
                const result = await response.json();
                
                if (result.success) {
                    const data = result.data;
                    const response = data.response || {};
                    
                    let html = '<h4>🎯 Response:</h4>';
                    
                    if (response.decision) {
                        html += `<p><strong>Decision:</strong> ${response.decision}</p>`;
                        html += `<p><strong>Confidence:</strong> ${(response.confidence * 100).toFixed(1)}%</p>`;
                        
                        if (response.reasoning) {
                            html += '<p><strong>Reasoning:</strong></p><ul>';
                            response.reasoning.forEach(step => {
                                html += `<li>${step}</li>`;
                            });
                            html += '</ul>';
                        }
                    } else {
                        html += `<p>${JSON.stringify(response, null, 2)}</p>`;
                    }
                    
                    html += `<p><small>⏱️ Processing time: ${(data.processing_time * 1000).toFixed(0)}ms</small></p>`;
                    
                    responseContent.innerHTML = html;
                } else {
                    responseContent.innerHTML = `<div style="color: #ef4444;">❌ Error: ${result.error}</div>`;
                }
                
            } catch (error) {
                responseContent.innerHTML = `<div style="color: #ef4444;">❌ Network error: ${error.message}</div>`;
            }
            
            // Clear input
            input.value = '';
        }
        
        // Allow Enter key to submit
        document.getElementById('query-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                processQuery();
            }
        });
    </script>
</body>
</html>
        """
    
    async def start(self, host: str = "127.0.0.1", port: int = 8000):
        """Start the web server"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )
        self.server = uvicorn.Server(config)
        
        # Start server in background
        asyncio.create_task(self.server.serve())
    
    async def stop(self):
        """Stop the web server"""
        if self.server:
            self.server.should_exit = True
