#!/usr/bin/env python3
"""
Simple Web Dashboard for Enhanced FSOT 2.0
==========================================

Lightweight web interface for monitoring and interacting with the Enhanced FSOT 2.0 system.
Uses Flask for simplicity and broad compatibility.

Author: GitHub Copilot
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Flask imports (optional dependency)
try:
    from flask import Flask, render_template_string, jsonify, request, send_from_directory
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleFSOTDashboard:
    """Simple web dashboard for FSOT system monitoring"""
    
    def __init__(self, port: int = 5000, debug: bool = False):
        self.port = port
        self.debug = debug
        self.app = None
        self.server_thread = None
        self.is_running = False
        
        # System status tracking
        self.system_status = {
            "startup_time": datetime.now().isoformat(),
            "current_time": datetime.now().isoformat(),
            "brain_status": "initialized",
            "learning_active": False,
            "api_discovery_active": False,
            "memory_system_active": False,
            "multimodal_active": False
        }
        
        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "learning_sessions": 0,
            "apis_discovered": 0,
            "knowledge_nodes": 0,
            "memory_entries": 0
        }
        
        # Recent activity log
        self.activity_log = []
        self.max_log_entries = 100
        
        if not FLASK_AVAILABLE:
            logger.warning("Flask not available - install with: pip install flask")
            logger.info("Dashboard will run in console mode only")
        else:
            self._setup_flask_app()
    
    def _setup_flask_app(self):
        """Setup Flask application with routes"""
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'fsot-dashboard-key'
        
        # Main dashboard route
        @self.app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML_TEMPLATE, 
                                        system_status=self.system_status,
                                        metrics=self.metrics,
                                        activity_log=self.activity_log[-10:])  # Last 10 entries
        
        # API endpoints
        @self.app.route('/api/status')
        def api_status():
            self._update_current_time()
            return jsonify(self.system_status)
        
        @self.app.route('/api/metrics')
        def api_metrics():
            return jsonify(self.metrics)
        
        @self.app.route('/api/activity')
        def api_activity():
            return jsonify(self.activity_log[-20:])  # Last 20 entries
        
        @self.app.route('/api/query', methods=['POST'])
        def api_query():
            try:
                data = request.get_json()
                query = data.get('query', '')
                
                # Log the query
                self.log_activity("user_query", f"User query: {query}")
                self.metrics["total_queries"] += 1
                
                # Simulate processing (replace with actual system integration)
                response = self._process_query(query)
                
                if response["success"]:
                    self.metrics["successful_queries"] += 1
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                return jsonify({"success": False, "error": str(e)})
        
        @self.app.route('/api/system/<action>', methods=['POST'])
        def api_system_action(action):
            try:
                if action == "start_learning":
                    self.system_status["learning_active"] = True
                    self.log_activity("system", "Autonomous learning started")
                    return jsonify({"success": True, "message": "Learning started"})
                
                elif action == "stop_learning":
                    self.system_status["learning_active"] = False
                    self.log_activity("system", "Autonomous learning stopped")
                    return jsonify({"success": True, "message": "Learning stopped"})
                
                elif action == "start_api_discovery":
                    self.system_status["api_discovery_active"] = True
                    self.log_activity("system", "API discovery started")
                    return jsonify({"success": True, "message": "API discovery started"})
                
                elif action == "stop_api_discovery":
                    self.system_status["api_discovery_active"] = False
                    self.log_activity("system", "API discovery stopped")
                    return jsonify({"success": True, "message": "API discovery stopped"})
                
                else:
                    return jsonify({"success": False, "error": "Unknown action"})
                    
            except Exception as e:
                logger.error(f"System action error: {e}")
                return jsonify({"success": False, "error": str(e)})
    
    def _process_query(self, query: str) -> Dict[str, Any]:
        """Process user query (placeholder for actual system integration)"""
        
        # This would integrate with the actual FSOT system
        # For now, provide a simple response
        
        response_templates = [
            "I understand you're asking about {query}. Let me search my knowledge base...",
            "That's an interesting question about {query}. Based on my current knowledge...",
            "For {query}, I can provide some insights from my learned concepts...",
            "Let me analyze {query} using my multi-modal processing capabilities..."
        ]
        
        import random
        template = random.choice(response_templates)
        mock_response = template.format(query=query)
        
        return {
            "success": True,
            "response": mock_response,
            "processing_time": round(random.uniform(0.5, 2.0), 2),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "sources": ["knowledge_graph", "web_search", "memory_system"]
        }
    
    def _update_current_time(self):
        """Update current time in system status"""
        self.system_status["current_time"] = datetime.now().isoformat()
    
    def log_activity(self, activity_type: str, message: str, details: Optional[Dict] = None):
        """Log system activity"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": activity_type,
            "message": message,
            "details": details or {}
        }
        
        self.activity_log.append(entry)
        
        # Keep log size manageable
        if len(self.activity_log) > self.max_log_entries:
            self.activity_log = self.activity_log[-self.max_log_entries:]
        
        logger.info(f"Activity logged: {activity_type} - {message}")
    
    def update_metrics(self, metric_updates: Dict[str, Any]):
        """Update system metrics"""
        for key, value in metric_updates.items():
            if key in self.metrics:
                self.metrics[key] = value
    
    def update_system_status(self, status_updates: Dict[str, Any]):
        """Update system status"""
        for key, value in status_updates.items():
            if key in self.system_status:
                self.system_status[key] = value
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        self._update_current_time()
        return self.system_status.copy()
    
    def start_dashboard(self):
        """Start the web dashboard server"""
        if not FLASK_AVAILABLE:
            logger.error("Cannot start web dashboard - Flask not available")
            self._start_console_dashboard()
            return
        
        if self.is_running:
            logger.warning("Dashboard already running")
            return
        
        def run_server():
            try:
                logger.info(f"Starting FSOT Dashboard on http://localhost:{self.port}")
                self.app.run(host='0.0.0.0', port=self.port, debug=self.debug, use_reloader=False)
            except Exception as e:
                logger.error(f"Dashboard server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.is_running = True
        
        self.log_activity("system", f"Web dashboard started on port {self.port}")
    
    def stop_dashboard(self):
        """Stop the web dashboard server"""
        self.is_running = False
        self.log_activity("system", "Web dashboard stopped")
        logger.info("Dashboard stopped")
    
    def _start_console_dashboard(self):
        """Start console-based dashboard as fallback"""
        logger.info("Starting console dashboard...")
        
        def console_loop():
            while True:
                try:
                    print("\n" + "="*60)
                    print("🧠 ENHANCED FSOT 2.0 CONSOLE DASHBOARD")
                    print("="*60)
                    
                    self._update_current_time()
                    
                    print(f"📊 System Status:")
                    for key, value in self.system_status.items():
                        status_icon = "✅" if "active" in key and value else "⏸️" if "active" in key else "ℹ️"
                        print(f"   {status_icon} {key}: {value}")
                    
                    print(f"\n📈 Metrics:")
                    for key, value in self.metrics.items():
                        print(f"   • {key}: {value}")
                    
                    print(f"\n📝 Recent Activity:")
                    for entry in self.activity_log[-5:]:
                        timestamp = entry["timestamp"][:19].replace("T", " ")
                        print(f"   [{timestamp}] {entry['type']}: {entry['message']}")
                    
                    print(f"\n⌨️  Commands: [q]uit, [r]efresh, [s]tatus")
                    
                    # Non-blocking input simulation
                    time.sleep(10)  # Refresh every 10 seconds
                    
                except KeyboardInterrupt:
                    print("\nConsole dashboard stopped")
                    break
                except Exception as e:
                    logger.error(f"Console dashboard error: {e}")
                    time.sleep(5)
        
        console_thread = threading.Thread(target=console_loop, daemon=True)
        console_thread.start()
        self.is_running = True

# HTML template for the web dashboard
DASHBOARD_HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced FSOT 2.0 Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 20px;
        }
        .header h1 {
            color: #667eea;
            margin: 0;
            font-size: 2.5em;
        }
        .header p {
            color: #666;
            margin: 10px 0 0 0;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .card h3 {
            margin-top: 0;
            color: #667eea;
            display: flex;
            align-items: center;
        }
        .card h3 .icon {
            margin-right: 10px;
            font-size: 1.2em;
        }
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #eee;
        }
        .status-item:last-child {
            border-bottom: none;
        }
        .status-value {
            font-weight: bold;
        }
        .status-active {
            color: #4CAF50;
        }
        .status-inactive {
            color: #f44336;
        }
        .metric-number {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            text-align: center;
            margin: 10px 0;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .chat-messages {
            height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 8px;
            background: #f9f9f9;
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
        }
        .chat-input button {
            padding: 12px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
        }
        .chat-input button:hover {
            background: #5a6fd8;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        .message.user {
            background: #e3f2fd;
            text-align: right;
        }
        .message.system {
            background: #f1f8e9;
        }
        .activity-item {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            font-size: 0.9em;
        }
        .activity-time {
            color: #666;
            font-size: 0.8em;
        }
        .control-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .control-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
        }
        .control-btn.start {
            background: #4CAF50;
            color: white;
        }
        .control-btn.stop {
            background: #f44336;
            color: white;
        }
        .auto-refresh {
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Enhanced FSOT 2.0 Dashboard</h1>
            <p>Real-time monitoring and control interface</p>
        </div>

        <div class="dashboard-grid">
            <!-- System Status Card -->
            <div class="card">
                <h3><span class="icon">🔧</span>System Status</h3>
                <div class="status-item">
                    <span>Brain Status:</span>
                    <span class="status-value">{{ system_status.brain_status }}</span>
                </div>
                <div class="status-item">
                    <span>Learning Active:</span>
                    <span class="status-value {{ 'status-active' if system_status.learning_active else 'status-inactive' }}">
                        {{ 'Yes' if system_status.learning_active else 'No' }}
                    </span>
                </div>
                <div class="status-item">
                    <span>API Discovery:</span>
                    <span class="status-value {{ 'status-active' if system_status.api_discovery_active else 'status-inactive' }}">
                        {{ 'Active' if system_status.api_discovery_active else 'Inactive' }}
                    </span>
                </div>
                <div class="status-item">
                    <span>Memory System:</span>
                    <span class="status-value {{ 'status-active' if system_status.memory_system_active else 'status-inactive' }}">
                        {{ 'Active' if system_status.memory_system_active else 'Inactive' }}
                    </span>
                </div>
                <div class="status-item">
                    <span>Multi-modal:</span>
                    <span class="status-value {{ 'status-active' if system_status.multimodal_active else 'status-inactive' }}">
                        {{ 'Active' if system_status.multimodal_active else 'Inactive' }}
                    </span>
                </div>
                
                <div class="control-buttons">
                    <button class="control-btn start" onclick="systemAction('start_learning')">Start Learning</button>
                    <button class="control-btn stop" onclick="systemAction('stop_learning')">Stop Learning</button>
                    <button class="control-btn start" onclick="systemAction('start_api_discovery')">Start API Discovery</button>
                    <button class="control-btn stop" onclick="systemAction('stop_api_discovery')">Stop API Discovery</button>
                </div>
            </div>

            <!-- Metrics Card -->
            <div class="card">
                <h3><span class="icon">📊</span>Performance Metrics</h3>
                <div style="text-align: center;">
                    <div>Total Queries</div>
                    <div class="metric-number">{{ metrics.total_queries }}</div>
                </div>
                <div class="status-item">
                    <span>Successful Queries:</span>
                    <span class="status-value">{{ metrics.successful_queries }}</span>
                </div>
                <div class="status-item">
                    <span>Learning Sessions:</span>
                    <span class="status-value">{{ metrics.learning_sessions }}</span>
                </div>
                <div class="status-item">
                    <span>APIs Discovered:</span>
                    <span class="status-value">{{ metrics.apis_discovered }}</span>
                </div>
                <div class="status-item">
                    <span>Knowledge Nodes:</span>
                    <span class="status-value">{{ metrics.knowledge_nodes }}</span>
                </div>
            </div>

            <!-- Recent Activity Card -->
            <div class="card">
                <h3><span class="icon">📝</span>Recent Activity</h3>
                {% for activity in activity_log %}
                <div class="activity-item">
                    <div>{{ activity.message }}</div>
                    <div class="activity-time">{{ activity.timestamp[:19] }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <!-- Chat Interface -->
        <div class="chat-container">
            <h3><span class="icon">💬</span>Interactive Chat</h3>
            <div class="chat-messages" id="chatMessages">
                <div class="message system">
                    <strong>System:</strong> Welcome to Enhanced FSOT 2.0! Ask me anything and I'll process it using my neuromorphic capabilities.
                </div>
            </div>
            <div class="chat-input">
                <input type="text" id="chatInput" placeholder="Ask me anything..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <div class="auto-refresh">
            <p>⟳ Dashboard updates automatically every 10 seconds</p>
        </div>
    </div>

    <script>
        // Auto-refresh dashboard
        setInterval(() => {
            location.reload();
        }, 10000);

        // Chat functionality
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            addMessage('user', message);
            input.value = '';
            
            // Send to API
            fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({query: message})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addMessage('system', data.response + 
                        ` (Confidence: ${(data.confidence * 100).toFixed(1)}%, Time: ${data.processing_time}s)`);
                } else {
                    addMessage('system', 'Error: ' + data.error);
                }
            })
            .catch(error => {
                addMessage('system', 'Connection error: ' + error);
            });
        }

        function addMessage(type, content) {
            const messages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            messageDiv.innerHTML = `<strong>${type === 'user' ? 'You' : 'FSOT'}:</strong> ${content}`;
            messages.appendChild(messageDiv);
            messages.scrollTop = messages.scrollHeight;
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function systemAction(action) {
            fetch(`/api/system/${action}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    addMessage('system', data.message);
                    setTimeout(() => location.reload(), 1000);
                } else {
                    addMessage('system', 'Error: ' + data.error);
                }
            })
            .catch(error => {
                addMessage('system', 'Connection error: ' + error);
            });
        }
    </script>
</body>
</html>
'''

# Global instance for easy access
_dashboard = None

def get_dashboard() -> SimpleFSOTDashboard:
    """Get or create global dashboard instance"""
    global _dashboard
    if _dashboard is None:
        _dashboard = SimpleFSOTDashboard()
    return _dashboard

if __name__ == "__main__":
    # Test the dashboard
    print("🌐 Testing Simple FSOT Dashboard")
    print("=" * 40)
    
    dashboard = SimpleFSOTDashboard()
    
    # Log some test activity
    dashboard.log_activity("system", "Dashboard test started")
    dashboard.log_activity("learning", "Test learning session completed")
    dashboard.log_activity("api_discovery", "Found 3 new APIs")
    
    # Update some metrics
    dashboard.update_metrics({
        "total_queries": 42,
        "successful_queries": 38,
        "learning_sessions": 15,
        "apis_discovered": 23
    })
    
    # Start dashboard
    if FLASK_AVAILABLE:
        print("✅ Flask available - starting web dashboard")
        dashboard.start_dashboard()
        print("🌐 Dashboard running on http://localhost:5000")
        print("Press Ctrl+C to stop...")
        
        try:
            # Keep running
            while dashboard.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            dashboard.stop_dashboard()
    else:
        print("⚠️ Flask not available - starting console dashboard")
        dashboard._start_console_dashboard()
        
        try:
            while dashboard.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDashboard stopped")
