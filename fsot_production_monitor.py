"""
FSOT Production Performance Monitor
==================================
Real-time production monitoring system for neuromorphic applications.
"""

import time
import json
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Optional psutil import for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available - using simulated system metrics")

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    latency_ms: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    accuracy: float
    error_rate: float
    network_id: str = ""
    operation_type: str = ""

@dataclass
class SystemHealth:
    """System health status."""
    status: str  # 'healthy', 'warning', 'critical'
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    uptime_hours: float

class FSOTProductionMonitor:
    """Production monitoring system for FSOT neuromorphic applications."""
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 monitoring_interval: float = 1.0,
                 retention_hours: int = 24):
        
        # Configuration
        self.monitoring_interval = monitoring_interval
        self.retention_hours = retention_hours
        self.max_samples = int(retention_hours * 3600 / monitoring_interval)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'latency_ms_max': 100.0,
            'memory_mb_max': 1000.0,
            'cpu_percent_max': 80.0,
            'error_rate_max': 0.05,
            'accuracy_min': 0.8
        }
        
        # Monitoring data
        self.metrics_history = deque(maxlen=self.max_samples)
        self.current_session = {
            'start_time': time.time(),
            'total_requests': 0,
            'total_errors': 0,
            'total_processing_time': 0.0
        }
        
        # Real-time tracking
        self.active_operations = {}
        self.performance_counters = defaultdict(list)
        
        # Alerting system
        self.alert_callbacks = []
        self.alert_history = deque(maxlen=1000)
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("🔍 FSOT Production Monitor initialized")
    
    def start_monitoring(self):
        """Start background monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._background_monitor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("📊 Background monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("⏹️ Background monitoring stopped")
    
    def _background_monitor(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_health = self._collect_system_health()
                
                # Store metrics
                current_metrics = PerformanceMetrics(
                    timestamp=time.time(),
                    latency_ms=float(np.mean(self.performance_counters['latency'])) if self.performance_counters['latency'] else 0.0,
                    throughput=self._calculate_throughput(),
                    memory_usage_mb=system_health.memory_usage,
                    cpu_usage_percent=system_health.cpu_usage,
                    accuracy=float(np.mean(self.performance_counters['accuracy'])) if self.performance_counters['accuracy'] else 1.0,
                    error_rate=self._calculate_error_rate(),
                    operation_type="background_monitor"
                )
                
                self.metrics_history.append(current_metrics)
                
                # Check alerts
                self._check_alerts(current_metrics, system_health)
                
                # Clear short-term counters
                self._cleanup_counters()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def track_inference(self, operation_name: str = "inference"):
        """Context manager for tracking inference operations."""
        return InferenceTracker(self, operation_name)
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log performance metrics."""
        timestamp = time.time()
        
        # Update session counters
        self.current_session['total_requests'] += 1
        if 'latency_ms' in metrics:
            self.current_session['total_processing_time'] += metrics['latency_ms'] / 1000.0
        
        # Store individual metrics
        for key, value in metrics.items():
            if key in ['latency_ms', 'accuracy', 'throughput', 'memory_usage']:
                self.performance_counters[key.replace('_ms', '').replace('_mb', '')].append(value)
        
        # Log errors
        if metrics.get('error', False):
            self.current_session['total_errors'] += 1
    
    def _collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network connections
                connections = len(psutil.net_connections())
                
                memory_mb = memory.used / (1024**2)  # MB
                disk_percent = disk.percent
            else:
                # Simulated metrics when psutil not available
                cpu_percent = np.random.uniform(10, 40)
                memory_mb = np.random.uniform(200, 800)
                disk_percent = np.random.uniform(20, 60)
                connections = np.random.randint(5, 50)
            
            # Uptime
            uptime = time.time() - self.current_session['start_time']
            uptime_hours = uptime / 3600.0
            
            # Determine status
            status = 'healthy'
            if (cpu_percent > self.alert_thresholds['cpu_percent_max'] or 
                memory_mb > self.alert_thresholds['memory_mb_max']):
                status = 'warning'
            if (cpu_percent > 95.0 or memory_mb > self.alert_thresholds['memory_mb_max'] * 1.2):
                status = 'critical'
            
            return SystemHealth(
                status=status,
                cpu_usage=cpu_percent,
                memory_usage=memory_mb,
                disk_usage=disk_percent,
                network_latency=0.0,  # Placeholder
                active_connections=connections,
                uptime_hours=uptime_hours
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system health: {e}")
            return SystemHealth(
                status='unknown',
                cpu_usage=0, memory_usage=0, disk_usage=0,
                network_latency=0, active_connections=0, uptime_hours=0
            )
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (requests per second)."""
        if not self.metrics_history:
            return 0.0
        
        # Calculate over last minute
        cutoff_time = time.time() - 60.0
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        time_span = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
        if time_span <= 0:
            return 0.0
        
        return len(recent_metrics) / time_span
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        if self.current_session['total_requests'] == 0:
            return 0.0
        
        return self.current_session['total_errors'] / self.current_session['total_requests']
    
    def _check_alerts(self, metrics: PerformanceMetrics, health: SystemHealth):
        """Check for alert conditions."""
        alerts = []
        
        # Performance alerts
        if metrics.latency_ms > self.alert_thresholds['latency_ms_max']:
            alerts.append({
                'type': 'HIGH_LATENCY',
                'severity': 'warning',
                'message': f"High latency: {metrics.latency_ms:.1f}ms",
                'threshold': self.alert_thresholds['latency_ms_max']
            })
        
        if metrics.memory_usage_mb > self.alert_thresholds['memory_mb_max']:
            alerts.append({
                'type': 'HIGH_MEMORY',
                'severity': 'warning',
                'message': f"High memory usage: {metrics.memory_usage_mb:.1f}MB",
                'threshold': self.alert_thresholds['memory_mb_max']
            })
        
        if metrics.cpu_usage_percent > self.alert_thresholds['cpu_percent_max']:
            alerts.append({
                'type': 'HIGH_CPU',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                'threshold': self.alert_thresholds['cpu_percent_max']
            })
        
        if metrics.error_rate > self.alert_thresholds['error_rate_max']:
            alerts.append({
                'type': 'HIGH_ERROR_RATE',
                'severity': 'critical',
                'message': f"High error rate: {metrics.error_rate:.1%}",
                'threshold': self.alert_thresholds['error_rate_max']
            })
        
        if metrics.accuracy < self.alert_thresholds['accuracy_min']:
            alerts.append({
                'type': 'LOW_ACCURACY',
                'severity': 'warning',
                'message': f"Low accuracy: {metrics.accuracy:.1%}",
                'threshold': self.alert_thresholds['accuracy_min']
            })
        
        # System health alerts
        if health.status == 'critical':
            alerts.append({
                'type': 'SYSTEM_CRITICAL',
                'severity': 'critical',
                'message': "System in critical state"
            })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert notification."""
        alert['timestamp'] = datetime.now().isoformat()
        self.alert_history.append(alert)
        
        # Log alert
        severity = alert['severity'].upper()
        self.logger.warning(f"🚨 [{severity}] {alert['message']}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add alert notification callback."""
        self.alert_callbacks.append(callback)
    
    def _cleanup_counters(self):
        """Clean up short-term performance counters."""
        # Keep only recent samples for real-time calculations
        max_samples = 100
        for key in self.performance_counters:
            if len(self.performance_counters[key]) > max_samples:
                self.performance_counters[key] = self.performance_counters[key][-max_samples:]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        system_health = self._collect_system_health()
        
        return {
            'timestamp': latest.timestamp,
            'latency_ms': latest.latency_ms,
            'throughput': latest.throughput,
            'memory_usage_mb': latest.memory_usage_mb,
            'cpu_usage_percent': latest.cpu_usage_percent,
            'accuracy': latest.accuracy,
            'error_rate': latest.error_rate,
            'system_status': system_health.status,
            'uptime_hours': system_health.uptime_hours,
            'total_requests': self.current_session['total_requests']
        }
    
    def get_historical_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            asdict(metrics) for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        recent_metrics = list(self.metrics_history)[-100:]  # Last 100 samples
        
        latencies = [m.latency_ms for m in recent_metrics]
        throughputs = [m.throughput for m in recent_metrics]
        accuracies = [m.accuracy for m in recent_metrics]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        cpu_usage = [m.cpu_usage_percent for m in recent_metrics]
        
        # System health
        system_health = self._collect_system_health()
        
        # Session statistics
        uptime = time.time() - self.current_session['start_time']
        avg_processing_time = (
            self.current_session['total_processing_time'] / 
            max(self.current_session['total_requests'], 1)
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_period_hours': uptime / 3600.0,
            'system_health': asdict(system_health),
            'performance_summary': {
                'total_requests': self.current_session['total_requests'],
                'total_errors': self.current_session['total_errors'],
                'error_rate': self._calculate_error_rate(),
                'avg_processing_time_ms': avg_processing_time * 1000,
                'current_throughput': self._calculate_throughput()
            },
            'performance_statistics': {
                'latency': {
                    'mean_ms': float(np.mean(latencies)) if latencies else 0,
                    'median_ms': float(np.median(latencies)) if latencies else 0,
                    'p95_ms': float(np.percentile(latencies, 95)) if latencies else 0,
                    'p99_ms': float(np.percentile(latencies, 99)) if latencies else 0,
                    'max_ms': float(np.max(latencies)) if latencies else 0
                },
                'throughput': {
                    'mean_rps': float(np.mean(throughputs)) if throughputs else 0,
                    'max_rps': float(np.max(throughputs)) if throughputs else 0,
                    'current_rps': self._calculate_throughput()
                },
                'accuracy': {
                    'mean': float(np.mean(accuracies)) if accuracies else 0,
                    'min': float(np.min(accuracies)) if accuracies else 0,
                    'current': accuracies[-1] if accuracies else 0
                },
                'resource_usage': {
                    'memory': {
                        'mean_mb': float(np.mean(memory_usage)) if memory_usage else 0,
                        'max_mb': float(np.max(memory_usage)) if memory_usage else 0,
                        'current_mb': system_health.memory_usage
                    },
                    'cpu': {
                        'mean_percent': float(np.mean(cpu_usage)) if cpu_usage else 0,
                        'max_percent': float(np.max(cpu_usage)) if cpu_usage else 0,
                        'current_percent': system_health.cpu_usage
                    }
                }
            },
            'alerts': {
                'active_alerts': len([a for a in self.alert_history if 
                                    datetime.fromisoformat(a['timestamp']) > 
                                    datetime.now() - timedelta(minutes=5)]),
                'total_alerts': len(self.alert_history),
                'recent_alerts': list(self.alert_history)[-10:] if self.alert_history else []
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.metrics_history:
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-50:]
        
        # Latency recommendations
        avg_latency = np.mean([m.latency_ms for m in recent_metrics])
        if avg_latency > self.alert_thresholds['latency_ms_max']:
            recommendations.append(
                f"High average latency ({avg_latency:.1f}ms). Consider optimizing network architecture."
            )
        
        # Memory recommendations
        avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        if avg_memory > self.alert_thresholds['memory_mb_max'] * 0.8:
            recommendations.append(
                f"Memory usage approaching limit ({avg_memory:.1f}MB). Consider reducing network size."
            )
        
        # CPU recommendations
        avg_cpu = np.mean([m.cpu_usage_percent for m in recent_metrics])
        if avg_cpu > self.alert_thresholds['cpu_percent_max'] * 0.8:
            recommendations.append(
                f"CPU usage high ({avg_cpu:.1f}%). Consider load balancing or hardware upgrade."
            )
        
        # Accuracy recommendations
        avg_accuracy = np.mean([m.accuracy for m in recent_metrics])
        if avg_accuracy < self.alert_thresholds['accuracy_min']:
            recommendations.append(
                f"Accuracy below threshold ({avg_accuracy:.1%}). Consider model retraining."
            )
        
        # Error rate recommendations
        error_rate = self._calculate_error_rate()
        if error_rate > self.alert_thresholds['error_rate_max']:
            recommendations.append(
                f"High error rate ({error_rate:.1%}). Review error logs and validate inputs."
            )
        
        # Throughput recommendations
        current_throughput = self._calculate_throughput()
        if current_throughput < 10.0:  # Less than 10 RPS
            recommendations.append(
                "Low throughput detected. Consider batch processing or connection pooling."
            )
        
        return recommendations
    
    def save_performance_report(self, filename: Optional[str] = None) -> str:
        """Save performance report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"FSOT_Performance_Report_{timestamp}.json"
        
        report = self.generate_performance_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📄 Performance report saved: {filename}")
        return filename
    
    def create_dashboard_data(self) -> Dict[str, Any]:
        """Create data for monitoring dashboard."""
        current_metrics = self.get_current_metrics()
        historical_data = self.get_historical_metrics(hours=1)
        
        # Time series data for charts
        timestamps = [m['timestamp'] for m in historical_data]
        latencies = [m['latency_ms'] for m in historical_data]
        throughputs = [m['throughput'] for m in historical_data]
        memory_usage = [m['memory_usage_mb'] for m in historical_data]
        cpu_usage = [m['cpu_usage_percent'] for m in historical_data]
        
        return {
            'current_status': current_metrics,
            'time_series': {
                'timestamps': timestamps,
                'latency_ms': latencies,
                'throughput_rps': throughputs,
                'memory_mb': memory_usage,
                'cpu_percent': cpu_usage
            },
            'alerts': {
                'recent': list(self.alert_history)[-5:],
                'count_last_hour': len([
                    a for a in self.alert_history
                    if datetime.fromisoformat(a['timestamp']) > 
                    datetime.now() - timedelta(hours=1)
                ])
            }
        }

class InferenceTracker:
    """Context manager for tracking inference operations."""
    
    def __init__(self, monitor: FSOTProductionMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.operation_id = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.operation_id = f"{self.operation_name}_{self.start_time}"
        self.monitor.active_operations[self.operation_id] = {
            'start_time': self.start_time,
            'operation_name': self.operation_name
        }
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.time()
            latency_ms = (end_time - self.start_time) * 1000
            
            # Log performance
            self.monitor.log_performance({
                'latency_ms': latency_ms,
                'operation_name': self.operation_name,
                'error': exc_type is not None
            })
        
        # Clean up
        if self.operation_id and self.operation_id in self.monitor.active_operations:
            del self.monitor.active_operations[self.operation_id]
    
    def get_latency(self) -> float:
        """Get current operation latency in milliseconds."""
        if self.start_time:
            return (time.time() - self.start_time) * 1000
        return 0.0

def main():
    """Demo of production monitoring system."""
    # Initialize monitor
    monitor = FSOTProductionMonitor()
    
    # Add alert callback
    def alert_handler(alert):
        print(f"🚨 ALERT: {alert['message']}")
    
    monitor.add_alert_callback(alert_handler)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Simulate production workload
        print("🚀 Simulating production workload...")
        
        for i in range(50):
            with monitor.track_inference("pattern_recognition"):
                # Simulate processing
                time.sleep(np.random.uniform(0.01, 0.1))
                
                # Log additional metrics
                monitor.log_performance({
                    'accuracy': np.random.uniform(0.85, 0.95),
                    'throughput': np.random.uniform(50, 100),
                    'memory_usage': np.random.uniform(100, 200)
                })
            
            if i % 10 == 0:
                print(f"Processed {i+1} requests...")
        
        # Generate report
        report_file = monitor.save_performance_report()
        
        # Print summary
        current_metrics = monitor.get_current_metrics()
        print("\n📊 Performance Summary:")
        print(f"Throughput: {current_metrics.get('throughput', 0):.1f} RPS")
        print(f"Average Latency: {current_metrics.get('latency_ms', 0):.1f}ms")
        print(f"Accuracy: {current_metrics.get('accuracy', 0):.1%}")
        print(f"Total Requests: {current_metrics.get('total_requests', 0)}")
        print(f"Report saved: {report_file}")
        
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
