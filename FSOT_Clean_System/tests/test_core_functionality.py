"""
Test Suite for FSOT 2.0 Clean System
Basic tests to ensure system functionality
"""

# Optional testing dependencies - graceful fallback if not installed
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    print("⚠️  pytest not available - tests disabled")
    PYTEST_AVAILABLE = False
    
import asyncio
from unittest.mock import Mock, patch

# Test imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import FSOTEngine, Domain, NeuralSignal, SignalType, Priority
from brain import FrontalCortex, BrainOrchestrator
from config import config

class TestFSOTEngine:
    """Test FSOT 2.0 mathematical engine"""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        engine = FSOTEngine()
        
        # Check constants are computed
        assert engine.phi > 1.6
        assert engine.e > 2.7
        assert engine.pi > 3.1
        assert engine.k > 0.0
    
    def test_domain_computation(self):
        """Test domain-specific computations"""
        engine = FSOTEngine()
        
        # Test different domains
        quantum_scalar = engine.compute_for_domain(Domain.QUANTUM)
        cognitive_scalar = engine.compute_for_domain(Domain.COGNITIVE)
        cosmological_scalar = engine.compute_for_domain(Domain.COSMOLOGICAL)
        
        # Results should be different for different domains
        assert quantum_scalar != cognitive_scalar
        assert cognitive_scalar != cosmological_scalar
        
        # Results should be finite numbers
        assert isinstance(quantum_scalar, (int, float))
        assert isinstance(cognitive_scalar, (int, float))
        assert isinstance(cosmological_scalar, (int, float))
    
    def test_result_interpretation(self):
        """Test result interpretation"""
        engine = FSOTEngine()
        scalar = engine.compute_for_domain(Domain.COGNITIVE)
        interpretation = engine.interpret_result(scalar, Domain.COGNITIVE)
        
        # Check interpretation structure
        assert 'scalar_value' in interpretation
        assert 'domain' in interpretation
        assert 'sign_meaning' in interpretation
        assert 'consciousness_contribution' in interpretation

class TestNeuralSignal:
    """Test neural signal system"""
    
    def test_signal_creation(self):
        """Test neural signal creation"""
        signal = NeuralSignal(
            source="test_source",
            target="test_target",
            signal_type=SignalType.COGNITIVE,
            data={"test": "data"}
        )
        
        assert signal.source == "test_source"
        assert signal.target == "test_target"
        assert signal.signal_type == SignalType.COGNITIVE
        assert signal.data == {"test": "data"}
        assert signal.priority == Priority.NORMAL  # Default
    
    def test_signal_serialization(self):
        """Test signal to/from dict conversion"""
        original = NeuralSignal(
            source="source",
            target="target",
            signal_type=SignalType.MEMORY,
            data={"key": "value"},
            priority=Priority.HIGH
        )
        
        # Convert to dict and back
        signal_dict = original.to_dict()
        restored = NeuralSignal.from_dict(signal_dict)
        
        assert restored.source == original.source
        assert restored.target == original.target
        assert restored.signal_type == original.signal_type
        assert restored.data == original.data
        assert restored.priority == original.priority

class TestBrainModules:
    """Test brain module functionality"""
    
    @pytest.mark.asyncio
    async def test_frontal_cortex_initialization(self):
        """Test frontal cortex initializes correctly"""
        frontal = FrontalCortex()
        
        assert frontal.name == "frontal_cortex"
        assert frontal.anatomical_region == "cerebrum"
        assert "decision_making" in frontal.functions
        assert frontal.activation_level == 0.0
        
        # Cleanup
        await frontal.shutdown()
    
    @pytest.mark.asyncio
    async def test_frontal_cortex_decision_making(self):
        """Test frontal cortex decision making"""
        frontal = FrontalCortex()
        
        # Create decision request signal
        signal = NeuralSignal(
            source="test",
            target="frontal_cortex",
            signal_type=SignalType.COGNITIVE,
            data={
                "decision_request": {
                    "options": ["option_a", "option_b", "option_c"],
                    "context": {"test": True},
                    "urgency": "normal"
                }
            },
            response_expected=True
        )
        
        # Process signal
        response = await frontal._process_signal_impl(signal)
        
        assert response is not None
        assert response.signal_type == SignalType.EXECUTIVE
        assert "decision" in response.data
        assert "confidence" in response.data
        assert response.data["decision"] in ["option_a", "option_b", "option_c"]
        
        # Cleanup
        await frontal.shutdown()

class TestBrainOrchestrator:
    """Test brain orchestrator"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly"""
        orchestrator = BrainOrchestrator()
        
        await orchestrator.initialize()
        
        assert orchestrator.is_initialized
        assert len(orchestrator.modules) > 0
        assert "frontal_cortex" in orchestrator.modules
        
        # Cleanup
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_query_processing(self):
        """Test query processing through orchestrator"""
        orchestrator = BrainOrchestrator()
        await orchestrator.initialize()
        
        # Process a test query
        result = await orchestrator.process_query("What should I do today?")
        
        assert "query" in result
        assert "response" in result
        assert "processing_time" in result
        assert "brain_state" in result
        
        # Check brain state
        brain_state = result["brain_state"]
        assert "overall_activation" in brain_state
        assert "processing_load" in brain_state
        assert "consciousness_level" in brain_state
        
        # Cleanup
        await orchestrator.shutdown()

class TestConfiguration:
    """Test configuration management"""
    
    def test_config_access(self):
        """Test configuration access"""
        # Test brain config
        assert hasattr(config.brain_config, 'consciousness_threshold')
        assert hasattr(config.brain_config, 'learning_rate')
        assert hasattr(config.brain_config, 'memory_capacity')
        
        # Test FSOT config
        assert hasattr(config.fsot_config, 'max_dimensions')
        assert hasattr(config.fsot_config, 'scaling_constant')
        
        # Test system config
        assert hasattr(config.system_config, 'log_level')
        assert hasattr(config.system_config, 'web_port')
    
    def test_config_updates(self):
        """Test configuration updates"""
        original_threshold = config.brain_config.consciousness_threshold
        
        # Update config
        config.update_brain_config(consciousness_threshold=0.8)
        assert config.brain_config.consciousness_threshold == 0.8
        
        # Restore original
        config.update_brain_config(consciousness_threshold=original_threshold)
        assert config.brain_config.consciousness_threshold == original_threshold

# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
