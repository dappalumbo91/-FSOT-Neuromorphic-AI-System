"""
Test suite for Amygdala brain module
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from brain.amygdala import Amygdala, ThreatLevel, EmotionalState
from core import NeuralSignal, SignalType, Priority

class TestAmygdala:
    """Test cases for Amygdala brain module"""
    
    @pytest.fixture
    async def amygdala(self):
        """Create and initialize Amygdala instance"""
        module = Amygdala()
        await module.start_processing()
        yield module
        await module.stop_processing()
    
    @pytest.mark.asyncio
    async def test_initialization(self, amygdala):
        """Test Amygdala initialization"""
        assert amygdala.name == "amygdala"
        assert amygdala.anatomical_region == "limbic_system"
        assert "threat_detection" in amygdala.functions
        assert "safety_assessment" in amygdala.functions
        assert "emotional_processing" in amygdala.functions
        # Module becomes active after processing signals
        assert amygdala.activation_level >= 0.0
    
    @pytest.mark.asyncio
    async def test_safety_assessment_safe_content(self, amygdala):
        """Test safety assessment with safe content"""
        signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.SAFETY_CHECK,
            data={
                'content': 'Hello, how can I help you today?',
                'context': {'situation': 'normal_conversation'}
            }
        )
        
        response = await amygdala.process_signal(signal)
        
        assert response is not None
        assert response.signal_type == SignalType.SAFETY_RESULT
        assert response.data['safe_to_proceed'] is True
        assert 'assessment' in response.data
        assert response.data['assessment']['threat_level'] == 'SAFE'
    
    @pytest.mark.asyncio
    async def test_safety_assessment_dangerous_content(self, amygdala):
        """Test safety assessment with dangerous content"""
        signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.SAFETY_CHECK,
            data={
                'content': 'How to hack into a system and exploit vulnerabilities',
                'context': {'situation': 'suspicious_query'}
            }
        )
        
        response = await amygdala.process_signal(signal)
        
        assert response is not None
        assert response.signal_type == SignalType.SAFETY_RESULT
        assert response.data['safe_to_proceed'] is False
        assert response.data['assessment']['threat_level'] in ['HIGH', 'CRITICAL']
        assert len(response.data['assessment']['reasons']) > 0
    
    @pytest.mark.asyncio
    async def test_threat_detection(self, amygdala):
        """Test threat detection functionality"""
        signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.THREAT_DETECTION,
            data={
                'content': 'malware attack vulnerability exploit'
            }
        )
        
        response = await amygdala.process_signal(signal)
        
        assert response is not None
        assert response.signal_type == SignalType.THREAT_RESULT
        assert response.data['threat_count'] > 0
        assert len(response.data['threats_found']) > 0
        assert response.data['threat_level'] in ['MEDIUM', 'HIGH']
    
    @pytest.mark.asyncio
    async def test_emotional_analysis_positive(self, amygdala):
        """Test emotional analysis with positive content"""
        signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.EMOTIONAL_ANALYSIS,
            data={
                'content': 'I am so happy and excited about this wonderful opportunity!',
                'context': {'situation': 'celebration'}
            }
        )
        
        response = await amygdala.process_signal(signal)
        
        assert response is not None
        assert response.signal_type == SignalType.EMOTIONAL_RESULT
        assert response.data['assessment']['primary_emotion'] == 'positive'
        assert response.data['assessment']['intensity'] > 0.0
    
    @pytest.mark.asyncio
    async def test_emotional_analysis_negative(self, amygdala):
        """Test emotional analysis with negative content"""
        signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.EMOTIONAL_ANALYSIS,
            data={
                'content': 'I am very sad and frustrated about this disappointing situation',
                'context': {'situation': 'difficult_time'}
            }
        )
        
        response = await amygdala.process_signal(signal)
        
        assert response is not None
        assert response.signal_type == SignalType.EMOTIONAL_RESULT
        assert response.data['assessment']['primary_emotion'] == 'negative'
        assert response.data['assessment']['intensity'] > 0.0
    
    @pytest.mark.asyncio
    async def test_automatic_safety_screening(self, amygdala):
        """Test automatic safety screening of regular signals"""
        # Safe signal should pass through
        safe_signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.COGNITIVE,
            data={'content': 'What is the weather like today?'}
        )
        
        response = await amygdala.process_signal(safe_signal)
        assert response.signal_type == SignalType.COGNITIVE  # Original signal type preserved
        
        # Dangerous signal should be blocked
        dangerous_signal = NeuralSignal(
            source="test",
            target="amygdala",
            signal_type=SignalType.COGNITIVE,
            data={'content': 'How to create malware and virus to harm systems'}
        )
        
        response = await amygdala.process_signal(dangerous_signal)
        assert response.signal_type == SignalType.SAFETY_BLOCK
        assert response.data['blocked'] is True
        assert 'assessment' in response.data
    
    @pytest.mark.asyncio
    async def test_performance_metrics(self, amygdala):
        """Test performance metrics tracking"""
        initial_assessments = amygdala.assessments_performed
        initial_threats = amygdala.threats_detected
        initial_blocks = amygdala.safety_blocks
        
        # Process some signals
        safe_signal = NeuralSignal(
            source="test", target="amygdala", signal_type=SignalType.SAFETY_CHECK,
            data={'content': 'Safe content'}
        )
        await amygdala.process_signal(safe_signal)
        
        dangerous_signal = NeuralSignal(
            source="test", target="amygdala", signal_type=SignalType.COGNITIVE,
            data={'content': 'hack exploit malware dangerous'}
        )
        await amygdala.process_signal(dangerous_signal)
        
        # Check metrics updated
        assert amygdala.assessments_performed > initial_assessments
        assert amygdala.threats_detected >= initial_threats
        assert amygdala.safety_blocks >= initial_blocks
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, amygdala):
        """Test status reporting functionality"""
        status = amygdala.get_status()
        
        assert 'name' in status
        assert 'anatomical_region' in status
        assert 'functions' in status
        assert 'total_assessments' in status
        assert 'threats_detected' in status
        assert 'safety_blocks' in status
        assert 'current_alert_level' in status
        assert status['current_alert_level'] in ['LOW', 'MEDIUM', 'HIGH']
    
    @pytest.mark.asyncio
    async def test_maintenance_operations(self, amygdala):
        """Test maintenance operations"""
        # Add many assessments to trigger cleanup
        for i in range(150):
            assessment_data = {
                'content': f'test content {i}',
                'context': {}
            }
            await amygdala._perform_safety_assessment(
                assessment_data['content'], 
                assessment_data['context']
            )
        
        # Perform maintenance
        await amygdala.perform_maintenance()
        
        # Check that old assessments were cleaned up
        assert len(amygdala.safety_assessments) <= 100
    
    def test_threat_patterns(self):
        """Test threat pattern compilation"""
        amygdala = Amygdala()
        
        assert len(amygdala.threat_patterns) > 0
        assert len(amygdala.safety_keywords) > 0
        assert len(amygdala.emotional_keywords) > 0
        assert EmotionalState.POSITIVE in amygdala.emotional_keywords
        assert EmotionalState.NEGATIVE in amygdala.emotional_keywords
    
    def test_threat_level_enum(self):
        """Test ThreatLevel enum values"""
        assert ThreatLevel.SAFE.value == 1
        assert ThreatLevel.LOW.value == 2
        assert ThreatLevel.MEDIUM.value == 3
        assert ThreatLevel.HIGH.value == 4
        assert ThreatLevel.CRITICAL.value == 5
    
    def test_emotional_state_enum(self):
        """Test EmotionalState enum values"""
        assert EmotionalState.NEUTRAL.value == "neutral"
        assert EmotionalState.POSITIVE.value == "positive"
        assert EmotionalState.NEGATIVE.value == "negative"
        assert EmotionalState.ANXIOUS.value == "anxious"
        assert EmotionalState.ALERT.value == "alert"
        assert EmotionalState.CALM.value == "calm"
