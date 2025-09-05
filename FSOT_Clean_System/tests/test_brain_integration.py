"""
Brain Integration Tests
Comprehensive testing of all 7 brain modules working together
"""

import pytest
import asyncio
from datetime import datetime

from brain.brain_orchestrator import BrainOrchestrator
from brain.cerebellum import Cerebellum
from brain.temporal_lobe import TemporalLobe  
from brain.occipital_lobe import OccipitalLobe
from brain.thalamus import Thalamus
from core import NeuralSignal, SignalType, Priority

class TestBrainIntegration:
    """Test integration of all brain modules"""
    
    @pytest.fixture
    async def brain_orchestrator(self):
        """Create initialized brain orchestrator"""
        orchestrator = BrainOrchestrator()
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_brain_orchestrator_initialization(self, brain_orchestrator):
        """Test that all 7 brain modules initialize properly"""
        status = await brain_orchestrator.get_status()
        
        assert status['initialized'] == True
        assert len(status['modules']) == 7
        
        # Check all expected modules are present
        expected_modules = [
            'frontal_cortex', 'hippocampus', 'amygdala',
            'cerebellum', 'temporal_lobe', 'occipital_lobe', 'thalamus'
        ]
        
        for module_name in expected_modules:
            assert module_name in status['modules']
            module_status = status['modules'][module_name]
            assert module_status['initialized'] == True
            assert module_status['processing'] == True
    
    @pytest.mark.asyncio
    async def test_thalamus_module_registration(self, brain_orchestrator):
        """Test that all modules are registered with thalamus"""
        thalamus = await brain_orchestrator.get_module('thalamus')
        assert thalamus is not None
        
        thalamus_status = thalamus.get_status()
        assert thalamus_status['registered_modules'] == 6  # All except thalamus itself
    
    @pytest.mark.asyncio
    async def test_motor_skill_learning(self, brain_orchestrator):
        """Test cerebellum motor skill learning"""
        cerebellum = await brain_orchestrator.get_module('cerebellum')
        assert cerebellum is not None
        
        # Test motor skill learning signal
        skill_signal = NeuralSignal(
            source='test',
            target='cerebellum',
            signal_type=SignalType.SKILL_LEARNING,
            data={
                'skill_learning': {
                    'skill_name': 'typing',
                    'practice_data': [0.5, 0.6, 0.7, 0.8],
                    'feedback': 'good_progress'
                }
            },
            priority=Priority.NORMAL,
            response_expected=True
        )
        
        response = await cerebellum.process_signal(skill_signal)
        assert response is not None
        assert 'skill_learning_result' in response.data
        
        result = response.data['skill_learning_result']
        assert result['skill_learned'] == True
        assert 'typing' in result['skill_name']
    
    @pytest.mark.asyncio
    async def test_language_comprehension(self, brain_orchestrator):
        """Test temporal lobe language processing"""
        temporal_lobe = await brain_orchestrator.get_module('temporal_lobe')
        assert temporal_lobe is not None
        
        # Test language comprehension
        language_signal = NeuralSignal(
            source='test',
            target='temporal_lobe',
            signal_type=SignalType.LANGUAGE_COMPREHENSION,
            data={
                'language_comprehension': {
                    'text': 'The weather is nice today',
                    'context': {},
                    'analysis_level': 'semantic'
                }
            },
            priority=Priority.NORMAL,
            response_expected=True
        )
        
        response = await temporal_lobe.process_signal(language_signal)
        assert response is not None
        assert 'comprehension_result' in response.data
        
        result = response.data['comprehension_result']
        assert result['understood'] == True
        assert 'sentiment' in result
        assert 'entities' in result
    
    @pytest.mark.asyncio
    async def test_visual_pattern_recognition(self, brain_orchestrator):
        """Test occipital lobe visual processing"""
        occipital_lobe = await brain_orchestrator.get_module('occipital_lobe')
        assert occipital_lobe is not None
        
        # Test pattern recognition
        visual_signal = NeuralSignal(
            source='test',
            target='occipital_lobe',
            signal_type=SignalType.PATTERN_RECOGNITION,
            data={
                'pattern_recognition': {
                    'image_data': 'simulated_image_data',
                    'recognition_type': 'geometric',
                    'confidence_threshold': 0.8
                }
            },
            priority=Priority.NORMAL,
            response_expected=True
        )
        
        response = await occipital_lobe.process_signal(visual_signal)
        assert response is not None
        assert 'pattern_result' in response.data
        
        result = response.data['pattern_result']
        assert 'patterns_found' in result
        assert 'confidence_scores' in result
    
    @pytest.mark.asyncio
    async def test_consciousness_coordination(self, brain_orchestrator):
        """Test thalamus consciousness coordination"""
        thalamus = await brain_orchestrator.get_module('thalamus')
        assert thalamus is not None
        
        # Test consciousness update
        consciousness_signal = NeuralSignal(
            source='test',
            target='thalamus',
            signal_type=SignalType.CONSCIOUSNESS_UPDATE,
            data={
                'consciousness': {
                    'type': 'level_update',
                    'level_change': 1,
                    'trigger': 'test_activation'
                }
            },
            priority=Priority.HIGH,
            response_expected=True
        )
        
        response = await thalamus.process_signal(consciousness_signal)
        assert response is not None
        assert 'consciousness_result' in response.data
        
        result = response.data['consciousness_result']
        assert 'level_changed' in result
        assert 'new_level' in result
    
    @pytest.mark.asyncio
    async def test_attention_control(self, brain_orchestrator):
        """Test thalamus attention control"""
        thalamus = await brain_orchestrator.get_module('thalamus')
        assert thalamus is not None
        
        # Test attention focusing
        attention_signal = NeuralSignal(
            source='test',
            target='thalamus',
            signal_type=SignalType.ATTENTION_CONTROL,
            data={
                'attention': {
                    'command': 'focus',
                    'target': 'frontal_cortex',
                    'parameters': {'intensity': 0.9}
                }
            },
            priority=Priority.HIGH,
            response_expected=True
        )
        
        response = await thalamus.process_signal(attention_signal)
        assert response is not None
        assert 'attention_result' in response.data
        
        result = response.data['attention_result']
        assert result['focus_set'] == True
        assert result['target'] == 'frontal_cortex'
    
    @pytest.mark.asyncio
    async def test_brain_state_query(self, brain_orchestrator):
        """Test comprehensive brain state retrieval"""
        thalamus = await brain_orchestrator.get_module('thalamus')
        assert thalamus is not None
        
        # Test brain state query
        state_signal = NeuralSignal(
            source='test',
            target='thalamus',
            signal_type=SignalType.BRAIN_STATE_QUERY,
            data={
                'query': {
                    'detail_level': 'detailed'
                }
            },
            priority=Priority.NORMAL,
            response_expected=True
        )
        
        response = await thalamus.process_signal(state_signal)
        assert response is not None
        assert 'brain_state' in response.data
        
        brain_state = response.data['brain_state']
        assert 'consciousness_level' in brain_state
        assert 'active_modules' in brain_state
        assert 'attention_distribution' in brain_state
        assert 'global_workspace' in brain_state
    
    @pytest.mark.asyncio
    async def test_inter_module_communication(self, brain_orchestrator):
        """Test communication between different brain modules"""
        # Get modules
        frontal_cortex = await brain_orchestrator.get_module('frontal_cortex')
        hippocampus = await brain_orchestrator.get_module('hippocampus')
        amygdala = await brain_orchestrator.get_module('amygdala')
        
        assert all([frontal_cortex, hippocampus, amygdala])
        
        # Test memory formation and safety check sequence
        memory_signal = NeuralSignal(
            source='test',
            target='hippocampus',
            signal_type=SignalType.MEMORY_STORE,
            data={
                'memory_store': {
                    'content': 'Important safety information',
                    'memory_type': 'episodic',
                    'importance': 0.9,
                    'tags': ['safety', 'important']
                }
            },
            priority=Priority.HIGH,
            response_expected=True
        )
        
        memory_response = await hippocampus.process_signal(memory_signal)
        assert memory_response is not None
        assert 'memory_stored' in memory_response.data['memory_result']
        
        # Test safety check
        safety_signal = NeuralSignal(
            source='test',
            target='amygdala',
            signal_type=SignalType.SAFETY_CHECK,
            data={
                'safety_check': {
                    'content': 'Important safety information',
                    'check_type': 'content_analysis'
                }
            },
            priority=Priority.HIGH,
            response_expected=True
        )
        
        safety_response = await amygdala.process_signal(safety_signal)
        assert safety_response is not None
        assert 'safety_assessment' in safety_response.data
    
    @pytest.mark.asyncio
    async def test_full_brain_query_processing(self, brain_orchestrator):
        """Test complete query processing through brain orchestrator"""
        result = await brain_orchestrator.process_query(
            "What is the weather like today?",
            context={'source': 'user', 'session_id': 'test_session'}
        )
        
        assert 'query' in result
        assert 'response' in result
        assert 'processing_time' in result
        assert 'brain_state' in result
        assert 'modules_involved' in result
        
        # Check brain state includes consciousness level
        brain_state = result['brain_state']
        assert 'overall_activation' in brain_state
        assert 'processing_load' in brain_state
        assert 'consciousness_level' in brain_state
    
    @pytest.mark.asyncio
    async def test_brain_module_health(self, brain_orchestrator):
        """Test brain module health monitoring"""
        status = await brain_orchestrator.get_status()
        
        # Check each module has health metrics
        for module_name, module_status in status['modules'].items():
            assert 'initialized' in module_status
            assert 'processing' in module_status
            assert 'activation_level' in module_status
            
            # Check module-specific capabilities
            if module_name == 'thalamus':
                assert 'registered_modules' in module_status
                assert 'consciousness_level' in module_status
                assert 'routing_efficiency' in module_status
            elif module_name == 'cerebellum':
                assert 'learned_skills' in module_status
                assert 'coordination_accuracy' in module_status
            elif module_name == 'temporal_lobe':
                assert 'conversation_contexts' in module_status
                assert 'language_understanding_accuracy' in module_status
            elif module_name == 'occipital_lobe':
                assert 'recognized_patterns' in module_status
                assert 'visual_processing_accuracy' in module_status

class TestAdvancedIntegration:
    """Advanced integration tests for complex scenarios"""
    
    @pytest.fixture
    async def brain_orchestrator(self):
        """Create initialized brain orchestrator"""
        orchestrator = BrainOrchestrator()
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_multi_modal_processing(self, brain_orchestrator):
        """Test processing involving multiple brain modules"""
        # Simulate a complex scenario involving visual, language, and motor processing
        
        # 1. Visual input processing
        occipital_lobe = await brain_orchestrator.get_module('occipital_lobe')
        visual_result = await occipital_lobe.process_signal(NeuralSignal(
            source='test',
            target='occipital_lobe',
            signal_type=SignalType.VISUAL_PROCESSING,
            data={'visual_processing': {'scene': 'kitchen_scene', 'objects': ['cup', 'table']}},
            priority=Priority.NORMAL,
            response_expected=True
        ))
        
        assert visual_result is not None
        
        # 2. Language processing of visual scene
        temporal_lobe = await brain_orchestrator.get_module('temporal_lobe')
        language_result = await temporal_lobe.process_signal(NeuralSignal(
            source='test',
            target='temporal_lobe',
            signal_type=SignalType.LANGUAGE_GENERATION,
            data={
                'language_generation': {
                    'intent': 'describe_scene',
                    'context': visual_result.data,
                    'style': 'descriptive'
                }
            },
            priority=Priority.NORMAL,
            response_expected=True
        ))
        
        assert language_result is not None
        
        # 3. Motor planning for interaction
        cerebellum = await brain_orchestrator.get_module('cerebellum')
        motor_result = await cerebellum.process_signal(NeuralSignal(
            source='test',
            target='cerebellum',
            signal_type=SignalType.MOTOR_COORDINATION,
            data={
                'motor_coordination': {
                    'action': 'reach_for_cup',
                    'target_position': {'x': 10, 'y': 20, 'z': 30},
                    'coordination_level': 'precise'
                }
            },
            priority=Priority.NORMAL,
            response_expected=True
        ))
        
        assert motor_result is not None
        
        # All modules should have responded successfully
        assert all([visual_result, language_result, motor_result])
    
    @pytest.mark.asyncio
    async def test_consciousness_state_transitions(self, brain_orchestrator):
        """Test consciousness state transitions through thalamus"""
        thalamus = await brain_orchestrator.get_module('thalamus')
        
        # Test progression through consciousness levels
        consciousness_levels = [
            ('level_update', 1),   # Increase to focused
            ('level_update', 1),   # Increase to hyperaware
            ('level_update', -2),  # Decrease to aware
        ]
        
        for update_type, level_change in consciousness_levels:
            response = await thalamus.process_signal(NeuralSignal(
                source='test',
                target='thalamus',
                signal_type=SignalType.CONSCIOUSNESS_UPDATE,
                data={
                    'consciousness': {
                        'type': update_type,
                        'level_change': level_change,
                        'trigger': 'test_sequence'
                    }
                },
                priority=Priority.HIGH,
                response_expected=True
            ))
            
            assert response is not None
            assert 'consciousness_result' in response.data
            assert response.data['current_level'] is not None
    
    @pytest.mark.asyncio
    async def test_brain_performance_under_load(self, brain_orchestrator):
        """Test brain performance under high signal load"""
        # Send multiple signals concurrently
        tasks = []
        
        for i in range(10):
            signal = NeuralSignal(
                source='test',
                target='frontal_cortex',
                signal_type=SignalType.COGNITIVE,
                data={'cognitive_task': f'task_{i}', 'complexity': 'medium'},
                priority=Priority.NORMAL,
                response_expected=True
            )
            
            frontal_cortex = await brain_orchestrator.get_module('frontal_cortex')
            task = asyncio.create_task(frontal_cortex.process_signal(signal))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that most tasks completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 8  # At least 80% success rate
        
        # Check brain state after load test
        status = await brain_orchestrator.get_status()
        assert status['processing_load'] <= 1.0  # Should still be manageable
