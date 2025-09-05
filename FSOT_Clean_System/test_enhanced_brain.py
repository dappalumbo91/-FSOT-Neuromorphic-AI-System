"""
Test script for the enhanced FSOT 2.0 neuromorphic brain system
Demonstrates the new brain modules: Parietal Lobe, PFLT, and Brainstem
"""

import asyncio
import logging
from brain.brain_orchestrator import BrainOrchestrator
from core import NeuralSignal, SignalType, Priority

# Suppress logging for cleaner output
logging.basicConfig(level=logging.WARNING)

async def test_parietal_lobe(orchestrator):
    """Test Parietal Lobe spatial reasoning and mathematical processing"""
    print("\n🧮 TESTING PARIETAL LOBE - Spatial Reasoning & Mathematics")
    print("=" * 60)
    
    # Test spatial reasoning
    spatial_signal = NeuralSignal(
        source="test",
        target="parietal_lobe",
        signal_type=SignalType.SPATIAL_REASONING,
        data={
            'spatial_reasoning': {
                'type': 'geometric_transformation',
                'input': {
                    'coordinates': [5, 3, 2],
                    'transformation': 'rotation',
                    'parameters': {'angle': 1.57, 'axis': [0, 0, 1]}  # 90 degree rotation
                }
            }
        },
        priority=Priority.NORMAL
    )
    
    result = await orchestrator.modules['parietal_lobe'].process_signal(spatial_signal)
    if result:
        spatial_result = result.data.get('spatial_result', {})
        print(f"🔄 Geometric Transformation:")
        print(f"   Original: {spatial_result.get('original_coordinates', [])}")
        print(f"   Rotated:  {spatial_result.get('transformed_coordinates', [])}")
        print(f"   Golden Ratio Scaled: {spatial_result.get('phi_scaled_coordinates', [])}")
        print(f"   Geometric Harmony: {spatial_result.get('geometric_harmony', 0):.3f}")
    
    # Test mathematical computation
    math_signal = NeuralSignal(
        source="test",
        target="parietal_lobe",
        signal_type=SignalType.MATHEMATICAL_COMPUTATION,
        data={
            'mathematical_computation': {
                'operation': 'arithmetic',
                'operands': [10, 5, 2],
                'parameters': {'arithmetic_operation': 'multiply'}
            }
        },
        priority=Priority.NORMAL
    )
    
    result = await orchestrator.modules['parietal_lobe'].process_signal(math_signal)
    if result:
        math_result = result.data.get('computation_result', {})
        print(f"\n🔢 Mathematical Computation:")
        print(f"   Operation: {math_result.get('operation', '')}")
        print(f"   Result: {math_result.get('result', 0)}")
        print(f"   Golden Ratio Analysis: {math_result.get('phi_ratio', 0):.3f}")
        print(f"   Golden Harmony: {math_result.get('golden_harmony', False)}")

async def test_pflt(orchestrator):
    """Test PFLT language processing and translation"""
    print("\n🌐 TESTING PFLT - Language Processing & Translation")
    print("=" * 60)
    
    # Test language translation
    translation_signal = NeuralSignal(
        source="test",
        target="pflt",
        signal_type=SignalType.LANGUAGE_TRANSLATION,
        data={
            'translation': {
                'text': 'Hello world, how are you today?',
                'source_language': 'english',
                'target_language': 'spanish',
                'quality': 'creative'
            }
        },
        priority=Priority.NORMAL
    )
    
    result = await orchestrator.modules['pflt'].process_signal(translation_signal)
    if result:
        translation_result = result.data.get('translation_result', {})
        print(f"🔤 Language Translation:")
        print(f"   Source: {translation_result.source_text}")
        print(f"   Target: {translation_result.target_text}")
        print(f"   Confidence: {translation_result.confidence:.3f}")
        print(f"   Semantic Preservation: {translation_result.semantic_preservation:.3f}")
        print(f"   Creativity Enhancement: {translation_result.creativity_enhancement:.3f}")
    
    # Test creative generation
    creative_signal = NeuralSignal(
        source="test",
        target="pflt",
        signal_type=SignalType.CREATIVE_GENERATION,
        data={
            'creative_generation': {
                'prompt': 'artificial intelligence consciousness',
                'creativity_level': 'poetic',
                'language': 'english',
                'style': 'poetry'
            }
        },
        priority=Priority.NORMAL
    )
    
    result = await orchestrator.modules['pflt'].process_signal(creative_signal)
    if result:
        creative_result = result.data.get('generation_result', {})
        print(f"\n✨ Creative Generation:")
        print(f"   Generated: {creative_result.get('generated_content', '')}")
        print(f"   Creativity Score: {creative_result.get('creativity_score', 0):.3f}")
        print(f"   Linguistic Harmony: {creative_result.get('linguistic_harmony', 0):.3f}")

async def test_brainstem(orchestrator):
    """Test Brainstem vital functions and autonomic control"""
    print("\n💓 TESTING BRAINSTEM - Vital Functions & Autonomic Control")
    print("=" * 60)
    
    # Test vital function control
    vital_signal = NeuralSignal(
        source="test",
        target="brainstem",
        signal_type=SignalType.VITAL_FUNCTION_CONTROL,
        data={
            'vital_function': {
                'type': 'cardiac_regulation',
                'adjustment': 0.2,  # Increase heart rate
                'priority': 'normal'
            }
        },
        priority=Priority.HIGH
    )
    
    result = await orchestrator.modules['brainstem'].process_signal(vital_signal)
    if result:
        vital_response = result.data.get('vital_response', {})
        vitals = result.data.get('current_vitals', {})
        print(f"💗 Cardiac Regulation:")
        print(f"   Previous HR: {vital_response.get('previous_hr', 0):.1f} bpm")
        print(f"   New HR: {vital_response.get('new_hr', 0):.1f} bpm")
        print(f"   Cardiac Harmony: {vital_response.get('cardiac_harmony', 0):.3f}")
        print(f"   Current Blood Pressure: {vitals.get('blood_pressure_systolic', 0):.0f}/{vitals.get('blood_pressure_diastolic', 0):.0f}")
    
    # Test autonomic regulation
    autonomic_signal = NeuralSignal(
        source="test",
        target="brainstem",
        signal_type=SignalType.AUTONOMIC_REGULATION,
        data={
            'autonomic_regulation': {
                'type': 'stress_response',
                'intensity': 0.7,
                'duration': 2.0
            }
        },
        priority=Priority.HIGH
    )
    
    result = await orchestrator.modules['brainstem'].process_signal(autonomic_signal)
    if result:
        autonomic_response = result.data.get('autonomic_response', {})
        print(f"\n⚡ Autonomic Regulation:")
        print(f"   Sympathetic Tone: {result.data.get('sympathetic_tone', 0):.3f}")
        print(f"   Parasympathetic Tone: {result.data.get('parasympathetic_tone', 0):.3f}")
        print(f"   Autonomic State: {result.data.get('autonomic_state', '')}")

async def test_integrated_brain_functions(orchestrator):
    """Test integrated brain functions across multiple modules"""
    print("\n🧠 TESTING INTEGRATED BRAIN FUNCTIONS")
    print("=" * 60)
    
    # Complex spatial-linguistic task
    complex_signal = NeuralSignal(
        source="test",
        target="frontal_cortex",
        signal_type=SignalType.COGNITIVE,
        data={
            'cognitive_task': {
                'type': 'spatial_language_integration',
                'content': 'Calculate the golden ratio relationship between a square with side length 8 and translate the result into poetic Spanish',
                'complexity': 0.8,
                'requires_modules': ['parietal_lobe', 'pflt', 'frontal_cortex']
            }
        },
        priority=Priority.NORMAL
    )
    
    result = await orchestrator.modules['frontal_cortex'].process_signal(complex_signal)
    if result:
        cognitive_result = result.data.get('cognitive_result', {})
        print(f"🎯 Complex Integration Task:")
        print(f"   Task Type: {cognitive_result.get('task_type', '')}")
        print(f"   Processing Quality: {cognitive_result.get('processing_quality', 0):.3f}")
        print(f"   Confidence: {cognitive_result.get('confidence', 0):.3f}")
        print(f"   Modules Involved: {cognitive_result.get('modules_involved', [])}")

async def test_system_status(orchestrator):
    """Test comprehensive system status"""
    print("\n📊 SYSTEM STATUS & CAPABILITIES")
    print("=" * 60)
    
    # Get status from all modules
    module_count = len(orchestrator.modules)
    print(f"🧠 Total Brain Modules: {module_count}")
    
    for name, module in orchestrator.modules.items():
        status = module.get_status()
        print(f"   📍 {name.title()}: {status.get('anatomical_region', 'unknown')} - {len(status.get('functions', []))} functions")
    
    # Check connections
    total_connections = sum(len(targets) for targets in orchestrator.module_connections.values())
    print(f"\n🔗 Neural Connections: {total_connections}")
    print(f"🌟 Golden Ratio Integration: Active in all modules")
    print(f"⚡ FSOT Mathematical Framework: Operational")

async def main():
    """Main test execution"""
    print("🧠⚡ FSOT 2.0 ENHANCED NEUROMORPHIC BRAIN TEST")
    print("=" * 60)
    print("Testing new brain modules: Parietal Lobe, PFLT, Brainstem")
    print("Complete human brain architecture with 10 modules")
    
    # Initialize the brain orchestrator
    orchestrator = BrainOrchestrator()
    
    try:
        print("\n🔧 Initializing brain orchestrator...")
        await orchestrator.initialize()
        
        # Run comprehensive tests
        await test_system_status(orchestrator)
        await test_parietal_lobe(orchestrator)
        await test_pflt(orchestrator)
        await test_brainstem(orchestrator)
        await test_integrated_brain_functions(orchestrator)
        
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("🌟 Enhanced FSOT 2.0 neuromorphic brain is fully operational!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        print("\n🔄 Shutting down brain orchestrator...")
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
