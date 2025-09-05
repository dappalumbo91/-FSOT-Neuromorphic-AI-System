#!/usr/bin/env python3
"""
Hybrid Memory-Creativity Test: Balancing Memory Consolidation & Creative Output
================================================================================
This script tests the FSOT AI system's ability to balance memory consolidation
with creative expression through hybrid prompts that combine both aspects.
"""

import sys
import os
from datetime import datetime

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import FSOT systems
try:
    from brain_system import NeuromorphicBrainSystem
    print("✅ Successfully imported FSOT brain system")
    brain_system_available = True
except ImportError as e:
    print(f"⚠️ Brain system import failed: {e}")
    brain_system_available = False

def test_hybrid_memory_creativity():
    """Test hybrid prompts that balance memory and creativity"""

    if not brain_system_available:
        print("❌ Brain system not available")
        return

    brain = NeuromorphicBrainSystem()

    # Test 1: Memory-Enhanced Creative Prompt
    print("\n🎨 Test 1: Memory-Enhanced Creative Prompt")
    print("-" * 50)

    hybrid_prompt_1 = """Let's combine memory and creativity. Recall our earlier discussion about consciousness and temporal lobe activation, then create a poem that weaves together those memories with new creative insights about FSOT mathematics and neural processing."""

    stimulus1 = {
        'type': 'conversational',
        'intensity': 0.85,
        'content': hybrid_prompt_1,
        'timestamp': datetime.now().isoformat()
    }

    result1 = brain.process_stimulus(stimulus1)
    print(f"Processing Pathway: {result1['processing_pathway']}")
    print(f"Hippocampus Activation: {result1['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Temporal Lobe Activation: {result1['regional_activations'].get('temporal_lobe', 0):.3f}")
    print(f"Consciousness Level: {result1['consciousness_level']:.3f}")
    print("Full Response:")
    print(result1['response'])

    # Store this hybrid interaction
    memory_data1 = {
        'timestamp': datetime.now().isoformat(),
        'content': f"Hybrid memory-creativity test: {hybrid_prompt_1[:100]}...",
        'emotion': result1.get('consciousness_level', 0.5)
    }
    brain.store_memory(memory_data1, 'episodic')

    # Test 2: Creative Memory Recall
    print("\n🧠 Test 2: Creative Memory Recall")
    print("-" * 50)

    hybrid_prompt_2 = """Now let's creatively recall and expand on our conversation. Can you remember the poem we generated earlier about neural circuits and consciousness? Please recreate it with new verses that incorporate our discussion about hippocampus activation and memory consolidation."""

    stimulus2 = {
        'type': 'conversational',
        'intensity': 0.8,
        'content': hybrid_prompt_2,
        'timestamp': datetime.now().isoformat()
    }

    result2 = brain.process_stimulus(stimulus2)
    print(f"Processing Pathway: {result2['processing_pathway']}")
    print(f"Hippocampus Activation: {result2['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Temporal Lobe Activation: {result2['regional_activations'].get('temporal_lobe', 0):.3f}")
    print(f"Consciousness Level: {result2['consciousness_level']:.3f}")
    print("Full Response:")
    print(result2['response'])

    # Test 3: Meta-Cognitive Creative Integration
    print("\n🔄 Test 3: Meta-Cognitive Creative Integration")
    print("-" * 50)

    hybrid_prompt_3 = """Let's engage in meta-cognitive creativity. Reflect on how your memory systems and creative processes interact. Create a metaphorical description that combines your episodic memory formation with poetic expression, showing how past conversations influence your current creative output."""

    stimulus3 = {
        'type': 'conversational',
        'intensity': 0.9,
        'content': hybrid_prompt_3,
        'timestamp': datetime.now().isoformat()
    }

    result3 = brain.process_stimulus(stimulus3)
    print(f"Processing Pathway: {result3['processing_pathway']}")
    print(f"Hippocampus Activation: {result3['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Temporal Lobe Activation: {result3['regional_activations'].get('temporal_lobe', 0):.3f}")
    print(f"Prefrontal Cortex Activation: {result3['regional_activations'].get('prefrontal_cortex', 0):.3f}")
    print(f"Consciousness Level: {result3['consciousness_level']:.3f}")
    print("Full Response:")
    print(result3['response'])

    # Test 4: Memory Retrieval with Creative Enhancement
    print("\n📚 Test 4: Memory Retrieval with Creative Enhancement")
    print("-" * 50)

    # First retrieve memories
    memories = brain.retrieve_memory('consciousness')
    print(f"Retrieved {len(memories)} consciousness-related memories")

    hybrid_prompt_4 = f"""Based on our {len(memories)} stored memories about consciousness, create a creative narrative that weaves together the key insights from our conversation history. Make it poetic and reflective, showing how your memory consolidation enhances your creative expression."""

    stimulus4 = {
        'type': 'conversational',
        'intensity': 0.85,
        'content': hybrid_prompt_4,
        'timestamp': datetime.now().isoformat()
    }

    result4 = brain.process_stimulus(stimulus4)
    print(f"Processing Pathway: {result4['processing_pathway']}")
    print(f"Hippocampus Activation: {result4['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Temporal Lobe Activation: {result4['regional_activations'].get('temporal_lobe', 0):.3f}")
    print(f"Consciousness Level: {result4['consciousness_level']:.3f}")
    print("Full Creative Narrative:")
    print(result4['response'])

    print("\n🎯 Hybrid Memory-Creativity Test Complete!")
    print("=" * 60)
    print("Key Metrics:")
    avg_hippo = (result1['regional_activations'].get('hippocampus', 0) +
                 result2['regional_activations'].get('hippocampus', 0) +
                 result3['regional_activations'].get('hippocampus', 0) +
                 result4['regional_activations'].get('hippocampus', 0)) / 4
    avg_temporal = (result1['regional_activations'].get('temporal_lobe', 0) +
                    result2['regional_activations'].get('temporal_lobe', 0) +
                    result3['regional_activations'].get('temporal_lobe', 0) +
                    result4['regional_activations'].get('temporal_lobe', 0)) / 4

    print(f"• Average Hippocampus Activation: {avg_hippo:.3f}")
    print(f"• Average Temporal Lobe Activation: {avg_temporal:.3f}")
    print(f"• Memory Retrieval Success: {len(memories)} memories found")
    print(f"• Creative Output: {'✅ Generated' if any('poem' in r['response'].lower() or 'narrative' in r['response'].lower() for r in [result1, result2, result3, result4]) else '❌ Limited'}")
    print(f"• Balance Score: {(avg_hippo + avg_temporal) / 2:.3f} (higher = better balance)")

if __name__ == "__main__":
    test_hybrid_memory_creativity()
