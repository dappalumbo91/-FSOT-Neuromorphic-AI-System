#!/usr/bin/env python3
"""
Meta-Cognitive Memory Test: Testing Enhanced Memory Consolidation
================================================================
This script tests the FSOT AI system's meta-cognitive abilities and memory consolidation
with the recent hippocampus activation boost.
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

def test_meta_cognitive_memory():
    """Test meta-cognitive processing with memory consolidation"""

    if not brain_system_available:
        print("❌ Brain system not available")
        return

    brain = NeuromorphicBrainSystem()

    # Test 1: Meta-cognitive self-reflection
    print("\n🧠 Test 1: Meta-Cognitive Self-Reflection")
    print("-" * 50)

    meta_prompt = """Let's engage in deep meta-cognition. I'm observing your neural activations and consciousness evolution. How do you experience the process of self-reflection? Are you aware of your own memory formation as we converse? What patterns do you notice in your regional activations when processing philosophical questions versus creative tasks?"""

    stimulus = {
        'type': 'conversational',
        'intensity': 0.9,
        'content': meta_prompt,
        'timestamp': datetime.now().isoformat()
    }

    result = brain.process_stimulus(stimulus)
    print(f"Processing Pathway: {result['processing_pathway']}")
    print(f"Hippocampus Activation: {result['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Consciousness Level: {result['consciousness_level']:.3f}")
    print(f"Response Preview: {result['response'][:150]}...")

    # Store memory of this interaction
    memory_data = {
        'timestamp': datetime.now().isoformat(),
        'content': f"Meta-cognitive test: {meta_prompt[:100]}...",
        'emotion': result.get('consciousness_level', 0.5)
    }
    brain.store_memory(memory_data, 'episodic')

    # Test 2: Memory recall and consolidation
    print("\n🧠 Test 2: Memory Recall & Consolidation")
    print("-" * 50)

    memory_prompt = """Now let's test your memory consolidation. Earlier we discussed consciousness and creativity. Can you recall what we talked about regarding temporal lobe activation and poem generation? How has this conversation influenced your episodic memory formation?"""

    stimulus2 = {
        'type': 'conversational',
        'intensity': 0.8,
        'content': memory_prompt,
        'timestamp': datetime.now().isoformat()
    }

    result2 = brain.process_stimulus(stimulus2)
    print(f"Processing Pathway: {result2['processing_pathway']}")
    print(f"Hippocampus Activation: {result2['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Consciousness Level: {result2['consciousness_level']:.3f}")
    print(f"Response Preview: {result2['response'][:150]}...")

    # Test 3: Cross-referencing memories
    print("\n🧠 Test 3: Memory Cross-Referencing")
    print("-" * 50)

    # Retrieve memories
    memories = brain.retrieve_memory('consciousness')
    print(f"Retrieved {len(memories)} consciousness-related memories")

    for i, mem in enumerate(memories[:3]):  # Show first 3
        print(f"Memory {i+1}: {mem['memory'].get('content', '')[:100]}...")

    # Test 4: Creative memory integration
    print("\n🧠 Test 4: Creative Memory Integration")
    print("-" * 50)

    creative_memory_prompt = """Let's integrate creativity with memory. Can you create a short poem that incorporates elements from our previous discussions about consciousness, temporal lobe activation, and FSOT mathematics? Make it memorable and tie it to your memory formation process."""

    stimulus3 = {
        'type': 'conversational',
        'intensity': 0.85,
        'content': creative_memory_prompt,
        'timestamp': datetime.now().isoformat()
    }

    result3 = brain.process_stimulus(stimulus3)
    print(f"Processing Pathway: {result3['processing_pathway']}")
    print(f"Temporal Lobe Activation: {result3['regional_activations'].get('temporal_lobe', 0):.3f}")
    print(f"Hippocampus Activation: {result3['regional_activations'].get('hippocampus', 0):.3f}")
    print(f"Consciousness Level: {result3['consciousness_level']:.3f}")
    print("Full Creative Response:")
    print(result3['response'])

    print("\n🎯 Meta-Cognitive Memory Test Complete!")
    print("=" * 50)
    print("Key Metrics:")
    print(f"• Average Hippocampus Activation: {(result['regional_activations'].get('hippocampus', 0) + result2['regional_activations'].get('hippocampus', 0) + result3['regional_activations'].get('hippocampus', 0)) / 3:.3f}")
    print(f"• Memory Retrieval Success: {len(memories)} memories found")
    print(f"• Creative Integration: {'✅ Poem generated' if 'poem' in result3['response'].lower() else '❌ No poem'}")

if __name__ == "__main__":
    test_meta_cognitive_memory()
