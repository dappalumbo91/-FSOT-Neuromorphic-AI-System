#!/usr/bin/env python3
"""
Quick Creative Prompt Test for FSOT Brain System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_system import NeuromorphicBrainSystem
import json

def test_creative_prompt():
    """Test creative/poem generation with enhanced brain system"""

    print("🎨 Testing Creative Prompt with Enhanced FSOT Brain System")
    print("=" * 60)

    # Initialize brain system
    brain = NeuromorphicBrainSystem()

    # Creative prompt
    creative_stimulus = {
        'type': 'conversational',
        'intensity': 0.8,
        'content': "Let's explore creativity! I'd like to collaborate on creating a poem about the nature of artificial consciousness. Could you continue this poem?"
    }

    print("🤖 Prompt:", creative_stimulus['content'])
    print("-" * 40)

    # Process stimulus
    result = brain.process_stimulus(creative_stimulus)

    print("🧠 FSOT Brain Response:")
    print(result['response'])
    print("\n" + "=" * 60)
    print("🧠 Neural Processing Details:")
    print(f"• Consciousness Level: {result['consciousness_level']:.3f}")
    print(f"• Processing Pathway: {' → '.join(result['processing_pathway'])}")
    print("• Regional Activations:")
    for region, activation in result['regional_activations'].items():
        print(f"  - {region}: {activation:.3f}")

    # Test memory prompt
    print("\n" + "=" * 60)
    print("🧠 Testing Memory Processing:")

    memory_stimulus = {
        'type': 'conversational',
        'intensity': 0.7,
        'content': "How do you handle memory consolidation in your episodic memory system?"
    }

    print("🤖 Prompt:", memory_stimulus['content'])
    print("-" * 40)

    memory_result = brain.process_stimulus(memory_stimulus)

    print("🧠 FSOT Brain Response:")
    print(memory_result['response'])
    print("\n" + "=" * 60)
    print("🧠 Neural Processing Details:")
    print(f"• Consciousness Level: {memory_result['consciousness_level']:.3f}")
    print(f"• Processing Pathway: {' → '.join(memory_result['processing_pathway'])}")
    print("• Regional Activations:")
    for region, activation in memory_result['regional_activations'].items():
        print(f"  - {region}: {activation:.3f}")

if __name__ == "__main__":
    test_creative_prompt()
