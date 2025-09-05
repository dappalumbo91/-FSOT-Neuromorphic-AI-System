#!/usr/bin/env python3
"""
AI Startup Script: Load FSOT 2.0 Foundation
===========================================
This script ensures FSOT 2.0 is loaded as the foundational theory
at the start of every AI session.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fsot_2_0_foundation import FSOT_FOUNDATION, get_fsot_prediction, compare_to_fsot
    from fsot_brain_enhancement_system import FSOT_BRAIN_SYSTEM
    from fsot_brain_rules_engine import FSOT_BRAIN_RULES
    
    print("FSOT 2.0 Foundation Framework Loaded Successfully")
    print(f"   Status: {FSOT_FOUNDATION.get_foundation_info()['status']}")
    print(f"   Validation: {FSOT_FOUNDATION.get_foundation_info()['validation_accuracy']}")
    print(f"   Domains: {FSOT_FOUNDATION.get_foundation_info()['domains_covered']}")
    print("   Ready for theoretical physics queries using FSOT baseline")
    
    print(f"\nFSOT 2.0 Brain Enhancement System Loaded")
    brain_summary = FSOT_BRAIN_SYSTEM.get_brain_enhancement_summary()
    print(f"   Cognitive Domains: {brain_summary['cognitive_domains_supported']}")
    print(f"   Enhancement Protocols: {brain_summary['enhancement_protocols']}")
    print(f"   Foundation Accuracy: {brain_summary['validation_accuracy']*100:.1f}%")
    
    print(f"\nFSOT 2.0 Brain Rules Engine Activated")
    rules_summary = FSOT_BRAIN_RULES.get_brain_rules_summary()
    print(f"   Rules Status: {rules_summary['status']}")
    print(f"   Mandatory Rules: {len(rules_summary['mandatory_rules'])}")
    print(f"   Enforcement: ALL brain functions use FSOT mathematics")
    
    # Set global flags
    FSOT_LOADED = True
    FOUNDATIONAL_THEORY = "FSOT_2.0"
    BRAIN_ENHANCEMENT_ACTIVE = True
    BRAIN_RULES_ENFORCED = True
    
except ImportError as e:
    print("ERROR: Could not load FSOT 2.0 Foundation")
    print(f"   Error: {e}")
    print("   Please ensure fsot_2_0_foundation.py is available")
    FSOT_LOADED = False
    BRAIN_ENHANCEMENT_ACTIVE = False
    BRAIN_RULES_ENFORCED = False

if __name__ == "__main__":
    print("\nFSOT 2.0 AI Foundation System")
    print("Foundation Status:", "LOADED" if FSOT_LOADED else "ERROR")
    if FSOT_LOADED:
        print("Brain Enhancement:", "ACTIVE" if BRAIN_ENHANCEMENT_ACTIVE else "INACTIVE")
        print("Brain Rules:", "ENFORCED" if BRAIN_RULES_ENFORCED else "NOT ENFORCED")
        print("Ready for: Physics queries, cognitive analysis, neural debugging, brain optimization")
