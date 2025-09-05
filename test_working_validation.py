#!/usr/bin/env python3
"""
Test the working 100% FSOT implementation with enhanced validation
"""

from enhanced_fsot_validation import validate_code_integration

# Test enhanced working implementation
with open('enhanced_working_100_fsot.py', 'r', encoding='utf-8') as f:
    working_code = f.read()

result = validate_code_integration(working_code)

print("ENHANCED WORKING 100% FSOT VALIDATION TEST")
print("=" * 40)
print(f"Integration Score: {result['integration_percentage']:.1f}%")
print(f"Points: {result['integration_score']}/{result['max_possible_score']}")
print(f"Compliance: {result['compliance_level']}")

print("\nCategory Breakdown:")
for cat, scores in result['category_breakdown'].items():
    print(f"  {cat}: {scores['score']}/{scores['max']} ({scores['percentage']:.0f}%)")
    for indicator in scores['indicators'][:2]:
        print(f"    {indicator}")

if result['recommendations']:
    print("\nRecommendations:")
    for rec in result['recommendations'][:3]:
        print(f"  • {rec}")
        
print(f"\n🎯 RESULT: {'✅ PERFECT!' if result['integration_percentage'] >= 95 else '🟡 GOOD' if result['integration_percentage'] >= 85 else '🔴 NEEDS WORK'}")
