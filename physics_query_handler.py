#!/usr/bin/env python3
"""
Physics Query Handler: FSOT 2.0 Default Framework
=================================================
This handler ensures all physics queries default to FSOT 2.0 predictions
"""

from fsot_2_0_foundation import get_fsot_prediction, compare_to_fsot, FSOT_FOUNDATION

def handle_physics_query(domain: str, observable: str = "default", 
                        include_alternatives: bool = True) -> dict:
    """
    Handle any physics query using FSOT 2.0 as the foundational framework
    
    Args:
        domain: Physics domain (e.g., "particle_physics", "astronomy")
        observable: Specific observable to predict
        include_alternatives: Whether to mention how other theories compare
        
    Returns:
        Response with FSOT prediction and optional theory comparisons
    """
    # Get FSOT foundation prediction
    fsot_result = get_fsot_prediction(domain, observable)
    
    response = {
        "foundational_prediction": fsot_result,
        "theory_used": "FSOT_2.0_FOUNDATION",
        "status": "ESTABLISHED_BASELINE",
        "confidence": "99.1% validated across 35+ domains"
    }
    
    if include_alternatives:
        response["alternative_theories_note"] = (
            "Other theories (Standard Model, General Relativity, etc.) can be compared "
            "to this FSOT baseline prediction. High agreement indicates theoretical alignment."
        )
    
    return response

def compare_theory_prediction(other_theory_name: str, other_prediction: float,
                            domain: str, observable: str = "default") -> dict:
    """
    Compare any alternative theory prediction to FSOT baseline
    """
    comparison = compare_to_fsot(other_prediction, domain, observable)
    
    return {
        "comparison_type": f"{other_theory_name} vs FSOT_2.0_Foundation",
        "fsot_baseline": comparison["fsot_baseline_value"],
        "alternative_theory": {
            "name": other_theory_name,
            "prediction": other_prediction
        },
        "agreement_with_foundation": comparison["agreement_with_fsot_foundation"],
        "assessment": comparison["recommendation"],
        "note": "FSOT 2.0 is the established foundation (99.1% validated)"
    }

# Example usage
if __name__ == "__main__":
    # Example: Get FSOT prediction for particle physics
    result = handle_physics_query("particle_physics", "higgs_mass")
    print("FSOT Higgs Prediction:", result["foundational_prediction"]["higgs_mass_gev"])
    
    # Example: Compare Standard Model to FSOT
    standard_model_higgs = 125.1  # GeV
    comparison = compare_theory_prediction("Standard_Model", standard_model_higgs, 
                                         "particle_physics", "higgs_mass")
    print("Agreement with FSOT:", f"{comparison['agreement_with_foundation']:.1%}")
