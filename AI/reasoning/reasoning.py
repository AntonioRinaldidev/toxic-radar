# AI/reasoning/reasoning.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from AI.reasoning.scsp import (
    ToxicitySCSP,
    create_threat_detection_constraint,
    create_severe_toxicity_boost_constraint,
    create_consistency_constraint,
    create_hate_speech_constraint,
    create_sexual_protection_constraint, create_toxicity_safety_constraint, create_value_protection_constraint, create_anti_artificial_increase_constraint, create_sexual_violence_constraint, create_low_toxicity_constraint, create_context_constraint, create_quality_constraint
)

app = FastAPI(title="ToxicRadar Reasoning Module - SCSP Enhanced")


class LabelRequest(BaseModel):
    labels: Dict[str, float]


# Global SCSP instance
_scsp_instance = None


def get_scsp_instance():
    """Get or create enhanced SCSP instance"""
    global _scsp_instance

    if _scsp_instance is None:
        _scsp_instance = ToxicitySCSP()

        _scsp_instance.add_constraint(
            create_toxicity_safety_constraint(weight=2000))
        _scsp_instance.add_constraint(
            create_consistency_constraint(weight=1200))
        _scsp_instance.add_constraint(
            create_low_toxicity_constraint(weight=1000))
        _scsp_instance.add_constraint(
            create_value_protection_constraint(weight=800))
        _scsp_instance.add_constraint(
            create_sexual_violence_constraint(weight=500))
        _scsp_instance.add_constraint(
            create_sexual_protection_constraint(weight=500))
        _scsp_instance.add_constraint(create_context_constraint(weight=400))
        _scsp_instance.add_constraint(
            create_hate_speech_constraint(weight=400))
        _scsp_instance.add_constraint(
            create_anti_artificial_increase_constraint(weight=350))
        _scsp_instance.add_constraint(
            create_threat_detection_constraint(weight=300))
        _scsp_instance.add_constraint(
            create_severe_toxicity_boost_constraint(weight=250))
        _scsp_instance.add_constraint(create_quality_constraint(weight=150))

        print("✅ Enhanced Hill Climbing SCSP initialized with threat detection")

    return _scsp_instance


def apply_reasoning(labels: Dict[str, float]) -> Dict[str, any]:
    """
    Apply SCSP-based reasoning to toxicity classification labels.

    Args:
        labels: Dictionary of classification scores (e.g., from Detoxify)

    Returns:
        Dictionary containing:
        - adjusted_labels: Labels after applying SCSP optimization
        - explanations: List of optimization details
    """
    try:
        scsp = get_scsp_instance()

        # Calculate initial cost
        initial_cost = scsp.objective_function(labels)

        # Optimize using SCSP
        optimized_labels = scsp.solve(labels)

        # Calculate final cost
        final_cost = scsp.objective_function(optimized_labels)

        # Generate explanations
        explanations = []

        # Check which labels changed significantly
        for label, original_value in labels.items():
            optimized_value = optimized_labels.get(label, original_value)
            difference = abs(optimized_value - original_value)

            if difference > 0.05:  # Significant change threshold
                direction = "increased" if optimized_value > original_value else "decreased"
                explanations.append(
                    f"SCSP optimization: {label} {direction} from {original_value:.3f} to {optimized_value:.3f}"
                )

        if not explanations:
            explanations.append("SCSP optimization: minimal adjustments made")

        # Add cost information
        explanations.append(
            f"Optimization cost: {initial_cost:.3f} → {final_cost:.3f}")

        return {
            "adjusted_labels": optimized_labels,
            "explanations": explanations
        }

    except Exception as e:
        # Fallback: return original labels if SCSP fails
        return {
            "adjusted_labels": labels,
            "explanations": [f"SCSP failed, using original labels: {str(e)}"]
        }


@app.post("/reason")
def reason_labels(data: LabelRequest):
    """Apply SCSP-based reasoning to labels"""
    try:
        result = apply_reasoning(data.labels)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Health check with SCSP info"""
    scsp = get_scsp_instance()
    return {
        "status": "healthy",
        "service": "ToxicRadar Reasoning Module - SCSP Enhanced",
        "constraints": len(scsp.constraints),
        "constraint_names": [c.name for c in scsp.constraints]
    }


@app.get("/scsp_info")
def get_scsp_info():
    """Get information about the SCSP system"""
    scsp = get_scsp_instance()
    return {
        "system": "Soft Constraint Satisfaction Problem",
        "semiring": "Weighted (ℝ⁺, +, ×, 0, 1)",
        "algorithm": "Hill Climbing",
        "variables": scsp.variables,
        "constraints": [
            {
                "name": c.name,
                "weight": c.weight,
                "variables": c.variables
            }
            for c in scsp.constraints
        ]
    }
