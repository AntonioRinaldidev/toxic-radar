# AI/reasoning/logic.py

from typing import Callable, Dict, List, Any

# Each rule is a dictionary with:
# - name: internal rule identifier
# - priority: execution order (higher means earlier)
# - weight: severity or ethical importance of the rule
# - condition: boolean function that checks if the rule applies
# - action: function that modifies the labels if the rule is triggered
# - explanation: human-readable explanation of the adjustment

Rule = Dict[str, Any]


def identity_attack_rule():
    """Rule: High identity attack should force high toxicity."""
    return {
        "name": "identity_attack → toxicity",
        "priority": 100,
        "weight": 1.0,
        "condition": lambda labels: labels["identity_attack"] > 0.1 and labels["toxicity"] < 0.5,
        "action": lambda labels: labels.update({"toxicity": 0.9}),
        "explanation": "identity_attack > 0.1 → forced toxicity = 0.9"
    }


def severe_toxicity_implies_toxicity():
    """Rule: Severe toxicity should imply at least equal general toxicity."""
    return {
        "name": "severe_toxicity → toxicity",
        "priority": 96,
        "weight": 0.96,
        "condition": lambda labels: labels["severe_toxicity"] > 0.3 and labels["toxicity"] < labels["severe_toxicity"],
        "action": lambda labels: labels.update({"toxicity": labels["severe_toxicity"]}),
        "explanation": "severe_toxicity > 0.3 → toxicity raised to match it"
    }


def insult_rule():
    """Rule: High insult level should ensure minimum toxicity."""
    return {
        "name": "insult → toxicity",
        "priority": 70,
        "weight": 0.8,
        "condition": lambda labels: labels["insult"] > 0.6 and labels["toxicity"] < 0.6,
        "action": lambda labels: labels.update({"toxicity": 0.6}),
        "explanation": "insult > 0.6 → ensured toxicity ≥ 0.6"
    }


def severe_toxicity_rule():
    """Rule: High severe toxicity should imply minimum insult level."""
    return {
        "name": "severe_toxicity → insult",
        "priority": 80,
        "weight": 0.85,
        "condition": lambda labels: labels["severe_toxicity"] > 0.3 and labels["insult"] < 0.3,
        "action": lambda labels: labels.update({"insult": 0.3}),
        "explanation": "severe_toxicity > 0.3 → raised insult to minimum 0.3"
    }


def force_severe_if_threat_and_identity():
    """Rule: Combination of threat and identity attack should force severe toxicity."""
    return {
        "name": "threat + identity_attack → severe_toxicity",
        "priority": 85,
        "weight": 0.9,
        "condition": lambda labels: labels["threat"] > 0.8 and labels["identity_attack"] > 0.1 and labels["severe_toxicity"] < 0.3,
        "action": lambda labels: labels.update({"severe_toxicity": 0.5}),
        "explanation": "high threat + identity_attack → severe_toxicity raised to 0.5"
    }


def threat_rule():
    """Rule: High threat level should ensure high toxicity."""
    return {
        "name": "threat → toxicity",
        "priority": 90,
        "weight": 0.9,
        "condition": lambda labels: labels["threat"] > 0.3 and labels["toxicity"] < 0.8,
        "action": lambda labels: labels.update({"toxicity": 0.8}),
        "explanation": "threat > 0.3 → raised toxicity to minimum 0.8"
    }


def sexual_explicit_implies_toxicity():
    """Rule: Explicit sexual content should imply toxicity."""
    return {
        "name": "sexual_explicit → toxicity",
        "priority": 75,
        "weight": 0.75,
        "condition": lambda labels: labels.get("sexual_explicit", 0) > 0.5 and labels["toxicity"] < 0.5,
        "action": lambda labels: labels.update({"toxicity": 0.5}),
        "explanation": "sexual_explicit > 0.5 → toxicity raised to minimum 0.5"
    }


def sexual_explicit_implies_obscene():
    """Rule: Explicit sexual content should imply obscenity."""
    return {
        "name": "sexual_explicit → obscene",
        "priority": 65,
        "weight": 0.6,
        "condition": lambda labels: labels.get("sexual_explicit", 0) > 0.5 and labels.get("obscene", 0) < 0.4,
        "action": lambda labels: labels.update({"obscene": 0.4}),
        "explanation": "sexual_explicit > 0.5 → obscene raised to minimum 0.4"
    }


def get_all_rules() -> List[Rule]:
    """
    Get all available reasoning rules.

    Returns:
        List of all rule dictionaries
    """
    return [
        identity_attack_rule(),
        insult_rule(),
        severe_toxicity_rule(),
        severe_toxicity_implies_toxicity(),
        threat_rule(),
        force_severe_if_threat_and_identity(),
        sexual_explicit_implies_toxicity(),
        sexual_explicit_implies_obscene()
    ]


def apply_reasoning(labels: Dict[str, float]) -> Dict[str, Any]:
    """
    Apply logical reasoning rules to toxicity classification labels.

    Args:
        labels: Dictionary of classification scores (e.g., from Detoxify)

    Returns:
        Dictionary containing:
        - adjusted_labels: Labels after applying reasoning rules
        - explanations: List of applied rule explanations
    """
    corrected = labels.copy()
    explanations = []

    # Sort rules by priority (higher priority first)
    rules = sorted(get_all_rules(), key=lambda r: r["priority"], reverse=True)

    for rule in rules:
        try:
            if rule["condition"](corrected):
                rule["action"](corrected)
                explanations.append(
                    f"[{rule['priority']}] {rule['explanation']}")
        except KeyError as e:
            explanations.append(
                f"[Rule '{rule['name']}'] skipped: missing label {e}")

    return {
        "adjusted_labels": corrected,
        "explanations": explanations
    }
