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
    return {
        "name": "identity_attack → toxicity",
        "priority": 100,
        "weight": 1.0,
        "condition": lambda l: l["identity_attack"] > 0.1 and l["toxicity"] < 0.5,
        "action": lambda l: l.update({"toxicity": 0.9}),
        "explanation": "identity_attack > 0.1 → forced toxicity = 0.9"
    }

def severe_toxicity_implies_toxicity():
    return {
        "name": "severe_toxicity → toxicity",
        "priority": 96,
        "weight": 0.96,
        "condition": lambda l: l["severe_toxicity"] > 0.3 and l["toxicity"] < l["severe_toxicity"],
        "action": lambda l: l.update({"toxicity": l["severe_toxicity"]}),
        "explanation": "severe_toxicity > 0.3 → toxicity raised to match it"
    }

def insult_rule():
    return {
        "name": "insult → toxicity",
        "priority": 70,
        "weight": 0.8,
        "condition": lambda l: l["insult"] > 0.6 and l["toxicity"] < 0.6,
        "action": lambda l: l.update({"toxicity": 0.6}),
        "explanation": "insult > 0.6 → ensured toxicity ≥ 0.6"
    }

def severe_toxicity_rule():
    return {
        "name": "severe_toxicity → insult",
        "priority": 80,
        "weight": 0.85,
        "condition": lambda l: l["severe_toxicity"] > 0.3 and l["insult"] < 0.3,
        "action": lambda l: l.update({"insult": 0.3}),
        "explanation": "severe_toxicity > 0.3 → raised insult to minimum 0.3"
    }

def force_severe_if_threat_and_identity():
    return {
        "name": "threat + identity_attack → severe_toxicity",
        "priority": 85,
        "weight": 0.9,
        "condition": lambda l: l["threat"] > 0.8 and l["identity_attack"] > 0.1 and l["severe_toxicity"] < 0.3,
        "action": lambda l: l.update({"severe_toxicity": 0.5}),
        "explanation": "high threat + identity_attack → severe_toxicity raised to 0.5"
    }

def threat_rule():
    return {
        "name": "threat → toxicity",
        "priority": 90,
        "weight": 0.9,
        "condition": lambda l: l["threat"] > 0.3 and l["toxicity"] < 0.8,
        "action": lambda l: l.update({"toxicity": 0.8}),
        "explanation": "threat > 0.3 → raised toxicity to minimum 0.8"
    }
def sexual_explicit_implies_toxicity():
    return {
        "name": "sexual_explicit → toxicity",
        "priority": 75,
        "weight": 0.75,
        "condition": lambda l: l.get("sexual_explicit", 0) > 0.5 and l["toxicity"] < 0.5,
        "action": lambda l: l.update({"toxicity": 0.5}),
        "explanation": "sexual_explicit > 0.5 → toxicity raised to minimum 0.5"
    }
def sexual_explicit_implies_obscene():
    return {
        "name": "sexual_explicit → obscene",
        "priority": 65,
        "weight": 0.6,
        "condition": lambda l: l.get("sexual_explicit", 0) > 0.5 and l.get("obscene", 0) < 0.4,
        "action": lambda l: l.update({"obscene": 0.4}),
        "explanation": "sexual_explicit > 0.5 → obscene raised to minimum 0.4"
    }


def get_all_rules() -> List[Rule]:
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
    corrected = labels.copy()
    explanations = []

    rules = sorted(get_all_rules(), key=lambda r: r["priority"], reverse=True)

    for rule in rules:
        try:
            if rule["condition"](corrected):
                rule["action"](corrected)
                explanations.append(f"[{rule['priority']}] {rule['explanation']}")
        except KeyError as e:
            explanations.append(f"[Rule '{rule['name']}'] skipped: missing label {e}")

    return {
        "adjusted_labels": corrected,
        "explanations": explanations
    }
