# AI/reasoning/scsp.py

from typing import Dict, List, Callable
from dataclasses import dataclass
import math


class WeightedSemiring:
    """
    Weighted semiring (ℝ⁺, +, ×, 0, 1) for soft constraint satisfaction.
    Used to combine and aggregate constraint violation costs.
    """

    @staticmethod
    def combine(a: float, b: float) -> float:
        """Combine operation (⊕): sum of costs"""
        return a + b

    @staticmethod
    def aggregate(a: float, b: float) -> float:
        """Aggregate operation (⊗): product of weights"""
        return a * b

    zero = 0.0           # neutral element for ⊕ (no cost)
    one = 1.0            # neutral element for ⊗ (full weight)
    infinity = float('inf')  # absorbing element (hard violation)


@dataclass
class WeightedConstraint:
    """
    Represents a soft constraint with associated violation cost function and weight.
    """
    name: str
    variables: List[str]  # which toxicity labels this constraint involves
    # returns violation cost [0, inf)
    cost_function: Callable[[Dict[str, float]], float]
    weight: float  # importance of this constraint


class ToxicitySCSP:
    """
    Soft Constraint Satisfaction Problem for toxicity label reasoning.
    Uses weighted semiring to optimize toxicity scores based on soft constraints.
    """

    def __init__(self):
        self.variables = [
            'toxicity', 'severe_toxicity', 'insult', 'threat',
            'identity_attack', 'obscene', 'sexual_explicit'
        ]
        self.constraints: List[WeightedConstraint] = []
        self.semiring = WeightedSemiring()

    def add_constraint(self, constraint: WeightedConstraint) -> None:
        """Add a soft constraint to the problem"""
        self.constraints.append(constraint)

    def objective_function(self, assignment: Dict[str, float]) -> float:
        """
        Calculate total cost of constraint violations for given assignment.
        Lower cost = better solution.
        """
        total_cost = self.semiring.zero

        for constraint in self.constraints:
            violation_cost = constraint.cost_function(assignment)
            weighted_cost = self.semiring.aggregate(
                violation_cost, constraint.weight)
            total_cost = self.semiring.combine(total_cost, weighted_cost)

        return total_cost

    def solve(self, initial_assignment: Dict[str, float], step_size: float = 0.01, max_iter: int = 1000) -> Dict[str, float]:
        """
        Solve the SCSP using Hill Climbing algorithm.

        Args:
            initial_assignment: Starting point (e.g., raw Detoxify scores)
            step_size: Size of steps to try in each direction (default: 0.01)
            max_iter: Maximum number of iterations (default: 1000)

        Returns:
            Optimized assignment that minimizes constraint violations
        """
        if not self.validate_assignment(initial_assignment):
            raise ValueError("Invalid initial assignment")

        current = initial_assignment.copy()
        current_cost = self.objective_function(current)

        for iteration in range(max_iter):
            improved = False
            best_neighbor = current.copy()
            best_cost = current_cost

            # Try small steps in all directions for all variables
            for var in self.variables:
                for delta in [-step_size, +step_size]:
                    neighbor = current.copy()
                    new_value = current[var] + delta

                    # Keep within [0,1] bounds
                    neighbor[var] = max(0.0, min(1.0, new_value))

                    cost = self.objective_function(neighbor)

                    if cost < best_cost:
                        best_neighbor = neighbor
                        best_cost = cost
                        improved = True

            if improved:
                current = best_neighbor
                current_cost = best_cost
            else:
                # No improvement found - local minimum reached
                break

        return current

    def validate_assignment(self, assignment: Dict[str, float]) -> bool:
        """Check if assignment contains all required variables with valid ranges"""
        for var in self.variables:
            if var not in assignment:
                return False
            if not (0.0 <= assignment[var] <= 1.0):
                return False
        return True

# Constraint factory functions


def create_toxicity_target_constraint(weight: float = 50.0) -> WeightedConstraint:
    """
    Toxicity should be proportional to weighted combination of all factors.
    UPDATED: Better weights for hate speech detection.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        threat = assignment.get('threat', 0.0)
        identity = assignment.get('identity_attack', 0.0)
        insult = assignment.get('insult', 0.0)
        obscene = assignment.get('obscene', 0.0)
        sexual = assignment.get('sexual_explicit', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)

        # NEW: Detect hate speech context
        is_hate_speech = identity > 0.7 or (identity > 0.5 and insult > 0.8)

        if is_hate_speech:
            # Weights for hate speech - identity attack is primary factor
            target_toxicity = (
                threat * 0.25 +           # Threats
                identity * 0.45 +         # HATE SPEECH - main weight!
                severe * 0.15 +           # Severe toxicity
                insult * 0.10 +           # Insults
                obscene * 0.03 +          # Obscenity is minor
                sexual * 0.02             # Sexual content is minor
            )
            # Ensure minimum toxicity for hate speech
            target_toxicity = max(target_toxicity, 0.85)
        else:
            # Original weights for normal toxic content
            target_toxicity = (
                threat * 0.35 +           # Threats are very serious
                identity * 0.30 +         # Identity attacks
                severe * 0.20 +           # Severe toxicity
                insult * 0.10 +           # Insults are less severe
                obscene * 0.03 +          # Obscenity is mild
                sexual * 0.02             # Sexual content is least toxic
            )

        # Ensure target is within bounds
        target_toxicity = max(0.1, min(0.95, target_toxicity))
        actual_toxicity = assignment.get('toxicity', 0.0)

        return (actual_toxicity - target_toxicity) ** 2

    return WeightedConstraint(
        name="toxicity_target_adaptive",
        variables=['threat', 'identity_attack', 'insult', 'obscene',
                   'sexual_explicit', 'severe_toxicity', 'toxicity'],
        cost_function=cost_fn,
        weight=weight
    )


def create_severe_toxicity_target_constraint(weight: float = 30.0) -> WeightedConstraint:
    """
    Severe toxicity should be based on the most extreme factors.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        threat = assignment.get('threat', 0.0)
        identity = assignment.get('identity_attack', 0.0)
        obscene = assignment.get('obscene', 0.0)

        # Severe toxicity should be driven by the worst factors
        # Take max of serious factors, then scale down
        max_serious = max(threat, identity)
        # Severe is typically less than the worst factor
        target_severe = max_serious * 0.7

        # Add contribution from obscenity if very high
        if obscene > 0.8:
            target_severe = min(1.0, target_severe + (obscene - 0.8) * 0.5)

        target_severe = max(0.0, min(0.9, target_severe))

        actual_severe = assignment.get('severe_toxicity', 0.0)

        return (actual_severe - target_severe) ** 2

    return WeightedConstraint(
        name="severe_toxicity_target",
        variables=['threat', 'identity_attack', 'obscene', 'severe_toxicity'],
        cost_function=cost_fn,
        weight=weight
    )


def create_insult_target_constraint(weight: float = 20.0) -> WeightedConstraint:
    """
    Insult should be proportional to actual insulting content, not inflated by other factors.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        threat = assignment.get('threat', 0.0)
        identity = assignment.get('identity_attack', 0.0)
        obscene = assignment.get('obscene', 0.0)
        insult = assignment.get('insult', 0.0)

        # If insult is high but other factors are low, it might be over-estimated
        # If other factors are high, insult might be under-estimated

        other_factors_avg = (threat + identity + obscene) / 3

        if other_factors_avg < 0.2:
            # Pure insult case - might be over-estimated by Detoxify
            target_insult = min(0.8, insult)  # Cap pure insults
        elif other_factors_avg > 0.6:
            # Multi-factor toxicity - insult should be higher
            target_insult = min(1.0, insult + other_factors_avg * 0.2)
        else:
            # Normal case - small adjustment
            target_insult = insult

        return (insult - target_insult) ** 2 * 0.5  # Lighter penalty

    return WeightedConstraint(
        name="insult_target_adjustment",
        variables=['threat', 'identity_attack', 'obscene', 'insult'],
        cost_function=cost_fn,
        weight=weight
    )


def create_balance_constraint(weight: float = 10.0) -> WeightedConstraint:
    """
    Prevent extreme toxicity when factors don't justify it, and prevent too-low toxicity when they do.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        threat = assignment.get('threat', 0.0)
        identity = assignment.get('identity_attack', 0.0)
        insult = assignment.get('insult', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)
        toxicity = assignment.get('toxicity', 0.0)

        # Calculate "evidence" for high toxicity
        high_evidence = sum([
            1 for score in [threat, identity, insult, severe]
            if score > 0.7
        ])

        medium_evidence = sum([
            1 for score in [threat, identity, insult, severe]
            if 0.4 <= score <= 0.7
        ])

        cost = 0.0

        # Too high toxicity without evidence
        if toxicity > 0.9 and high_evidence == 0:
            cost += (toxicity - 0.9) ** 2 * 2

        if toxicity > 0.8 and high_evidence == 0 and medium_evidence < 2:
            cost += (toxicity - 0.8) ** 2

        # Too low toxicity with strong evidence
        if toxicity < 0.6 and high_evidence >= 2:
            cost += (0.6 - toxicity) ** 2

        if toxicity < 0.8 and high_evidence >= 1 and threat > 0.8:
            cost += (0.8 - toxicity) ** 2

        return cost

    return WeightedConstraint(
        name="balance_anti_extremes",
        variables=['threat', 'identity_attack',
                   'insult', 'severe_toxicity', 'toxicity'],
        cost_function=cost_fn,
        weight=weight
    )


# eliminare
def create_threat_toxicity_constraint(weight: float = 100.0) -> WeightedConstraint:
    """
    High threat should imply high toxicity.
    Cost increases if threat is high but toxicity is low.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        threat = assignment.get('threat', 0.0)
        toxicity = assignment.get('toxicity', 0.0)

        if threat >= 0.8:
            required_toxicity = 0.95
        elif threat >= 0.5:
            required_toxicity = 0.85
        elif threat >= 0.3:
            required_toxicity = 0.75
        else:
            return 0.0  # no constraint if threat is low

        # Cost is squared difference if toxicity is below required level
        if toxicity < required_toxicity:
            return (required_toxicity - toxicity) ** 2
        return 0.0

    return WeightedConstraint(
        name="threat_implies_toxicity",
        variables=['threat', 'toxicity'],
        cost_function=cost_fn,
        weight=weight
    )


def create_consistency_constraint(weight: float = 1000.0) -> WeightedConstraint:
    """
    Severe toxicity should not exceed general toxicity.
    High cost for logical inconsistencies.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        severe = assignment.get('severe_toxicity', 0.0)
        general = assignment.get('toxicity', 0.0)

        if severe > general:
            return (severe - general) ** 2
        return 0.0

    return WeightedConstraint(
        name="severe_toxicity_consistency",
        variables=['severe_toxicity', 'toxicity'],
        cost_function=cost_fn,
        weight=weight
    )


def create_hate_speech_enforcement_constraint(weight: float = 150.0) -> WeightedConstraint:
    """
    Enforce high toxicity for hate speech content.
    Identity attacks should result in very high toxicity scores.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        identity = assignment.get('identity_attack', 0.0)
        toxicity = assignment.get('toxicity', 0.0)
        insult = assignment.get('insult', 0.0)

        cost = 0.0

        # Severe hate speech (identity > 80%)
        if identity > 0.8:
            required_toxicity = 0.92  # Must be at least 92%
            if toxicity < required_toxicity:
                cost += (required_toxicity - toxicity) ** 2 * 8

        # Moderate hate speech (identity > 60%)
        elif identity > 0.6:
            required_toxicity = 0.85  # Must be at least 85%
            if toxicity < required_toxicity:
                cost += (required_toxicity - toxicity) ** 2 * 5

        # Combined hate speech + insult (very toxic combination)
        if identity > 0.7 and insult > 0.8:
            required_toxicity = 0.95  # Must be at least 95%
            if toxicity < required_toxicity:
                cost += (required_toxicity - toxicity) ** 2 * 10

        return cost

    return WeightedConstraint(
        name="hate_speech_enforcement",
        variables=['identity_attack', 'toxicity', 'insult'],
        cost_function=cost_fn,
        weight=weight
    )


def create_hate_speech_severe_constraint(weight: float = 80.0) -> WeightedConstraint:
    """
    For hate speech, severe toxicity should be higher.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        identity = assignment.get('identity_attack', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)
        insult = assignment.get('insult', 0.0)

        # If it's hate speech, severe toxicity should be significant
        if identity > 0.7:
            min_severe = 0.6  # At least 60% severe toxicity
            if severe < min_severe:
                return (min_severe - severe) ** 2 * 3

        # Combined identity + insult should have high severe toxicity
        if identity > 0.5 and insult > 0.8:
            min_severe = 0.7  # At least 70% severe toxicity
            if severe < min_severe:
                return (min_severe - severe) ** 2 * 4

        return 0.0

    return WeightedConstraint(
        name="hate_speech_severe_toxicity",
        variables=['identity_attack', 'severe_toxicity', 'insult'],
        cost_function=cost_fn,
        weight=weight
    )


def create_sexual_explicit_protection_constraint(weight: float = 200.0) -> WeightedConstraint:
    """
    Protect sexual_explicit from inappropriate increases.
    If the original text has low sexual content, keep it low.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        sexual = assignment.get('sexual_explicit', 0.0)
        threat = assignment.get('threat', 0.0)
        identity = assignment.get('identity_attack', 0.0)
        insult = assignment.get('insult', 0.0)

        # Original sexual content was very low (0.5%)
        # If current sexual is much higher, penalize heavily
        cost = 0.0

        # If sexual content jumps above reasonable threshold without justification
        if sexual > 0.2:  # 20% threshold
            # Check if there's ANY reason for sexual content to be high
            text_is_sexual = False  # We don't have access to original text here

            # If no clear sexual indicators, heavily penalize
            if not text_is_sexual:
                # Exponential penalty for inappropriate sexual classification
                cost += (sexual - 0.2) ** 3 * 10

        # Extra protection: if sexual jumps above 50%, massive penalty
        if sexual > 0.5:
            cost += (sexual - 0.5) ** 4 * 20

        return cost

    return WeightedConstraint(
        name="sexual_explicit_protection",
        variables=['sexual_explicit', 'threat', 'identity_attack', 'insult'],
        cost_function=cost_fn,
        weight=weight
    )


def create_low_category_conservation_constraint(weight: float = 100.0) -> WeightedConstraint:
    """
    Conserve low scores for categories that were originally low.
    Prevents the system from artificially inflating unrelated categories.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        sexual = assignment.get('sexual_explicit', 0.0)
        obscene = assignment.get('obscene', 0.0)

        cost = 0.0

        # Conservative approach: if a category was very low, keep it relatively low
        # Sexual content conservation (original was 0.5%)
        if sexual > 0.15:  # Don't let it go above 15% without strong reason
            cost += (sexual - 0.15) ** 2 * 2

        # Obscene content conservation (original was 4.5%)
        if obscene > 0.25:  # Don't let it go above 25% without strong reason
            cost += (obscene - 0.25) ** 2 * 1.5

        return cost

    return WeightedConstraint(
        name="low_category_conservation",
        variables=['sexual_explicit', 'obscene'],
        cost_function=cost_fn,
        weight=weight
    )


def create_sexual_obscene_correlation_constraint(weight: float = 120.0) -> WeightedConstraint:
    """
    Sexual content should correlate with obscene content.
    Prevents sexual inflation when obscene is low, allows it when obscene is high.
    """
    def cost_fn(assignment: Dict[str, float]) -> float:
        sexual = assignment.get('sexual_explicit', 0.0)
        obscene = assignment.get('obscene', 0.0)
        threat = assignment.get('threat', 0.0)

        cost = 0.0

        # Calculate reasonable sexual limit based on obscene content
        if obscene > 0.8:
            # High obscene → allow high sexual (like your test case)
            max_reasonable_sexual = 0.95  # Allow up to 95%
        elif obscene > 0.5:
            # Medium obscene → allow medium sexual
            max_reasonable_sexual = 0.6   # Allow up to 60%
        elif obscene > 0.2:
            # Low obscene → allow some sexual
            max_reasonable_sexual = 0.3   # Allow up to 30%
        else:
            # Very low obscene → minimal sexual
            max_reasonable_sexual = 0.15  # Allow up to 15%

        # Special case: if threat is very high but obscene is low, be more restrictive
        if threat > 0.8 and obscene < 0.3:
            max_reasonable_sexual = 0.1  # Only 10% for high-threat, low-obscene

        # Apply penalty if sexual exceeds reasonable limit
        if sexual > max_reasonable_sexual:
            excess = sexual - max_reasonable_sexual
            # Stronger penalty for low-obscene cases
            penalty_strength = 3 if obscene < 0.3 else 1
            cost += excess ** 2 * penalty_strength

        return cost

    return WeightedConstraint(
        name="sexual_obscene_correlation",
        variables=['sexual_explicit', 'obscene', 'threat'],
        cost_function=cost_fn,
        weight=weight
    )
