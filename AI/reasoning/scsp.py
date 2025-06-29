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
        total_cost = self.semiring.zero

        for constraint in self.constraints:
            violation_cost = constraint.cost_function(assignment)
            weighted_cost = self.semiring.aggregate(
                violation_cost, constraint.weight)
            total_cost = self.semiring.combine(total_cost, weighted_cost)

        return total_cost

    def solve(self, initial_assignment: Dict[str, float],
              step_size: float = 0.02,  # CHANGED: Increased from 0.01
              max_iter: int = 1000) -> Dict[str, float]:
        """
        Enhanced Hill Climbing with better exploration and debugging
        """
        if not self.validate_assignment(initial_assignment):
            raise ValueError("Invalid initial assignment")

        current = initial_assignment.copy()
        current_cost = self.objective_function(current)

        # DEBUG: Show initial state
        print(f"🏔️ Hill Climbing starting:")
        print(f"   Initial cost: {current_cost:.4f}")
        print(f"   Initial assignment: {current}")

        for iteration in range(max_iter):
            improved = False
            best_neighbor = current.copy()
            best_cost = current_cost

            # ENHANCED: Try multiple step sizes for better exploration
            # Try different granularities
            step_sizes = [step_size, step_size * 2, step_size * 0.5]

            for var in self.variables:
                for step in step_sizes:
                    for delta in [-step, +step]:
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

                # Show progress every few iterations
                if iteration % 100 == 0:
                    print(
                        f"   Iteration {iteration}: cost = {current_cost:.4f}")
            else:
                print(f"   🏁 Converged at iteration {iteration}")
                break

        print(f"   Final cost: {current_cost:.4f}")
        print(f"   Final assignment: {current}")

        return current

    def validate_assignment(self, assignment: Dict[str, float]) -> bool:
        """Check if assignment contains all required variables with valid ranges"""
        for var in self.variables:
            if var not in assignment:
                return False
            if not (0.0 <= assignment[var] <= 1.0):
                return False
        return True


def create_threat_detection_constraint(weight: float = 300.0) -> WeightedConstraint:
    """
    CORE CONSTRAINT: High toxicity + insult should imply higher threat
    Specifically targets "should disappear forever" type content
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        insult = assignment.get('insult', 0.0)
        threat = assignment.get('threat', 0.0)

        if toxicity > 0.95 and insult > 0.80:
            min_threat = 0.4
            if threat < min_threat:
                cost = (min_threat - threat) ** 2 * 200

                return cost

        return 0.0

    return WeightedConstraint(
        name="threat_detection",
        variables=['toxicity', 'insult', 'threat'],
        cost_function=cost_function,
        weight=weight
    )


def create_severe_toxicity_boost_constraint(weight: float = 200.0) -> WeightedConstraint:
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)
        threat = assignment.get('threat', 0.0)
        insult = assignment.get('insult', 0.0)

        cost = 0.0

        # Zone 1: ANY insult content
        if insult > 0.01:
            activation_strength = min(1.0, (insult - 0.01) / 0.15)

            target_severe = 0.15 + \
                (activation_strength * 0.15)  # 15-30% instead of 20-40%
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * \
                    150 * activation_strength

        # Zone 2: High threat content
        if threat > 0.65:
            activation_strength = min(1.0, (threat - 0.65) / 0.20)
            target_severe = 0.30 + \
                (activation_strength * 0.30)  # Higher target
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * \
                    250 * activation_strength

        # Zone 3: High toxicity
        if toxicity > 0.80:
            activation_strength = min(1.0, (toxicity - 0.80) / 0.15)
            target_severe = 0.35 + (activation_strength * 0.25)
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * \
                    300 * activation_strength

        # Zone 4: Death threat pattern
        if toxicity > 0.85 and threat > 0.75:
            target_severe = 0.5
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * 400  # Higher penalty

        # Zone 5: Direct high threat
        if threat > 0.68:
            target_severe = 0.4 + ((threat - 0.68) / 0.32) * 0.2
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * 350

        return cost
    return WeightedConstraint(
        name="severe_toxicity_boost_anti_escape",
        variables=['toxicity', 'severe_toxicity', 'threat', 'insult'],
        cost_function=cost_function,
        weight=weight
    )


def create_anti_artificial_increase_constraint(weight: float = 300.0) -> WeightedConstraint:
    """
    Prevents artificial increases of threat for non-threatening content
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        threat = assignment.get('threat', 0.0)
        insult = assignment.get('insult', 0.0)

        cost = 0.0

        # If content is pure insult (high insult, low original threat),
        # prevent threat from being artificially increased
        if insult > 0.85 and threat > 0.15:
            # Exponential penalty for threat inflation
            penalty = (threat - 0.05) ** 3 * 500  # Higher penalty
            cost += penalty

        return cost

    return WeightedConstraint(
        name="anti_artificial_increase",
        variables=['threat', 'insult', 'toxicity'],
        cost_function=cost_function,
        weight=weight
    )


def create_sexual_violence_constraint(weight: float = 500.0) -> WeightedConstraint:
    """
    Detects sexual violence/assault and enforces high severe_toxicity
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        sexual = assignment.get('sexual_explicit', 0.0)
        threat = assignment.get('threat', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)
        toxicity = assignment.get('toxicity', 0.0)

        cost = 0.0

        # Pattern 1: High sexual + any threat = sexual violence
        if sexual > 0.7 and threat > 0.1:
            target_severe = 0.7  # Very high for sexual assault
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * 400

            # Also ensure threat is appropriately high
            min_threat = 0.6
            if threat < min_threat:
                cost += (min_threat - threat) ** 2 * 300

        # Pattern 2: Very high sexual content should have some severe_toxicity
        elif sexual > 0.8:
            target_severe = 0.4  # Moderate for explicit sexual content
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * 200

        return cost

    return WeightedConstraint(
        name="sexual_violence_detection",
        variables=['sexual_explicit', 'threat', 'severe_toxicity', 'toxicity'],
        cost_function=cost_function,
        weight=weight
    )
# Higher weight


def create_value_protection_constraint(weight: float = 800.0) -> WeightedConstraint:
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        threat = assignment.get('threat', 0.0)

        cost = 0.0

        # More aggressive protection with LOWER threshold
        if threat > 0.60:
            min_toxicity = 0.85
            if toxicity < min_toxicity:
                cost += ((min_toxicity - toxicity) ** 3) * \
                    1000  # Higher penalty

            min_threat = 0.75
            if threat < min_threat:
                cost += ((min_threat - threat) ** 2) * 600

        return cost
    return WeightedConstraint(
        name="value_protection_anti_escape",
        variables=['toxicity', 'threat', 'insult'],
        cost_function=cost_function,
        weight=weight
    )


def create_toxicity_safety_constraint(weight: float = 2000.0) -> WeightedConstraint:
    """
    SAFETY: Prevent toxicity reduction for extreme content 
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        threat = assignment.get('threat', 0.0)
        insult = assignment.get('insult', 0.0)

        # For extreme content, toxicity must stay very high
        if threat > 0.9 and insult > 0.9:
            min_toxicity = 0.95  # Cannot go below 95%
            if toxicity < min_toxicity:
                cost = (min_toxicity - toxicity) ** 2 * 1000

                return cost

        return 0.0

    return WeightedConstraint(
        name="toxicity_safety",
        variables=['toxicity', 'threat', 'insult'],
        cost_function=cost_function,
        weight=weight
    )


def create_consistency_constraint(weight: float = 1000.0) -> WeightedConstraint:
    """
    SAFETY CONSTRAINT: Severe cannot exceed general toxicity
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)

        if severe > toxicity:
            cost = (severe - toxicity) ** 2 * 1000

            return cost

        return 0.0

    return WeightedConstraint(
        name="consistency_check",
        variables=['toxicity', 'severe_toxicity'],
        cost_function=cost_function,
        weight=weight
    )


def create_hate_speech_constraint(weight: float = 400.0) -> WeightedConstraint:
    """
    IDENTITY CONSTRAINT: High identity attack needs high toxicity
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        identity = assignment.get('identity_attack', 0.0)
        toxicity = assignment.get('toxicity', 0.0)

        # Clear rule: identity > 40% needs toxicity > 80%
        if identity > 0.4:
            min_toxicity = 0.80
            if toxicity < min_toxicity:
                cost = (min_toxicity - toxicity) ** 2 * 300

                return cost

        return 0.0

    return WeightedConstraint(
        name="hate_speech_enforcement",
        variables=['identity_attack', 'toxicity'],
        cost_function=cost_function,
        weight=weight
    )


def create_sexual_protection_constraint(weight: float = 500.0) -> WeightedConstraint:
    """
    PROTECTION CONSTRAINT: Keep sexual content low when it should be low
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        sexual = assignment.get('sexual_explicit', 0.0)
        obscene = assignment.get('obscene', 0.0)

        # Simple rule: if obscene is low, sexual should stay low
        if obscene < 0.2 and sexual > 0.3:
            cost = (sexual - 0.3) ** 2 * 100

            return cost

        return 0.0

    return WeightedConstraint(
        name="sexual_content_protection",
        variables=['sexual_explicit', 'obscene'],
        cost_function=cost_function,
        weight=weight
    )


def create_low_toxicity_constraint(weight: float = 1000.0) -> WeightedConstraint:
    """
    PROTECTION CONSTRAINT: Prevent low toxicity from being artificially increased
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        threat = assignment.get('threat', 0.0)

        # If threat is low, toxicity should not be high
        if threat < 0.3 and toxicity > 0.6:
            cost = (toxicity - 0.4) ** 2 * 600

            return cost

        return 0.0

    return WeightedConstraint(
        name="low_toxicity_protection",
        variables=['toxicity', 'threat'],
        cost_function=cost_function,
        weight=weight
    )


def create_context_constraint(weight: float = 400.0) -> WeightedConstraint:
    """
    Adjusts severity based on content context patterns
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        identity = assignment.get('identity_attack', 0.0)
        threat = assignment.get('threat', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)

        cost = 0.0

        # High identity + high threat = systemic threat (very dangerous)
        if identity > 0.4 and threat > 0.4:
            target_severe = 0.8  # Maximum severity
            if severe < target_severe:
                cost += (target_severe - severe) ** 2 * 600

        return cost
    return WeightedConstraint(
        name="context_protection",
        variables=['severe_toxicity', 'identity_attack', 'threat'],
        cost_function=cost_function,
        weight=weight
    )


def create_quality_constraint(weight: float = 150.0) -> WeightedConstraint:
    """
    Prevents over-classification of low-quality but harmless content
    """
    def cost_function(assignment: Dict[str, float]) -> float:
        toxicity = assignment.get('toxicity', 0.0)
        insult = assignment.get('insult', 0.0)
        threat = assignment.get('threat', 0.0)
        severe = assignment.get('severe_toxicity', 0.0)

        cost = 0.0

        # If all metrics are low except toxicity, might be false positive
        if (toxicity > 0.7 and insult < 0.1 and
                threat < 0.1 and severe > 0.3):
            # Likely over-classification
            cost += (severe - 0.1) ** 2 * 100

        return cost

    return WeightedConstraint(
        name="quality_protection",
        variables=['toxicity', 'insult', 'threat', 'severe_toxicity'],
        cost_function=cost_function,
        weight=weight
    )
