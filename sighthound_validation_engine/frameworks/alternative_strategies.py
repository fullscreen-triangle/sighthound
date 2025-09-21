"""
Alternative Strategy Validator

Implements the Black Sea alternative experience networks for validating
all possible alternative race strategies simultaneously through strategic
impossibility optimization.
"""

import numpy as np
import asyncio
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from itertools import product

logger = logging.getLogger(__name__)

@dataclass
class AlternativeStrategy:
    """Individual alternative race strategy."""
    strategy_id: str
    pacing_profile: List[float]
    energy_distribution: List[float]
    breathing_pattern: List[float]
    stride_parameters: Dict[str, float]
    predicted_performance: Dict[str, float]
    optimality_score: float

@dataclass
class StrategyValidation:
    """Validation result for an alternative strategy."""
    strategy: AlternativeStrategy
    predicted_time: float
    performance_improvement: float
    feasibility_score: float
    validation_confidence: float

class AlternativeStrategyValidator:
    """
    Alternative Strategy Validator using Black Sea methodology.
    
    Validates all possible alternative race strategies simultaneously
    through strategic impossibility optimization, enabling access to
    information about strategies athletes DIDN'T use.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.strategies_count = config.get('alternative_strategies_count', 10_000)
        self.race_duration = config.get('race_duration', 45.0)
        
        # Strategy parameter ranges
        self.strategy_parameters = {
            'pacing_profiles': ['even', 'positive_split', 'negative_split', 'surge', 'kick'],
            'energy_systems': ['aerobic_dominant', 'anaerobic_dominant', 'mixed', 'power_reserve'],
            'breathing_patterns': ['rhythmic_2_2', 'rhythmic_3_3', 'adaptive', 'power_breathing'],
            'stride_strategies': ['consistent', 'progressive', 'variable', 'optimal_frequency']
        }
        
    async def generate_alternative_strategies(self, 
                                            athlete_data: Dict,
                                            num_strategies: int) -> List[AlternativeStrategy]:
        """
        Generate alternative strategies for athlete analysis.
        
        Args:
            athlete_data: Complete athlete dataset
            num_strategies: Number of alternative strategies to generate
            
        Returns:
            List of alternative strategies
        """
        logger.info(f"Generating {num_strategies} alternative strategies")
        
        strategies = []
        athlete_capabilities = self._analyze_athlete_capabilities(athlete_data)
        
        for i in range(num_strategies):
            strategy = await self._generate_single_strategy(
                i, athlete_capabilities, athlete_data
            )
            strategies.append(strategy)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1}/{num_strategies} strategies")
        
        return strategies
    
    def _analyze_athlete_capabilities(self, athlete_data: Dict) -> Dict:
        """Analyze athlete capabilities to constrain strategy generation."""
        
        biometrics = athlete_data.get('biometrics', {})
        performance = athlete_data.get('performance', {})
        
        capabilities = {
            'vo2_capacity': biometrics.get('base_vo2', 65.0),
            'lactate_threshold': biometrics.get('base_lactate', 8.0),
            'anaerobic_power': performance.get('performance_index', 85.0),
            'endurance_capacity': biometrics.get('metabolic', {}).get('ventilatory_threshold', 70.0),
            'neuromuscular_power': athlete_data.get('physiology', {}).get('neuromuscular', {}).get('power_output', 1000),
            'efficiency_rating': athlete_data.get('biomechanics', {}).get('running_effectiveness', 0.85)
        }
        
        return capabilities
    
    async def _generate_single_strategy(self, 
                                       strategy_index: int,
                                       capabilities: Dict,
                                       athlete_data: Dict) -> AlternativeStrategy:
        """Generate a single alternative strategy."""
        
        strategy_id = f"strategy_{strategy_index:05d}"
        
        # Generate pacing profile
        pacing_profile = self._generate_pacing_profile(capabilities)
        
        # Generate energy distribution
        energy_distribution = self._generate_energy_distribution(capabilities)
        
        # Generate breathing pattern
        breathing_pattern = self._generate_breathing_pattern(capabilities)
        
        # Generate stride parameters
        stride_parameters = self._generate_stride_parameters(capabilities, athlete_data)
        
        # Predict performance for this strategy
        predicted_performance = await self._predict_strategy_performance(
            pacing_profile, energy_distribution, breathing_pattern, 
            stride_parameters, capabilities
        )
        
        # Calculate optimality score
        optimality_score = self._calculate_optimality_score(
            predicted_performance, capabilities
        )
        
        return AlternativeStrategy(
            strategy_id=strategy_id,
            pacing_profile=pacing_profile,
            energy_distribution=energy_distribution,
            breathing_pattern=breathing_pattern,
            stride_parameters=stride_parameters,
            predicted_performance=predicted_performance,
            optimality_score=optimality_score
        )
    
    def _generate_pacing_profile(self, capabilities: Dict) -> List[float]:
        """Generate pacing profile based on athlete capabilities."""
        
        # Split 400m into 8 segments (50m each)
        segments = 8
        base_pace = 100.0 / (self.race_duration / 4)  # Base pace in m/s
        
        # Pacing strategy variants
        strategies = {
            'even': [1.0] * segments,
            'positive_split': [1.05, 1.02, 1.0, 0.98, 0.96, 0.94, 0.92, 0.90],
            'negative_split': [0.90, 0.92, 0.95, 0.98, 1.02, 1.05, 1.08, 1.10],
            'surge': [1.0, 1.15, 1.10, 0.95, 0.90, 0.95, 1.05, 1.10],
            'kick': [1.0, 1.0, 0.98, 0.96, 0.94, 0.98, 1.08, 1.15]
        }
        
        # Select strategy based on capabilities
        vo2_ratio = capabilities['vo2_capacity'] / 65.0  # Normalized
        lactate_tolerance = capabilities['lactate_threshold'] / 8.0  # Normalized
        
        if vo2_ratio > 1.1 and lactate_tolerance > 1.1:
            # High aerobic and anaerobic capacity
            strategy_type = np.random.choice(['negative_split', 'surge', 'kick'], p=[0.4, 0.3, 0.3])
        elif vo2_ratio > 1.05:
            # Good aerobic capacity
            strategy_type = np.random.choice(['even', 'negative_split'], p=[0.6, 0.4])
        elif lactate_tolerance > 1.05:
            # Good anaerobic capacity
            strategy_type = np.random.choice(['positive_split', 'kick'], p=[0.7, 0.3])
        else:
            # Conservative approach
            strategy_type = np.random.choice(['even', 'positive_split'], p=[0.8, 0.2])
        
        base_profile = strategies[strategy_type]
        
        # Add individual variation
        variation = np.random.uniform(0.95, 1.05, segments)
        pacing_profile = [base_pace * profile * var for profile, var in zip(base_profile, variation)]
        
        return pacing_profile
    
    def _generate_energy_distribution(self, capabilities: Dict) -> List[float]:
        """Generate energy system utilization distribution."""
        
        # Energy systems: aerobic, anaerobic_alactic, anaerobic_lactic, neuromuscular
        base_distribution = [0.60, 0.15, 0.20, 0.05]  # 400m typical
        
        # Adjust based on capabilities
        vo2_factor = capabilities['vo2_capacity'] / 65.0
        lactate_factor = capabilities['lactate_threshold'] / 8.0
        power_factor = capabilities['neuromuscular_power'] / 1000.0
        
        adjusted_distribution = [
            base_distribution[0] * vo2_factor * np.random.uniform(0.9, 1.1),
            base_distribution[1] * power_factor * np.random.uniform(0.8, 1.2),
            base_distribution[2] * lactate_factor * np.random.uniform(0.9, 1.1),
            base_distribution[3] * power_factor * np.random.uniform(0.7, 1.3)
        ]
        
        # Normalize to sum to 1.0
        total = sum(adjusted_distribution)
        energy_distribution = [dist / total for dist in adjusted_distribution]
        
        return energy_distribution
    
    def _generate_breathing_pattern(self, capabilities: Dict) -> List[float]:
        """Generate breathing pattern strategy."""
        
        # Breathing rates throughout race (breaths per minute)
        base_rate = 40.0  # Base respiratory rate during 400m
        
        # Pattern strategies
        patterns = {
            'rhythmic_2_2': [base_rate * f for f in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]],
            'rhythmic_3_3': [base_rate * f for f in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]],
            'adaptive': [base_rate * f for f in [1.0, 1.2, 1.1, 1.3, 1.2, 1.4, 1.6, 1.8]],
            'power_breathing': [base_rate * f for f in [1.1, 1.0, 1.2, 1.1, 1.5, 1.3, 1.7, 1.9]]
        }
        
        # Select pattern based on efficiency
        efficiency = capabilities['efficiency_rating']
        
        if efficiency > 0.9:
            pattern_type = np.random.choice(['rhythmic_2_2', 'adaptive'], p=[0.6, 0.4])
        elif efficiency > 0.8:
            pattern_type = np.random.choice(['rhythmic_3_3', 'adaptive'], p=[0.7, 0.3])
        else:
            pattern_type = np.random.choice(['adaptive', 'power_breathing'], p=[0.6, 0.4])
        
        breathing_pattern = patterns[pattern_type]
        
        # Add individual variation
        variation = np.random.uniform(0.95, 1.05, len(breathing_pattern))
        return [rate * var for rate, var in zip(breathing_pattern, variation)]
    
    def _generate_stride_parameters(self, capabilities: Dict, athlete_data: Dict) -> Dict[str, float]:
        """Generate stride parameter strategy."""
        
        biomechanics = athlete_data.get('biomechanics', {})
        base_stride_length = biomechanics.get('stride_length', 2.2)
        base_stride_frequency = biomechanics.get('stride_frequency', 4.5)
        
        # Strategy variants
        strategies = ['consistent', 'progressive', 'variable', 'optimal_frequency']
        strategy_type = np.random.choice(strategies)
        
        if strategy_type == 'consistent':
            stride_length_factor = np.random.uniform(0.98, 1.02)
            stride_frequency_factor = np.random.uniform(0.98, 1.02)
        elif strategy_type == 'progressive':
            stride_length_factor = np.random.uniform(0.95, 1.05)
            stride_frequency_factor = np.random.uniform(1.00, 1.08)
        elif strategy_type == 'variable':
            stride_length_factor = np.random.uniform(0.92, 1.08)
            stride_frequency_factor = np.random.uniform(0.95, 1.10)
        else:  # optimal_frequency
            # Optimize for power capabilities
            power_ratio = capabilities['neuromuscular_power'] / 1000.0
            stride_length_factor = np.random.uniform(0.95, 1.00 + 0.1 * power_ratio)
            stride_frequency_factor = np.random.uniform(1.00, 1.05 + 0.05 * power_ratio)
        
        return {
            'stride_length': base_stride_length * stride_length_factor,
            'stride_frequency': base_stride_frequency * stride_frequency_factor,
            'ground_contact_time': biomechanics.get('ground_contact_time', 0.08) * np.random.uniform(0.95, 1.05),
            'vertical_oscillation': biomechanics.get('vertical_oscillation', 0.06) * np.random.uniform(0.90, 1.10),
            'strategy_type': strategy_type
        }
    
    async def _predict_strategy_performance(self,
                                          pacing_profile: List[float],
                                          energy_distribution: List[float],
                                          breathing_pattern: List[float],
                                          stride_parameters: Dict[str, float],
                                          capabilities: Dict) -> Dict[str, float]:
        """Predict performance for this strategy combination."""
        
        # Calculate theoretical race time based on strategy parameters
        segment_times = []
        accumulated_fatigue = 0.0
        
        for i, (pace, breathing_rate) in enumerate(zip(pacing_profile, breathing_pattern)):
            segment_distance = 50.0  # meters per segment
            
            # Base time for this segment
            base_time = segment_distance / pace
            
            # Energy system efficiency
            aerobic_contribution = energy_distribution[0] * (capabilities['vo2_capacity'] / 65.0)
            anaerobic_contribution = (energy_distribution[1] + energy_distribution[2]) * (capabilities['lactate_threshold'] / 8.0)
            power_contribution = energy_distribution[3] * (capabilities['neuromuscular_power'] / 1000.0)
            
            total_efficiency = (aerobic_contribution + anaerobic_contribution + power_contribution) / 3.0
            
            # Fatigue accumulation
            segment_stress = pace * breathing_rate / (capabilities['efficiency_rating'] * 1000)
            accumulated_fatigue += segment_stress
            fatigue_factor = 1.0 + accumulated_fatigue * 0.1
            
            # Stride efficiency
            stride_efficiency = min(1.0, stride_parameters['stride_length'] * stride_parameters['stride_frequency'] / 10.0)
            
            # Adjusted segment time
            adjusted_time = base_time * fatigue_factor / (total_efficiency * stride_efficiency)
            segment_times.append(adjusted_time)
        
        total_time = sum(segment_times)
        
        # Calculate additional performance metrics
        average_speed = 400.0 / total_time  # m/s
        peak_speed = 50.0 / min(segment_times)  # m/s
        speed_endurance = segment_times[-1] / segment_times[0]  # Final vs initial segment ratio
        
        # Energy cost estimation
        total_energy_cost = sum(energy_distribution[i] * capabilities[cap] for i, cap in enumerate([
            'vo2_capacity', 'anaerobic_power', 'lactate_threshold', 'neuromuscular_power'
        ]))
        
        return {
            'predicted_time': total_time,
            'segment_times': segment_times,
            'average_speed': average_speed,
            'peak_speed': peak_speed,
            'speed_endurance': speed_endurance,
            'energy_cost': total_energy_cost / 4,  # Normalize
            'efficiency_score': total_efficiency,
            'fatigue_accumulation': accumulated_fatigue
        }
    
    def _calculate_optimality_score(self, 
                                   performance: Dict[str, float], 
                                   capabilities: Dict) -> float:
        """Calculate optimality score for strategy."""
        
        # Base score from predicted time (lower is better)
        time_score = max(0.0, 1.0 - (performance['predicted_time'] - 40.0) / 10.0)  # Normalize around 45s ± 5s
        
        # Efficiency score
        efficiency_score = performance['efficiency_score']
        
        # Speed endurance score (closer to 1.0 is better)
        endurance_score = 1.0 - abs(performance['speed_endurance'] - 1.0)
        
        # Energy utilization score
        energy_score = 1.0 - performance['energy_cost'] / capabilities['vo2_capacity']
        energy_score = max(0.0, min(1.0, energy_score))
        
        # Combined optimality score
        weights = [0.4, 0.3, 0.2, 0.1]
        scores = [time_score, efficiency_score, endurance_score, energy_score]
        
        optimality_score = sum(w * s for w, s in zip(weights, scores))
        
        return np.clip(optimality_score, 0.0, 1.0)
    
    async def validate_strategies(self, 
                                strategies: List[AlternativeStrategy],
                                actual_performance: Dict) -> List[StrategyValidation]:
        """Validate alternative strategies against actual performance."""
        
        logger.info(f"Validating {len(strategies)} alternative strategies")
        
        validations = []
        actual_time = actual_performance.get('race_time', 45.0)
        
        for i, strategy in enumerate(strategies):
            validation = await self._validate_single_strategy(
                strategy, actual_time, actual_performance
            )
            validations.append(validation)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Validated {i + 1}/{len(strategies)} strategies")
        
        return validations
    
    async def _validate_single_strategy(self,
                                       strategy: AlternativeStrategy,
                                       actual_time: float,
                                       actual_performance: Dict) -> StrategyValidation:
        """Validate a single alternative strategy."""
        
        predicted_time = strategy.predicted_performance['predicted_time']
        
        # Calculate performance improvement
        time_improvement = actual_time - predicted_time
        performance_improvement = time_improvement / actual_time * 100  # Percentage
        
        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility_score(strategy, actual_performance)
        
        # Calculate validation confidence
        validation_confidence = self._calculate_validation_confidence(
            strategy, predicted_time, actual_time, feasibility_score
        )
        
        return StrategyValidation(
            strategy=strategy,
            predicted_time=predicted_time,
            performance_improvement=performance_improvement,
            feasibility_score=feasibility_score,
            validation_confidence=validation_confidence
        )
    
    def _calculate_feasibility_score(self, 
                                    strategy: AlternativeStrategy, 
                                    actual_performance: Dict) -> float:
        """Calculate feasibility score for strategy."""
        
        # Check if strategy parameters are within realistic bounds
        feasibility_factors = []
        
        # Pacing feasibility
        pace_variation = np.std(strategy.pacing_profile) / np.mean(strategy.pacing_profile)
        pace_feasibility = max(0.0, 1.0 - pace_variation * 2)  # Lower variation is more feasible
        feasibility_factors.append(pace_feasibility)
        
        # Energy distribution feasibility
        energy_balance = abs(sum(strategy.energy_distribution) - 1.0)
        energy_feasibility = max(0.0, 1.0 - energy_balance * 10)
        feasibility_factors.append(energy_feasibility)
        
        # Stride parameter feasibility
        stride_length = strategy.stride_parameters['stride_length']
        stride_freq = strategy.stride_parameters['stride_frequency']
        stride_feasibility = 1.0 if 1.5 <= stride_length <= 3.0 and 3.5 <= stride_freq <= 6.0 else 0.5
        feasibility_factors.append(stride_feasibility)
        
        # Optimality score as feasibility indicator
        optimality_feasibility = strategy.optimality_score
        feasibility_factors.append(optimality_feasibility)
        
        return np.mean(feasibility_factors)
    
    def _calculate_validation_confidence(self,
                                       strategy: AlternativeStrategy,
                                       predicted_time: float,
                                       actual_time: float,
                                       feasibility_score: float) -> float:
        """Calculate confidence in strategy validation."""
        
        # Prediction accuracy
        time_difference = abs(predicted_time - actual_time)
        prediction_accuracy = max(0.0, 1.0 - time_difference / 10.0)  # Within 10 seconds
        
        # Strategy optimality
        optimality_confidence = strategy.optimality_score
        
        # Feasibility confidence
        feasibility_confidence = feasibility_score
        
        # Combined confidence
        confidence_factors = [prediction_accuracy, optimality_confidence, feasibility_confidence]
        weights = [0.5, 0.3, 0.2]
        
        validation_confidence = sum(w * f for w, f in zip(weights, confidence_factors))
        
        return np.clip(validation_confidence, 0.0, 0.99)
    
    async def find_optimal_strategy(self, 
                                   strategy_validations: List[StrategyValidation]) -> StrategyValidation:
        """Find the optimal strategy from validations."""
        
        # Sort by combination of performance improvement and feasibility
        def strategy_score(validation: StrategyValidation) -> float:
            improvement_score = max(0.0, validation.performance_improvement / 10.0)  # Normalize to ~1.0
            feasibility_weight = validation.feasibility_score
            confidence_weight = validation.validation_confidence
            
            return improvement_score * feasibility_weight * confidence_weight
        
        optimal_validation = max(strategy_validations, key=strategy_score)
        
        logger.info(f"Optimal strategy found: {optimal_validation.strategy.strategy_id}")
        logger.info(f"  Predicted improvement: {optimal_validation.performance_improvement:.2f}%")
        logger.info(f"  Feasibility score: {optimal_validation.feasibility_score:.3f}")
        logger.info(f"  Validation confidence: {optimal_validation.validation_confidence:.3f}")
        
        return optimal_validation
