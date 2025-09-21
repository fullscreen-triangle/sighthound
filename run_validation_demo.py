#!/usr/bin/env python3
"""
Demonstration Script for Metacognitive Athletic Positioning Validation

This script runs the complete experimental validation framework using Olympic athlete data
to prove bidirectional relationships between consciousness-aware biometrics and geolocation.

Usage:
    python run_validation_demo.py [--athlete-index 0] [--time-points 50] [--verbose]
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from experimental_validation_framework import MetacognitiveAthleticValidator, ValidationResult


async def run_single_athlete_validation(athlete_index: int = 0, 
                                      time_points: int = 50,
                                      verbose: bool = True) -> ValidationResult:
    """
    Run validation for a single athlete
    
    Args:
        athlete_index: Index of athlete in database (0-based)
        time_points: Number of time points to analyze (seconds)
        verbose: Whether to print detailed results
        
    Returns:
        ValidationResult with comprehensive metrics
    """
    print("🚀 Initializing Metacognitive Athletic Positioning Validation Framework")
    print("="*80)
    
    try:
        # Initialize validator
        validator = MetacognitiveAthleticValidator()
        
        # Run complete validation
        result = await validator.run_complete_validation(
            athlete_index=athlete_index,
            num_time_points=time_points
        )
        
        if verbose:
            print("\n🎉 VALIDATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            
        return result
        
    except Exception as e:
        print(f"❌ ERROR during validation: {str(e)}")
        print("This might be due to missing Olympic data files or Autobahn integration issues.")
        print("Please ensure all data files are present in public/olympics/")
        raise


async def run_multi_athlete_comparison(num_athletes: int = 5,
                                     time_points: int = 25) -> dict:
    """
    Run validation across multiple athletes for comparison
    
    Args:
        num_athletes: Number of athletes to validate
        time_points: Number of time points per athlete
        
    Returns:
        Dictionary of results for each athlete
    """
    print(f"🏃‍♂️ Running Multi-Athlete Validation ({num_athletes} athletes)")
    print("="*80)
    
    validator = MetacognitiveAthleticValidator()
    results = {}
    
    for i in range(num_athletes):
        print(f"\n--- Athlete {i+1}/{num_athletes} ---")
        try:
            result = await validator.run_complete_validation(
                athlete_index=i,
                num_time_points=time_points
            )
            results[f"athlete_{i}"] = result
        except Exception as e:
            print(f"⚠️ Warning: Failed to validate athlete {i}: {str(e)}")
            results[f"athlete_{i}"] = None
    
    # Print comparison summary
    print("\n📊 MULTI-ATHLETE COMPARISON SUMMARY")
    print("="*50)
    
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if successful_results:
        # Calculate averages
        avg_position_accuracy = sum(r.position_accuracy for r in successful_results.values()) / len(successful_results)
        avg_biometric_accuracy = sum(sum(r.biometric_accuracy.values()) for r in successful_results.values()) / (len(successful_results) * 4)
        avg_consciousness_score = sum(r.metacognitive_validation_score for r in successful_results.values()) / len(successful_results)
        
        print(f"Successful validations: {len(successful_results)}/{num_athletes}")
        print(f"Average position accuracy: {avg_position_accuracy:.3f}")
        print(f"Average biometric accuracy: {avg_biometric_accuracy:.3f}")
        print(f"Average consciousness score: {avg_consciousness_score:.3f}")
    else:
        print("❌ No successful validations completed")
    
    return results


def save_results_to_file(results: ValidationResult, filename: str = "validation_results.json"):
    """Save validation results to JSON file"""
    try:
        # Convert ValidationResult to dictionary for JSON serialization
        results_dict = {
            "position_accuracy": results.position_accuracy,
            "position_rmse": results.position_rmse,
            "biometric_accuracy": results.biometric_accuracy,
            "consciousness_enhanced_position_accuracy": results.consciousness_enhanced_position_accuracy,
            "consciousness_improvement_factor": results.consciousness_improvement_factor,
            "oscillatory_pattern_accuracy": results.oscillatory_pattern_accuracy,
            "temporal_synchronization_accuracy": results.temporal_synchronization_accuracy,
            "bidirectional_correlation": results.bidirectional_correlation,
            "metacognitive_validation_score": results.metacognitive_validation_score,
            "framework_completeness": results.framework_completeness,
            "predicted_positions": results.predicted_positions[:10],  # Save first 10 positions
            "validation_timestamp": "2024-12-21T00:00:00Z"
        }
        
        with open(filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not save results to file: {str(e)}")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run Metacognitive Athletic Positioning Validation Framework"
    )
    parser.add_argument(
        "--athlete-index", 
        type=int, 
        default=0, 
        help="Index of athlete to validate (default: 0)"
    )
    parser.add_argument(
        "--time-points", 
        type=int, 
        default=50, 
        help="Number of time points to analyze (default: 50)"
    )
    parser.add_argument(
        "--multi-athlete", 
        type=int, 
        help="Run validation on multiple athletes (specify number)"
    )
    parser.add_argument(
        "--save-results", 
        action="store_true", 
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=True,
        help="Print verbose output (default: True)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.multi_athlete:
            # Run multi-athlete validation
            results = await run_multi_athlete_comparison(
                num_athletes=args.multi_athlete,
                time_points=args.time_points
            )
            print(f"\n✅ Multi-athlete validation completed for {args.multi_athlete} athletes")
            
        else:
            # Run single athlete validation
            result = await run_single_athlete_validation(
                athlete_index=args.athlete_index,
                time_points=args.time_points,
                verbose=args.verbose
            )
            
            if args.save_results:
                save_results_to_file(result)
            
            print(f"\n✅ Validation completed successfully!")
            print(f"Final Framework Completeness: {result.framework_completeness*100:.1f}%")
            print(f"Metacognitive Validation Score: {result.metacognitive_validation_score:.3f}")
            
    except Exception as e:
        print(f"\n❌ Validation failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure Olympic data files are present in public/olympics/")
        print("2. Check that required Python packages are installed")
        print("3. Verify Autobahn integration is properly configured")
        sys.exit(1)


if __name__ == "__main__":
    print("🧠 Metacognitive Athletic Positioning Validation Framework")
    print("Proving bidirectional relationships between consciousness and geolocation")
    print("="*80)
    
    # Run the validation
    asyncio.run(main())
