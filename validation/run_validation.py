"""
Master Validation Script

Runs the complete circular validation pipeline for the GPS-Weather unification.

This validates both papers:
1. cynegeticus-positioning-script.tex (categorical GPS)
2. ober-atmos-scripting.tex (partition dynamics weather)

Through a single unified validation demonstrating that position and weather
are dual aspects of atmospheric partition geometry.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "cynegeticus" / "src"))

from weather.circular_validation import CircularValidation


def main():
    """Run complete validation pipeline"""

    print("""
    ========================================================================

             CIRCULAR VALIDATION FRAMEWORK
             GPS <-> Weather Unification via Partition Geometry

      Validates:
      - Cynegeticus Positioning (molecular-based categorical GPS)
      - Ober Atmospheric Scripting (partition dynamics weather)

      Method: Trajectory Completion (Observation = Computing = Processing)

    ========================================================================
    """)

    # Configuration
    geojson_path = "c:/Users/kundai/Documents/geosciences/sighthound/cynegeticus/public/comprehensive_gps_multiprecision_20251013_053445.geojson"
    output_dir = "c:/Users/kundai/Documents/geosciences/sighthound/validation/results"

    # Validate paths
    if not Path(geojson_path).exists():
        print(f"❌ Error: GeoJSON file not found at {geojson_path}")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run validation
    print("Initializing validation framework...\n")
    validation = CircularValidation(geojson_path, munich_coords=(48.183, 11.357))

    try:
        # Execute complete validation
        results = validation.run_complete_validation()

        # Print summary
        validation.print_summary()

        # Save all results
        validation.save_all_results(output_dir)

        print(f"""
    ========================================================================

      VALIDATION COMPLETE

      Results saved to:
      {output_dir}

      Files generated:
      - complete_circular_validation.json (full results)
      - validation_summary.csv (key metrics)
      - atmospheric_state_from_gps.json/csv
      - weather_forecast_from_gps.json/csv
      - actual_weather_data.json/csv
      - gps_from_weather.json/csv

    ========================================================================
        """)

        # Return success code
        return 0

    except Exception as e:
        print(f"\n[ERROR] Validation failed with error:")
        print(f"   {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
