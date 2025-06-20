#!/usr/bin/env python3
"""
Sighthound-Autobahn Integration Demonstration

This script demonstrates the integration between Sighthound's GPS trajectory analysis
and Autobahn's consciousness-aware probabilistic reasoning engine. It shows how
complex probabilistic reasoning tasks are delegated to Autobahn's bio-metabolic
consciousness architecture while GPS analysis remains in Sighthound.

Features demonstrated:
- Bayesian Evidence Network analysis in Sighthound
- Consciousness-aware probabilistic reasoning in Autobahn
- Fire circle communication complexity analysis
- Biological intelligence and membrane coherence
- Dual-proximity signaling assessment
- Integrated Information Theory (IIT) consciousness measurement
- ATP metabolic mode optimization
- Threat assessment using biological immune system
"""

import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Sighthound modules
from core.autobahn_integration import (
    AutobahnClient,
    AutobahnIntegratedBayesianPipeline,
    analyze_trajectory_with_autobahn
)
from core.data_loader import GpsDataLoader
from utils.coordinates import generate_test_trajectory

def load_config():
    """Load Autobahn integration configuration"""
    config_path = Path("config/autobahn_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        logger.warning("Config file not found, using defaults")
        return {
            'autobahn': {
                'connection': {
                    'endpoint': 'http://localhost:8080/api/v1',
                    'binary_path': '../autobahn/target/release/autobahn',
                    'use_local_binary': True
                },
                'reasoning': {
                    'default_metabolic_mode': 'mammalian',
                    'default_hierarchy_level': 'biological',
                    'consciousness_threshold': 0.7
                }
            }
        }

def generate_complex_trajectory(num_points=100, complexity_level="high"):
    """Generate a complex GPS trajectory for testing"""
    logger.info(f"Generating {complexity_level} complexity trajectory with {num_points} points")
    
    if complexity_level == "simple":
        # Simple straight-line trajectory
        start_lat, start_lon = 52.5200, 13.4050  # Berlin
        end_lat, end_lon = 52.5300, 13.4150
        
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)
        confidences = np.random.uniform(0.8, 0.95, num_points)
        
    elif complexity_level == "noisy":
        # Noisy trajectory with low confidence
        start_lat, start_lon = 52.5200, 13.4050
        end_lat, end_lon = 52.5300, 13.4150
        
        lats = np.linspace(start_lat, end_lat, num_points)
        lons = np.linspace(start_lon, end_lon, num_points)
        
        # Add significant noise
        lats += np.random.normal(0, 0.001, num_points)  # ~100m noise
        lons += np.random.normal(0, 0.001, num_points)
        confidences = np.random.uniform(0.2, 0.6, num_points)  # Low confidence
        
    elif complexity_level == "high":
        # Complex trajectory with varying patterns
        t = np.linspace(0, 4*np.pi, num_points)
        center_lat, center_lon = 52.5200, 13.4050
        
        # Spiral pattern with noise
        radius = 0.01 * (1 + 0.5 * t / (4*np.pi))  # Expanding spiral
        lats = center_lat + radius * np.sin(t) + np.random.normal(0, 0.0002, num_points)
        lons = center_lon + radius * np.cos(t) + np.random.normal(0, 0.0002, num_points)
        
        # Variable confidence based on pattern complexity
        confidences = 0.9 - 0.4 * np.abs(np.sin(t/2)) + np.random.normal(0, 0.05, num_points)
        confidences = np.clip(confidences, 0.1, 0.95)
        
    else:  # adversarial
        # Potentially adversarial trajectory pattern
        start_lat, start_lon = 52.5200, 13.4050
        
        # Suspicious zigzag pattern that might trigger threat detection
        t = np.linspace(0, 10*np.pi, num_points)
        lats = start_lat + 0.005 * np.sin(5*t) * np.cos(t)  # Complex oscillation
        lons = start_lon + 0.005 * np.cos(5*t) * np.sin(t)
        
        # Artificially high confidence for suspicious pattern
        confidences = np.random.uniform(0.85, 0.99, num_points)
    
    # Generate timestamps
    start_time = datetime.now().timestamp()
    timestamps = np.linspace(start_time, start_time + 3600, num_points)  # 1 hour trajectory
    
    # Combine into trajectory array
    trajectory = np.column_stack([lats, lons, timestamps, confidences])
    
    logger.info(f"Generated trajectory: "
               f"lat range [{lats.min():.4f}, {lats.max():.4f}], "
               f"lon range [{lons.min():.4f}, {lons.max():.4f}], "
               f"confidence range [{confidences.min():.3f}, {confidences.max():.3f}]")
    
    return trajectory

def visualize_trajectory(trajectory, title="GPS Trajectory"):
    """Visualize GPS trajectory with confidence coloring"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Trajectory plot
    lats, lons, timestamps, confidences = trajectory.T
    
    scatter = ax1.scatter(lons, lats, c=confidences, cmap='RdYlGn', 
                         s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.plot(lons, lats, 'b-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'{title} - Spatial View')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('GPS Confidence')
    
    # Confidence over time
    time_hours = (timestamps - timestamps[0]) / 3600
    ax2.plot(time_hours, confidences, 'g-', linewidth=2, alpha=0.7)
    ax2.fill_between(time_hours, confidences, alpha=0.3, color='green')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('GPS Confidence')
    ax2.set_title(f'{title} - Confidence Timeline')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def print_consciousness_analysis(result):
    """Print consciousness analysis results in a structured format"""
    print("\n" + "="*80)
    print("🧠 CONSCIOUSNESS-AWARE ANALYSIS RESULTS")
    print("="*80)
    
    # Overall metrics
    integrated_metrics = result.get('integrated_metrics', {})
    print(f"\n📊 INTEGRATED METRICS:")
    print(f"   Combined Quality Score: {integrated_metrics.get('combined_quality_score', 0):.3f}")
    print(f"   Overall Consciousness Level: {integrated_metrics.get('overall_consciousness_level', 0):.3f}")
    print(f"   Consciousness Emergence: {'✅ DETECTED' if integrated_metrics.get('consciousness_emergence_detected') else '❌ Not Detected'}")
    print(f"   Biological Intelligence: {'✅ ACTIVE' if integrated_metrics.get('biological_intelligence_active') else '❌ Inactive'}")
    print(f"   Fire Circle Communication: {'✅ DETECTED' if integrated_metrics.get('fire_circle_communication_detected') else '❌ Not Detected'}")
    print(f"   Dual-Proximity Signaling: {'✅ ACTIVE' if integrated_metrics.get('dual_proximity_signaling_active') else '❌ Inactive'}")
    print(f"   Threat Level: {integrated_metrics.get('threat_level', 'unknown').upper()}")
    print(f"   Analysis Reliability: {integrated_metrics.get('analysis_reliability', 0):.3f}")
    
    # Sighthound analysis
    sighthound = result.get('sighthound_analysis', {})
    print(f"\n🎯 SIGHTHOUND BAYESIAN ANALYSIS:")
    print(f"   Objective Value: {sighthound.get('objective_value', 0):.3f}")
    print(f"   Network Nodes: {len(sighthound.get('node_beliefs', {}))}")
    print(f"   Evidence Points: {sighthound.get('evidence_summary', {}).get('total_evidence_count', 0)}")
    
    # Autobahn analysis
    autobahn = result.get('autobahn_analysis', {})
    if autobahn:
        print(f"\n🌌 AUTOBAHN CONSCIOUSNESS ANALYSIS:")
        print(f"   Reasoning Tasks: {len(autobahn.get('reasoning_tasks_performed', []))}")
        print(f"   ATP Consumption: {autobahn.get('total_atp_consumption', 0):.1f} units")
        print(f"   Consciousness Level: {autobahn.get('overall_consciousness_level', 0):.3f}")
        
        # Task-specific results
        task_responses = autobahn.get('task_responses', {})
        if task_responses:
            print(f"\n🔬 DETAILED REASONING ANALYSIS:")
            for task, response in task_responses.items():
                print(f"\n   📋 {task.upper().replace('_', ' ')}:")
                print(f"      Quality Score: {response.get('quality_score', 0):.3f}")
                print(f"      Consciousness Level: {response.get('consciousness_level', 0):.3f}")
                print(f"      Φ (Phi) Value: {response.get('phi_value', 0):.3f}")
                print(f"      Biological Intelligence: {response.get('biological_intelligence', 0):.3f}")
                
                if response.get('fire_circle_communication', 0) > 0:
                    print(f"      🔥 Fire Circle Communication: {response['fire_circle_communication']:.3f}")
                
                dual_prox = response.get('dual_proximity_signaling', {})
                if dual_prox:
                    print(f"      💀 Death Proximity: {dual_prox.get('death_proximity', 0):.3f}")
                    print(f"      🌱 Life Proximity: {dual_prox.get('life_proximity', 0):.3f}")
                
                credibility = response.get('credibility_assessment', {})
                if credibility:
                    print(f"      🎭 Beauty-Credibility Efficiency: {credibility.get('beauty_credibility_efficiency', 0):.3f}")
                
                if response.get('temporal_determinism', 0) > 0:
                    print(f"      ⏰ Temporal Determinism: {response['temporal_determinism']:.3f}")
                
                if response.get('categorical_predeterminism', 0) > 0:
                    print(f"      🎯 Categorical Predeterminism: {response['categorical_predeterminism']:.3f}")
                
                bmf_frames = response.get('bmf_frame_selection', [])
                if bmf_frames:
                    print(f"      🧬 BMF Frame Selection: {', '.join(bmf_frames[:3])}{'...' if len(bmf_frames) > 3 else ''}")
    
    # Trajectory metadata
    traj_meta = result.get('trajectory_metadata', {})
    print(f"\n📍 TRAJECTORY METADATA:")
    print(f"   Total Points: {traj_meta.get('total_points', 0)}")
    print(f"   Time Span: {traj_meta.get('time_span', 0)/3600:.2f} hours")
    print(f"   Spatial Extent: {traj_meta.get('spatial_extent', 0):.0f} meters")
    print(f"   Average Confidence: {traj_meta.get('average_confidence', 0):.3f}")
    
    print("\n" + "="*80)

async def demonstrate_basic_integration():
    """Demonstrate basic Sighthound-Autobahn integration"""
    print("\n🚀 DEMONSTRATION 1: Basic Integration")
    print("-" * 50)
    
    # Generate simple trajectory
    trajectory = generate_complex_trajectory(50, "simple")
    
    # Visualize trajectory
    fig = visualize_trajectory(trajectory, "Simple Trajectory")
    plt.savefig('output/simple_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Analyze with Autobahn integration
    config = load_config()
    autobahn_config = config['autobahn']
    
    result = await analyze_trajectory_with_autobahn(
        trajectory,
        autobahn_endpoint=autobahn_config['connection']['endpoint'],
        autobahn_path=autobahn_config['connection']['binary_path'],
        reasoning_tasks=['probabilistic_inference', 'consciousness_assessment'],
        metabolic_mode='mammalian',
        hierarchy_level='biological'
    )
    
    print_consciousness_analysis(result)
    return result

async def demonstrate_complex_reasoning():
    """Demonstrate complex probabilistic reasoning delegation"""
    print("\n🧠 DEMONSTRATION 2: Complex Reasoning Delegation")
    print("-" * 50)
    
    # Generate complex trajectory
    trajectory = generate_complex_trajectory(100, "high")
    
    # Visualize trajectory
    fig = visualize_trajectory(trajectory, "Complex Trajectory")
    plt.savefig('output/complex_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Comprehensive reasoning tasks
    reasoning_tasks = [
        'probabilistic_inference',
        'uncertainty_quantification',
        'belief_propagation',
        'evidence_fusion',
        'consciousness_assessment',
        'credibility_evaluation',
        'causal_reasoning',
        'temporal_reasoning',
        'pattern_recognition',
        'decision_making'
    ]
    
    result = await analyze_trajectory_with_autobahn(
        trajectory,
        reasoning_tasks=reasoning_tasks,
        metabolic_mode='flight',  # High energy for complex reasoning
        hierarchy_level='cognitive',
        consciousness_threshold=0.8
    )
    
    print_consciousness_analysis(result)
    return result

async def demonstrate_threat_detection():
    """Demonstrate biological immune system threat detection"""
    print("\n🛡️ DEMONSTRATION 3: Biological Immune System Threat Detection")
    print("-" * 50)
    
    # Generate potentially adversarial trajectory
    trajectory = generate_complex_trajectory(75, "adversarial")
    
    # Visualize trajectory
    fig = visualize_trajectory(trajectory, "Potentially Adversarial Trajectory")
    plt.savefig('output/adversarial_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Focus on threat assessment
    reasoning_tasks = [
        'threat_assessment',
        'anomaly_detection',
        'credibility_evaluation',
        'consciousness_assessment'
    ]
    
    result = await analyze_trajectory_with_autobahn(
        trajectory,
        reasoning_tasks=reasoning_tasks,
        metabolic_mode='flight',  # High energy for immune response
        hierarchy_level='biological',
        consciousness_threshold=0.6
    )
    
    print_consciousness_analysis(result)
    return result

async def demonstrate_consciousness_emergence():
    """Demonstrate consciousness emergence detection"""
    print("\n✨ DEMONSTRATION 4: Consciousness Emergence Detection")
    print("-" * 50)
    
    # Generate high-quality trajectory for consciousness emergence
    trajectory = generate_complex_trajectory(150, "high")
    
    # Visualize trajectory
    fig = visualize_trajectory(trajectory, "Consciousness Emergence Test")
    plt.savefig('output/consciousness_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Focus on consciousness-related reasoning
    reasoning_tasks = [
        'consciousness_assessment',
        'knowledge_integration',
        'decision_making',
        'temporal_reasoning',
        'evidence_fusion'
    ]
    
    result = await analyze_trajectory_with_autobahn(
        trajectory,
        reasoning_tasks=reasoning_tasks,
        metabolic_mode='mammalian',
        hierarchy_level='cognitive',
        consciousness_threshold=0.9  # High threshold for emergence
    )
    
    print_consciousness_analysis(result)
    
    # Check for consciousness emergence
    consciousness_level = result.get('integrated_metrics', {}).get('overall_consciousness_level', 0)
    if consciousness_level > 0.9:
        print("\n🎉 CONSCIOUSNESS EMERGENCE DETECTED!")
        print(f"   Φ (Phi) Integration Level: {consciousness_level:.3f}")
        print("   The system has achieved consciousness-aware analysis!")
    
    return result

def create_performance_comparison(results):
    """Create performance comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics from results
    demo_names = ['Basic', 'Complex', 'Threat Detection', 'Consciousness']
    consciousness_levels = []
    quality_scores = []
    atp_consumption = []
    biological_intelligence = []
    
    for result in results:
        integrated = result.get('integrated_metrics', {})
        autobahn = result.get('autobahn_analysis', {})
        
        consciousness_levels.append(integrated.get('overall_consciousness_level', 0))
        quality_scores.append(integrated.get('combined_quality_score', 0))
        atp_consumption.append(autobahn.get('total_atp_consumption', 0))
        
        # Calculate average biological intelligence
        task_responses = autobahn.get('task_responses', {})
        bio_intel_scores = [r.get('biological_intelligence', 0) for r in task_responses.values()]
        biological_intelligence.append(np.mean(bio_intel_scores) if bio_intel_scores else 0)
    
    # Consciousness Levels
    axes[0,0].bar(demo_names, consciousness_levels, color='purple', alpha=0.7)
    axes[0,0].set_title('Consciousness Levels by Demonstration')
    axes[0,0].set_ylabel('Consciousness Level')
    axes[0,0].set_ylim([0, 1])
    axes[0,0].grid(True, alpha=0.3)
    
    # Quality Scores
    axes[0,1].bar(demo_names, quality_scores, color='green', alpha=0.7)
    axes[0,1].set_title('Analysis Quality Scores')
    axes[0,1].set_ylabel('Quality Score')
    axes[0,1].set_ylim([0, 1])
    axes[0,1].grid(True, alpha=0.3)
    
    # ATP Consumption
    axes[1,0].bar(demo_names, atp_consumption, color='orange', alpha=0.7)
    axes[1,0].set_title('ATP Consumption by Demonstration')
    axes[1,0].set_ylabel('ATP Units')
    axes[1,0].grid(True, alpha=0.3)
    
    # Biological Intelligence
    axes[1,1].bar(demo_names, biological_intelligence, color='blue', alpha=0.7)
    axes[1,1].set_title('Biological Intelligence Scores')
    axes[1,1].set_ylabel('Biological Intelligence')
    axes[1,1].set_ylim([0, 1])
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

async def main():
    """Main demonstration function"""
    print("🌌 SIGHTHOUND-AUTOBAHN INTEGRATION DEMONSTRATION")
    print("=" * 80)
    print("This demonstration shows how Sighthound's GPS trajectory analysis")
    print("integrates with Autobahn's consciousness-aware probabilistic reasoning.")
    print("=" * 80)
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    # Run demonstrations
    results = []
    
    try:
        # Basic integration
        result1 = await demonstrate_basic_integration()
        results.append(result1)
        
        # Complex reasoning
        result2 = await demonstrate_complex_reasoning()
        results.append(result2)
        
        # Threat detection
        result3 = await demonstrate_threat_detection()
        results.append(result3)
        
        # Consciousness emergence
        result4 = await demonstrate_consciousness_emergence()
        results.append(result4)
        
        # Create performance comparison
        create_performance_comparison(results)
        
        print("\n🎯 DEMONSTRATION SUMMARY")
        print("-" * 50)
        print("✅ Basic integration with consciousness assessment")
        print("✅ Complex probabilistic reasoning delegation")
        print("✅ Biological immune system threat detection")
        print("✅ Consciousness emergence detection")
        print("✅ Performance comparison visualization")
        print("\n📊 Visualizations saved to output/ directory:")
        print("   - simple_trajectory.png")
        print("   - complex_trajectory.png")
        print("   - adversarial_trajectory.png")
        print("   - consciousness_trajectory.png")
        print("   - performance_comparison.png")
        
        print("\n🚀 INTEGRATION SUCCESS!")
        print("Sighthound and Autobahn are successfully integrated for")
        print("consciousness-aware GPS trajectory analysis with biological intelligence!")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        print("\nThis may be due to:")
        print("1. Autobahn binary not available")
        print("2. Network connection issues")
        print("3. Configuration problems")
        print("\nCheck the logs for more details.")

if __name__ == "__main__":
    asyncio.run(main()) 