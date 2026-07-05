"""
FWDC Algorithm Validation and Experiments
Tests the algorithm on various graph configurations and records results.
"""

import json
import time
import random
import math
from typing import Dict, List
from fwdc_algorithm import Graph, FWDC, FuzzyInterval
import os


def experiment_small_graph() -> Dict:
    """Test on a small 4-node graph (toy example from paper)."""
    print("=== Experiment 1: Small 4-Node Graph ===")

    graph = Graph(beta0=0.5)

    # Add nodes: s, a, b, t
    graph.add_node(0, "s", 0.0, 0.0)
    graph.add_node(1, "a", 1.0, 0.0)
    graph.add_node(2, "b", 0.0, 1.0)
    graph.add_node(3, "t", 1.0, 1.0)

    # Synthesize edges (on-demand will happen during search)
    # Manually for illustration:
    edges = [
        (0, 1),  # s -> a
        (0, 2),  # s -> b
        (1, 3),  # a -> t
        (2, 3),  # b -> t
        (1, 2),  # a -> b
    ]

    for u, v in edges:
        graph.synthesize_edge(u, v)

    # Run FWDC
    fwdc = FWDC(graph)
    start_time = time.time()
    result = fwdc.find_shortest_path(source=0, sink=3)
    elapsed = time.time() - start_time

    result['experiment'] = 'small_4_node_graph'
    result['nodes'] = 4
    result['edges'] = len(graph.edges)
    result['time_seconds'] = elapsed
    result['beta0'] = graph.beta0

    print(f"Path: {result['path']}")
    print(f"Cost: [{result['cost_min']:.4f}, {result['cost_max']:.4f}]")
    print(f"Optimality gap: {result['optimality_gap']:.4f}")
    print(f"Ruled out: {result['ruled_out_nodes']}")
    print(f"Time: {elapsed:.4f}s")
    print()

    return result


def experiment_grid_graph(size: int) -> Dict:
    """Test on a grid graph of size x size."""
    print(f"=== Experiment 2: {size}x{size} Grid Graph ===")

    graph = Graph(beta0=0.3)

    # Create grid nodes
    node_id = 0
    node_map = {}
    for i in range(size):
        for j in range(size):
            node_map[(i, j)] = node_id
            graph.add_node(node_id, f"n_{i}_{j}", float(i), float(j))
            node_id += 1

    # Add edges (4-connectivity)
    edge_count = 0
    for i in range(size):
        for j in range(size):
            current_id = node_map[(i, j)]
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    neighbor_id = node_map[(ni, nj)]
                    graph.synthesize_edge(current_id, neighbor_id)
                    edge_count += 1

    # Source: top-left, Sink: bottom-right
    source = node_map[(0, 0)]
    sink = node_map[(size-1, size-1)]

    fwdc = FWDC(graph)
    start_time = time.time()
    result = fwdc.find_shortest_path(source=source, sink=sink)
    elapsed = time.time() - start_time

    result['experiment'] = f'grid_{size}x{size}'
    result['nodes'] = len(graph.nodes)
    result['edges'] = len(graph.edges)
    result['time_seconds'] = elapsed
    result['beta0'] = graph.beta0

    print(f"Grid size: {size}x{size}")
    print(f"Nodes: {result['nodes']}, Edges: {result['edges']}")
    print(f"Cost: [{result['cost_min']:.4f}, {result['cost_max']:.4f}]")
    print(f"Optimality gap: {result['optimality_gap']:.4f}")
    print(f"Ruled out: {result['num_ruled_out']}")
    print(f"Synthesized edges: {result['synthesized_edges']}")
    print(f"Time: {elapsed:.4f}s")
    print()

    return result


def experiment_random_graph(num_nodes: int, edge_density: float) -> Dict:
    """Test on a random graph."""
    print(f"=== Experiment 3: Random Graph ({num_nodes} nodes, {edge_density:.1%} density) ===")

    graph = Graph(beta0=0.2)

    # Create random nodes
    random.seed(42)
    for i in range(num_nodes):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        graph.add_node(i, f"n_{i}", x, y)

    # Add edges based on density
    edge_count = 0
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if random.random() < edge_density:
                graph.synthesize_edge(i, j)
                graph.synthesize_edge(j, i)  # Bidirectional
                edge_count += 1

    # Source: node 0, Sink: node (num_nodes-1)
    source = 0
    sink = num_nodes - 1

    fwdc = FWDC(graph)
    start_time = time.time()
    result = fwdc.find_shortest_path(source=source, sink=sink)
    elapsed = time.time() - start_time

    result['experiment'] = f'random_{num_nodes}_nodes'
    result['nodes'] = len(graph.nodes)
    result['edges'] = len(graph.edges)
    result['edge_density'] = edge_density
    result['time_seconds'] = elapsed
    result['beta0'] = graph.beta0

    print(f"Nodes: {result['nodes']}, Edges: {result['edges']}")
    print(f"Cost: [{result['cost_min']:.4f}, {result['cost_max']:.4f}]")
    print(f"Optimality gap: {result['optimality_gap']:.4f}")
    print(f"Ruled out: {result['num_ruled_out']}")
    print(f"Synthesized edges: {result['synthesized_edges']}")
    print(f"Time: {elapsed:.4f}s")
    print()

    return result


def experiment_beta0_sensitivity() -> Dict:
    """Test sensitivity to beta0 parameter."""
    print("=== Experiment 4: Beta0 Sensitivity ===")

    results_by_beta = {}

    graph_base = Graph(beta0=0.1)
    # Use a fixed graph
    for i in range(10):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        graph_base.add_node(i, f"n_{i}", x, y)

    for i in range(10):
        for j in range(i+1, 10):
            if random.random() < 0.3:
                graph_base.synthesize_edge(i, j)
                graph_base.synthesize_edge(j, i)

    for beta0 in [0.05, 0.1, 0.2, 0.5, 1.0]:
        graph = Graph(beta0=beta0)
        graph.nodes = graph_base.nodes.copy()
        graph.edges = {k: FuzzyInterval(v.lower, v.upper) for k, v in graph_base.edges.items()}

        fwdc = FWDC(graph)
        start_time = time.time()
        result = fwdc.find_shortest_path(source=0, sink=9)
        elapsed = time.time() - start_time

        result['beta0'] = beta0
        result['time_seconds'] = elapsed
        results_by_beta[str(beta0)] = result

        print(f"Beta0={beta0}: Iterations={result['iterations']}, "
              f"Ruled out={result['num_ruled_out']}, Gap={result['optimality_gap']:.4f}, "
              f"Time={elapsed:.4f}s")

    print()
    return {
        'experiment': 'beta0_sensitivity',
        'results_by_beta': results_by_beta
    }


def experiment_scaling() -> Dict:
    """Test algorithm scaling with graph size."""
    print("=== Experiment 5: Scaling Analysis ===")

    scaling_results = []

    for size in [5, 10, 15]:
        graph = Graph(beta0=0.2)

        # Create grid
        node_id = 0
        node_map = {}
        for i in range(size):
            for j in range(size):
                node_map[(i, j)] = node_id
                graph.add_node(node_id, f"n_{i}_{j}", float(i), float(j))
                node_id += 1

        # Add 4-connectivity
        for i in range(size):
            for j in range(size):
                current_id = node_map[(i, j)]
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < size and nj < size:
                        neighbor_id = node_map[(ni, nj)]
                        graph.synthesize_edge(current_id, neighbor_id)
                        graph.synthesize_edge(neighbor_id, current_id)

        source = node_map[(0, 0)]
        sink = node_map[(size-1, size-1)]

        fwdc = FWDC(graph)
        start_time = time.time()
        result = fwdc.find_shortest_path(source=source, sink=sink)
        elapsed = time.time() - start_time

        record = {
            'grid_size': size,
            'nodes': len(graph.nodes),
            'edges': len(graph.edges),
            'synthesized_edges': result['synthesized_edges'],
            'iterations': result['iterations'],
            'ruled_out': result['num_ruled_out'],
            'time_seconds': elapsed,
            'cost_gap': result['optimality_gap']
        }

        scaling_results.append(record)

        print(f"Grid {size}x{size}: Nodes={record['nodes']}, "
              f"Synthesized={record['synthesized_edges']}/{record['edges']}, "
              f"Time={elapsed:.4f}s")

    print()
    return {
        'experiment': 'scaling',
        'results': scaling_results
    }


def experiment_on_demand_efficiency() -> Dict:
    """Verify on-demand synthesis reduces storage."""
    print("=== Experiment 6: On-Demand Synthesis Efficiency ===")

    size = 20
    graph = Graph(beta0=0.15)

    # Create grid
    node_id = 0
    node_map = {}
    for i in range(size):
        for j in range(size):
            node_map[(i, j)] = node_id
            graph.add_node(node_id, f"n_{i}_{j}", float(i), float(j))
            node_id += 1

    # Add 4-connectivity
    total_possible_edges = 0
    for i in range(size):
        for j in range(size):
            current_id = node_map[(i, j)]
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < size and 0 <= nj < size:
                    total_possible_edges += 1

    # Run algorithm
    source = node_map[(0, 0)]
    sink = node_map[(size-1, size-1)]

    fwdc = FWDC(graph)
    start_time = time.time()
    result = fwdc.find_shortest_path(source=source, sink=sink)
    elapsed = time.time() - start_time

    synthesized = result['synthesized_edges']
    efficiency_ratio = synthesized / total_possible_edges

    print(f"Grid {size}x{size}")
    print(f"Total possible edges: {total_possible_edges}")
    print(f"Synthesized edges: {synthesized}")
    print(f"Efficiency ratio: {efficiency_ratio:.2%}")
    print(f"Storage reduction: {(1 - efficiency_ratio)*100:.1f}%")
    print(f"Time: {elapsed:.4f}s")
    print()

    return {
        'experiment': 'on_demand_efficiency',
        'grid_size': size,
        'total_possible_edges': total_possible_edges,
        'synthesized_edges': synthesized,
        'efficiency_ratio': efficiency_ratio,
        'time_seconds': elapsed
    }


def run_all_experiments() -> Dict:
    """Run all experiments and collect results."""
    print("\n" + "="*60)
    print("FWDC Algorithm Validation Suite")
    print("="*60 + "\n")

    all_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'experiments': []
    }

    # Experiment 1: Small graph
    try:
        result = experiment_small_graph()
        all_results['experiments'].append(result)
    except Exception as e:
        print(f"Experiment 1 failed: {e}\n")

    # Experiment 2: Grid graphs
    for size in [5, 10]:
        try:
            result = experiment_grid_graph(size)
            all_results['experiments'].append(result)
        except Exception as e:
            print(f"Experiment 2 (grid {size}x{size}) failed: {e}\n")

    # Experiment 3: Random graphs
    for num_nodes, density in [(20, 0.2), (30, 0.15)]:
        try:
            result = experiment_random_graph(num_nodes, density)
            all_results['experiments'].append(result)
        except Exception as e:
            print(f"Experiment 3 (random {num_nodes} nodes) failed: {e}\n")

    # Experiment 4: Beta0 sensitivity
    try:
        result = experiment_beta0_sensitivity()
        all_results['experiments'].append(result)
    except Exception as e:
        print(f"Experiment 4 failed: {e}\n")

    # Experiment 5: Scaling
    try:
        result = experiment_scaling()
        all_results['experiments'].append(result)
    except Exception as e:
        print(f"Experiment 5 failed: {e}\n")

    # Experiment 6: On-demand efficiency
    try:
        result = experiment_on_demand_efficiency()
        all_results['experiments'].append(result)
    except Exception as e:
        print(f"Experiment 6 failed: {e}\n")

    return all_results


if __name__ == '__main__':
    # Run experiments
    results = run_all_experiments()

    # Save results to JSON
    output_file = 'fwdc_validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("="*60)
    print(f"Results saved to {output_file}")
    print("="*60)

    # Print summary statistics
    print("\nSummary Statistics:")
    for exp in results['experiments']:
        if 'experiment' in exp:
            print(f"  {exp.get('experiment', 'unknown')}: "
                  f"time={exp.get('time_seconds', 'N/A'):.4f if isinstance(exp.get('time_seconds'), float) else 'N/A'}s")
