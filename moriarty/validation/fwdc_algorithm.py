"""
Fuzzy-Weighted Deterministic Closure (FWDC) Shortest Path Algorithm
Core implementation for validation and testing.
"""

import heapq
import math
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import json


@dataclass
class FuzzyInterval:
    """Represents an interval [lower, upper] for fuzzy weights."""
    lower: float
    upper: float

    def __init__(self, lower: float, upper: float):
        assert lower <= upper, f"Invalid interval: [{lower}, {upper}]"
        self.lower = lower
        self.upper = upper

    def width(self) -> float:
        """Width of the interval."""
        return self.upper - self.lower

    def midpoint(self) -> float:
        """Midpoint of the interval."""
        return (self.lower + self.upper) / 2

    def is_separated_by(self, other: 'FuzzyInterval', beta0: float) -> bool:
        """Check if two intervals are beta0-separated."""
        return self.lower > other.upper + beta0 or other.lower > self.upper + beta0

    def __repr__(self) -> str:
        return f"[{self.lower:.4f}, {self.upper:.4f}]"


@dataclass
class Node:
    """Represents a node in the graph with coordinates."""
    id: int
    name: str
    x: float
    y: float

    def euclidean_distance(self, other: 'Node') -> float:
        """Compute Euclidean distance to another node."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


class Graph:
    """Weighted directed graph with fuzzy edges."""

    def __init__(self, beta0: float):
        """Initialize graph with resolution floor beta0."""
        self.beta0 = beta0
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], FuzzyInterval] = {}

    def add_node(self, node_id: int, name: str, x: float, y: float):
        """Add a node to the graph."""
        self.nodes[node_id] = Node(node_id, name, x, y)

    def synthesize_edge(self, u_id: int, v_id: int) -> FuzzyInterval:
        """Synthesize fuzzy weight for edge (u, v) on-demand."""
        if (u_id, v_id) in self.edges:
            return self.edges[(u_id, v_id)]

        u = self.nodes[u_id]
        v = self.nodes[v_id]

        # Intrinsic distance
        d = u.euclidean_distance(v)

        # Fuzzy weight: [d - beta0, d + beta0]
        weight = FuzzyInterval(max(0, d - self.beta0), d + self.beta0)
        self.edges[(u_id, v_id)] = weight

        return weight

    def get_neighbors(self, node_id: int) -> List[int]:
        """Get neighbors of a node."""
        neighbors = set()
        for (u, v) in self.edges.keys():
            if u == node_id:
                neighbors.add(v)
            elif v == node_id:
                neighbors.add(u)
        return list(neighbors)

    def dijkstra_with_bounds(self, source: int, sink: int, excluded_node: Optional[int] = None,
                            use_lower: bool = True) -> float:
        """
        Dijkstra's algorithm using either lower or upper bounds of fuzzy weights.
        Optionally exclude a node from the search.
        """
        if excluded_node is not None and source == excluded_node:
            return float('inf')

        dist = {node_id: float('inf') for node_id in self.nodes}
        dist[source] = 0
        pq = [(0, source)]
        visited = set()

        while pq:
            d, u = heapq.heappop(pq)

            if u in visited:
                continue
            visited.add(u)

            if u == sink:
                return d

            # Explore neighbors
            for v_id in self.nodes:
                if (u, v_id) not in self.edges and (v_id, u) not in self.edges:
                    continue

                if v_id in visited or (excluded_node is not None and v_id == excluded_node):
                    continue

                # Get edge weight (handle both directions)
                if (u, v_id) in self.edges:
                    edge_weight = self.edges[(u, v_id)]
                else:
                    edge_weight = self.edges.get((v_id, u))
                    if edge_weight is None:
                        continue

                # Use lower or upper bound
                cost = edge_weight.lower if use_lower else edge_weight.upper

                if dist[u] + cost < dist[v_id]:
                    dist[v_id] = dist[u] + cost
                    heapq.heappush(pq, (dist[v_id], v_id))

        return float('inf')

    def compute_separation_cost(self, node_id: int, source: int, sink: int) -> Tuple[float, float]:
        """
        Compute fuzzy separation cost of a node.
        Returns (sigma_min, sigma_max) = cost to avoid the node.
        """
        # sigma_min: minimum cost to avoid node (using lower bounds)
        sigma_min = self.dijkstra_with_bounds(source, sink, excluded_node=node_id, use_lower=True)

        # sigma_max: maximum cost to avoid node (using upper bounds)
        sigma_max = self.dijkstra_with_bounds(source, sink, excluded_node=node_id, use_lower=False)

        return (sigma_min, sigma_max)


class FWDC:
    """Fuzzy-Weighted Deterministic Closure shortest path algorithm."""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.weights_dict: Dict[Tuple[int, int], FuzzyInterval] = {}
        self.ruled_out: Set[int] = set()
        self.synthesized_edges_count = 0
        self.iterations = 0
        self.separation_costs: Dict[int, Tuple[float, float]] = {}

    def find_shortest_path(self, source: int, sink: int,
                          initial_path: Optional[List[int]] = None) -> Dict:
        """
        Find shortest path using FWDC algorithm.

        Returns:
            Dict with 'path', 'cost_min', 'cost_max', 'ruled_out', 'optimality_gap', 'iterations'
        """
        # Initial path: if not provided, use greedy heuristic
        if initial_path is None:
            initial_path = self._greedy_path(source, sink)

        current_path = initial_path
        self.iterations = 0
        uncertain_nodes = set(self.graph.nodes.keys()) - {source, sink}

        while uncertain_nodes:
            self.iterations += 1
            improved = False

            # Synthesize edges for all nodes in current path
            for node_id in current_path:
                if node_id not in uncertain_nodes:
                    continue

                # Synthesize edges to neighbors
                for neighbor_id in self.graph.nodes:
                    if neighbor_id == node_id:
                        continue

                    edge_key = (node_id, neighbor_id)
                    if edge_key not in self.weights_dict:
                        weight = self.graph.synthesize_edge(node_id, neighbor_id)
                        self.weights_dict[edge_key] = weight
                        self.synthesized_edges_count += 1

                # Compute separation cost
                sigma_min, sigma_max = self.graph.compute_separation_cost(node_id, source, sink)
                self.separation_costs[node_id] = (sigma_min, sigma_max)

            # Find worst node in current path (highest sigma_min)
            worst_node = max(
                (n for n in current_path if n != source and n != sink),
                key=lambda n: self.separation_costs.get(n, (0, 0))[0],
                default=None
            )

            if worst_node is None:
                break

            worst_sigma = FuzzyInterval(*self.separation_costs[worst_node])

            # Check if any uncertain node has deterministically separated regions
            for node_id in list(uncertain_nodes):
                if node_id in self.separation_costs:
                    node_sigma = FuzzyInterval(*self.separation_costs[node_id])

                    # Check for beta0-separation
                    if node_sigma.is_separated_by(worst_sigma, self.graph.beta0):
                        # Node is ruled out
                        self.ruled_out.add(node_id)
                        uncertain_nodes.discard(node_id)

                        # Recompute path avoiding this node
                        current_path = self._recompute_path_avoiding(source, sink, self.ruled_out)
                        improved = True
                        break

            if not improved:
                # No deterministic separation found
                break

        # Compute final cost
        cost_min = self._compute_path_cost(current_path, use_lower=True)
        cost_max = self._compute_path_cost(current_path, use_lower=False)

        # Optimality gap
        optimality_gap = cost_max - cost_min if cost_min != float('inf') else float('inf')

        return {
            'path': current_path,
            'cost_min': cost_min,
            'cost_max': cost_max,
            'optimality_gap': optimality_gap,
            'ruled_out_nodes': list(self.ruled_out),
            'num_ruled_out': len(self.ruled_out),
            'synthesized_edges': self.synthesized_edges_count,
            'iterations': self.iterations
        }

    def _greedy_path(self, source: int, sink: int) -> List[int]:
        """Simple greedy path heuristic (nearest neighbor)."""
        path = [source]
        current = source
        visited = {source}

        while current != sink:
            # Find nearest unvisited neighbor to sink
            best_next = None
            best_dist = float('inf')

            for node_id in self.graph.nodes:
                if node_id not in visited:
                    # Distance to sink
                    dist_to_sink = self.graph.nodes[current].euclidean_distance(self.graph.nodes[node_id])
                    if dist_to_sink < best_dist:
                        best_dist = dist_to_sink
                        best_next = node_id

            if best_next is None:
                # If no unvisited neighbors, go directly to sink
                path.append(sink)
                break
            else:
                path.append(best_next)
                visited.add(best_next)
                current = best_next

        return path

    def _recompute_path_avoiding(self, source: int, sink: int, excluded: Set[int]) -> List[int]:
        """Recompute path avoiding a set of nodes using Dijkstra."""
        # Simplified: use BFS to find any path avoiding excluded nodes
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            node, path = queue.popleft()

            if node == sink:
                return path

            for next_node in self.graph.nodes:
                if next_node not in visited and next_node not in excluded:
                    if (node, next_node) in self.graph.edges or (next_node, node) in self.graph.edges:
                        visited.add(next_node)
                        queue.append((next_node, path + [next_node]))

        # Fallback: return original path
        return [source, sink]

    def _compute_path_cost(self, path: List[int], use_lower: bool = True) -> float:
        """Compute total cost of a path."""
        total = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]

            # Get edge weight
            if (u, v) in self.graph.edges:
                weight = self.graph.edges[(u, v)]
            elif (v, u) in self.graph.edges:
                weight = self.graph.edges[(v, u)]
            else:
                return float('inf')

            cost = weight.lower if use_lower else weight.upper
            total += cost

        return total
