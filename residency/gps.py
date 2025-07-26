from collections import defaultdict
import heapq


class Graph:
    """
    A directed, weighted graph to represent a road network with safety features.
    Each edge can include:
      - weight: the cost to traverse the edge (e.g., distance or time)
      - crowd level: low/medium/high (affects route preference)
      - safety tip: a learner-specific instructor message
      - blind spot: indicates if the transition has a blind spot
    """

    def __init__(self):
        # Adjacency list maps each node to a list of tuples:
        # (neighbor, weight, crowd_level, tip, blind_spot)
        self.adj_list = defaultdict(list)

    def add_edge(self, u, v, weight, crowd='low', tip='', blind_spot=False):
        """
        Adds a directed edge from node `u` to node `v`.

        :param u: Source node
        :param v: Destination node
        :param weight: Cost to traverse the edge (must be non-negative)
        :param crowd: Crowd level ('low', 'medium', 'high'), defaults to 'low'
        :param tip: Optional instructor tip for learner mode
        :param blind_spot: Boolean flag for blind spot alert
        :raises ValueError: if a negative weight is provided
        """
        if weight < 0:
            raise ValueError("Negative weights are not supported in Dijkstra's algorithm.")
        self.adj_list[u].append((v, weight, crowd, tip, blind_spot))

    def dijkstra(self, start, end, learner_mode=False):
        """
        Computes the shortest path from `start` to `end` using Dijkstra's algorithm.
        When `learner_mode` is enabled, it also collects safety tips and applies
        crowd-based bias to avoid high-traffic areas.
        Blind spot alerts are always collected for the final path.
        """
        # Early exit if start is invalid
        if start not in self.adj_list and start != end:
            return [], [], []

        # Handle same start and end node
        if start == end:
            return [start], [], []

        # Standard Dijkstra setup
        distances = {node: float('inf') for node in self.adj_list}
        distances[start] = 0
        previous = {node: None for node in self.adj_list}
        pq = [(0, start)]

        while pq:
            dist_u, u = heapq.heappop(pq)
            if dist_u > distances[u]:
                continue
            if u == end:
                break
            for v, w, crowd, tip, blind_spot in self.adj_list[u]:
                # In learner mode, skip blind spot edges if alternative exists
                # We'll handle this by adding a large penalty instead of skipping entirely
                if learner_mode:
                    crowd_bias = 0 if crowd == 'low' else w * 0.5
                    blind_spot_penalty = w * 10 if blind_spot else 0  # Heavy penalty for blind spots
                    bias = crowd_bias + blind_spot_penalty
                else:
                    bias = 0
                
                new_dist = dist_u + w + bias
                if new_dist < distances.get(v, float('inf')):
                    distances[v] = new_dist
                    previous[v] = u
                    heapq.heappush(pq, (new_dist, v))

        # Reconstruct path
        path = []
        cur = end
        while cur is not None:
            path.append(cur)
            cur = previous.get(cur)
        path.reverse()

        if not path or path[0] != start:
            return [], [], []

        # Collect tips and blind spot alerts along the chosen path
        tips = []
        blind_alerts = []
        for i in range(1, len(path)):
            u, v = path[i-1], path[i]
            for neigh, _, _, t, b in self.adj_list[u]:
                if neigh == v:
                    if learner_mode and t:
                        tips.append(t)
                    if b:
                        blind_alerts.append(f"Blind spot at {u} to {v}")
                    break

        return path, tips, blind_alerts