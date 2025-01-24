import heapq
import matplotlib.pyplot as plt
import unittest

class Road:
    """Define road with length, speed limit, and destination
    (beginning of next road)"""
    def __init__(self, destination, distance, speed_limit):
        """Initialize road"""
        self.destination = destination
        self.distance = distance
        self.speed_limit = speed_limit

    def travel_time(self):
        """Return time required to travel across road at maximum speed"""
        if self.speed_limit == 0:
            raise ZeroDivisionError("Speed limit must be greater than zero")
        return self.distance / self.speed_limit

class Graph:
    """Use graph to represent roads and intersections at specific points"""
    def __init__(self):
        """Initialize graph and store positions for visualization"""
        self.edges = {}
        self.positions = {}

    def add_road(self, start, end, distance, speed_limit):
        """Add a new road to the graph"""
        if start not in self.edges: # Initialize start if doesn't already exist
            self.edges[start] = []
        if end not in self.edges: # Initialize end if doesn't already exist
            self.edges[end] = []
        self.edges[start].append(Road(destination=end, distance=distance,
                                      speed_limit=speed_limit))
        self.edges[end].append(Road(destination=start, distance=distance,
                                    speed_limit=speed_limit))

    def set_position(self, intersection, x, y):
        """Set intersection's position at given coordinates"""
        self.positions[intersection] = (x, y)

    def dijkstra(self, start, goal, optimize_for='time'):
        """Dijkstra's algorithm finds shortest path considering
        speed limits and length of roads. Can optimize for distance or time"""
        # Check that desired start and end locations are actually in the graph
        if start not in self.edges or goal not in self.edges:
            raise ValueError("Start or goal node not in the graph.")

        queue = [(0, start, [])]
        shortest_paths = {start: 0}

        # Use greedy algorithm to find shortest path by looping through every node until all have been processed
        while queue:
            # Pop minimum node
            cur_value, cur_node, path = heapq.heappop(queue)
            path = path + [cur_node]

            if cur_node == goal: # Goal case
                return cur_value, path

            # Iterate over all neighboring nodes
            for road in self.edges[cur_node]:
                increment = road.travel_time() if optimize_for == 'time' \
                else road.distance
                if increment < 0: # Check for negative values
                  raise ValueError("Distance or travel time is negative")
                new_value = cur_value + increment

                # Update path and push to queue if neighbor is shorter
                if road.destination not in shortest_paths or new_value < \
                shortest_paths[road.destination]:
                    shortest_paths[road.destination] = new_value
                    heapq.heappush(queue, (new_value, road.destination, path))

        return float("inf"), [] # If no path possible

    def path_details(self, path):
        """Print out path details"""
        print("Path details:")
        for index in range(len(path) - 1):
            for road in self.edges[path[index]]:
                if road.destination == path[index + 1]:
                    print(f"{path[index]} -> {path[index+1]}, \
                    Distance: {road.distance} mi, \
                    Speed Limit: {road.speed_limit} mph, \
                    Travel Time: {road.travel_time():.2f} h")
                    break

    def visualize(self, path=None):
        """Make plot showing whole graph and shortest path"""
        plt.figure(figsize=(10, 6))

        # Plot all roads in light gray
        for start, roads in self.edges.items():
            for road in roads:
                start_pos = self.positions.get(start, (0, 0))
                end_pos = self.positions.get(road.destination, (0, 0))
                plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                         'gray', linestyle='--')

        # Highlight shortest path blue
        if path:
            for i in range(len(path) - 1):
                start, end = path[i], path[i + 1]
                start_pos, end_pos = self.positions[start], self.positions[end]
                plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                         'blue', linewidth=2)

        # Highlight interesction points red if in shortest path, black otherwise
        for intersection, pos in self.positions.items():
            plt.plot(pos[0], pos[1], 'o', color="red"
                     if intersection in path else "black")
            plt.text(pos[0], pos[1], f' {intersection}', fontsize=12,
                     verticalalignment='center', color="blue"
                     if intersection in path else "black")

        plt.title("Shortest Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

class Test(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        do_graph(self.graph)
        self.expected_path = ["A", "C", "D", "E"]
        self.expected_time = 0.77

    def test_path(self):
        self.assertEqual(self.expected_path, self.graph.dijkstra("A", "E")[1])

    def test_time(self):
        self.assertAlmostEqual(self.expected_time,
                               self.graph.dijkstra("A", "E")[0], delta=0.1)

def do_graph(graph):
    """Add list of roads and intersections to graph"""
    roads = [
        ("A", "B", 10, 60),
        ("A", "C", 15, 40),
        ("B", "C", 12, 50),
        ("B", "D", 15, 30),
        ("C", "D", 10, 70),
        ("C", "E", 25, 60),
        ("D", "E", 20, 80)]
    positions = {
        "A": (0, 0),
        "B": (10, 10),
        "C": (15, 5),
        "D": (25, 10),
        "E": (30, 0)}
    for start, end, distance, speed_limit in roads:
        graph.add_road(start, end, distance, speed_limit)
    for intersection, pos in positions.items():
        graph.set_position(intersection, *pos)

def main():
    """Run optimization"""
    graph = Graph()
    do_graph(graph)

    start = "A"
    goal = "E"
    optimize_for = "distance"  # Can optimize time or distance

    time, path = graph.dijkstra(start, goal, optimize_for)

    if path:
        print(f"Shortest Path from {start} to {goal} (Optimized for {optimize_for}):", end = " ")
        for item in path:
            print(item, end=" ")
        print(f"\nTotal {optimize_for}: {time:.2f}")

        graph.path_details(path)
        graph.visualize(path=path)
    else:
        print("No path possible from", start, "to", goal)

if __name__ == "__main__":
    main()
    unittest.main(argv=[''], exit=False)
