using System;
using System.Collections.Generic;
using System.Linq;

namespace AStarAlgorithm
{
    public class Node
    {
        public int X { get; }
        public int Y { get; }
        public string Name { get; }
        public Dictionary<Node, (double RealDistance, double EuclideanDistance)> NeighborCosts { get; }

        public Node(int x, int y, string name)
        {
            X = x;
            Y = y;
            Name = name;
            NeighborCosts = new Dictionary<Node, (double RealDistance, double EuclideanDistance)>();
        }

        public void AddNeighbor(Node neighbor, double realDistance)
        {
            double euclideanDistance = CalculateEuclideanDistance(neighbor);
            NeighborCosts[neighbor] = (realDistance, euclideanDistance);
        }

        public (double RealDistance, double EuclideanDistance) GetNeighborCost(Node neighbor)
        {
            return NeighborCosts.TryGetValue(neighbor, out var cost) ? cost : (double.PositiveInfinity, double.PositiveInfinity);
        }

        public override bool Equals(object? obj)
        {
            if (obj is null || GetType() != obj.GetType())
            {
                return false;
            }

            var other = (Node)obj;
            return X == other.X && Y == other.Y && Name == other.Name;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(X, Y, Name);
        }

        public override string ToString()
        {
            return Name;
        }

        private double CalculateEuclideanDistance(Node other)
        {
            return Math.Sqrt(Math.Pow(X - other.X, 2) + Math.Pow(Y - other.Y, 2));
        }
    }

    public class Graph
    {
        private readonly Dictionary<Node, List<Node>> adjacencyList;

        public Graph()
        {
            adjacencyList = new Dictionary<Node, List<Node>>();
        }

        public void AddEdge(Node source, Node destination, double realDistance)
        {
            if (!adjacencyList.TryGetValue(source, out var neighbors))
            {
                neighbors = new List<Node>();
                adjacencyList[source] = neighbors;
            }
            neighbors.Add(destination);
            source.AddNeighbor(destination, realDistance);

            if (!adjacencyList.TryGetValue(destination, out neighbors))
            {
                neighbors = new List<Node>();
                adjacencyList[destination] = neighbors;
            }
            neighbors.Add(source);
            destination.AddNeighbor(source, realDistance);
        }

        public IEnumerable<Node> GetNeighbors(Node node)
        {
            return adjacencyList.TryGetValue(node, out var neighbors) ? neighbors : Enumerable.Empty<Node>();
        }

        public IEnumerable<Node> GetNodes()
        {
            return adjacencyList.Keys;
        }
    }

    public class AStarPathfinding
    {
        public List<Node> FindPath(Graph graph, Node start, List<Node> objectives, Func<Node, Node, double> heuristic, out double totalCost, out HashSet<Node> openNodes, out HashSet<Node> closedNodes, bool useInadmissibleHeuristic = false)
        {
            var path = new List<Node>();
            var visited = new HashSet<Node>();
            var currentPosition = start;
            totalCost = 0;
            openNodes = new HashSet<Node>();
            closedNodes = new HashSet<Node>();

            while (objectives.Any())
            {
                var objectiveCosts = new Dictionary<Node, double>();
                foreach (var objective in objectives)
                {
                    var (partialPath, partialCost) = FindShortestPath(graph, currentPosition, objective, heuristic, out var partialTotalCost, openNodes, closedNodes, useInadmissibleHeuristic);
                    objectiveCosts[objective] = partialTotalCost;
                }

                var nextObjective = objectiveCosts.OrderBy(kv => kv.Value).First().Key;

                var (partialPathToObjective, partialCostToObjective) = FindShortestPath(graph, currentPosition, nextObjective, heuristic, out var partialTotalCostToObjective, openNodes, closedNodes, useInadmissibleHeuristic);

                if (path.Count > 0 && partialPathToObjective.Count > 0 && path.Last().Equals(partialPathToObjective.First()))
                {
                    partialPathToObjective.RemoveAt(0);
                }

                path.AddRange(partialPathToObjective);
                visited.Add(nextObjective);
                currentPosition = nextObjective;
                objectives.Remove(nextObjective);

                totalCost += partialCostToObjective;

                // Break if returning to the starting point and no objectives remain
                if (currentPosition.Equals(start) && !objectives.Any())
                {
                    break;
                }
            }

            var (returnPath, returnCost) = FindShortestPath(graph, currentPosition, start, heuristic, out var returnTotalCost, openNodes, closedNodes, useInadmissibleHeuristic);
            if (returnPath.Any())
            {
                returnPath.RemoveAt(0);
            }
            path.AddRange(returnPath);
            totalCost += returnCost;

            return path;
        }

        private (List<Node>, double) FindShortestPath(Graph graph, Node start, Node goal, Func<Node, Node, double> heuristic, out double totalCost, HashSet<Node> openNodes, HashSet<Node> closedNodes, bool useInadmissibleHeuristic)
        {
            var openSet = new HashSet<Node>();
            var closedSet = new HashSet<Node>();
            var gScore = new Dictionary<Node, double>();
            var fScore = new Dictionary<Node, double>();
            var cameFrom = new Dictionary<Node, Node>();

            foreach (var node in graph.GetNodes())
            {
                gScore[node] = double.PositiveInfinity;
                fScore[node] = double.PositiveInfinity;
            }

            gScore[start] = 0;
            fScore[start] = heuristic(start, goal) * (useInadmissibleHeuristic ? 21 : 1);
            openSet.Add(start);
            openNodes.Add(start);

            while (openSet.Any())
            {
                var current = GetLowestFScoreNode(openSet, fScore);

                if (current.Equals(goal))
                {
                    totalCost = gScore[current];
                    return (ReconstructPath(cameFrom, current), totalCost);
                }

                openSet.Remove(current);
                closedSet.Add(current);
                closedNodes.Add(current);

                foreach (var neighbor in graph.GetNeighbors(current))
                {
                    if (closedSet.Contains(neighbor))
                        continue;

                    var tentativeGScore = gScore[current] + CalculateCost(current, neighbor);

                    if (tentativeGScore < gScore[neighbor])
                    {
                        cameFrom[neighbor] = current;
                        gScore[neighbor] = tentativeGScore;
                        fScore[neighbor] = gScore[neighbor] + heuristic(neighbor, goal) * (useInadmissibleHeuristic ? 20 : 1);

                        if (!openSet.Contains(neighbor))
                        {
                            openSet.Add(neighbor);
                            openNodes.Add(neighbor);
                        }
                    }
                }
            }

            totalCost = 0;
            return (new List<Node>(), totalCost);
        }

        private Node GetLowestFScoreNode(HashSet<Node> openSet, Dictionary<Node, double> fScore)
        {
            return openSet.OrderBy(node => fScore[node]).First();
        }

        private List<Node> ReconstructPath(Dictionary<Node, Node> cameFrom, Node current)
        {
            var path = new List<Node> { current };
            while (cameFrom.TryGetValue(current, out var previous))
            {
                current = previous;
                path.Insert(0, current);
            }
            return path;
        }

        private double CalculateCost(Node current, Node neighbor)
        {
            return current.GetNeighborCost(neighbor).RealDistance;
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var graph = new Graph();
            var node1 = new Node(4, 8, "Entrada/Saida");
            var node2 = new Node(8, 2, "Grãos");
            var node3 = new Node(12, 4, "Bebidas");
            var node4 = new Node(10, 6, "Biscoitos");
            var node5 = new Node(8, 8, "Condimentos");
            var node6 = new Node(8, 10, "Doces");
            var node7 = new Node(6, 11, "Higiene");
            var node8 = new Node(14, 3, "Frutas");
            var node9 = new Node(14, 7, "Laticinios");
            var node10 = new Node(14, 11, "Carnes");

            graph.AddEdge(node1, node2, 10);
            graph.AddEdge(node1, node3, 12);
            graph.AddEdge(node1, node4, 8);
            graph.AddEdge(node1, node5, 4);
            graph.AddEdge(node1, node6, 6);
            graph.AddEdge(node1, node7, 5);
            graph.AddEdge(node2, node3, 6);
            graph.AddEdge(node2, node8, 7);
            graph.AddEdge(node3, node4, 4);
            graph.AddEdge(node3, node8, 3);
            graph.AddEdge(node8, node9, 4);
            graph.AddEdge(node4, node5, 4);
            graph.AddEdge(node4, node9, 5);
            graph.AddEdge(node5, node6, 2);
            graph.AddEdge(node5, node9, 7);
            graph.AddEdge(node9, node10, 4);
            graph.AddEdge(node6, node7, 3);
            graph.AddEdge(node6, node10, 7);
            graph.AddEdge(node7, node10, 10);

            var nodes = new List<Node> { node1, node2, node3, node4, node5, node6, node7, node8, node9, node10 };

            // Interface de seleção de nós objetivos
            var selectedObjectives = SelectObjectives(nodes);

            var startNode = node1;

            var pathfinder = new AStarPathfinding();

            Console.WriteLine("Deseja utilizar a heurística inadmissível? (s/n)");
            var useInadmissibleHeuristic = Console.ReadLine().Trim().ToLower() == "s";

            var path = pathfinder.FindPath(graph, startNode, selectedObjectives, Heuristic, out var totalCost, out var openNodes, out var closedNodes, useInadmissibleHeuristic);

            Console.Clear();
            Console.WriteLine("Caminho encontrado:");
            foreach (var node in path)
            {
                Console.WriteLine(node);
            }

            Console.WriteLine($"\nCusto total: {totalCost}");
            Console.WriteLine("\nNós abertos:");
            foreach (var node in openNodes)
            {
                Console.WriteLine(node);
            }

            Console.WriteLine("\nNós fechados:");
            foreach (var node in closedNodes)
            {
                Console.WriteLine(node);
            }
        }

        static List<Node> SelectObjectives(List<Node> nodes)
        {
            var objectives = new List<Node>();
            while (true)
            {
                Console.Clear();
                Console.WriteLine("Selecione os nós objetivos (pressione Enter para finalizar):");
                for (int i = 0; i < nodes.Count; i++)
                {
                    Console.WriteLine($"{i + 1}. {nodes[i]}");
                }

                var input = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(input))
                {
                    break;
                }

                if (int.TryParse(input, out int selectedIndex) && selectedIndex > 0 && selectedIndex <= nodes.Count)
                {
                    var selectedNode = nodes[selectedIndex - 1];
                    if (!objectives.Contains(selectedNode))
                    {
                        objectives.Add(selectedNode);
                    }
                }
            }

            return objectives;
        }

        static double Heuristic(Node node, Node goal)
        {
            return Math.Sqrt(Math.Pow(node.X - goal.X, 2) + Math.Pow(node.Y - goal.Y, 2));
        }
    }
}