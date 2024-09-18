import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import networkx as nx

# Step 1: Voronoi Diagram Generation
def generate_voronoi(image, num_points=100):
    height, width, _ = image.shape
    points = np.random.rand(num_points, 2) * [width, height]
    
    vor = spatial.Voronoi(points)
    voronoi_plot_2d(vor)
    
    return vor, points

def voronoi_plot_2d(vor):
    fig, ax = plt.subplots()
    spatial.voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange', line_width=2)
    plt.show()

# Step 2: Graph Construction
def create_graph(points):
    G = nx.complete_graph(len(points))
    
    # Add positions to the nodes for visualization later
    pos = {i: (points[i][0], points[i][1]) for i in range(len(points))}
    
    # Visualize the complete graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='grey')
    plt.title("Complete Graph from Voronoi Cells")
    plt.show()
    
    return G, pos

# Step 3: Graph Optimization
def optimize_graph(G, pos):
    # Convert G.edges() to a list to use with np.random.choice
    edge_list = list(G.edges())
    
    # Get the number of edges to remove
    num_edges_to_remove = int(0.5 * len(edge_list))
    
    # Randomly choose indices of the edges to remove
    edge_indices_to_remove = np.random.choice(len(edge_list), size=num_edges_to_remove, replace=False)
    
    # Use the selected indices to get the edges to remove
    edges_to_remove = [edge_list[i] for i in edge_indices_to_remove]
    
    # Remove the selected edges
    G.remove_edges_from(edges_to_remove)
    
    # Assign random values to nodes
    node_values = {i: np.random.randint(-5, 5) for i in G.nodes()}
    
    # Visualize the optimized graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_color=[node_values[i] for i in G.nodes()], node_size=500, edge_color='grey', cmap=plt.cm.coolwarm)
    plt.title("Optimized Graph with Random Node Values")
    plt.show()
    
    return node_values



# Step 4: Ford-Fulkerson Flow Calculation
def ford_fulkerson_flow(G, node_values, pos):
    G_flow = nx.DiGraph()

    # Create a source and sink
    sources = [node for node in G.nodes if node_values[node] > 0]
    sinks = [node for node in G.nodes if node_values[node] < 0]

    for u, v in G.edges():
        G_flow.add_edge(u, v, capacity=np.random.randint(1, 10))  # Random capacities
    
    # Apply Ford-Fulkerson
    flow_dict = nx.maximum_flow_value(G_flow, _s=min(sources), _t=max(sinks))
    
    # Visualize the flow network
    plt.figure(figsize=(8, 8))
    nx.draw(G_flow, pos, with_labels=True, node_color='lightgreen', node_size=500, edge_color='red')
    plt.title(f"Flow Network with Maximum Flow: {flow_dict}")
    plt.show()

    return flow_dict

# Driver Code
def main():
    # Load an image
    image = cv2.imread('./bangalore-district-map.jpg')
    
    # Step 1: Generate Voronoi Diagram
    vor, points = generate_voronoi(image, num_points=50)
    
    # Step 2: Create Graph from Voronoi
    G, pos = create_graph(points)
    
    # Step 3: Optimize Graph and Assign Node Values
    node_values = optimize_graph(G, pos)
    
    # Step 4: Apply Ford-Fulkerson and Visualize Flow
    ford_fulkerson_flow(G, node_values, pos)

if __name__ == "__main__":
    main()
