import os
import shutil
import glob
from pathlib import Path
import re
import subprocess
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, url_for
from collections import defaultdict
from matplotlib.patches import Wedge
import numpy as np
from node2vec import Node2Vec
import skfuzzy as fuzz
from cdlib import NodeClustering
from ahn_link import link_communities
from slpaalgorithm import find_communities
from slpav2 import SLPA
from cdlib.algorithms import (
    slpa,
    kclique,
    conga
)
from cdlib.evaluation import (
    overlapping_normalized_mutual_information_MGH,
    omega,
    f1,
    modularity_overlap,
    link_modularity
)

from evaluations import ( shen_modularity, lazar_modularity, NF1, mgh_onmi)

app = Flask(__name__)

GRAPH_PATH = "network.dat"
COMMUNITY_PATH = "community.dat"
COUNTER_FILE = "static/plot_counter.txt"
LFR_OUTPUT_DIR = 'lfr_output/'

node_embeddings_cache = None


@app.route("/generateEmbedding", methods=["POST"])
def generate_embedding():
    global node_embeddings_cache
    

    print("Generating embedding")
    # Read the graph data
    G = nx.read_edgelist(GRAPH_PATH, nodetype=int)
    dimensions = int(request.json.get("dimensions", 8))
    walk_length = int(request.json.get("walk_length", 50))
    num_walks = int(request.json.get("num_walks", 10))
    p = float(request.json.get("p", 1))
    q = float(request.json.get("q", 1))

    # Generate node embeddings using Node2Vec
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1, p=p, q=q)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Overwrite the previous embeddings with new ones
    node_embeddings_cache = np.array([model.wv[str(node)] for node in G.nodes()])

    return jsonify({
        "success": True,
        "message": "Embeddings generated and cached successfully."
    })


@app.route('/get_available_graphs')
def get_available_graphs_endpoint():
    graphs = get_available_graphs()  # Funkce pro získání seznamu složek
    return jsonify(graphs)

# Function to generate a unique folder name by adding an index
def get_unique_folder_name(base_folder):
    index = 1
    folder_name = base_folder
    while os.path.exists(folder_name):
        folder_name = f"{base_folder}_{index}"
        index += 1
    return folder_name

def get_available_graphs():
    return [d for d in os.listdir(LFR_OUTPUT_DIR) if os.path.isdir(os.path.join(LFR_OUTPUT_DIR, d))]

@app.route('/load_graph/<folder_name>', methods=['POST'])
def load_graph_endpoint(folder_name):

    # Construct paths for the network.dat and community.dat files in the selected folder
    folder_path = os.path.join(LFR_OUTPUT_DIR, folder_name)
    network_file_path = os.path.join(folder_path, 'network.dat')
    community_file_path = os.path.join(folder_path, 'community.dat')

    # Check if the folder exists and contains the required files
    if os.path.exists(folder_path) and os.path.exists(network_file_path) and os.path.exists(community_file_path):
        try:
            # Copy the network.dat and community.dat files from the selected folder to the root directory
            shutil.copy(network_file_path, GRAPH_PATH)
            shutil.copy(community_file_path, COMMUNITY_PATH)

            return 'Graf byl úspěšně načten.', 200
        except Exception as e:
            return f'Chyba při načítání grafu: {str(e)}', 400
    else:
        return 'Chyba: Soubory neexistují v dané složce.', 400


def get_next_plot_index():
    # Read the last index
    if os.path.exists(COUNTER_FILE):
        with open(COUNTER_FILE, "r") as f:
            index = int(f.read().strip())
    else:
        index = 0

    # Increment and save back
    with open(COUNTER_FILE, "w") as f:
        f.write(str(index + 1))

    return index



def align_partitions(detected, ground_truth):
    """ Ensure both partitions contain the same nodes. """
    detected_nodes = set(node for comm in detected for node in comm)
    ground_truth_nodes = set(node for comm in ground_truth for node in comm)

    all_nodes = detected_nodes.union(ground_truth_nodes)  # Get all unique nodes

    # Add missing nodes as singleton communities
    for node in all_nodes:
        if node not in detected_nodes:
            detected.append(frozenset([node]))  # Add as singleton
        if node not in ground_truth_nodes:
            ground_truth.append(frozenset([node]))  # Add as singleton

    return detected, ground_truth


def compute_metrics(graph, detected_communities, ground_truth_communities):
    """
    Compute evaluation metrics for community detection.
    """
    
    #print(ground_truth_communities)
    print(len(ground_truth_communities))

    shen = shen_modularity(graph, detected_communities)
    lazar_score = lazar_modularity(graph, detected_communities)
    my_f1_score = NF1(detected_communities, ground_truth_communities).get_f1()
    my_onmi = mgh_onmi(detected_communities, ground_truth_communities)

    #my_f1_score = f1score(detected_communities, ground_truth_communities)
    
    l_detected_nc, l_ground_truth_nc = align_partitions(detected_communities, ground_truth_communities)


    detected_nc = NodeClustering([list(comm) for comm in l_detected_nc], graph, "Detected")
    ground_truth_nc = NodeClustering(l_ground_truth_nc, graph, "Ground Truth")

    # Compute Metrics
    onmi_score = overlapping_normalized_mutual_information_MGH(detected_nc, ground_truth_nc).score
    f1_score = f1(detected_nc, ground_truth_nc).score
    modularity_score = modularity_overlap(graph, detected_nc).score
    omega_score = omega(detected_nc, ground_truth_nc).score


    #print("Shen: ")
    #print(shen)
    #print("Lazar: ")
    #print(lazar_score)
    #print(modularity_score)
    #print("F1")
    #print(f1_score)
    #print(my_f1_score)
    #print("ONMI")
    #print(onmi_score)
    #print(my_onmi)

    return {
        "onmi": round(onmi_score,4),
        "omega": round(omega_score,4),
        "f1": round(f1_score, 4),
        "lazar": round(lazar_score, 4),
        "shen2": round (shen, 4)
    }




# Read ground truth communities from community.dat
def load_ground_truth():
    ground_truth = defaultdict(set)
    with open(COMMUNITY_PATH, "r") as f:
        for line in f:
            parts = list(map(int, line.split()))
            node = parts[0]
            communities = parts[1:]
            for comm in communities:
                ground_truth[comm].add(node)

    return [set(nodes) for nodes in ground_truth.values()]



@app.route("/check_glibc")
def check_glibc():
    try:
        result = subprocess.run(["ldd", "--version"], capture_output=True, text=True)
        return f"<pre>{result.stdout}</pre>"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/generate', methods=['POST'])
def generate_graph():
    global LFR_PARAMETERS
    params = request.get_json()
    LFR_PARAMETERS = params.copy()

    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH) 
    if os.path.exists(COMMUNITY_PATH):
        os.remove(COMMUNITY_PATH) 


    cmd = ["./benchmark"]

    required_params = ["N", "k", "maxk", "mu"]

    # Add required parameters
    for param in required_params:
        if param in params and params[param].strip():
            cmd.append(f"-{param}")
            cmd.append(str(params[param]))
        else:
            return jsonify({"success": False, "message": f"Error: Missing required parameter '{param}'"})

    # Add optional parameters if provided
    optional_params = ["t1", "t2", "minc", "maxc", "on", "om"]
    for param in optional_params:
        if param in params and params[param].strip():
            cmd.append(f"-{param}")
            cmd.append(str(params[param]))

    try:
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)
        output = result.stdout.strip()

        if os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH):

            base_folder_name = os.path.join(LFR_OUTPUT_DIR, '_'.join(f"{key}_{value}" for key, value in params.items()))

            # Get a unique folder name if the base folder already exists
            output_folder = get_unique_folder_name(base_folder_name)
            os.makedirs(output_folder)  # Create the folder

            # Copy the generated files into the unique output folder
            shutil.copy(GRAPH_PATH, os.path.join(output_folder, 'network.dat'))
            shutil.copy(COMMUNITY_PATH, os.path.join(output_folder, 'community.dat'))

            match = re.search(r"\*{60,}\n(.*?)\*{60,}", output, re.DOTALL)
            info = match.group(1).strip() if match else "No benchmark details found."
            
            ground_truth_communities = load_ground_truth()
            num_communities = len(ground_truth_communities)

            return jsonify({
                "success": True,
                "message": "Graph successfully generated!",
                "benchmark_info": info,
                "num_communities": num_communities,  
                "LFR_parameters": LFR_PARAMETERS 
            })

        else:
            error_message = result.stderr.strip() if result.stderr else "Unknown error occurred."
            return jsonify({"success": False, "message": f"Error: {error_message}"})

    except Exception as e:
        return jsonify({"success": False, "message": f"Exception occurred: {str(e)}"})


import subprocess
from pathlib import Path

def run_oslom(graph_path: str, directed=False, extra_args=None, copra_runs=0,
              oslom_f_l_i=10, oslom_h_l_i=50, oslom_p_v_t=0.1, oslom_infomap=0, oslom_louvain=0,
              oslom_copra=0, oslom_singler=0, seed=10):
    graph_path = Path(graph_path).resolve()
    
    # Choose the right binary (undirected flag)
    algo = "./oslom_dir" if directed else "./oslom_undir"
    
    # Build the OSLOM command (always using unweighted -uw)
    cmd = [algo, "-f", str(graph_path), "-uw", "-seed", str(seed)]
    
    # Add COPRA if requested
    if copra_runs > 0:
        cmd += ["-copra", str(copra_runs)]

    # Add Infomap if requested
    if oslom_infomap > 0:
        cmd += ["-infomap", str(oslom_infomap)]
        
    # Add Louvain if requested
    if oslom_louvain > 0:
        cmd += ["-louvain", str(oslom_louvain)]
        
    # Add COPRA N if requested
    if oslom_copra > 0:
        cmd += ["-copra", str(oslom_copra)]
        
    # Add Singler flag if requested
    if oslom_singler > 0:
        cmd += ["-singlet"]

    # Add OSLOM-specific parameters
    cmd += [
        "-r", str(oslom_f_l_i),  # First level iterations
        "-hr", str(oslom_h_l_i),  # Higher level iterations
        "-t", str(oslom_p_v_t),   # P-value threshold
    ]
    
    # Add any extra arguments provided
    if extra_args:
        cmd += extra_args

    # Run OSLOM
    subprocess.run(cmd, check=True)
    
    # OSLOM output folder
    output_dir = graph_path.parent / f"{graph_path.name}_oslo_files"
    tp_path = output_dir / "tp"
    
    # Check if the output file exists
    if not tp_path.exists():
        raise FileNotFoundError(f"'tp' file not found at: {tp_path}")
    
    # Parse tp file
    communities = []
    with tp_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue  # Skip comment lines
            nodes = set(map(int, line.strip().split()))
            communities.append(nodes)

    return communities

@app.route("/reset", methods=["POST"])
def reset():

    for filepath in glob.glob("static/community_plot_*.svg"):
        os.remove(filepath)
    
    with open(COUNTER_FILE, "w") as f:
        f.write(str(0))

    main_plot = "static/community_plot.svg"
    if os.path.exists(main_plot):
        os.remove(main_plot)

    return jsonify({"success": True, "message": "Results deleted."})

@app.route("/detect", methods=["POST"])
def detect():
    if not (os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH)):
        return jsonify({"success": False, "message": "Graph has not been generated yet!"})

    G = nx.read_edgelist(GRAPH_PATH, nodetype=int)
    ground_truth_communities = load_ground_truth()

    algorithm = request.json.get("algorithm")
    N = int(request.json.get("N_iter", 1))
    params = {}

    all_results = []

    for i in range(N):
        if algorithm == "slpa" or algorithm == "slpa_cdlib":
            T = int(request.json.get("T", 40))
            r = float(request.json.get("r", 0.3))
            if algorithm == "slpa":
                detected_communities = find_communities(G, T, r)
            else:
                detected_communities = set(frozenset(c) for c in slpa(G, T, r).communities)
            params = {"T": T, "r": r}

        elif algorithm == "n2vfcm":
            clusters = int(request.json.get("clusters", 4))
            m = float(request.json.get("m", 1.5))
            threshold = float(request.json.get("threshold", 0.3))
            detected_communities = fuzzy_cmeans(G, clusters, m, threshold)
            params = {"m": m, "c": clusters, "t": threshold}
            #params = {"dim": dimensions, "wl": walk_length, "nw": num_walks, "p": p, "m": m, "c": clusters, "t": threshold}

        elif algorithm == "conga":
            number_communities = int(request.json.get("number_community", 4))
            detected_communities = set(frozenset(c) for c in conga(G, number_communities).communities)
            params = {"number_communities": number_communities}

        elif algorithm == "oslom":
            oslom_f_l_i = int(request.json.get("oslom_f_l_i", 10))
            oslom_h_l_i = int(request.json.get("oslom_h_l_i", 50))
            oslom_p_v_t = float(request.json.get("oslom_p_v_t", 0.1))
            oslom_infomap = int(request.json.get("oslom_infomap", 0))
            oslom_louvain = int(request.json.get("oslom_louvain", 0))
            oslom_copra = int(request.json.get("oslom_copra", 0))
            oslom_singler = int(request.json.get("oslom_singler", 0))
            detected_communities = run_oslom(
                GRAPH_PATH,
                directed=False,
                oslom_f_l_i=oslom_f_l_i,
                oslom_h_l_i=oslom_h_l_i,
                oslom_p_v_t=oslom_p_v_t,
                oslom_infomap=oslom_infomap,
                oslom_louvain=oslom_louvain,
                oslom_copra=oslom_copra,
                oslom_singler=oslom_singler,
            )
            params = {
                "r": oslom_f_l_i, "hr": oslom_h_l_i, "t": oslom_p_v_t,
                "i": oslom_infomap, "l": oslom_louvain, "c": oslom_copra, "s": oslom_singler
            }

        elif algorithm == "linkcom":
            min_com_size = int(request.json.get("ahn_min_com", 2))
            method = request.json.get("method", "single")
            use_threshold = request.json.get("ahn_use_threshold", False)
            ahn_threshold = float(request.json.get("ahn_threshold", 0.0)) if use_threshold else None
            if ahn_threshold is not None:
                edge2cid, cid2edges, cid2nodes = link_communities(G, threshold=ahn_threshold, linkage=method)
                params = {"min": min_com_size, "linkage": method, "t": ahn_threshold}
            else:
                edge2cid, _, _, cid2edges, cid2nodes = link_communities(G, threshold=ahn_threshold, linkage=method)
                params = {"min": min_com_size, "linkage": method}
            detected_communities = [frozenset(nodes) for nodes in cid2nodes.values() if len(nodes) >= min_com_size]
            #print(detected_communities)

        elif algorithm == "linkcom_original":
            use_threshold = request.json.get("ahn_original_use_threshold", False)
            ahn_threshold = float(request.json.get("ahn_original_threshold", None)) if use_threshold else None
            cmd = (
                ["python3", "ahn_original.py", "-t", str(ahn_threshold), GRAPH_PATH]
                if ahn_threshold
                else ["python3", "ahn_original.py", GRAPH_PATH]
            )
            try:
                subprocess.run(cmd, check=True)
                result_file = glob.glob("comm2nodes.txt")
                if not result_file:
                    print("no file")
                    continue
                result_file = result_file[0]
                detected_communities = read_comm2nodes(result_file)
            except subprocess.CalledProcessError:
                print(f"Failed to run linkcom_original with threshold {ahn_threshold}")
                continue

        # Plot communities
        index = get_next_plot_index()
        image_path = plot_communities(G, ground_truth_communities, detected_communities, index)
        image_url = url_for("static", filename=f"community_plot_{index}.svg")

        # Metrics
        result_metrics = compute_metrics(G, detected_communities, ground_truth_communities)
        result_metrics_rounded = {k: round(v, 2) for k, v in result_metrics.items()}

        # Store results
        all_results.append({
            "algorithm": algorithm,
            "params": params,
            "image_url": image_url,
            **result_metrics_rounded
        })

    return jsonify({
        "success": True,
        "message": f"{algorithm} run {N} times.",
        "results": all_results
    })


def read_comm2nodes(file_path):
    communities = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                node_ids = set(map(int, parts[1:]))
                communities.append(node_ids)

    communities = [com for com in communities if len(com) >= 3]

    return communities


@app.route('/optimize', methods=['POST'])
def optimize():
    if not (os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH)):
        return jsonify({"success": False, "message": "Graph has not been generated yet!"})

    G = nx.read_edgelist(GRAPH_PATH, nodetype=int)
    ground_truth_communities = load_ground_truth()

    algorithm = request.json.get("algorithm", "slpa") 
    metric_to_optimize = request.json.get('metric', 'lazar')

    best_score = float('-inf')
    best_params = None
    best_communities = None

    if algorithm == "linkcom_original":
        possible_thresholds = [round(x, 2) for x in list(np.arange(0.1, 0.91, 0.05))]
        for threshold in possible_thresholds:
           
            cmd = [
                "python3", "ahn_original.py",
                "-t", str(threshold),
                GRAPH_PATH
            ]

            try:
                subprocess.run(cmd, check=True)
                basename = os.path.splitext(os.path.basename(GRAPH_PATH))[0]

                result_file = glob.glob("comm2nodes.txt")
                if not result_file:
                    print("no file")
                    continue  
                result_file = result_file[0]

                detected_communities = read_comm2nodes(result_file)

                result_metrics = compute_metrics(G, detected_communities, ground_truth_communities)
                score = result_metrics.get(metric_to_optimize, 0)

                if score > best_score:
                    best_score = score
                    best_params = {"threshold": threshold}
                    best_communities = detected_communities

            except subprocess.CalledProcessError:
                print(f"Failed to run linkcom_original with threshold {threshold}")
                continue

    elif algorithm == "slpa":
        possible_T = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
        possible_r = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        for T in possible_T:
            for r in possible_r:
                detected_communities = find_communities(G, T, r)
                result_metrics = compute_metrics(G, detected_communities, ground_truth_communities)
                score = result_metrics.get(metric_to_optimize, 0)

                if score > best_score:
                    best_score = score
                    best_params = {"T": T, "r": r}
                    best_communities = detected_communities

    elif algorithm == "linkcom":
        thresholds = [round(x, 2) for x in list(np.arange(0.1, 0.91, 0.05))]
        
        for threshold in thresholds:
            edge2cid, best_partition, cid2nodes = link_communities(G, threshold=threshold)
            detected_communities = [frozenset(nodes) for nodes in cid2nodes.values() if len(nodes) >= 3]
            result_metrics = compute_metrics(G, detected_communities, ground_truth_communities)
            score = result_metrics.get(metric_to_optimize, 0)

            if score > best_score:
                best_score = score
                best_params = {"threshold": threshold}
                best_communities = detected_communities

    else:
        return jsonify({"success": False, "message": f"Unsupported algorithm '{algorithm}' for optimization."})

    if best_params:
        return jsonify({
            "success": True,
            "message": f"Optimal {algorithm.upper()} parameters found based on {metric_to_optimize}!",
            "best_params": best_params,
            "best_score": best_score
        })
    else:
        return jsonify({"success": False, "message": f"No optimal parameters found for {algorithm}!"})





def fuzzy_cmeans(G, clusters, m, threshold):

    #node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1, p=p, q=q)
    #model = node2vec.fit(window=10, min_count=1, batch_words=4)
    global node_embeddings_cache

    node_embeddings = node_embeddings_cache #np.array([model.wv[str(node)] for node in G.nodes()])
    
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=node_embeddings.T, 
        c=clusters, 
        m=m, 
        error=0.005, 
        maxiter=500, 
        init=None
    )

    threshold = threshold 
    node_communities = defaultdict(set)

    for i, node in enumerate(G.nodes()):
        for cluster in range(clusters):
            if u.T[i, cluster] > threshold:   
                node_communities[node].add(cluster)

    communities = defaultdict(set)
    for node, comms in node_communities.items():
        for comm in comms:
            communities[comm].add(node)

    return [set(nodes) for nodes in communities.values()]

def plot_communities(G, ground_truth, detected, index=None):
    pos = nx.spring_layout(G, seed=10)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes[0].axis('off')
    axes[1].axis('off')

    axes[0].set_title("Ground Truth Communities")
    draw_colored_communities(G, ground_truth, pos, ax=axes[0])

    axes[1].set_title("Detected Communities")
    draw_colored_communities(G, detected, pos, ax=axes[1])

    plt.tight_layout()
    filename = f"static/community_plot_{index}.svg" if index is not None else "static/community_plot.svg"
    plt.savefig(filename, format='svg')
    plt.close()
    return filename



def generate_colors(n):
    cmap = plt.get_cmap("tab20")  # Works well for up to 20 communities
    return [cmap(i % 20) for i in range(n)]  # Repeats if n > 20





def draw_colored_communities(G, communities, pos, ax=None):
    node_communities = {node: set() for node in G.nodes}
    for i, comm in enumerate(communities):
        for node in comm:
            node_communities[node].add(i)



    color_palette = ["#bfb100",
"#3d66ef",
"#68df62",
"#f42e9e",
"#02ad31",
"#f991ff",
"#1e7800",
"#c1004d",
"#1bdcd5",
"#df312b",
"#57d7ec",
"#8b3a0c",
"#66dab5",
"#922e47",
"#aad457",
"#524892",
"#f7bd3f",
"#007e71",
"#ff7a86",
"#019f5e",
"#ff9dc2",
"#9ed57c",
"#ffada3",
"#155f3a",
"#ffa15a",
"#4f5711",
"#efbf65",
"#6d4d16",
"#93a772",
"#8e6d00"]

    community_colors = {i: color_palette[i % len(color_palette)] for i in range(len(communities))}

    if ax is None:
        ax = plt.gca()

    nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax)

    node_radius = 0.05
    if len(G) > 50:
        node_radius = 0.02

    for node, (x, y) in pos.items():
        comms = node_communities[node]
        num_comms = len(comms)

        if num_comms == 1:
            color = community_colors[next(iter(comms))]
            circle = plt.Circle((x, y), node_radius, color=color, ec="black", lw=0.2, zorder=3)
            ax.add_patch(circle)

        elif num_comms > 1:
            angles = [i * (360 / num_comms) for i in range(num_comms)]
            sorted_comms = sorted(comms)
            for i, comm in enumerate(sorted_comms):
                color = community_colors[comm]
                wedge = Wedge((x, y), node_radius, angles[i], angles[(i + 1) % num_comms],
                              facecolor=color, ec="black", lw=0.2, zorder=3)
                ax.add_patch(wedge)
        #ax.text(x, y + node_radius * 1.1, str(node), fontsize=6, ha='center', va='center', zorder=4)
    
    ax.set_aspect("equal")

if __name__ == "__main__":
    app.run(debug=True)
