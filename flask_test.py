import os
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
from demonalgo import DemonAlgorithm
from cdlib import NodeClustering
from ahn_link import link_communities
from slpaalgorithm import find_communities
from slpav2 import SLPA
from cdlib.algorithms import (
    slpa,
    demon,
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

app = Flask(__name__)

GRAPH_PATH = "network.dat"
COMMUNITY_PATH = "community.dat"
COUNTER_FILE = "static/plot_counter.txt"

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

def compute_onmi(graph, detected_communities, ground_truth_communities):
    """
    Compute Overlapping Normalized Mutual Information (oNMI)
    between detected and ground truth communities.
    """

    detected_nc = NodeClustering(detected_communities, graph, "Detected")
    ground_truth_nc = NodeClustering(ground_truth_communities, graph, "Ground Truth")

    onmi_score = overlapping_normalized_mutual_information_MGH(detected_nc, ground_truth_nc)
    
    return onmi_score.score


# Function to run LFR benchmark
"""
def generate_graph(params):

    # Check if the graph and community files already exist
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)  # Delete the existing network.dat
    if os.path.exists(COMMUNITY_PATH):
        os.remove(COMMUNITY_PATH)  # Delete the existing community.dat

    cmd = ["./benchmark"]

    # Required parameters
    required_params = ["N", "k", "maxk", "mu"]
    
    # Add required parameters
    for param in required_params:
        if param in params and params[param].strip():
            cmd.append(f"-{param}")
            cmd.append(str(params[param]))
        else:
            return f"Error: Missing required parameter '{param}'"

    # Add optional parameters if provided
    optional_params = ["t1", "t2", "minc", "maxc", "on", "om"]
    for param in optional_params:
        if param in params and params[param].strip():
            cmd.append(f"-{param}")
            cmd.append(str(params[param]))

    try:
        # Run the command and capture both stdout & stderr
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)
        output = result.stdout.strip()  # Remove leading/trailing spaces

        # If both files exist, extract useful benchmark info
        if os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH):
            match = re.search(r"\*{60,}\n(.*?)\*{60,}", output, re.DOTALL)
            info = match.group(1).strip() if match else "No benchmark details found."
            return True, output, info  # Successful case

        else:
            # If files are missing, assume failure & return stderr
            error_message = result.stderr.strip() if result.stderr else "Unknown error occurred."
            return False, output, error_message  # Failed case

    except Exception as e:
        return "", f"Exception occurred: {str(e)}"  # Handle unexpected errors

    #except subprocess.CalledProcessError as e:
    #    return e.stdout + "\n" + e.stderr, "Error: Benchmark execution failed!"
    
    subprocess.run(cmd, check=False )

    if os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH):
        return "Graph successfully generated!"
    else:
        return "Error: Graph generation failed."
    """

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


    l_detected_nc, l_ground_truth_nc = align_partitions(detected_communities, ground_truth_communities)

    detected_nc = NodeClustering([list(comm) for comm in l_detected_nc], graph, "Detected")
    ground_truth_nc = NodeClustering(l_ground_truth_nc, graph, "Ground Truth")

    # Compute Metrics
    onmi_score = overlapping_normalized_mutual_information_MGH(detected_nc, ground_truth_nc).score
    f1_score = f1(detected_nc, ground_truth_nc).score
    modularity_score = modularity_overlap(graph, detected_nc).score
    omega_score = omega(detected_nc, ground_truth_nc).score

    return {
        "onmi": round(onmi_score,4),
        "omega": round(omega_score,4),
        "f1": round(f1_score, 4),
        "modularity": round(modularity_score, 4)
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

    return [frozenset(nodes) for nodes in ground_truth.values()]

"""
@app.route("/generate", methods=["POST"])
def generate_graph():
    data = request.json  # Get parameters from frontend
    cmd = ["./benchmark", f"-N {data['N']}", f"-k {data['k']}", f"-maxk {data['maxk']}", f"-mu {data['mu']}"]

    success, output, message = run_lfr_benchmark(cmd)

    return jsonify({
        "success": success,
        "message": message
    })

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    benchmark_info = ""
    full_output = ""
    if request.method == "POST":
        if "generate" in request.form:
            params = {key: request.form[key] for key in request.form}

            full_output, benchmark_info = generate_graph(params)

            if os.path.exists("network.dat") and os.path.exists("community.dat"):
                message = "Graph successfully generated!"
            else:
                message = "Error: Graph generation failed! Check parameters."

            return render_template("index.html", message=message, benchmark_info=benchmark_info, full_output=full_output)

        if "detect" in request.form:
            G = nx.read_edgelist(GRAPH_PATH, nodetype=int)
            ground_truth_communities = load_ground_truth()

            algorithm = request.form["algorithm"]

            if algorithm == "slpa":
                T = int(request.form.get("T", 40))
                r = float(request.form.get("r", 0.3))
                detected_communities = find_communities(G, T, r)

            elif algorithm == "node2vec_fcm":
                dimensions = int(request.form.get("dimensions", 4))
                walk_length = int(request.form.get("walk_length", 30))
                num_walks = int(request.form.get("num_walks", 30))
                clusters = int(request.form.get("clusters", 4))
                detected_communities = node2vec_fuzzy_cmeans(G, dimensions, walk_length, num_walks, clusters)

            elif algorithm == "demon":
                epsilon = float(request.form['epsilon'])
                min_community_size = int(request.form['min_community_size'])
                demon_algorithm = DemonAlgorithm(epsilon=epsilon, min_community_size=min_community_size)
                detected_communities = demon_algorithm.execute(G)
            
        

            result_metrics = compute_metrics(G, detected_communities, ground_truth_communities)
            


            plot_communities(G, ground_truth_communities, detected_communities, result_metrics)
            return render_template("index.html", message="Detection complete! Check the plots.")
    else:
        # When the page is first opened (GET request), check if parameters are present
        if not request.form.get("algorithm"):
            message = "SLPA algorithm is selected by default"

    return render_template("index.html", message="")

"""

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

"""
@app.route('/generate', methods=['POST'])
def generate_graph():
    
    global LFR_PARAMETERS
    params = request.get_json()
    LFR_PARAMETERS = params.copy() 

    # Check if the graph and community files already exist
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)  # Delete the existing network.dat
    if os.path.exists(COMMUNITY_PATH):
        os.remove(COMMUNITY_PATH)  # Delete the existing community.dat

    cmd = ["./benchmark"]

    # Required parameters
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
        # Run the command and capture both stdout & stderr
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)
        output = result.stdout.strip()  # Remove leading/trailing spaces

        # If both files exist, extract useful benchmark info
        if os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH):
            match = re.search(r"\*{60,}\n(.*?)\*{60,}", output, re.DOTALL)
            info = match.group(1).strip() if match else "No benchmark details found."
            
            # Get the number of communities
            ground_truth_communities = load_ground_truth()
            num_communities = len(ground_truth_communities)

            return jsonify({
                "success": True,
                "message": "Graph successfully generated!",
                "benchmark_info": info,
                "num_communities": num_communities  # Return the number of communities
            })

        else:
            # If files are missing, assume failure & return stderr
            error_message = result.stderr.strip() if result.stderr else "Unknown error occurred."
            return jsonify({"success": False, "message": f"Error: {error_message}"})

    except Exception as e:
        return jsonify({"success": False, "message": f"Exception occurred: {str(e)}"})

"""

@app.route('/generate', methods=['POST'])
def generate_graph():
    global LFR_PARAMETERS
    params = request.get_json()
    LFR_PARAMETERS = params.copy()

    # Check if the graph and community files already exist
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)  # Delete the existing network.dat
    if os.path.exists(COMMUNITY_PATH):
        os.remove(COMMUNITY_PATH)  # Delete the existing community.dat

    cmd = ["./benchmark"]

    # Required parameters
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
        # Run the command and capture both stdout & stderr
        result = subprocess.run(cmd, shell=False, capture_output=True, text=True, check=False)
        output = result.stdout.strip()  # Remove leading/trailing spaces

        # If both files exist, extract useful benchmark info
        if os.path.exists(GRAPH_PATH) and os.path.exists(COMMUNITY_PATH):
            match = re.search(r"\*{60,}\n(.*?)\*{60,}", output, re.DOTALL)
            info = match.group(1).strip() if match else "No benchmark details found."
            
            # Get the number of communities
            ground_truth_communities = load_ground_truth()
            num_communities = len(ground_truth_communities)

            # Return the LFR parameters along with the other results
            return jsonify({
                "success": True,
                "message": "Graph successfully generated!",
                "benchmark_info": info,
                "num_communities": num_communities,  # Return the number of communities
                "LFR_parameters": LFR_PARAMETERS  # Return the LFR parameters to use on the frontend
            })

        else:
            # If files are missing, assume failure & return stderr
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
            nodes = list(map(int, line.strip().split()))
            communities.append(nodes)

    return communities

@app.route("/reset", methods=["POST"])
def reset():

    for filepath in glob.glob("static/community_plot_*.png"):
        os.remove(filepath)
    
    with open(COUNTER_FILE, "w") as f:
        f.write(str(0))

    main_plot = "static/community_plot.png"
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
    N = int(request.json.get("N", 1))
    best_by = request.json.get("best_by", "modularity")  # Metric chosen for selecting best result
    params = {}

    best_result = None
    best_score = float("-inf")  # Best score for the chosen metric
    all_results = []
    image_paths = []

    for i in range(N):
        if algorithm == "slpa" or algorithm == "slpa_cdlib":
            T = int(request.json.get("T", 40))
            r = float(request.json.get("r", 0.3))
            if algorithm == "slpa":
                detected_communities = find_communities(G, T, r)
            else:
                detected_communities = slpa(G, T, r).communities
            params = {"T": T, "r": r}


        elif algorithm == "n2vfcm":
            dimensions = int(request.json.get("dimensions", 8))
            walk_length = int(request.json.get("walk_length", 50))
            num_walks = int(request.json.get("num_walks", 10))
            clusters = int(request.json.get("clusters", 4))
            p = float(request.json.get("p", 1))
            q = float(request.json.get("q", 1))
            m = float(request.json.get("m", 1.5))
            threshold = float(request.json.get("threshold", 0.3))
            detected_communities = node2vec_fuzzy_cmeans(G, dimensions, walk_length, num_walks, clusters, p, q, m, threshold)
            params = {"dim": dimensions, "wl": walk_length, "nw": num_walks, "p" : p, "q": q, "c": clusters, "t": threshold }

        elif algorithm == "demon":
            epsilon = float(request.json.get("epsilon", 0.1))
            min_community_size = int(request.json.get("min_community_size", 3))
            detected_communities = DemonAlgorithm(epsilon=epsilon, min_community_size=min_community_size).execute(G)
            params = {"epsilon": epsilon, "mincomsize": min_community_size}
        elif algorithm == "conga":
            number_communities = int(request.json.get("number_community", 4))
            detected_communities = conga(G, number_communities).communities
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
                "r": oslom_f_l_i,
                "hr": oslom_h_l_i,
                "t": oslom_p_v_t,
                "i": oslom_infomap,
                "l": oslom_louvain,
                "c": oslom_copra,
                "s": oslom_singler
            }

            
        
        elif algorithm == "linkcom":
            
            min_com_size = int(request.json.get("ahn_min_com", 2))  
            method = request.json.get("method", "single")  
            use_threshold = request.json.get("ahn_use_threshold", False) 


            ahn_threshold = float(request.json.get("ahn_threshold", 0.0)) if use_threshold else None
            print(ahn_threshold)
            print(method)
            
            #edge2cid, _, _,_, cid2edges, cid2nodes, = link_communities(G, threshold=ahn_threshold, linkage=method)
            #edge2cid, best_S, best_D, best_partition, cid2nodes = link_communities(G, threshold=ahn_threshold, linkage=method)
            
            # Set the parameters based on whether the threshold is used
            if ahn_threshold is not None:
                edge2cid, cid2edges, cid2nodes = link_communities(G, threshold=ahn_threshold, linkage=method)
                params = {"min": min_com_size, "linkage": method, "t": ahn_threshold}
            else:
                edge2cid, _, _,cid2edges, cid2nodes = link_communities(G, threshold=ahn_threshold, linkage=method)
                params = {"min": min_com_size, "linkage": method}
            #filter out primitive communities
            detected_communities = [frozenset(nodes) for nodes in cid2nodes.values() if len(nodes) >= min_com_size]

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
                # linkcom_original will create an output file like: network_thrS0.5_thrD0.45.edge2comm.txt
                basename = os.path.splitext(os.path.basename(GRAPH_PATH))[0]
    
                result_file = glob.glob("comm2nodes.txt")
                if not result_file:
                    print("no file")
                    continue  # skip if no file
                result_file = result_file[0]

                 # Load communities
                detected_communities = read_comm2nodes(result_file)

            except subprocess.CalledProcessError:
                print(f"Failed to run linkcom_original with threshold {ahn_threshold}")
                continue
            
        index = get_next_plot_index()
        image_path = plot_communities(G, ground_truth_communities, detected_communities, index)

        image_url = url_for("static", filename=f"community_plot_{index}.png")
        image_paths.append({
            "image_url": image_url,
            "index": index
        })

        
        result_metrics = compute_metrics(G, detected_communities, ground_truth_communities)
        result_metrics_rounded = {
            k: round(v, 2) for k, v in result_metrics.items()
        }

        score = result_metrics[best_by]  # Get the score based on the selected metric
        all_results.append({
            "algorithm": algorithm,
            "params": params,
            "image_url": image_url,
            **result_metrics_rounded
        })

        # Check if this result is the best based on the chosen metric
        if score > best_score:
            best_score = score
            best_result = detected_communities
            best_result_metrics = result_metrics_rounded


    return jsonify({
        "success": True,
        "message": f"{algorithm} run {N} times. With {best_by} {best_score}.",
        "image_url": image_url,
        "results": all_results,
        "best_metric": best_by,
        "best_score": best_score
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
    metric_to_optimize = request.json.get('metric', 'modularity')

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
                detected_communities = slpa(G, T, r).communities
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





def node2vec_fuzzy_cmeans(G, dimensions, walk_length, num_walks, clusters,p, q, m, threshold):

    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=1, p=p, q=q)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    
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

    return [frozenset(nodes) for nodes in communities.values()]

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
    filename = f"static/community_plot_{index}.png" if index is not None else "static/community_plot.png"
    plt.savefig(filename)
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

    num_communities = 20  # Adjust based on your needs

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

    #ax.set_xlim(-1.1, 1.1)
    #ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")

if __name__ == "__main__":
    app.run(debug=True)
