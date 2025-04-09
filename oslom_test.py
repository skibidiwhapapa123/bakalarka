def run_oslom(graph_path: str, is_weighted=False, directed=False, extra_args=None, copra_runs=0):
    from pathlib import Path
    import subprocess

    graph_path = Path(graph_path).resolve()
    
    # Choose the right binary
    algo = "./oslom_dir" if directed else "./oslom_undir"
    weight_flag = "-w" if is_weighted else "-uw"
    
    # Build command
    cmd = [algo, "-f", str(graph_path), weight_flag]

    # Add COPRA if requested
    if copra_runs > 0:
        cmd += ["-copra", str(copra_runs)]

    # Add other arguments
    if extra_args:
        cmd += extra_args

    # Run OSLOM
    subprocess.run(cmd, check=True)
    
    # OSLOM output folder
    output_dir = graph_path.parent / f"{graph_path.name}_oslo_files"
    tp_path = output_dir / "tp"
    
    if not tp_path.exists():
        raise FileNotFoundError(f"'tp' file not found at: {tp_path}")
    
    # Parse tp file
    communities = []
    with tp_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            nodes = list(map(int, line.strip().split()))
            communities.append(nodes)

    return communities

if __name__ == "__main__":
    graph_file = "network.dat"
    
    # Extra settings: 5 runs, no hierarchy, fixed seed
    extra = ["-r", "5", "-hr", "5", "-seed", "10"]
    
    try:
        communities = run_oslom(graph_file, is_weighted=False, copra_runs=0, extra_args=extra)
        print("Detected communities with COPRA init:")
        for i, com in enumerate(communities):
            print(f"Community {i}: {com}")
    except Exception as e:
        print("Error:", e)
