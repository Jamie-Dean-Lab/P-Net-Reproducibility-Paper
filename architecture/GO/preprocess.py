import json
import os

import pandas as pd
from collections import defaultdict, deque


def build_GO_pnet_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cx_path = os.path.join(script_dir, 'DrugCell Hierarchy.cx')
    hierarchy_out = os.path.join(script_dir, 'go_hierarchy.tsv')
    gmt_out = os.path.join(script_dir, 'go_gene_sets.gmt')

    if os.path.exists(hierarchy_out) and os.path.exists(gmt_out):
        return

    # Load CX file
    print("Loading CX file...")
    with open(cx_path, 'r') as f:
        cx_data = json.load(f)
    aspects = {list(a.keys())[0]: list(a.values())[0] for a in cx_data}

    # Parse nodes
    node_names = {n['@id']: n['n'] for n in aspects['nodes']}
    node_namespace = {}
    node_fullname = {}
    for a in aspects['nodeAttributes']:
        if a['n'] == 'namespace':
            node_namespace[a['po']] = a['v']
        if a['n'] == 'fullname':
            node_fullname[a['po']] = a['v']

    go_nodes = {k: v for k, v in node_names.items() if k in node_namespace}
    gene_nodes = {k: v for k, v in node_names.items() if k not in node_namespace}
    print(f"GO term nodes: {len(go_nodes)}")
    print(f"Gene nodes: {len(gene_nodes)}")

    # Parse edges
    edge_attrs = {}
    for a in aspects['edgeAttributes']:
        if a['po'] not in edge_attrs:
            edge_attrs[a['po']] = {}
        edge_attrs[a['po']][a['n']] = a['v']

    gene_edges = []
    hierarchy_edges = []
    for e in aspects['edges']:
        attrs = edge_attrs.get(e['@id'], {})
        if attrs.get('Column 3') == 'gene':
            gene_edges.append(e)
        else:
            hierarchy_edges.append(e)

    print(f"Gene-GO edges: {len(gene_edges)}")
    print(f"GO-GO edges: {len(hierarchy_edges)}")

    # Build hierarchy TSV (child -> parent format)
    hierarchy_rows = []
    for e in hierarchy_edges:
        parent = node_names[e['s']]
        child = node_names[e['t']]
        hierarchy_rows.append({'child': child, 'parent': parent})

    hierarchy_df = pd.DataFrame(hierarchy_rows)
    hierarchy_df.to_csv(hierarchy_out, sep='\t', index=False, header=False)

    # Build GMT file
    name_to_id = {v: k for k, v in node_names.items()}

    go_genes = defaultdict(list)
    for e in gene_edges:
        go_term = node_names[e['s']]
        gene = node_names[e['t']]
        go_genes[go_term].append(gene)

    with open(gmt_out, 'w') as f:
        for go_term, genes in go_genes.items():
            node_id = name_to_id.get(go_term)
            fullname = node_fullname.get(node_id, go_term)
            line = '\t'.join([fullname, go_term, 'GO Biological Process'] + genes)
            f.write(line + '\n')

    print(f"Wrote GMT file: {gmt_out} ({len(go_genes)} GO terms)")

    # Check single root
    all_children = set(hierarchy_df['child'])
    all_parents = set(hierarchy_df['parent'])
    roots = all_parents - all_children
    if len(roots) == 1:
        print(f"Single root found: {roots}")
    else:
        print(f"Expected 1 root, found {len(roots)}: {roots}")

    # Check all gene-annotated GO terms are in hierarchy
    hierarchy_terms = all_children | all_parents
    annotated_terms = set(go_genes.keys())
    orphaned = annotated_terms - hierarchy_terms
    if len(orphaned) == 0:
        print(f"All {len(annotated_terms)} gene-annotated GO terms are in hierarchy")
    else:
        print(f"{len(orphaned)} gene-annotated GO terms not in hierarchy: {orphaned}")

    # Check hierarchy depth using BFS
    children_map = defaultdict(list)
    for _, row in hierarchy_df.iterrows():
        children_map[row['parent']].append(row['child'])

    root = list(roots)[0] if roots else None
    if root:
        depths = {}
        queue = deque([(root, 0)])
        while queue:
            node, depth = queue.popleft()
            if node in depths:
                continue
            depths[node] = depth
            for child in children_map[node]:
                queue.append((child, depth + 1))

        max_depth = max(depths.values())
        depth_counts = defaultdict(int)
        for d in depths.values():
            depth_counts[d] += 1

        print(f"Max hierarchy depth: {max_depth}")
        print(f"Nodes per depth: { {d: depth_counts[d] for d in sorted(depth_counts)} }")

    # Check GMT has genes
    all_go_genes = set(g for genes in go_genes.values() for g in genes)
    print(f"Total unique genes in GMT: {len(all_go_genes)}")
    if len(all_go_genes) == 0:
        print("No genes found in GMT file")

    # Check terminal nodes have gene annotations
    terminal_nodes = all_children - all_parents
    annotated_terminals = terminal_nodes & annotated_terms
    print(f"Terminal GO terms with gene annotations: {len(annotated_terminals)}/{len(terminal_nodes)}")