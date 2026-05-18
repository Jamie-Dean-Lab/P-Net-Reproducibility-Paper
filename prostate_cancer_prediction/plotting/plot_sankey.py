import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.graph_objects as go



def plot_sankey(pnet_run_dir, n_hidden_layers, figures_dir, dataset_id_mappings):
    """
    Generates a Sankey diagram visualising the P-Net model's feature importance flow
    from input genomic features through gene and pathway layers to the outcome node.

    Reproduces the visualisation from Elmarakeby et al. (2021). Loads pre-computed
    DeepLIFT importance scores and trained model link weights, selects the top N most
    important nodes per layer (genes by coef_combined, pathways by coef), and builds
    a layered edge graph connecting inputs (mutation, amplification, deletion) -> genes
    -> pathway hierarchy (6 layers) -> outcome.

    Non-selected nodes are collapsed into 'residual' nodes per layer rather than
    dropped, preserving total flow. Edge weights for pathway layers (1+) are adjusted
    via adjust_values() which reweights each edge as the minimum of source-normalised
    and target-normalised importance, ensuring edges only appear thick when both
    endpoints are important. First layer (input->gene) edges are NOT passed through
    adjust_values(), matching the original which constructs these separately and
    concatenates them after adjustment.

    Node y-positions are computed from flow values (max of incoming/outgoing), with
    residual nodes forced to the bottom of each layer. Output saved as PDF, PNG, HTML.
    """

    # load pre-computed DeepLIFT importance scores saved by get_deeplift_global()
    # keys: 'inputs', 'h0'..'h5' — one DataFrame per layer with columns
    # [coef, coef_graph, coef_combined, coef_combined_zscore, coef_combined2,
    #  feature_importance_adjusted]
    deeplift = {
        fn.split("_")[-1].replace(".csv", ""): pd.read_csv(f"{pnet_run_dir}/{fn}", index_col=0)
        for fn in os.listdir(pnet_run_dir) if fn.startswith("feature_importance_")
    }

    # load trained model kernel weight matrices saved by get_deeplift_global()
    # keys: 0..5 — link_weights[0] = inputs->genes (27687x9229),
    #              link_weights[i] = layer i-1 -> layer i for i=1..5
    link_weights = {
        int(fn.split("_")[-1].replace(".csv", "")): pd.read_csv(f"{pnet_run_dir}/{fn}", index_col=0)
        for fn in os.listdir(pnet_run_dir) if fn.startswith("link_weights_")
    }

    print(f"\n=== plot_sankey ===")
    print(f"deeplift keys: {sorted(deeplift.keys())}")
    print(f"link_weights keys: {sorted(link_weights.keys())}")
    for k, v in deeplift.items():
        print(f"  deeplift['{k}'] shape={v.shape}, columns={v.columns.tolist()}")

    # pathway ID -> full name mapping for display labels
    pathwaynames = pd.read_csv(dataset_id_mappings, sep="\t", index_col=0, header=None)
    pathwaynames.columns = ["name", "species"]
    id_to_name = pathwaynames["name"].to_dict()

    # number of top nodes to show per layer
    # [genes, h1_pathways, h2_pathways, h3_pathways, h4_pathways, h5_pathways, h6_pathways]
    nlargest = [10, 10, 10, 10, 6, 6, 6]

    # -------------------------------------------------------------------
    # 1. select top N important nodes per layer
    #    matching original: genes ranked by coef_combined (degree-adjusted),
    #    pathways ranked by raw coef — these are the nodes that will be shown
    #    explicitly; all others collapse to residual nodes
    # -------------------------------------------------------------------
    print(f"\n--- Section 1: Node selection ---")

    # genes: use coef_combined (degree-adjusted importance), filtered to only genes
    # that appear in the Reactome graph (link_weights[1] rows = connected genes)
    gene_importance = deeplift["h0"]["coef_combined"].copy()
    gene_importance = gene_importance[gene_importance.index.isin(link_weights[1].index)]
    print(f"  gene_importance after Reactome filter: {len(gene_importance)} genes")

    high_nodes = {}
    high_nodes[0] = ["mut_important", "cnv_amp", "cnv_del"]  # input layer always shows all 3
    high_nodes[1] = gene_importance.nlargest(nlargest[0]).index.tolist()
    print(f"  high_nodes[1] (genes): {high_nodes[1]}")

    # pathways: use raw coef (not degree-adjusted) matching original
    for i in range(1, n_hidden_layers + 1):
        k = f"h{i}"
        if k in deeplift:
            high_nodes[i + 1] = deeplift[k]["coef"].nlargest(nlargest[i]).index.tolist()
            print(f"  high_nodes[{i+1}] (layer {k}): {high_nodes[i+1]}")

    # -------------------------------------------------------------------
    # 2. node importance lookup
    #    matching original: for each node, importance = log(1 + 100 * coef / layer_sum)
    #    computed over ALL nodes in each layer (not just top N) so the denominator
    #    reflects the full layer's total importance — this normalises across layers
    #    and applies a log compression to reduce the effect of extreme outliers
    # -------------------------------------------------------------------
    print(f"\n--- Section 2: Node importance ---")
    node_importance = {}

    # gene layer: normalise coef across all 9229 genes
    gi_all = deeplift["h0"]["coef"].clip(lower=0)
    layer_sum = gi_all.sum()
    gi_norm_all = np.log(1. + 100. * gi_all / layer_sum) if layer_sum > 0 else gi_all
    node_importance.update(gi_norm_all.to_dict())
    print(f"  h0 (genes): layer_sum={layer_sum:.4f}, "
          f"importance range=[{gi_norm_all.min():.4f}, {gi_norm_all.max():.4f}]")

    # pathway layers: same normalisation per layer
    for i in range(1, n_hidden_layers + 1):
        k = f"h{i}"
        if k in deeplift:
            pi_all = deeplift[k]["coef"].clip(lower=0)
            layer_sum = pi_all.sum()
            pi_norm_all = np.log(1. + 100. * pi_all / layer_sum) if layer_sum > 0 else pi_all
            node_importance.update(pi_norm_all.to_dict())
            print(f"  {k}: layer_sum={layer_sum:.4f}, "
                  f"importance range=[{pi_norm_all.min():.4f}, {pi_norm_all.max():.4f}]")

    # residual (others) nodes get a small fixed importance matching original coef=1
    # this ensures adjust_values error correction terms are non-zero for these nodes
    for i in range(1, n_hidden_layers + 2):
        node_importance[f"others{i}"] = 0.1

    # input nodes (mutation/amplification/deletion) get fixed importance = log(1 + 100/layer_sum)
    # matching original high_nodes_df.loc['mutation'] = [1, 0] i.e. coef=1
    gene_layer_sum = deeplift["h0"]["coef"].clip(lower=0).sum()
    input_importance_val = float(np.log(1. + 100. * 1.0 / gene_layer_sum)) if gene_layer_sum > 0 else 1.0
    for input_id in ["mut_important", "cnv_amp", "cnv_del"]:
        node_importance[input_id] = input_importance_val
    print(f"  input node importance: {input_importance_val:.4f}")
    print(f"  Total nodes in importance lookup: {len(node_importance)}")
    print(f"  Sample high-node importances:")
    for g in high_nodes[1][:3]:
        print(f"    {g}: {node_importance.get(g, 'MISSING'):.4f}")

    # -------------------------------------------------------------------
    # 3. assemble edges
    #    layer 0 (input->gene): the original loads gradient_importance_0.csv which
    #      has a two-level multi-index (gene, input_type) and a single 'coef' column.
    #      our saved file has a flat index like 'cnv_amp_RECK' with a single value
    #      column, so we parse the prefix to recover (input_type, gene) and reconstruct
    #      the same long-form edge table the original gets from reset_index().
    #    layers 1+ (gene->pathway, pathway->pathway): use trained kernel weights
    #      filtered to edges touching at least one high-importance node, with
    #      non-important nodes collapsed to residual
    # -------------------------------------------------------------------
    print(f"\n--- Section 3: Edge assembly ---")
    all_edges = []

    # --- layer 0: inputs -> genes ---
    print(f"  Layer 0: parsing flat input importance index -> (source, target) edges")

    # take the first (and only) column regardless of its name — original saves as 'coef'
    # but our file may have saved it under a different name; content is what matters
    input_imp_series = deeplift["inputs"].iloc[:, 0].copy().abs()
    print(f"  input_imp_series shape: {input_imp_series.shape}")
    print(f"  input_imp_series index[:5]: {input_imp_series.index.tolist()[:5]}")
    print(f"  input_imp_series column used: {deeplift['inputs'].columns[0]!r}")
    print(f"  input_imp_series value range: [{input_imp_series.min():.6f}, {input_imp_series.max():.6f}]")
    print(f"  input_imp_series non-zero count: {(input_imp_series != 0).sum()}")

    # parse flat index 'cnv_amp_RECK' -> source='cnv_amp', target='RECK'
    # must anchor on known prefixes because 'mut_important' contains an underscore,
    # so a naive split('_', 1) would give ('mut', 'important_RECK') — wrong
    parsed = input_imp_series.index.str.extract(r"^(mut_important|cnv_amp|cnv_del)_(.+)$")
    parsed.columns = ["source", "target"]
    parsed["value"] = input_imp_series.values
    n_unparsed = parsed["source"].isna().sum()
    print(f"  Parsed {len(parsed)} rows, {n_unparsed} failed to match prefix pattern")
    if n_unparsed > 0:
        failed = input_imp_series.index[parsed["source"].isna()].tolist()
        print(f"  WARNING: unparsed rows (first 5): {failed[:5]}")

    # drop rows that didn't match — shouldn't happen with clean data
    first_edges = parsed.dropna(subset=["source"]).copy()
    first_edges["value"] = first_edges["value"].abs()
    first_edges["layer"] = 0

    # sanity: each gene should appear exactly 3 times (one per input type)
    gene_counts = first_edges.groupby("target").size()
    print(f"  Gene entry counts — expected all 3: "
          f"min={gene_counts.min()}, max={gene_counts.max()}, "
          f"n_genes={len(gene_counts)}")
    if (gene_counts != 3).any():
        bad = gene_counts[gene_counts != 3].index.tolist()
        print(f"  WARNING: {len(bad)} genes without exactly 3 input types (first 5): {bad[:5]}")

    # verify all 3 input types are present
    print(f"  Input types found: {sorted(first_edges['source'].unique().tolist())}")

    # per-input-type importance totals before any collapsing
    print(f"  Per-input-type importance totals (raw, before zero-drop):\n"
          f"  {first_edges.groupby('source')['value'].sum().sort_values(ascending=False).to_string()}")

    # drop zero-value edges — original only keeps nonzero connections
    n_before = len(first_edges)
    first_edges = first_edges[first_edges["value"] != 0].copy()
    print(f"  Dropped {n_before - len(first_edges)} zero-value edges, {len(first_edges)} remain")
    print(f"  first_edges before collapsing: {len(first_edges)} edges")

    # collapse non-top genes to others1 residual — matching original get_first_layer
    # with include_others=True which maps non-top genes to 'others1' then groups
    first_edges["target"] = first_edges["target"].map(
        lambda x: x if x in high_nodes[1] else "others1"
    )
    first_edges = first_edges.groupby(["source", "target"])["value"].sum().reset_index()
    first_edges["layer"] = 0
    print(f"  first_edges after collapsing to others1: {len(first_edges)} edges")
    print(f"  others1 edges: {(first_edges['target'] == 'others1').sum()}")
    print(f"  top-gene edges: {(first_edges['target'] != 'others1').sum()}")

    # normalize each edge by the total input to its target gene (so all edges
    # into a gene sum to 1), then scale by gene importance * 150
    # others1 gets fixed importance=10.0 matching original df.coef.fillna(10.0)
    gene_imp_raw = np.log(1. + deeplift["h0"]["coef_combined"].clip(lower=0).abs())
    gene_imp_raw["others1"] = 10.0
    first_edges["value"] = first_edges["value"] / first_edges.groupby("target")["value"].transform("sum")
    first_edges["gene_imp"] = first_edges["target"].map(gene_imp_raw.to_dict()).fillna(0)
    first_edges["value"] = first_edges["value"] * first_edges["gene_imp"] * 150.
    first_edges = first_edges[first_edges["value"] > 0][["source", "target", "value", "layer"]]

    input_totals = first_edges.groupby("source")["value"].sum().sort_values(ascending=False)
    print(f"  Input type totals (layer 0, no adjust_values applied):\n{input_totals.to_string()}")
    print(f"  others1 edges after scaling: {(first_edges['target'] == 'others1').sum()}")

    # layer 0 edges are NOT passed through adjust_values — matching original which
    # concatenates first_layer_df AFTER adjust_values is applied to pathway edges
    all_edges.append(first_edges)

    # --- layers 1+: genes->pathways and pathway->pathway ---
    # matching original filter_connections(add_unk=True):
    # keep any edge where source OR target is a high-importance node,
    # then collapse non-high nodes to their layer's residual node
    pathway_edges = []
    for i in range(n_hidden_layers):
        lw = link_weights[i + 1].abs()  # trained kernel weights for this layer transition
        src_high = set(high_nodes[i + 1])   # important source nodes
        tgt_high = set(high_nodes[i + 2])   # important target nodes
        others_src = f"others{i + 1}"       # residual label for non-important sources
        others_tgt = f"others{i + 2}"       # residual label for non-important targets

        # melt weight matrix to edge list, drop zero-weight edges
        df_layer = lw.copy()
        df_layer.index.name = "source"
        edges = df_layer.reset_index().melt(id_vars="source", var_name="target", value_name="value")
        edges = edges[edges["value"] != 0].copy()

        # keep edges where at least one end is a high node (add_unk=True behaviour)
        ind1 = edges["source"].isin(src_high)
        ind2 = edges["target"].isin(tgt_high)
        edges = edges[ind1 | ind2].copy()

        # collapse non-high nodes to residual labels
        edges["source"] = edges["source"].map(lambda x: x if x in src_high else others_src)
        edges["target"] = edges["target"].map(lambda x: x if x in tgt_high else others_tgt)

        # sum duplicate edges created by collapsing (many non-high nodes -> one residual)
        edges = edges.groupby(["source", "target"])["value"].sum().reset_index()
        edges["layer"] = i + 1

        print(f"  Layer {i+1}: {len(edges)} edges, "
              f"sources={sorted(edges['source'].unique().tolist())[:5]}..., "
              f"targets={sorted(edges['target'].unique().tolist())[:5]}...")
        pathway_edges.append(edges)

    pathway_edges_df = pd.concat(pathway_edges, ignore_index=True)
    print(f"  Total pathway edges before adjust_values: {len(pathway_edges_df)}")

    # -------------------------------------------------------------------
    # 4. adjust_values on pathway edges only (layers 1+)
    #    matching original which runs adjust_values before concatenating first layer
    #
    #    for each edge, computes:
    #      A = (edge_value / source_total_outflow) * source_importance
    #      B = (edge_value / target_total_inflow)  * target_importance
    #      value_final = min(A, B)  — bottlenecked by the less important endpoint
    #
    #    then corrects residual node edges by adding the error between actual
    #    and expected fan-in/fan-out, ensuring flow conservation
    # -------------------------------------------------------------------
    print(f"\n--- Section 4: adjust_values (pathway edges only) ---")
    df_pw = pathway_edges_df.copy()
    df_pw["value_abs"] = df_pw["value"].abs()

    # normalise each edge by total flow through its source and target
    df_pw["child_sum_target"] = df_pw.groupby("target")["value_abs"].transform("sum")
    df_pw["child_sum_source"] = df_pw.groupby("source")["value_abs"].transform("sum")
    df_pw["value_normalized_by_target"] = 100. * df_pw["value_abs"] / df_pw["child_sum_target"]
    df_pw["value_normalized_by_source"] = 100. * df_pw["value_abs"] / df_pw["child_sum_source"]

    # look up pre-computed node importance for source and target of each edge
    df_pw["target_importance"] = df_pw["target"].map(node_importance).fillna(0)
    df_pw["source_importance"] = df_pw["source"].map(node_importance).fillna(0)

    missing_src = df_pw[df_pw["source_importance"] == 0]["source"].unique()
    missing_tgt = df_pw[df_pw["target_importance"] == 0]["target"].unique()
    if len(missing_src) > 0:
        print(f"  WARNING: zero source_importance for: {missing_src.tolist()}")
    if len(missing_tgt) > 0:
        print(f"  WARNING: zero target_importance for: {missing_tgt.tolist()}")

    # A and B are the importance-weighted normalised edge values
    # taking the min ensures an edge is only large if BOTH endpoints are important
    df_pw["A"] = df_pw["value_normalized_by_source"] * df_pw["source_importance"]
    df_pw["B"] = df_pw["value_normalized_by_target"] * df_pw["target_importance"]
    df_pw["value_final"] = df_pw[["A", "B"]].min(axis=1)

    # compute flow conservation error for each node
    # residual nodes absorb the error to preserve total flow
    df_pw["source_fan_out"] = df_pw.groupby("source")["value_final"].transform("sum")
    df_pw["source_fan_out_error"] = np.abs(df_pw["source_fan_out"] - 100. * df_pw["source_importance"])
    df_pw["target_fan_in"] = df_pw.groupby("target")["value_final"].transform("sum")
    df_pw["target_fan_in_error"] = np.abs(df_pw["target_fan_in"] - 100. * df_pw["target_importance"])

    # apply error correction to residual node edges
    df_pw["value_final_corrected"] = df_pw["value_final"]
    ind_src = df_pw["source"].str.contains("others", na=False)
    df_pw.loc[ind_src, "value_final_corrected"] = (
        df_pw.loc[ind_src, "value_final"] + df_pw.loc[ind_src, "target_fan_in_error"]
    )
    ind_tgt = df_pw["target"].str.contains("others", na=False)
    df_pw.loc[ind_tgt, "value_final_corrected"] = (
        df_pw.loc[ind_tgt, "value_final_corrected"] + df_pw.loc[ind_tgt, "source_fan_out_error"]
    )
    df_pw["value"] = df_pw["value_final_corrected"]
    df_pw = df_pw[df_pw["value"] > 0][["source", "target", "value", "layer"]].copy()
    print(f"  Pathway edges after adjust_values: {len(df_pw)}")
    print(f"  Value range: [{df_pw['value'].min():.4f}, {df_pw['value'].max():.4f}]")

    # concatenate unadjusted first layer + adjusted pathway edges — matching original
    df = pd.concat([first_edges, df_pw], ignore_index=True)
    print(f"  Total edges after concat: {len(df)}")

    input_totals_final = df[df["source"].isin(["mut_important", "cnv_amp", "cnv_del"])]
    input_totals_final = input_totals_final.groupby("source")["value"].sum().sort_values(ascending=False)
    print(f"  Input type totals (final):\n{input_totals_final.to_string()}")

    # -------------------------------------------------------------------
    # 5. build node list
    #    only include residual (others) nodes that actually have edges after
    #    adjust_values — avoids isolated floating nodes in the diagram
    #    each node is keyed by (layer_idx, node_id) to guarantee uniqueness
    #    even when the same pathway appears in adjacent layers
    # -------------------------------------------------------------------
    print(f"\n--- Section 5: Node list ---")

    # find which residual nodes survived the adjust_values filter
    others_with_edges = set(
        df[df["source"].str.startswith("others", na=False)]["source"].tolist() +
        df[df["target"].str.startswith("others", na=False)]["target"].tolist()
    )
    print(f"  others nodes with edges: {sorted(others_with_edges)}")

    # build ordered node list per layer — important nodes first, residual at end
    layer_nodes = {}
    layer_nodes[0] = ["mut_important", "cnv_amp", "cnv_del"]
    layer_nodes[1] = high_nodes[1] + (["others1"] if "others1" in others_with_edges else [])
    for i in range(1, n_hidden_layers + 1):
        others_id = f"others{i + 1}"
        layer_nodes[i + 1] = high_nodes[i + 1] + ([others_id] if others_id in others_with_edges else [])
    root_layer = n_hidden_layers + 2
    layer_nodes[root_layer] = ["root"]  # outcome node, connects from last pathway layer

    for layer_idx, nodes in sorted(layer_nodes.items()):
        print(f"  layer_nodes[{layer_idx}]: {len(nodes)} nodes — {nodes}")

    # assign a unique integer index to each (layer, node) pair
    # this prevents self-loops when the same pathway ID appears in adjacent layers
    all_node_ids = []
    node_to_idx = {}
    for layer_idx in sorted(layer_nodes.keys()):
        for node_id in layer_nodes[layer_idx]:
            key = (layer_idx, node_id)
            if key not in node_to_idx:
                node_to_idx[key] = len(all_node_ids)
                all_node_ids.append(node_id)
    print(f"  Total nodes: {len(all_node_ids)}")

    # convert internal IDs to display labels (pathway IDs -> full names)
    display_name_map = {
        "mut_important": "mutation",
        "cnv_amp": "amplification",
        "cnv_del": "deletion",
        "root": "outcome",
    }
    all_node_labels = []
    for node_id in all_node_ids:
        if node_id in display_name_map:
            all_node_labels.append(display_name_map[node_id])
        elif node_id.startswith("others"):
            all_node_labels.append("residual")
        else:
            all_node_labels.append(id_to_name.get(node_id, node_id))

    # -------------------------------------------------------------------
    # 6. encode edges as integer source/target indices for plotly
    #    also connect the last pathway layer to the root/outcome node
    #    using raw coef importance as edge weight
    # -------------------------------------------------------------------
    print(f"\n--- Section 6: Edge encoding ---")
    diagram_source = []
    diagram_target = []
    diagram_values = []
    missing_keys = []

    for _, row in df.iterrows():
        layer_idx = int(row["layer"])
        src_key = (layer_idx, row["source"])
        tgt_key = (layer_idx + 1, row["target"])
        if src_key in node_to_idx and tgt_key in node_to_idx:
            diagram_source.append(node_to_idx[src_key])
            diagram_target.append(node_to_idx[tgt_key])
            diagram_values.append(row["value"])
        else:
            missing_keys.append((src_key, src_key in node_to_idx,
                                  tgt_key, tgt_key in node_to_idx))

    if missing_keys:
        print(f"  WARNING: {len(missing_keys)} edges dropped due to missing keys:")
        for mk in missing_keys[:5]:
            print(f"    src={mk[0]} found={mk[1]}, tgt={mk[2]} found={mk[3]}")

    print(f"  Encoded edges (excl. root): {len(diagram_source)}")

    # connect each last-layer node to the root/outcome node
    # edge weight = raw coef importance of that pathway node
    last_layer_idx = n_hidden_layers + 1
    root_idx = node_to_idx[(root_layer, "root")]
    last_importance = deeplift[f"h{n_hidden_layers}"]["coef"]
    n_root_edges = 0
    for node_id in layer_nodes[last_layer_idx]:
        key = (last_layer_idx, node_id)
        if key in node_to_idx:
            # residual node gets small fixed weight so it appears in the diagram
            imp = 1.0 if node_id.startswith("others") else float(last_importance.get(node_id, 0))
            if imp > 0:
                diagram_source.append(node_to_idx[key])
                diagram_target.append(root_idx)
                diagram_values.append(imp)
                n_root_edges += 1
    print(f"  Root edges added: {n_root_edges}")
    print(f"  Total encoded edges: {len(diagram_source)}")

    # -------------------------------------------------------------------
    # 7. node x/y positions
    #    x: replicates original get_x_y() which hard-codes layers 0,1,2 then
    #       uses np.linspace(0.14, 1, 6, endpoint=False) for layers 3-7
    #    y: flow-weighted cumulative positioning within each layer
    #       matching original get_x_y() logic:
    #       - node flow = max(total outgoing, total incoming) per node
    #       - residual nodes zeroed before sort (forced to bottom) then restored
    #       - y = (cumsum - 0.5*flow) / (1.5 * layer_total)
    #       - root fixed at y=0.33
    # -------------------------------------------------------------------
    print(f"\n--- Section 7: Node positions ---")

    # replicate original's x position logic exactly:
    #   layer 0 -> 0.01, layer 1 -> 0.08, layer 2 -> 0.14
    #   layers 3..7 -> np.linspace(0.14, 1, 6, endpoint=False)[1:]
    #   (linspace gives 6 values from 0.14 to ~0.98; skip index 0 = 0.14 already used)
    xs_linspace = np.linspace(0.14, 1, 6, endpoint=False)
    print(f"  linspace x values (layers 3+): {xs_linspace[1:].tolist()}")
    x_positions_map = {
        0: 0.01,
        1: 0.08,
        2: 0.14,
    }
    for i, x_val in enumerate(xs_linspace[1:]):
        x_positions_map[i + 3] = float(x_val)  # layers 3, 4, 5, 6, 7
    print(f"  x_positions_map: { {k: round(v, 4) for k, v in sorted(x_positions_map.items())} }")

    # root layer uses the last entry in the map (layer 7 = last pathway + 1)
    # original sets root x via the linspace loop which covers up to layer 7,
    # so root_layer (=8) falls back — we explicitly set it to 0.99 to match plotly's
    # right-edge convention (original rescales x to [0.01, 0.98] in get_data_trace)
    x_positions_map[root_layer] = 0.99

    n_nodes = len(all_node_labels)

    # matching original get_x_y: node weight = max(source_sum, target_sum)
    # this means input nodes (source only) use their outgoing flow,
    # output nodes (target only) use their incoming flow,
    # and intermediate nodes use whichever direction carries more flow
    source_flow = np.zeros(n_nodes)
    target_flow = np.zeros(n_nodes)
    for src, tgt, val in zip(diagram_source, diagram_target, diagram_values):
        source_flow[src] += abs(val)
        target_flow[tgt] += abs(val)
    node_flow = np.maximum(source_flow, target_flow)
    print(f"  node_flow range: [{node_flow.min():.2f}, {node_flow.max():.2f}]")

    x_pos = [0.0] * n_nodes
    y_pos = [0.0] * n_nodes

    for layer_idx in sorted(layer_nodes.keys()):
        nodes = layer_nodes[layer_idx]
        idxs = [node_to_idx[(layer_idx, n)] for n in nodes]
        flows = node_flow[idxs].copy()

        print(f"  Layer {layer_idx} flows (pre-sort):")
        for n, f in zip(nodes, flows):
            print(f"    {n}: {f:.2f}")

        # zero out residual nodes before sorting so they fall to the bottom,
        # then restore their flows for the cumsum calculation
        # this matches original get_x_y which does the same with 'others' nodes
        is_others = np.array([n.startswith("others") or n == "root" for n in nodes])
        others_flows = flows[is_others].copy()
        flows[is_others] = 0.0
        sort_order = np.argsort(flows)[::-1]  # descending: most important at top
        flows[is_others] = others_flows        # restore residual flows
        flows = flows[sort_order]              # apply sort order

        print(f"  Layer {layer_idx} sort order: {[nodes[i] for i in sort_order]}")

        # cumulative y positioning: each node centred on its cumulative flow fraction
        # divided by 1.5 to compress nodes into top 2/3 of the plot area
        layer_total = flows.sum() or 1.0
        cumulative = np.cumsum(flows)
        ys = (cumulative - 0.5 * flows) / (1.5 * layer_total)

        x_val = x_positions_map.get(layer_idx, 0.99)
        for rank, orig_rank in enumerate(sort_order):
            node_idx = idxs[orig_rank]
            x_pos[node_idx] = x_val
            y_pos[node_idx] = float(ys[rank])
            print(f"    rank={rank} node={nodes[orig_rank]} x={x_val:.4f} y={ys[rank]:.3f}")

    # root/outcome node fixed at y=0.33 matching original
    y_pos[root_idx] = 0.33
    print(f"  root y fixed at 0.33")
    print(f"  x_pos range: [{min(x_pos):.4f}, {max(x_pos):.4f}]")
    print(f"  y_pos range: [{min(y_pos):.4f}, {max(y_pos):.4f}]")

    # -------------------------------------------------------------------
    # 8. node and edge colours
    #    matching original get_node_colors_ordered():
    #    - important nodes: Reds colormap, darkest at top (most important)
    #    - residual nodes: light grey rgba(232,232,232,0.5)
    #    - input types: fixed colours (cyan=mutation, orange=amplification,
    #      dark blue=deletion) matching original special-case colouring
    #    - outcome node: mid-red
    #    - edges involving 'others' nodes: grey rgba(192,192,192,0.2)
    #      matching original get_edge_colors with remove_others=False
    #    - all other edges: source node colour at alpha=0.2
    # -------------------------------------------------------------------
    print(f"\n--- Section 8: Colours ---")
    cmap = plt.cm.Reds
    node_colours = ['rgba(128,128,128,0.5)'] * n_nodes  # default grey

    for layer_idx in sorted(layer_nodes.keys()):
        nodes = layer_nodes[layer_idx]
        # colour important nodes with Reds gradient: darkest = most important (top)
        important_nodes = [n for n in nodes if not n.startswith("others") and n != "root"]
        n = len(important_nodes)
        colour_idx = np.linspace(1.0, 0.0, n)  # 1.0=dark, 0.0=light
        for j, node_id in enumerate(important_nodes):
            key = (layer_idx, node_id)
            if key in node_to_idx:
                colors = list(cmap(colour_idx[j]))
                colors = [int(255 * c) for c in colors[:3]]
                node_colours[node_to_idx[key]] = f'rgba({colors[0]},{colors[1]},{colors[2]},0.7)'
        # residual nodes: light grey
        for node_id in nodes:
            if node_id.startswith("others"):
                key = (layer_idx, node_id)
                if key in node_to_idx:
                    node_colours[node_to_idx[key]] = 'rgba(232,232,232,0.5)'

    # outcome node: mid-red
    node_colours[root_idx] = 'rgba(255,100,100,0.7)'

    # input type fixed colours matching original special cases
    for node_id, colour in [
        ("mut_important", 'rgba(105,189,210,0.7)'),   # cyan/teal for mutation
        ("cnv_amp", 'rgba(224,123,57,0.7)'),           # orange for amplification
        ("cnv_del", 'rgba(1,55,148,0.7)')              # dark blue for deletion
    ]:
        key = (0, node_id)
        if key in node_to_idx:
            node_colours[node_to_idx[key]] = colour
            print(f"  Set {node_id} colour to {colour}")

    # edge colours — matching original get_edge_colors(remove_others=False):
    #   edges involving 'others' nodes -> grey rgba(192,192,192,0.2)
    #   all other edges -> source node colour at alpha=0.2
    edge_colours = []
    for src, tgt in zip(diagram_source, diagram_target):
        src_id = all_node_ids[src]
        tgt_id = all_node_ids[tgt]
        if "others" in src_id or "others" in tgt_id:
            edge_colours.append('rgba(192,192,192,0.2)')
        else:
            base = node_colours[src]
            # replace alpha component robustly using known alpha values
            edge_colours.append(base.replace(',0.7)', ',0.2)').replace(',0.5)', ',0.2)'))

    print(f"  others edges (grey): {sum(1 for c in edge_colours if c == 'rgba(192,192,192,0.2)')}")
    print(f"  coloured edges: {sum(1 for c in edge_colours if c != 'rgba(192,192,192,0.2)')}")

    # -------------------------------------------------------------------
    # 9. render and save
    #    two outputs matching original:
    #    - static PDF/PNG at 600px width (scale=5 for high-res PNG)
    #    - interactive HTML at 1200px width with larger font
    # -------------------------------------------------------------------
    print(f"\n--- Section 9: Render ---")
    print(f"  Total nodes: {len(all_node_labels)}")
    print(f"  Total edges: {len(diagram_source)}")

    scale = 1.0
    width = 600. / scale
    height = 0.5 * width / scale

    data_trace = dict(
        type='sankey',
        arrangement='snap',      # snap nodes to specified x/y positions
        domain=dict(x=[0, 1.], y=[0, 1.]),
        orientation="h",         # left to right flow
        valueformat=".0f",
        node=dict(
            pad=2,               # minimal padding between nodes
            thickness=10,        # node bar width in pixels
            line=dict(color="white", width=0.5),
            label=all_node_labels,
            x=x_pos,
            y=y_pos,
            color=node_colours,
        ),
        link=dict(
            source=diagram_source,
            target=diagram_target,
            value=diagram_values,
            color=edge_colours,
        )
    )

    layout = dict(
        height=height,
        width=width,
        margin=dict(l=0, r=0, b=0.1, t=8),
        font=dict(size=6, family='Arial')
    )

    fig = go.Figure(dict(data=[data_trace], layout=layout))
    fig.write_image(f"{figures_dir}/sankey.pdf", scale=1, width=width, height=height, format='pdf')
    fig.write_image(f"{figures_dir}/sankey.png", scale=5, width=width, height=height, format='png')
    print(f"  Saved sankey.pdf and sankey.png")

    # interactive HTML at larger size for exploration
    scale = 0.5
    width_html = 600. / scale
    height_html = 0.5 * width_html
    layout_html = dict(
        height=height_html,
        width=width_html,
        margin=dict(l=0, r=0, b=0.1, t=8),
        font=dict(size=12, family='Arial')
    )
    fig_html = go.Figure(dict(data=[data_trace], layout=layout_html))
    fig_html.write_html(f"{figures_dir}/sankey.html")
    print(f"  Saved sankey.html")
    print(f"=== plot_sankey complete ===\n")