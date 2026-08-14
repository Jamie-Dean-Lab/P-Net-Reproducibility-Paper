import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from architecture.pnet_model import (
    PNetArchitectureGenerator,
    get_layer_maps,
    get_map_from_layer,
)


def _write(tmp, name, text):
    path = os.path.join(tmp, name)
    with open(path, "w") as f:
        f.write(text)
    return path


class TestGetMapFromLayer(unittest.TestCase):

    def setUp(self):
        self.map = get_map_from_layer({"p1": ["g1", "g2"], "p2": ["g2", "g3"]})

    def test_rows_are_genes_columns_are_pathways(self):
        self.assertEqual(list(self.map.index), ["g1", "g2", "g3"])
        self.assertEqual(list(self.map.columns), ["p1", "p2"])

    def test_membership_encoded_as_one(self):
        self.assertEqual(self.map.loc["g1", "p1"], 1.0)
        self.assertEqual(self.map.loc["g3", "p2"], 1.0)

    def test_non_membership_encoded_as_zero(self):
        self.assertEqual(self.map.loc["g1", "p2"], 0.0)
        self.assertEqual(self.map.loc["g3", "p1"], 0.0)

    def test_shared_gene_belongs_to_both_pathways(self):
        self.assertTrue((self.map.loc["g2"] == 1.0).all())

    def test_genes_deduplicated_across_pathways(self):
        self.assertEqual(len(self.map.index), len(set(self.map.index)))

    def test_edge_count_matches_input(self):
        self.assertEqual(self.map.values.sum(), 4)

    def test_matrix_is_binary(self):
        self.assertTrue(np.isin(self.map.values, [0.0, 1.0]).all())

    def test_empty_pathway_gives_all_zero_column(self):
        m = get_map_from_layer({"p1": ["g1"], "empty": []})
        self.assertEqual(m["empty"].sum(), 0.0)


class TestGetLayerMaps(unittest.TestCase):

    def setUp(self):
        self.layers = [{"root": ["p1", "p2"]}, {"p1": ["g1", "g2"], "p2": ["g2", "g3"]}]
        self.genes = ["g1", "g2", "gX"]
        self.maps = get_layer_maps(self.genes, self.layers, False)

    def test_layer_order_is_reversed_so_genes_come_first(self):
        self.assertEqual(list(self.maps[0].index), ["g1", "g2", "gX"])
        self.assertEqual(list(self.maps[-1].columns), ["root"])

    def test_rows_restricted_to_supplied_genes(self):
        self.assertEqual(list(self.maps[0].index), self.genes)

    def test_gene_with_no_annotation_kept_as_zero_row(self):
        self.assertEqual(self.maps[0].loc["gX"].sum(), 0.0)

    def test_genes_absent_from_the_list_are_dropped(self):
        self.assertNotIn("g3", self.maps[0].index)

    def test_columns_chain_into_the_next_layer_index(self):
        for lower, upper in zip(self.maps, self.maps[1:]):
            self.assertEqual(list(lower.columns), list(upper.index))

    def test_no_nan_left_after_filtering(self):
        self.assertFalse(any(m.isna().any().any() for m in self.maps))

    def test_index_and_columns_sorted(self):
        for m in self.maps:
            self.assertEqual(list(m.index), sorted(m.index))
            self.assertEqual(list(m.columns), sorted(m.columns))

    def test_one_map_per_input_layer(self):
        self.assertEqual(len(self.maps), len(self.layers))

    def test_connectivity_preserved_from_source_layers(self):
        self.assertEqual(self.maps[0].loc["g1", "p1"], 1.0)
        self.assertEqual(self.maps[0].loc["g2", "p2"], 1.0)
        self.assertEqual(self.maps[0].loc["g1", "p2"], 0.0)

    def test_add_unk_genes_adds_a_column(self):
        maps = get_layer_maps(self.genes, self.layers, True)
        self.assertIn("UNK", maps[0].columns)

    def test_unk_flags_only_unannotated_genes(self):
        maps = get_layer_maps(self.genes, self.layers, True)
        self.assertEqual(maps[0].loc["gX", "UNK"], 1)
        self.assertEqual(maps[0].loc["g1", "UNK"], 0)

    def test_unk_absent_when_flag_is_false(self):
        self.assertNotIn("UNK", self.maps[0].columns)


class TestLoadGmt(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = PNetArchitectureGenerator()
        self.path = _write(self.tmp, "x.gmt",
                           "NameA\tPWY1\tsrc\tgA\tgB\nNameB\tPWY2\tsrc\tgB\tgC\n")

    def test_returns_long_group_gene_frame(self):
        df = self.gen.load_gmt(self.path, genes_col=3, pathway_col=1)
        self.assertEqual(list(df.columns), ["group", "gene"])

    def test_one_row_per_pathway_gene_pair(self):
        df = self.gen.load_gmt(self.path, genes_col=3, pathway_col=1)
        self.assertEqual(len(df), 4)

    def test_pathway_column_index_respected(self):
        df = self.gen.load_gmt(self.path, genes_col=3, pathway_col=1)
        self.assertEqual(sorted(df["group"].unique()), ["PWY1", "PWY2"])

    def test_genes_column_index_respected(self):
        df = self.gen.load_gmt(self.path, genes_col=3, pathway_col=1)
        self.assertEqual(sorted(df["gene"].unique()), ["gA", "gB", "gC"])

    def test_gene_shared_by_two_pathways_appears_twice(self):
        df = self.gen.load_gmt(self.path, genes_col=3, pathway_col=1)
        self.assertEqual((df["gene"] == "gB").sum(), 2)

    def test_reactome_column_layout(self):
        path = _write(self.tmp, "r.gmt", "PWY1\tsrc\tgA\tgB\n")
        df = self.gen.load_gmt(path, genes_col=1, pathway_col=0)
        self.assertEqual(sorted(df["gene"].unique()), ["gA", "gB", "src"])

    def test_copy_suffix_stripped(self):
        path = _write(self.tmp, "c.gmt", "NameA\tPWY1_copy1\tsrc\tgA\n")
        df = self.gen.load_gmt(path, genes_col=3, pathway_col=1)
        self.assertEqual(df["group"].tolist(), ["PWY1"])


class TestGetNetworkx(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = PNetArchitectureGenerator()
        self.reactome = _write(self.tmp, "rel.txt",
                               "R-HSA-2\tR-HSA-1\nR-HSA-3\tR-HSA-1\n"
                               "R-HSA-4\tR-HSA-2\nOTHER-1\tOTHER-2\n")

    def test_non_human_reactome_rows_filtered_out(self):
        net = self.gen.get_networkx(self.reactome, "reactome")
        self.assertNotIn("OTHER-1", net.nodes())

    def test_reactome_edges_run_child_to_parent(self):
        net = self.gen.get_networkx(self.reactome, "reactome")
        self.assertIn(("R-HSA-2", "R-HSA-1"), net.edges())

    def test_root_node_added(self):
        net = self.gen.get_networkx(self.reactome, "reactome")
        self.assertIn("root", net.nodes())

    def test_root_connects_to_every_source_node(self):
        net = self.gen.get_networkx(self.reactome, "reactome")
        self.assertEqual(sorted(net.successors("root")), ["R-HSA-3", "R-HSA-4"])

    def test_root_has_no_predecessors(self):
        net = self.gen.get_networkx(self.reactome, "reactome")
        self.assertEqual(list(net.predecessors("root")), [])

    def test_every_node_reachable_from_root(self):
        import networkx as nx
        net = self.gen.get_networkx(self.reactome, "reactome")
        reachable = nx.descendants(net, "root") | {"root"}
        self.assertEqual(reachable, set(net.nodes()))

    def test_graph_named_after_the_dataset(self):
        self.assertEqual(self.gen.get_networkx(self.reactome, "reactome").name, "reactome")

    def test_go_rows_filtered_to_go_prefix(self):
        path = _write(self.tmp, "go.txt", "GO:0002\tGO:0001\nNOTGO\tGO:0001\n")
        net = self.gen.get_networkx(path, "go")
        self.assertNotIn("NOTGO", net.nodes())

    def test_go_edges_run_parent_to_child(self):
        path = _write(self.tmp, "go.txt", "GO:0002\tGO:0001\n")
        net = self.gen.get_networkx(path, "go")
        self.assertIn(("GO:0001", "GO:0002"), net.edges())

    def test_go_biological_process_root_removed(self):
        path = _write(self.tmp, "go.txt", "GO:0002\tGO:0001\nGO:0001\tGO:0008150\n")
        net = self.gen.get_networkx(path, "go")
        self.assertNotIn("GO:0008150", net.nodes())

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            self.gen.get_networkx(self.reactome, "kegg")


class TestHierarchyLevels(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = PNetArchitectureGenerator()
        path = _write(self.tmp, "rel.txt",
                      "R-HSA-2\tR-HSA-1\nR-HSA-3\tR-HSA-1\nR-HSA-4\tR-HSA-2\n")
        self.net = self.gen.get_networkx(path, "reactome")

    def test_level_zero_is_the_root_alone(self):
        self.assertEqual(self.gen.get_nodes_at_level(self.net, 0), ["root"])

    def test_level_one_is_the_roots_children(self):
        self.assertEqual(sorted(self.gen.get_nodes_at_level(self.net, 1)),
                         ["R-HSA-3", "R-HSA-4"])

    def test_levels_exclude_closer_nodes(self):
        level2 = set(self.gen.get_nodes_at_level(self.net, 2))
        self.assertNotIn("root", level2)
        self.assertFalse(level2 & set(self.gen.get_nodes_at_level(self.net, 1)))

    def test_terminals_have_no_successors(self):
        for node in self.gen.get_terminals(self.net):
            self.assertEqual(list(self.net.successors(node)), [])


class TestCompleteNetwork(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = PNetArchitectureGenerator()
        # Branch A is three deep, branch B only two.
        path = _write(self.tmp, "rel.txt",
                      "R-HSA-A1\tR-HSA-A2\nR-HSA-A2\tR-HSA-A3\nR-HSA-B1\tR-HSA-B2\n")
        self.net = self.gen.get_networkx(path, "reactome")
        self.completed = self.gen.get_completed_network(self.net, 3)

    def test_short_branch_padded_with_copy_nodes(self):
        self.assertIn("R-HSA-B2_copy1", self.completed.nodes())

    def test_deep_branch_left_alone(self):
        self.assertFalse([n for n in self.completed.nodes()
                          if n.startswith("R-HSA-A") and "_copy" in n])

    def test_original_edges_preserved(self):
        for edge in self.net.edges():
            self.assertIn(edge, self.completed.edges())

    def test_every_branch_reaches_requested_depth(self):
        layers = self.gen.get_layers_from_net(self.completed, 3)
        self.assertEqual(len(layers), 3)
        self.assertTrue(all(layers))

    def test_copy_suffix_stripped_from_layer_dicts(self):
        layers = self.gen.get_layers_from_net(self.completed, 3)
        for layer in layers:
            for parent, children in layer.items():
                self.assertNotIn("_copy", parent)
                self.assertFalse([c for c in children if "_copy" in c])

    def test_padded_pathway_maps_to_itself(self):
        layers = self.gen.get_layers_from_net(self.completed, 3)
        self.assertEqual(layers[2]["R-HSA-B2"], ["R-HSA-B2"])

    def test_add_edges_builds_a_chain(self):
        import networkx as nx
        g = nx.DiGraph()
        self.gen.add_edges(g, "n", 3)
        self.assertEqual(sorted(g.edges()),
                         [("n", "n_copy1"), ("n_copy1", "n_copy2"), ("n_copy2", "n_copy3")])


class TestGetLayers(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = PNetArchitectureGenerator()
        path = _write(self.tmp, "rel.txt", "R-HSA-2\tR-HSA-1\nR-HSA-3\tR-HSA-1\n")
        self.net = self.gen.get_networkx(path, "reactome")
        self.gmt = _write(self.tmp, "x.gmt",
                          "NameA\tR-HSA-1\tsrc\tgA\tgB\tgZ\n")

    def test_gene_layer_appended_below_the_pathway_layers(self):
        layers = self.gen.get_layers(self.net, 2, self.gmt, ["gA", "gB"])
        self.assertEqual(len(layers), 3)

    def test_gene_layer_restricted_to_supplied_alignment_ids(self):
        layers = self.gen.get_layers(self.net, 2, self.gmt, ["gA", "gB"])
        self.assertEqual(sorted(layers[-1]["R-HSA-1"]), ["gA", "gB"])

    def test_genes_absent_from_the_dataset_excluded(self):
        layers = self.gen.get_layers(self.net, 2, self.gmt, ["gA"])
        self.assertNotIn("gZ", list(layers[-1]["R-HSA-1"]))

    def test_pathway_with_no_matching_genes_kept_as_empty(self):
        layers = self.gen.get_layers(self.net, 2, self.gmt, ["nothing"])
        self.assertEqual(len(layers[-1]["R-HSA-1"]), 0)


class TestEndToEndMapConsistency(unittest.TestCase):
    """The hierarchy must survive the whole build as a consistent chain of maps."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.gen = PNetArchitectureGenerator()
        path = _write(self.tmp, "rel.txt",
                      "R-HSA-2\tR-HSA-1\nR-HSA-3\tR-HSA-1\nR-HSA-4\tR-HSA-2\n")
        self.net = self.gen.get_networkx(path, "reactome")
        self.gmt = _write(self.tmp, "x.gmt",
                          "A\tR-HSA-1\tsrc\tg1\tg2\nB\tR-HSA-2\tsrc\tg2\tg3\n")
        self.genes = ["g1", "g2", "g3"]
        layers = self.gen.get_layers(self.net, 2, self.gmt, self.genes)
        self.maps = get_layer_maps(sorted(self.genes), layers, False)

    def test_first_map_is_indexed_by_genes(self):
        self.assertEqual(list(self.maps[0].index), self.genes)

    def test_adjacent_maps_share_a_dimension(self):
        for lower, upper in zip(self.maps, self.maps[1:]):
            self.assertEqual(lower.shape[1], upper.shape[0])
            self.assertEqual(list(lower.columns), list(upper.index))

    def test_all_maps_binary(self):
        for m in self.maps:
            self.assertTrue(np.isin(m.values, [0.0, 1.0]).all())

    def test_network_is_sparser_than_dense_equivalent(self):
        for m in self.maps:
            self.assertLessEqual(m.values.sum(), m.shape[0] * m.shape[1])

    def test_every_gene_row_present_even_if_unconnected(self):
        self.assertEqual(len(self.maps[0].index), len(self.genes))


if __name__ == "__main__":
    unittest.main()
