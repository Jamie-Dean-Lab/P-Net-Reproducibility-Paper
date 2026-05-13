from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import pandas as pd


FIGURE_WIDTH = 1800
FIGURE_HEIGHT = 1000  # increase this if nodes still overlap
MIN_PX_GAP = 50       # minimum pixels between node centres — increase if text overlaps


class SankeyDiagram:

    def __init__(self, labels, weights):
        self.labels = labels
        self.weights = weights
        self.number_of_layers = len(self.labels)
        self.residual_labels_indexes = {}
        self.important_labels_indexes = {}

    def _get_layer_keys(self):
        return sorted(self.labels.keys(), key=lambda x: int(x.split("_")[1]))

    def _scale_network(self, number_of_important_points):
        sorted_weights = []
        layer_keys = self._get_layer_keys()

        for idx, layer in enumerate(tqdm(reversed(layer_keys), total=len(layer_keys), colour='red')):
            i = int(layer.split("_")[1])
            labels = self.labels[layer]
            weights = self.weights[layer]
            important_points = number_of_important_points[i]
            total_importances = np.abs(weights).sum(axis=1)
            sorted_label_indexes = np.argsort(total_importances)[::-1]
            sorted_weights.append(pd.DataFrame({
                "Label": labels[sorted_label_indexes],
                "Weight": weights.sum(axis=1)[sorted_label_indexes],
                "Layer": [layer] * len(labels)
            }))

            # layer_0 (inputs) — show all nodes explicitly, no residual
            if i == 0:
                self.labels[layer] = labels[sorted_label_indexes]
                self.weights[layer] = weights[sorted_label_indexes, :]
                self.residual_labels_indexes[layer] = np.array([], dtype=int)
                self.important_labels_indexes[layer] = sorted_label_indexes
                continue

            selected_labels = []
            line_length = 33
            for label in labels[sorted_label_indexes[:important_points]]:
                lines = []
                for j in range(0, len(label), line_length):
                    if j + line_length > len(label):
                        lines.append(label[j:])
                    else:
                        lines.append(label[j:j + line_length])
                selected_labels.append("<br />".join(lines))

            selected_weights = weights[sorted_label_indexes[:important_points], :]
            residual_weights = weights[sorted_label_indexes[important_points:], :]
            total_residual_weights = residual_weights.sum(axis=0)
            self.labels[layer] = np.concatenate([selected_labels, np.array(["Residual"])])
            self.weights[layer] = np.vstack([selected_weights, total_residual_weights])
            self.residual_labels_indexes[layer] = sorted_label_indexes[important_points:]
            self.important_labels_indexes[layer] = sorted_label_indexes[:important_points]

            next_layer = f"layer_{i + 1}"
            if next_layer in self.labels:
                next_layer_important = self.important_labels_indexes[next_layer]
                next_layer_residual = self.residual_labels_indexes[next_layer]
                scaled_weights = self.weights[layer][:, next_layer_important]
                if len(next_layer_residual) > 0:
                    residual_links = self.weights[layer][:, next_layer_residual].sum(axis=1).reshape(-1, 1)
                    self.weights[layer] = np.hstack([scaled_weights, residual_links])
                else:
                    self.weights[layer] = scaled_weights

        # after the main loop, trim layer_0 columns to match scaled layer_1
        layer_1_important = self.important_labels_indexes["layer_1"]
        layer_1_residual = self.residual_labels_indexes["layer_1"]
        scaled = self.weights["layer_0"][:, layer_1_important]
        if len(layer_1_residual) > 0:
            residual_links = self.weights["layer_0"][:, layer_1_residual].sum(axis=1).reshape(-1, 1)
            self.weights["layer_0"] = np.hstack([scaled, residual_links])
        else:
            self.weights["layer_0"] = scaled

        return pd.concat(sorted_weights)

    def _get_colours(self, layer_keys, diagram_labels, diagram_source, diagram_values):
        import matplotlib.pyplot as plt

        cmap = plt.cm.Reds
        n_nodes = len(diagram_labels)
        node_colours = ['rgba(100,100,100,0.7)'] * n_nodes

        for layer in layer_keys:
            idxs = self.diagram_indexes[layer]
            labels = self.labels[layer]
            n = len(labels)

            # per-layer gradient: each layer gets its own full dark→mid range
            colour_idx = np.linspace(0.99, 0.3, n)
            for j, (idx, label) in enumerate(zip(idxs, labels)):
                if 'Residual' in str(label):
                    node_colours[idx] = 'rgba(160,160,160,0.8)'
                else:
                    r, g, b, _ = cmap(colour_idx[j])
                    node_colours[idx] = f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.7)'

        # outcome node
        outcome_idx = len(diagram_labels) - 1
        r, g, b, _ = cmap(0.8)
        node_colours[outcome_idx] = f'rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.7)'

        # edge colours: match source node at low alpha
        edge_colours = []
        for src in diagram_source:
            base = node_colours[src]
            edge_colours.append(base.replace('0.7', '0.2').replace('0.8', '0.2'))

        return node_colours, edge_colours

    def _compute_node_positions(self, diagram_source, diagram_target, diagram_values, layer_keys):
        x_positions_map = {
            0: 0.01, 1: 0.1, 2: 0.16, 3: 0.32, 4: 0.48, 5: 0.64, 6: 0.8,
        }

        n_nodes = sum(len(self.labels[l]) for l in layer_keys) + 1
        node_flow = np.zeros(n_nodes)
        for src, tgt, val in zip(diagram_source, diagram_target, diagram_values):
            node_flow[src] += abs(val)
            node_flow[tgt] += abs(val)

        x_positions = []
        y_positions = []

        for layer in layer_keys:
            i = int(layer.split("_")[1])
            idxs = self.diagram_indexes[layer]
            flows = node_flow[idxs].copy()
            labels = self.labels[layer]

            # zero out residual before sorting so it falls to the bottom, then restore
            is_residual = np.array(['Residual' in str(l) for l in labels])
            residual_flows = flows[is_residual].copy()
            flows[is_residual] = 0.0
            sort_order = np.argsort(flows)[::-1]
            flows[is_residual] = residual_flows
            flows = flows[sort_order]

            # cumulative-sum y positioning, compressed by 1.5x so nodes don't overflow
            layer_total = flows.sum() or 1.0
            cumulative = np.cumsum(flows)
            ys = (cumulative - 0.5 * flows) / (1.5 * layer_total)

            x_positions.extend([x_positions_map[i]] * len(flows))
            y_positions.extend(ys.tolist())

        # outcome node
        x_positions.append(0.99)
        y_positions.append(0.33)

        return x_positions, y_positions

    def plot(self, number_of_important_points, save_path):
        sorted_weights = self._scale_network(number_of_important_points)
        sorted_weights.to_csv(f"{save_path}/deeplift_weights.csv")

        layer_keys = self._get_layer_keys()
        self.diagram_indexes = {}
        diagram_labels = []
        diagram_source = []
        diagram_target = []
        diagram_values = []
        total_number_of_nodes = 0

        for layer in layer_keys:
            input_labels = self.labels[layer].tolist()
            current_number_of_nodes = len(input_labels)
            diagram_labels.extend(input_labels)
            self.diagram_indexes[layer] = total_number_of_nodes + np.array(range(current_number_of_nodes))
            total_number_of_nodes += current_number_of_nodes

        for idx, layer in enumerate(layer_keys[:-1]):
            next_layer = layer_keys[idx + 1]
            weights = self.weights[layer].copy()
            weights /= weights.sum()
            current_number_of_nodes = weights.shape[0]
            next_number_of_nodes = weights.shape[1]
            current_indexes = self.diagram_indexes[layer].tolist()
            next_indexes = self.diagram_indexes[next_layer].tolist()
            diagram_source.extend([item for item in current_indexes for _ in range(next_number_of_nodes)])
            diagram_target.extend(next_indexes * current_number_of_nodes)
            diagram_values.extend(weights.flatten().tolist())

        last_layer = layer_keys[-1]
        diagram_labels.append("outcome")
        weights = self.weights[last_layer].copy()
        weights /= weights.sum()
        outcome_index = total_number_of_nodes
        diagram_source.extend(self.diagram_indexes[last_layer].tolist())
        diagram_target.extend([outcome_index] * len(self.diagram_indexes[last_layer]))
        diagram_values.extend(weights.flatten().tolist())

        x_positions, y_positions = self._compute_node_positions(
            diagram_source, diagram_target, diagram_values, layer_keys
        )

        node_colours, edge_colours = self._get_colours(
            layer_keys, diagram_labels, diagram_source, diagram_values
        )

        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=10,
                line=dict(color="black", width=0.5),
                label=diagram_labels,
                color=node_colours,
                x=x_positions,
                y=y_positions,
            ),
            link=dict(
                source=diagram_source,
                target=diagram_target,
                value=diagram_values,
                color=edge_colours,
            )
        )])

        fig.update_layout(
            font_size=14,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            margin={"b": 20, "l": 20, "r": 20, "t": 40}
        )

        fig.write_image(f"{save_path}/sankey.jpg")