from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go
import kaleido
import pandas as pd

class SankeyDiagram:

    def __init__(self, labels, weights):
        self.labels = labels
        self.weights = weights
        self.number_of_layers = len(self.labels)
        self.residual_labels_indexes = {}
        self.important_labels_indexes = {}

    def _scale_network(self, number_of_important_points):
        self.labels_to_plot = {}
        self.weights_to_plot = {}
        sorted_weights = []
        for i in tqdm(range(self.number_of_layers, 0, -1), total=self.number_of_layers, colour='red'):
            layer = f"layer_{i}"
            labels = self.labels[layer]
            weights = self.weights[layer]
            important_points = number_of_important_points[i-1]
            total_importances = np.abs(weights).sum(axis=1)
            sorted_label_indexes = np.argsort(total_importances)[::-1]
            sorted_weights.append(pd.DataFrame({"Label" : labels[sorted_label_indexes], 
                                                "Weight" : weights.sum(axis=1)[sorted_label_indexes],
                                                "Layer" : [layer] * len(labels)}))
            selected_labels = []
            line_length = 33
            for label in labels[sorted_label_indexes[:important_points]]:
                lines = []
                for j in range(0, len(label), line_length):
                    if j + line_length > len(label):
                        lines.append(label[j:])
                    else:
                        lines.append(label[j:j+line_length])
                selected_labels.append("<br />".join(lines))
            selected_weights = weights[sorted_label_indexes[:important_points], :]
            residual_labels = labels[sorted_label_indexes[important_points:]]
            residual_weights = weights[sorted_label_indexes[important_points:], :]
            total_residual_weights = residual_weights.sum(axis=0)
            self.labels[layer] = np.concatenate([selected_labels, np.array([f"Residual_{i}"])])
            self.weights[layer] = np.vstack([selected_weights, total_residual_weights])
            self.residual_labels_indexes[layer] = sorted_label_indexes[important_points:]
            self.important_labels_indexes[layer] = sorted_label_indexes[:important_points]
            # adjust the weights to account for the fact that the next layer has
            # a residual component as well
            if i < self.number_of_layers:
                next_layer = f"layer_{i+1}"
                next_layer_important_labels_indexes = self.important_labels_indexes[next_layer]
                next_layer_residual_labels_indexes = self.residual_labels_indexes[next_layer]
                scaled_weights = self.weights[layer][:, next_layer_important_labels_indexes]
                residual_links = self.weights[layer][:, next_layer_residual_labels_indexes].sum(axis=1)
                residual_links = residual_links.reshape(-1, 1)
                self.weights[layer] = np.hstack([scaled_weights, residual_links])
            
        return pd.concat(sorted_weights)

    def plot(self, number_of_important_points, save_path):
        sorted_weights = self._scale_network(number_of_important_points)
        sorted_weights.to_csv(f"{save_path}/deeplift_weights.csv")
        self.diagram_indexes = {}
        diagram_labels = []
        diagram_source = []
        diagram_target = []
        diagram_values = []

        total_number_of_nodes = 0

        for i in range(1, self.number_of_layers+1):
            layer = f"layer_{i}"
            input_labels = self.labels[layer].tolist()
            current_number_of_nodes = len(input_labels)
            diagram_labels.extend(input_labels)
            self.diagram_indexes[layer] = total_number_of_nodes + np.array(range(current_number_of_nodes))
            total_number_of_nodes += current_number_of_nodes
        
        for i in range(1, self.number_of_layers):
            layer = f"layer_{i}"
            next_layer = f"layer_{i+1}"
            weights = self.weights[layer]
            weights /= weights.sum()
            current_number_of_nodes = weights.shape[0]
            next_number_of_nodes = weights.shape[1]
            current_indexes = self.diagram_indexes[layer].tolist()
            next_indexes = self.diagram_indexes[next_layer].tolist()
            diagram_source.extend([item for item in current_indexes for _ in range(next_number_of_nodes)])
            diagram_target.extend(next_indexes*current_number_of_nodes)
            diagram_values.extend(weights.flatten().tolist())

        last_layer = f"layer_{self.number_of_layers}"
        diagram_labels.append("outcome")
        weights = self.weights[last_layer]
        weights /= weights.sum()
        outcome_index = total_number_of_nodes
        diagram_source.extend(self.diagram_indexes[last_layer].tolist())
        diagram_target.extend([outcome_index] * len(self.diagram_indexes[last_layer]))
        diagram_values.extend(weights.flatten().tolist())

        fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 20,
        line = dict(color = "black", width = 0.5),
        label = diagram_labels,
        color = "blue",
        ),
        link = dict(
        source = diagram_source,
        target = diagram_target,
        value = diagram_values
    ))])
        fig.update_layout(
            title_text="P-NET Prostate Cancer Primary vs Metastatic Gene and Pathway Weights",
            font_size=9,
            width=1200,
            height=600,
            margin={"b" : 20, "r" : 20, "l" : 20, "r" : 20}
        )
        print(diagram_source)
        print(diagram_target)
        fig.write_image(f"{save_path}/sankey.jpg")