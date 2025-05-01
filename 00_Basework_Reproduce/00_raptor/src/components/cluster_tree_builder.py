import logging
import pickle
import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Set

from .cluster_utils import ClusteringAlgorithm, RAPTOR_Clustering
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_structures import Node, Tree
from .utils import (distances_from_embeddings, get_children, get_embeddings,
                    get_node_list, get_text,
                    indices_of_nearest_neighbors_from_distances, split_text)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        clustering_algorithm=RAPTOR_Clustering,  # Default to RAPTOR clustering
        clustering_params={},  # Pass additional params as a dict
        report_directory="./reports",  # Directory for generated reports
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension
        self.clustering_algorithm = clustering_algorithm
        self.clustering_params = clustering_params
        self.report_directory = report_directory

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = f"""
        Reduction Dimension: {self.reduction_dimension}
        Clustering Algorithm: {self.clustering_algorithm.__name__}
        Clustering Parameters: {self.clustering_params}
        Report Directory: {self.report_directory}
        """
        return base_summary + cluster_tree_summary


class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension
        self.clustering_algorithm = config.clustering_algorithm
        self.clustering_params = config.clustering_params
        self.report_directory = config.report_directory
        self.node_hierarchy = {}  # Store hierarchy information for report

        logging.info(
            f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}"
        )

    def construct_tree(
        self,
        current_level_nodes: Dict[int, Node],
        all_tree_nodes: Dict[int, Node],
        layer_to_nodes: Dict[int, List[Node]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)
        self.node_hierarchy = {}  # Reset hierarchy information

        def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )
            
            # Store the hierarchy information for reporting
            with lock:
                new_level_nodes[next_node_index] = new_parent_node
                self.node_hierarchy[next_node_index] = {
                    "level": layer + 1,
                    "summary": summarized_text[:100] + "..." if len(summarized_text) > 100 else summarized_text,
                    "children": [node.index for node in cluster],
                }

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.clustering_algorithm.perform_clustering(
                node_list_current_layer,
                self.cluster_embedding_model,
                reduction_dimension=self.reduction_dimension,
                **self.clustering_params,
            )

            lock = Lock()

            summarization_length = self.summarization_length
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(
                            process_cluster,
                            cluster,
                            new_level_nodes,
                            next_node_index,
                            summarization_length,
                            lock,
                        )
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(
                        cluster,
                        new_level_nodes,
                        next_node_index,
                        summarization_length,
                        lock,
                    )
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(
                all_nodes=all_tree_nodes,
                root_nodes=layer_to_nodes[layer + 1],
                leaf_nodes=layer_to_nodes[0],
                num_layers=layer + 1,
                layer_to_nodes=layer_to_nodes,
            )
        
        # Generate the hierarchy report at the end of tree construction
        self.generate_report(tree, all_tree_nodes, layer_to_nodes)
        
        return current_level_nodes
        
    def generate_report(self, tree: Tree, all_nodes: Dict[int, Node], layer_to_nodes: Dict[int, List[Node]]):
        """
        Generate a detailed report of the cluster hierarchy showing which nodes were used for each summary.
        
        Args:
            tree: The constructed tree
            all_nodes: Dictionary of all nodes in the tree
            layer_to_nodes: Mapping from layer to list of nodes
        """
        logging.info("Generating cluster hierarchy report")
        
        # Create report directory if it doesn't exist
        os.makedirs(self.report_directory, exist_ok=True)
        
        # Create a report structure
        report = {
            "total_layers": tree.num_layers,
            "total_nodes": len(all_nodes),
            "leaf_nodes_count": len(tree.leaf_nodes),
            "layers": {}
        }
        
        # Populate report with layer information
        for layer_num, nodes in layer_to_nodes.items():
            report["layers"][f"layer_{layer_num}"] = {
                "node_count": len(nodes),
                "nodes": {}
            }
            
            # For each node in this layer
            for node in nodes:
                # Skip leaf nodes (layer 0) as they don't have children from clustering
                if layer_num == 0:
                    node_info = {
                        "index": node.index,
                        "text_snippet": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                    }
                else:
                    # Get the children of this node and their information
                    children_indices = list(node.children)
                    children_nodes = [all_nodes.get(idx) for idx in children_indices if idx in all_nodes]
                    children_info = []
                    
                    for child in children_nodes:
                        if child:
                            children_info.append({
                                "index": child.index,
                                "text_snippet": child.text[:100] + "..." if len(child.text) > 100 else child.text,
                            })
                    
                    node_info = {
                        "index": node.index,
                        "text_snippet": node.text[:100] + "..." if len(node.text) > 100 else node.text,
                        "children_count": len(children_indices),
                        "children": children_info
                    }
                
                report["layers"][f"layer_{layer_num}"]["nodes"][f"node_{node.index}"] = node_info
        
        # Save the report to a JSON file
        report_file_path = os.path.join(self.report_directory, "cluster_hierarchy_report.json")
        with open(report_file_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Cluster hierarchy report saved to {report_file_path}")
        
        # Generate a more readable text report
        text_report_path = os.path.join(self.report_directory, "cluster_hierarchy_report.txt")
        with open(text_report_path, 'w') as f:
            f.write("RAPTOR CLUSTER HIERARCHY REPORT\n")
            f.write("=============================\n\n")
            f.write(f"Total Layers: {tree.num_layers}\n")
            f.write(f"Total Nodes: {len(all_nodes)}\n")
            f.write(f"Leaf Nodes: {len(tree.leaf_nodes)}\n\n")
            
            # For each layer (from top to bottom)
            for layer_num in range(tree.num_layers, -1, -1):
                f.write(f"LAYER {layer_num}\n")
                f.write(f"--------\n")
                
                if layer_num in layer_to_nodes:
                    nodes = layer_to_nodes[layer_num]
                    f.write(f"Number of nodes: {len(nodes)}\n\n")
                    
                    for node in nodes:
                        f.write(f"Node {node.index}: {node.text[:100]}...\n")
                        
                        if layer_num > 0:  # Skip for leaf nodes
                            children = list(node.children)
                            f.write(f"  Children ({len(children)}): {', '.join(map(str, children))}\n\n")
                
                f.write("\n")
        
        logging.info(f"Text report saved to {text_report_path}")
