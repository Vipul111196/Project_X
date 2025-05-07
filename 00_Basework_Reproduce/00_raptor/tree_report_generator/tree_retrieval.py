#!/usr/bin/env python3
"""
RAPTOR Tree Retrieval Tool

This script allows retrieving subtrees from a RAPTOR tree structure.
It enables selecting a node at a specific level and retrieving all nodes below it,
tracing the path from the selected node back to the leaf nodes.
"""

import os
import sys
import json
import logging
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

# Import tree structures
from src.components.tree_structures import Node, Tree

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def load_tree(tree_path: str) -> Tree:
    """
    Load a previously serialized Tree object.
    
    Args:
        tree_path (str): Path to the pickled Tree object
        
    Returns:
        Tree: The loaded Tree object
    """
    try:
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
            logging.info(f"Successfully loaded tree from {tree_path}")
            return tree
    except Exception as e:
        logging.error(f"Failed to load tree from {tree_path}: {e}")
        sys.exit(1)


def get_node_info(node: Node) -> Dict[str, Any]:
    """
    Get basic information about a node.
    
    Args:
        node (Node): The node to get information for
        
    Returns:
        Dict: Information about the node
    """
    return {
        "index": node.index,
        "text": node.text[:200] + "..." if len(node.text) > 200 else node.text,
        "children_count": len(node.children) if hasattr(node, "children") else 0
    }


def find_node_by_index(tree: Tree, node_index: int) -> Optional[Node]:
    """
    Find a node by its index in the tree.
    
    Args:
        tree (Tree): The tree to search in
        node_index (int): The index of the node to find
        
    Returns:
        Optional[Node]: The node if found, None otherwise
    """
    if node_index in tree.all_nodes:
        return tree.all_nodes[node_index]
    return None


def find_node_layer(tree: Tree, node_index: int) -> Optional[int]:
    """
    Find the layer number for a given node index.
    
    Args:
        tree (Tree): The tree to search in
        node_index (int): The index of the node to find
        
    Returns:
        Optional[int]: The layer number if found, None otherwise
    """
    for layer_num, nodes in tree.layer_to_nodes.items():
        for node in nodes:
            if node.index == node_index:
                return layer_num
    return None


def get_subtree(tree: Tree, node_index: int) -> Dict[str, Any]:
    """
    Get a subtree rooted at the specified node.
    
    Args:
        tree (Tree): The complete tree
        node_index (int): The index of the root node for the subtree
        
    Returns:
        Dict: A dictionary representing the subtree
    """
    node = find_node_by_index(tree, node_index)
    if not node:
        logging.error(f"Node with index {node_index} not found in tree")
        return {}

    # Find the layer this node belongs to
    layer = find_node_layer(tree, node_index)
    if layer is None:
        logging.error(f"Couldn't determine layer for node {node_index}")
        return {}

    # Create a subtree data structure
    subtree = {
        "root": get_node_info(node),
        "layer": layer,
        "children": []
    }

    # If this is not a leaf node, recursively build the subtree
    if layer > 0:
        # Get direct children
        direct_children = [tree.all_nodes[child_idx] for child_idx in node.children 
                           if child_idx in tree.all_nodes]
        
        # For each direct child
        for child in direct_children:
            child_layer = find_node_layer(tree, child.index)
            
            # If this is a leaf node (layer 0), just add it to our results
            if child_layer == 0:
                subtree["children"].append({
                    "node": get_node_info(child),
                    "layer": 0,
                    "children": []
                })
            else:
                # Otherwise, get its subtree recursively
                child_subtree = get_subtree(tree, child.index)
                if child_subtree:
                    subtree["children"].append(child_subtree)

    return subtree


def generate_printable_subtree(subtree: Dict[str, Any], indent_level: int = 0) -> str:
    """
    Generate a printable representation of the subtree.
    
    Args:
        subtree (Dict): The subtree to print
        indent_level (int): Current indentation level
        
    Returns:
        str: A string representation of the subtree
    """
    indent = "  " * indent_level
    result = ""
    
    # Print the root information
    if "root" in subtree:
        node = subtree["root"]
        result += f"{indent}[Node {node['index']}] (Layer {subtree['layer']}): {node['text']}\n"
    elif "node" in subtree:
        node = subtree["node"]
        result += f"{indent}[Node {node['index']}] (Layer {subtree['layer']}): {node['text']}\n"
    else:
        return result
    
    # Print all children recursively
    for child in subtree["children"]:
        result += generate_printable_subtree(child, indent_level + 1)
    
    return result


def save_subtree_report(subtree: Dict[str, Any], output_file: str) -> None:
    """
    Save subtree information to an output file.
    
    Args:
        subtree (Dict): The subtree data
        output_file (str): Path to the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write JSON version
    json_path = output_file
    with open(json_path, 'w') as f:
        json.dump(subtree, f, indent=2)
    logging.info(f"Saved subtree report in JSON format to {json_path}")
    
    # Write text version
    text_path = output_file.replace('.json', '.txt')
    with open(text_path, 'w') as f:
        f.write("RAPTOR SUBTREE REPORT\n")
        f.write("====================\n\n")
        
        if "root" in subtree:
            node = subtree["root"]
            f.write(f"Root Node: {node['index']} (Layer {subtree['layer']})\n")
            f.write(f"Summary: {node['text']}\n\n")
            f.write("Subtree Structure:\n")
            f.write("----------------\n")
            f.write(generate_printable_subtree(subtree))
    
    logging.info(f"Saved subtree report in text format to {text_path}")


def list_available_nodes(tree: Tree, layer: int = None) -> None:
    """
    List all available nodes, optionally filtered by layer.
    
    Args:
        tree (Tree): The tree to get nodes from
        layer (int, optional): If provided, only nodes from this layer are shown
    """
    if layer is not None:
        if layer not in tree.layer_to_nodes:
            logging.error(f"Layer {layer} does not exist in the tree")
            return
        
        nodes = tree.layer_to_nodes[layer]
        print(f"\nAvailable nodes at Layer {layer}:")
        for i, node in enumerate(nodes):
            print(f"  Node {node.index}: {node.text[:100]}...")
    else:
        print("\nAvailable layers and node counts:")
        for layer_num, nodes in tree.layer_to_nodes.items():
            print(f"  Layer {layer_num}: {len(nodes)} nodes")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Retrieve subtrees from a RAPTOR tree")
    parser.add_argument("tree_path", type=str, help="Path to the pickled Tree object")
    parser.add_argument("--node", type=int, help="Index of the node to retrieve subtree for")
    parser.add_argument("--layer", type=int, help="Filter available nodes by layer")
    parser.add_argument("--list", action="store_true", help="List available nodes")
    parser.add_argument("--output", type=str, default="./reports/subtree_report.json", 
                       help="Path for the output report file (default: ./reports/subtree_report.json)")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Load the tree
    tree = load_tree(args.tree_path)
    
    # If listing mode is enabled, just list available nodes and exit
    if args.list:
        list_available_nodes(tree, args.layer)
        return
    
    # If no node index is provided, print usage and exit
    if args.node is None:
        logging.error("No node index provided. Use --node to specify a node or --list to see available nodes")
        return
    
    # Get the subtree for the specified node
    subtree = get_subtree(tree, args.node)
    if not subtree:
        return
    
    # Create reports directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save the subtree report
    save_subtree_report(subtree, args.output)


if __name__ == "__main__":
    main()
