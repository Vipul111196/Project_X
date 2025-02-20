# Description: This file contains the function to build a graph document from the fused triples data.
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document  # So you can set a minimal source doc

def build_graph_document_from_triples(fused_triples):
    graph_doc = []
    node_map = {}
    relationships = []

    # for (subject, predicate, obj) in fused_triples:
    #     if subject not in node_map:
    #         node_map[subject] = Node(id=subject, type="Concept", properties={})
    #     if obj not in node_map:
    #         node_map[obj] = Node(id=obj, type="Concept", properties={})
    #     relationships.append( Relationship(
    #         source=node_map[subject],
    #         target=node_map[obj],
    #         type=predicate,
    #         properties={}
    #     ) )

    for triple in fused_triples:
        # Extract subject, predicate, and object from the dictionary
        subject = triple["s"]
        predicate = triple["p"]
        obj = triple["o"]

        if subject not in node_map:
            node_map[subject] = Node(id=subject, type="Concept", properties={})
        if obj not in node_map:
            node_map[obj] = Node(id=obj, type="Concept", properties={})
        relationships.append(
            Relationship(
                source=node_map[subject],
                target=node_map[obj],
                type=predicate,
                properties={}
            )
        )

    # Convert dict to list
    all_nodes = list(node_map.values())

    # minimal Document for source
    minimal_src = Document(page_content="Fused triplets data")
    # Build a single GraphDocument
    doc = GraphDocument(
        nodes=all_nodes,
        relationships=relationships,
        source=minimal_src
    )

    graph_doc.append(doc)

    return graph_doc


## Another Method

# from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
# from langchain_core.documents import Document

# def build_graph_document_from_triples(fused_triples):
#     # We’ll track Nodes by their unique IDs so we don’t create duplicates
#     nodes_dict = {}
#     relationships = []

#     for (subject, relation, obj) in fused_triples:
#         # If the subject or object Node does not exist yet, create it
#         if subject not in nodes_dict:
#             # You can pick a type, e.g., "Concept", or something domain-specific
#             nodes_dict[subject] = Node(id=subject, type="Concept")

#         if obj not in nodes_dict:
#             nodes_dict[obj] = Node(id=obj, type="Concept")

#         # Create a Relationship from subject -> object
#         rel = Relationship(
#             source=nodes_dict[subject],
#             target=nodes_dict[obj],
#             type=relation
#         )
#         relationships.append(rel)


    # for triple in fused_triples:
    #     # Extract subject, predicate, and object from the dictionary
    #     subject = triple["s"]
    #     predicate = triple["p"]
    #     obj = triple["o"]

    #     if subject not in node_map:
    #         node_map[subject] = Node(id=subject, type="Concept", properties={})
    #     if obj not in node_map:
    #         node_map[obj] = Node(id=obj, type="Concept", properties={})
    #     relationships.append(
    #         Relationship(
    #             source=node_map[subject],
    #             target=node_map[obj],
    #             type=predicate,
    #             properties={}
    #         )
    #     )

#     # Convert our dictionary into a list of Nodes
#     nodes_list = list(nodes_dict.values())

#     # Instead of source=None, create a minimal Document
#     minimal_doc = Document(page_content="Placeholder for fused triples")

#     # Create the GraphDocument
#     graph_doc = GraphDocument(
#         nodes=nodes_list,
#         relationships=relationships,
#         source=minimal_doc  # Optional: You can attach a Document as the source
#     )
#     return graph_doc
