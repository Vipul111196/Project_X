# from rdflib import Graph
# from rdflib.namespace import RDF, OWL, RDFS
# from neo4j_graphrag.experimental.components.schema import (
#     SchemaBuilder,
#     SchemaEntity,
#     SchemaProperty,
#     SchemaRelation,
#     SchemaConfig
# )
# from typing import Dict, List

# def getLocalPart(uri):
#   pos = -1
#   pos = uri.rfind('#') 
#   if pos < 0 :
#     pos = uri.rfind('/')  
#   if pos < 0 :
#     pos = uri.rindex(':')
#   return uri[pos+1:]



# def getNLOntology(g):
#   result = ''
#   definedcats = []

#   result += '\nNode Labels:\n'
#   for cat in g.subjects(RDF.type, OWL.Class):  
#     result += getLocalPart(cat)
#     definedcats.append(cat)
#     for desc in g.objects(cat,RDFS.comment):
#         result += ': ' + desc + '\n'
#   extracats = {}
#   for cat in g.objects(None,RDFS.domain):
#      if not cat in definedcats:
#         extracats[cat] = None
#   for cat in g.objects(None,RDFS.range):
#      if not (cat.startswith("http://www.w3.org/2001/XMLSchema#") or cat in definedcats):
#         extracats[cat] = None   
  
#   for xtracat in extracats.keys():
#      result += getLocalPart(cat) + ":\n"

#   result += '\nNode Properties:\n'
#   for att in g.subjects(RDF.type, OWL.DatatypeProperty):  
#     result += getLocalPart(att)
#     for dom in g.objects(att,RDFS.domain):
#         result += ': Attribute that applies to entities of type ' + getLocalPart(dom)  
#     for desc in g.objects(att,RDFS.comment):
#         result += '. It represents ' + desc + '\n'

#   result += '\nRelationships:\n'
#   for att in g.subjects(RDF.type, OWL.ObjectProperty):  
#     result += getLocalPart(att)
#     for dom in g.objects(att,RDFS.domain):
#         result += ': Relationship that connects entities of type ' + getLocalPart(dom)
#     for ran in g.objects(att,RDFS.range):
#         result += ' to entities of type ' + getLocalPart(ran)
#     for desc in g.objects(att,RDFS.comment):
#         result += '. It represents ' + desc + '\n'
#   return result



# def getPropertiesForClass(g, cat):
#   props = []
#   for dtp in g.subjects(RDFS.domain,cat):
#     if (dtp, RDF.type, OWL.DatatypeProperty) in g:
#       propName = getLocalPart(dtp)
#       propDesc = next(g.objects(dtp,RDFS.comment),"") 
#       props.append(SchemaProperty(name=propName, type="STRING", description=propDesc))
#   return props

# def getSchemaFromOnto(g) -> SchemaConfig:
#   schema_builder = SchemaBuilder()
#   classes = {}
#   entities =[]
#   rels =[]
#   triples = []
  
#   for cat in g.subjects(RDF.type, OWL.Class):  
#     classes[cat] = None
#     label = getLocalPart(cat)
#     props = getPropertiesForClass(g, cat)
#     entities.append(SchemaEntity(label=label, 
#                  description=next(g.objects(cat,RDFS.comment),""),
#                  properties=props))
#   for cat in g.objects(None,RDFS.domain):
#      if not cat in classes.keys():
#         classes[cat] = None
#         label = getLocalPart(cat)
#         props = getPropertiesForClass(g, cat)
#         entities.append(SchemaEntity(label=label, 
#                     description=next(g.objects(cat,RDFS.comment),""),
#                     properties=props))
#   for cat in g.objects(None,RDFS.range):
#      if not (cat.startswith("http://www.w3.org/2001/XMLSchema#") or cat in classes.keys()):
#         classes[cat] = None
#         label = getLocalPart(cat)
#         props = getPropertiesForClass(g, cat)
#         entities.append(SchemaEntity(label=label, 
#                     description=next(g.objects(cat,RDFS.comment),""),
#                     properties=props))   
  
#   for op in g.subjects(RDF.type, OWL.ObjectProperty):  
#     relname = getLocalPart(op)
#     rels.append(SchemaRelation(label=relname, 
#                                properties = [],
#                                description=next(g.objects(op,RDFS.comment), "")))
    
#   for op in g.subjects(RDF.type, OWL.ObjectProperty):  
#     relname = getLocalPart(op)
#     doms = []
#     rans = []
#     for dom in g.objects(op,RDFS.domain):
#         if dom in classes.keys():
#           doms.append(getLocalPart(dom))
#     for ran in g.objects(op,RDFS.range):
#         if ran in classes.keys():
#           rans.append(getLocalPart(ran))
#     for d in doms:
#        for r in rans:
#           triples.append((d,relname,r))
    
#   return schema_builder.create_schema_model(entities=entities, 
#                      relations=rels,
#                      potential_schema=triples)


# def getPKs(g):
#   keys = []
#   for k in g.subjects(RDF.type, OWL.InverseFunctionalProperty):  
#     keys.append(getLocalPart(k))
#   return keys

# def extract_schema_elements(rdf_graph: Graph) -> Dict[str, List[str]]:
#     """Return a dict with keys 'classes', 'object_properties', 'data_properties'."""
#     classes = []
#     obj_props = []
#     data_props = []

#     for s, p, o in rdf_graph:
#         # If this triple says s is an OWL Class
#         if p == RDF.type and o == OWL.Class:
#             # maybe also get a label
#             label = rdf_graph.value(s, RDFS.label)
#             name = label if label else s.split("#")[-1]  # fallback
#             classes.append(str(name))

#         # If this triple says s is an OWL ObjectProperty
#         if p == RDF.type and o == OWL.ObjectProperty:
#             label = rdf_graph.value(s, RDFS.label)
#             name = label if label else s.split("#")[-1]
#             obj_props.append(str(name))

#         # If this triple says s is an OWL DatatypeProperty
#         if p == RDF.type and o == OWL.DatatypeProperty:
#             label = rdf_graph.value(s, RDFS.label)
#             name = label if label else s.split("#")[-1]
#             data_props.append(str(name))

#     return {
#         "classes": list(set(classes)), 
#         "object_properties": list(set(obj_props)),
#         "data_properties": list(set(data_props))
#     }


from rdflib import Graph, RDF, OWL, RDFS
from neo4j_graphrag.experimental.components.schema import (
    SchemaBuilder,
    SchemaEntity,
    SchemaProperty,
    SchemaRelation,
    SchemaConfig
)
from typing import Dict, List
from langchain_community.document_loaders import PyMuPDFLoader 
import os


def getLocalPart(uri):
    uri = str(uri)
    for sep in ['#', '/', ':']:
        if sep in uri:
            return uri.split(sep)[-1]
    return uri


def getPropertiesForClass(g, cat):
    props = []
    for dtp in g.subjects(RDFS.domain, cat):
        if (dtp, RDF.type, OWL.DatatypeProperty) in g:
            propName = getLocalPart(dtp)
            propDesc = next(g.objects(dtp, RDFS.comment), "")
            props.append(SchemaProperty(name=propName, type="STRING", description=propDesc))
    return props


def getSchemaFromOnto(g: Graph) -> SchemaConfig:
    schema_builder = SchemaBuilder()
    classes = {}
    entities = []
    rels = []
    triples = []

    # Extract Classes
    for cat in g.subjects(RDF.type, OWL.Class):
        classes[cat] = None
        label = getLocalPart(cat)
        props = getPropertiesForClass(g, cat)
        entities.append(SchemaEntity(label=label,
                                     description=next(g.objects(cat, RDFS.comment), ""),
                                     properties=props))

    # Add entities from domain/range that may not be OWL.Classes
    for cat in list(g.objects(None, RDFS.domain)) + list(g.objects(None, RDFS.range)):
        if isinstance(cat, str) and cat.startswith("http://www.w3.org/2001/XMLSchema#"):
            continue
        if cat not in classes:
            classes[cat] = None
            label = getLocalPart(cat)
            props = getPropertiesForClass(g, cat)
            entities.append(SchemaEntity(label=label,
                                         description=next(g.objects(cat, RDFS.comment), ""),
                                         properties=props))

    # Extract ObjectProperties (Relationships)
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        relname = getLocalPart(op)
        description = next(g.objects(op, RDFS.comment), "")
        rels.append(SchemaRelation(label=relname, properties=[], description=description))

    # Extract relation triples
    for op in g.subjects(RDF.type, OWL.ObjectProperty):
        relname = getLocalPart(op)
        doms = [getLocalPart(dom) for dom in g.objects(op, RDFS.domain) if dom in classes]
        rans = [getLocalPart(ran) for ran in g.objects(op, RDFS.range) if ran in classes]
        for d in doms:
            for r in rans:
                triples.append((d, relname, r))

    return schema_builder.create_schema_model(
        entities=entities,
        relations=rels,
        potential_schema=triples
    )

def process_pdfs_in_directory(directory_path):  

    documents = []

    for filename in os.listdir(directory_path):  
        if filename.endswith(".pdf"):  
            file_path = os.path.join(directory_path, filename) 
            pdf_loader = PyMuPDFLoader(file_path=file_path)
            document = pdf_loader.load()
            print(f"File loading done for: {filename}")
            documents.append(document)

    return documents