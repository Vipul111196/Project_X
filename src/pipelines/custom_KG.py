import re
import json
from typing import List
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

def sanitize_key(key: str) -> str:
    """
    Replace invalid Neo4j property characters with underscores.
    - Removes spaces, punctuation, etc.
    - If it starts with a digit, prepend an underscore.
    """
    # Replace all non-(alphanumeric/underscore) with underscore
    safe_key = re.sub(r'[^0-9A-Za-z_]', '_', key)
    # If it starts with digit, prepend underscore
    if safe_key and safe_key[0].isdigit():
        safe_key = f"_{safe_key}"
    return safe_key

class CustomKGPipeline:
    def __init__(
        self,
        driver,
        embedder,
        kg_llm,
        classes,
        object_properties,
        data_properties,
        prompt_template_path,
        neo4j_database=None
    ):
        self.driver = driver
        self.embedder = embedder
        self.kg_llm = kg_llm
        self.classes = classes
        self.object_properties = object_properties
        self.data_properties = data_properties
        self.neo4j_database = neo4j_database
        self.prompt_template_path = prompt_template_path

    def _create_chunk_nodes(self, docs: List[Document]):
        with self.driver.session(database=self.neo4j_database) as session:
            print("📦 Creating Chunk & Summary nodes...")

            # Phase 1: Create all nodes first
            for doc in docs:
                meta = doc.metadata
                node_id = meta.get("id")
                node_type = meta.get("type")
                props = {
                    "id": node_id,
                    "text": doc.page_content,
                    "type": node_type,
                    "iteration": meta.get("iteration"),
                    "source": meta.get("source"),
                    "embedding": self.embedder.embed_query(doc.page_content)
                }

                session.run(
                    f"""
                    MERGE (n:{node_type.capitalize()} {{id: $id}})
                    SET n += $props
                    """,
                    id=node_id,
                    props=props
                )

            # Phase 2: Create relationships (NEXT_CHUNK and PARENT_OF)
            for doc in docs:
                meta = doc.metadata
                node_id = meta.get("id")
                node_type = meta.get("type")
                parent_id = meta.get("parent_id")
                children_ids = meta.get("children_ids", [])
                adjacent_ids = meta.get("adjacent_ids", [])

                # Link to parent
                if parent_id:
                    session.run(
                        """
                        MATCH (parent {id: $parent_id}), (child {id: $child_id})
                        MERGE (parent)-[:PARENT_OF]->(child)
                        """,
                        parent_id=parent_id,
                        child_id=node_id
                    )

                # Link to children
                for child_id in children_ids:
                    session.run(
                        """
                        MATCH (parent {id: $parent_id}), (child {id: $child_id})
                        MERGE (parent)-[:PARENT_OF]->(child)
                        """,
                        parent_id=node_id,
                        child_id=child_id
                    )

                # Link adjacent chunks
                if node_type == "chunk":
                    for adj_id in adjacent_ids:
                        print(f"🔗 NEXT_CHUNK: {node_id} → {adj_id}")
                        session.run(
                            """
                            MATCH (c1:Chunk {id: $c1}), (c2:Chunk {id: $c2})
                            MERGE (c1)-[:NEXT_CHUNK]->(c2)
                            """,
                            c1=node_id,
                            c2=adj_id
                        )

    def _prompt_template(self):
        """
        Loads a textual prompt template from file and
        returns a ChatPromptTemplate object.
        """
        with open(self.prompt_template_path, "r") as f:
            template_for_extracting_triples = f.read()
        return ChatPromptTemplate.from_template(template_for_extracting_triples)

    def _create_kg_from_chunks(self, docs: List[Document]):
        """
        For each chunk, calls the LLM to extract triples,
        """
        for i, doc in enumerate(docs):
            try:
                # Build a chain that:
                # 1) Renders your prompt
                # 2) Passes it to the LLM
                # 3) Parses the LLM response as a raw string (StrOutputParser)
                llm_chain = self._prompt_template() | self.kg_llm | StrOutputParser()
                result = llm_chain.invoke({
                    'cls_str': self.classes,
                    'rel_str': self.object_properties,
                    'attr_str': self.data_properties,
                    'chunk_text': doc.page_content
                })

                result_obj = json.loads(result)
                triples = result_obj["triples"]

                with self.driver.session(database=self.neo4j_database) as session:
                    for t in triples:
                        s_txt = t["subject"]
                        s_type = t["subject_type"]
                        o_txt = t["object"]
                        o_type = t["object_type"]
                        rel = t["relation"]

                        s_attr_list = t.get("subject_attributes", [])
                        s_attr_dict = {pair["key"]: pair["value"] for pair in s_attr_list}
                        o_attr_list = t.get("object_attributes", [])
                        o_attr_dict = {pair["key"]: pair["value"] for pair in o_attr_list}

                        s_attr_sanitized = {}
                        for orig_k, val in s_attr_dict.items():
                            safe_k = sanitize_key(orig_k)
                            s_attr_sanitized[safe_k] = val

                        o_attr_sanitized = {}
                        for orig_k, val in o_attr_dict.items():
                            safe_k = sanitize_key(orig_k)
                            o_attr_sanitized[safe_k] = val

                        cypher = f"""
                        MERGE (subj:{s_type.replace(' ', '_')} {{name: $sName}})
                          ON CREATE SET subj.entity_type = $sType
                        MERGE (obj:{o_type.replace(' ', '_')} {{name: $oName}})
                          ON CREATE SET obj.entity_type = $oType
                        MERGE (subj)-[r:{rel.upper().replace(' ', '_')}]->(obj)
                        WITH subj, obj
                        MATCH (c:Chunk {{id: $chunk_id}})
                        MERGE (subj)-[:MENTIONED_IN]->(c)
                        MERGE (obj)-[:MENTIONED_IN]->(c)
                        """

                        params = {
                            "sName": s_txt,
                            "sType": s_type,
                            "oName": o_txt,
                            "oType": o_type,
                            "chunk_id": doc.metadata["id"]
                        }

                        for idx, (k, v) in enumerate(s_attr_sanitized.items()):
                            cypher += f"\nSET subj.{k} = coalesce(subj.{k}, $sa{idx})"
                            params[f"sa{idx}"] = v

                        for idx, (k, v) in enumerate(o_attr_sanitized.items()):
                            cypher += f"\nSET obj.{k} = coalesce(obj.{k}, $oa{idx})"
                            params[f"oa{idx}"] = v

                        session.run(cypher, params)

            except Exception as e:
                print(f"Error processing chunk {i}: {e}")


    def _deduplicate_entities(self):
        """
        Merges duplicate nodes that share the same (name, entity_type)
        by combining their properties via apoc.refactor.mergeNodes.
        """
        with self.driver.session(database=self.neo4j_database) as session:
            session.run("""
            MATCH (e1), (e2)
            WHERE e1.name = e2.name AND e1.entity_type = e2.entity_type AND id(e1) < id(e2)
            CALL apoc.refactor.mergeNodes([e1, e2], {properties:'combine'}) YIELD node
            RETURN count(node)
            """)

    def run(self, docs: List[Document]):
        """
        High-level orchestration method:
          1) Create chunk nodes
          2) Extract triples from each chunk & store them
          3) Deduplicate entities
        """
        print("📄 Creating chunk nodes...")
        self._create_chunk_nodes(docs)
        print("🔍 Extracting triples...")
        self._create_kg_from_chunks(docs)
        print("🧹 Deduplicating entities...")
        self._deduplicate_entities()
        print("✅ Done.")

