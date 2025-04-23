from dotenv import load_dotenv
import os
import yaml
import pickle
from src.components.utils import process_pdfs_in_directory
from langchain_core.documents import Document
import nest_asyncio 
nest_asyncio.apply()
from tqdm import tqdm
import time

## RAPTOR PROCESS STARTS FROM HERE

import numpy as np
import umap
import pandas as pd
from sklearn.mixture import GaussianMixture
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from src.components.document_processor import DocumentPreprocessor
from dotenv import find_dotenv, load_dotenv
import tiktoken
import os

load_dotenv(find_dotenv())

load_dotenv()

# --------------------------
# Node class for graph tree
# --------------------------
class Node:
    def __init__(self, id, text, type, iteration, metadata=None):
        self.id = id
        self.text = text
        self.type = type  # 'chunk' or 'summary'
        self.iteration = iteration
        self.metadata = metadata or {}
        self.parent_id = None
        self.children_ids = []
        self.adjacent_ids = []

# --------------------------
# Main summarizer class
# --------------------------
class TextClusterSummarizer:
    def __init__(
        self,
        token_limit,
        model_name,
        embedding_model_name,
        chunks,
    ):
        print("Initializing TextClusterSummarizer...")
        self.token_limit = token_limit
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name, api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_model = ChatOpenAI(temperature=0, model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
        self.iteration_summaries = []

        # Store all chunk and summary nodes
        self.nodes = {}  # id → Node
        self.chunk_counter = 0
        self.summary_counter = 0
        self.chunks = chunks

    def embed_texts(self, texts):
        print("Embedding texts...")
        return [self.embedding_model.embed_query(txt) for txt in tqdm(texts, desc="Embedding texts")]

    def reduce_dimensions(self, embeddings, dim, n_neighbors=None, metric="cosine"):
        print(f"Reducing dimensions to {dim}...")
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))

    def cluster_embeddings(self, embeddings, threshold, max_allowed=None, random_state=0):
        print("Clustering embeddings...")
        optimal_clusters = self.get_optimal_clusters(embeddings)

        if max_allowed is not None:
            n_clusters = min(optimal_clusters, max_allowed)
            print(f"Optimal clusters: {optimal_clusters}, capped to: {n_clusters}")
        else:
            n_clusters = optimal_clusters
            print(f"Optimal clusters: {n_clusters} (no cap applied)")

        gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(embeddings)
        probs = gm.predict_proba(embeddings)
        return [np.where(prob > threshold)[0] for prob in probs], n_clusters

    def get_optimal_clusters(self, embeddings, max_clusters=50, random_state=1234):
        print("Calculating optimal number of clusters...")
        max_clusters = min(max_clusters, len(embeddings))
        bics = [
            GaussianMixture(n_components=n, random_state=random_state)
            .fit(embeddings)
            .bic(embeddings)
            for n in range(1, max_clusters)
        ]
        print(f"Optimal number of clusters: {np.argmin(bics) + 1}")
        return np.argmin(bics) + 1

    def format_cluster_texts(self, df):
        print("Formatting cluster texts...")
        clustered_texts = {}
        for cluster in df["Cluster"].unique():
            cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
            clustered_texts[cluster] = " --- ".join(cluster_texts)
        return clustered_texts

    def split_and_summarize_cluster(self, cluster_id, cluster_texts, iteration, chain):
        """
        Splits a cluster into parts if it exceeds max token limit,
        summarizes each part, then merges into one parent summary node.
        """
        text_to_node_id = {
            node.text: node_id for node_id, node in self.nodes.items()
            if node.text in cluster_texts and node.iteration == iteration - 1
        }

        effective_limit = self.token_limit - 2000

        # Calculate token count of whole cluster
        combined_text = " --- ".join(cluster_texts)
        total_tokens = self.num_tokens_from_string(combined_text)

        if total_tokens <= effective_limit:
            # Normal single summary
            summary_text = chain.invoke({"text": combined_text})
            summary_id = f"summary_{self.summary_counter:04}"
            summary_node = Node(
                id=summary_id,
                text=summary_text,
                type="summary",
                iteration=iteration,
            )
            child_ids = [text_to_node_id[t] for t in cluster_texts]
            summary_node.children_ids = child_ids
            for cid in child_ids:
                self.nodes[cid].parent_id = summary_id
            self.nodes[summary_id] = summary_node
            self.summary_counter += 1
            return [summary_node]

        # Split and summarize in parts
        print(f"Cluster {cluster_id} is too large ({total_tokens} tokens). Splitting...")

        summaries = []
        current_chunk = []
        current_tokens = 0

        def flush():
            nonlocal current_chunk, current_tokens, summaries
            if current_chunk:
                sub_text = " --- ".join(current_chunk)
                sub_summary = chain.invoke({"text": sub_text})
                sub_id = f"summary_{self.summary_counter:04}"
                sub_node = Node(
                    id=sub_id,
                    text=sub_summary,
                    type="summary",
                    iteration=iteration,
                )
                child_ids = [text_to_node_id[t] for t in current_chunk]
                sub_node.children_ids = child_ids
                for cid in child_ids:
                    self.nodes[cid].parent_id = sub_id
                self.nodes[sub_id] = sub_node
                self.summary_counter += 1
                summaries.append(sub_node)
                current_chunk = []
                current_tokens = 0

        for t in cluster_texts:
            t_tokens = self.num_tokens_from_string(t)
            if current_tokens + t_tokens > effective_limit // 2:
                flush()
            current_chunk.append(t)
            current_tokens += t_tokens

        flush()

        # Merge summaries into one final summary node
        merged_text = " --- ".join([n.text for n in summaries])
        final_summary = chain.invoke({"text": merged_text})
        final_id = f"summary_{self.summary_counter:04}"
        final_node = Node(
            id=final_id,
            text=final_summary,
            type="summary",
            iteration=iteration,
        )
        for n in summaries:
            final_node.children_ids.append(n.id)
            self.nodes[n.id].parent_id = final_id
        self.nodes[final_id] = final_node
        self.summary_counter += 1

        return [final_node]


    def run(self):
        print("Running TextClusterSummarizer...")
        start_time = time.time()

        docs = self.chunks
        if not docs:
            raise ValueError("No documents provided for summarization.")
        texts = [doc.page_content for doc in docs]
        all_summaries = texts

        print("Registering chunk nodes...")
        for i, doc in enumerate(tqdm(docs, desc="Registering chunks")):
            node_id = f"chunk_{self.chunk_counter:04}"
            node = Node(
                id=node_id,
                text=doc.page_content,
                type="chunk",
                iteration=0,
                metadata=doc.metadata,
            )
            self.nodes[node_id] = node
            self.chunk_counter += 1

        chunk_ids = list(self.nodes.keys())
        for i in range(1, len(chunk_ids)):
            self.nodes[chunk_ids[i - 1]].adjacent_ids.append(chunk_ids[i])

        iteration = 1
        self.iteration_summaries.append({"iteration": 0, "texts": texts, "summaries": []})
        max_allowed_clusters = None

        while True:
            print(f"\nIteration {iteration}")
            print("Embedding texts...")
            embeddings = [self.embedding_model.embed_query(txt) for txt in tqdm(all_summaries, desc="Embedding texts")]

            n_neighbors = min(int((len(embeddings) - 1) ** 0.5), len(embeddings) - 1)

            if n_neighbors < 2:
                print("Not enough data points for UMAP reduction.")

                if len(all_summaries) > 1:
                    print("Creating a final summary from remaining points...")

                    chain = ChatPromptTemplate.from_template(
                        "You are an assistant to create a detailed summary of the text input provided.\nText:\n{text}"
                    ) | self.chat_model | StrOutputParser()

                    final_nodes = self.split_and_summarize_cluster(
                        cluster_id="final",
                        cluster_texts=all_summaries,
                        iteration=iteration,
                        chain=chain
                    )
                    all_summaries = [final_nodes[-1].text]

                    self.iteration_summaries.append({
                        "iteration": iteration,
                        "texts": all_summaries,
                        "summaries": all_summaries,
                    })

                break

            embeddings_reduced = self.reduce_dimensions(embeddings, dim=2, n_neighbors=n_neighbors)
            labels, num_clusters = self.cluster_embeddings(embeddings_reduced, threshold=0.5, max_allowed=max_allowed_clusters)
            max_allowed_clusters = max(1, num_clusters // 2)

            if num_clusters == 1:
                print("Reduced to a single cluster. Stopping iterations.")
                break

            simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]
            df = pd.DataFrame({
                "Text": all_summaries,
                "Embedding": list(embeddings_reduced),
                "Cluster": simple_labels,
            })

            clustered_texts = self.format_cluster_texts(df)

            summaries = {}
            print("Summarizing clusters...")
            chain = ChatPromptTemplate.from_template(
                "You are an assistant to create a detailed summary of the text input provided.\nText:\n{text}"
            ) | self.chat_model | StrOutputParser()

            for cluster, text in tqdm(clustered_texts.items(), desc=f"Summarizing iteration {iteration}"):
                cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
                summary_nodes = self.split_and_summarize_cluster(
                    cluster_id=cluster,
                    cluster_texts=cluster_texts,
                    iteration=iteration,
                    chain=chain
                )
                for node in summary_nodes:
                    summaries[cluster] = node.text

            all_summaries = list(summaries.values())
            self.iteration_summaries.append({
                "iteration": iteration,
                "texts": all_summaries,
                "summaries": list(summaries.values()),
            })
            iteration += 1

        final_summary = all_summaries[0] if all_summaries else ""

        print("Packaging LangChain Documents...")
        documents = []
        for node in self.nodes.values():
            documents.append(Document(
                page_content=node.text,
                metadata={
                    "id": node.id,
                    "type": node.type,
                    "iteration": node.iteration,
                    "parent_id": node.parent_id,
                    "children_ids": node.children_ids,
                    "adjacent_ids": node.adjacent_ids if node.type == "chunk" else [],
                    "source": node.metadata.get("source", None)
                }
            ))

        total_time = time.time() - start_time
        print(f"\n✅ TextClusterSummarizer complete in {total_time:.2f} seconds.")

        return {
            "initial_texts": texts,
            "iteration_summaries": self.iteration_summaries,
            "final_summary": final_summary,
            "nodes": self.nodes,
            "documents": documents,
        }