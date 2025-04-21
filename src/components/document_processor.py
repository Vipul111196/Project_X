import os
import pickle
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentPreprocessor:
    def __init__(self, config, pdf_processor):
        """
        config: dictionary of config parameters
        pdf_processor: function like `process_pdfs_in_directory(path)` that returns List[List[Document]]
        """
        self.config = config
        self.process_pdfs_in_directory = pdf_processor

        self.base_docs: List[Document] = []
        self.chunks: List[Document] = []

    def load_or_process_documents(self):
        """Either loads preprocessed PDF documents or processes them from scratch"""
        if self.config['data_preprocessing']['already_preprocessed'] == 0:
            docs_path = self.config['data_preprocessing']['directory_path']
            all_docs = self.process_pdfs_in_directory(docs_path)

            # Flatten all_docs and store in base_docs
            self.base_docs = [doc for doc_list in all_docs for doc in doc_list]
            print(f"Total pages extracted from PDFs: {len(self.base_docs)}")

            file_path = self.config['data_preprocessing']['data_preprocessed_path']
            with open(file_path, 'wb') as file:
                pickle.dump(self.base_docs, file)

            print(f"Base documents saved as pickle to: {file_path}")
        else:
            file_path = self.config['data_preprocessing']['data_preprocessed_path']
            with open(file_path, 'rb') as file:
                self.base_docs = pickle.load(file)

            print(f"Base documents loaded from: {file_path}, total: {len(self.base_docs)}")

    def load_or_chunk_documents(self):
        """Either chunks base docs or loads already chunked docs"""
        if self.config['data_processing']['chunking']['chunking_required'] == 1:
            chunk_size = self.config['data_processing']['chunking']['chunk_size']
            chunk_overlap = self.config['data_processing']['chunking']['chunk_overlap']
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            self.chunks = splitter.split_documents(self.base_docs)

            for doc in self.chunks:
                if 'source' in doc.metadata:
                    del doc.metadata['source']
                file_path = doc.metadata.get('file_path', '')
                doc.metadata['file_name'] = os.path.basename(file_path)
                if 'file_path' in doc.metadata:
                    del doc.metadata['file_path']

            chunk_file_path = self.config['data_processing']['chunking']['chunk_preprocessed_path']
            with open(chunk_file_path, 'wb') as file:
                pickle.dump(self.chunks, file)

            print(f"Chunks saved as pickle to: {chunk_file_path}, total chunks: {len(self.chunks)}")
        else:
            chunk_file_path = self.config['data_processing']['chunking']['chunk_preprocessed_path']
            with open(chunk_file_path, 'rb') as file:
                self.chunks = pickle.load(file)

            print(f"Chunks loaded from: {chunk_file_path}, total: {len(self.chunks)}")

    def run(self):
        self.load_or_process_documents()
        self.load_or_chunk_documents()
        return self.chunks
