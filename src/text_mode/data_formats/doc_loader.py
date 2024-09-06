import glob
import os
import warnings
warnings.filterwarnings("ignore")
from multiprocessing import Pool
from typing import List
from src.text_mode.chunking import ChunkingStrategy
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader, 
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if "text/html content not found in email" in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (
        TextLoader,
        {"encoding": "utf8"},
    ),  # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> Document:
    
    ext = "." + file_path.rsplit(".", 1)[-1]
    
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
    
        loader = loader_class(file_path, **loader_args)   
            
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(
    source_directory: str, ignored_files: List[str] = []
) -> List[Document]:
    if '.' in source_directory.split('\\')[-1]:
        return [load_single_document(source_directory)]
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_directory, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        for i, doc in enumerate(
            pool.imap_unordered(load_single_document, filtered_files)
        ):
            results.append(doc)
    return results


def process_documents(
    source_directory, ignored_files: List[str] = []
) -> List[Document]:
    documents = load_documents(source_directory, ignored_files)
    
    if not documents:
        print("No new documents to load")
        exit(0)
    texts = ChunkingStrategy(documents).split_texts()
    return texts

