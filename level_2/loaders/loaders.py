import os
from io import BytesIO

import fitz
from level_2.chunkers.chunkers import chunk_data
from langchain.document_loaders import PyPDFLoader

import requests
def _document_loader( observation: str, loader_settings: dict):
    # Check the format of the document
    document_format = loader_settings.get("format", "text")

    if document_format == "PDF":
        if loader_settings.get("source") == "url":
            pdf_response = requests.get(loader_settings["path"])
            pdf_stream = BytesIO(pdf_response.content)
            with fitz.open(stream=pdf_stream, filetype='pdf') as doc:
                file_content = "".join(page.get_text() for page in doc)
            return chunk_data(chunk_strategy= 'VANILLA', source_data=file_content)
        elif loader_settings.get("source") == "file":
            # Process the PDF using PyPDFLoader
            # might need adapting for different loaders + OCR
            # need to test the path
            loader = PyPDFLoader(loader_settings["path"])
            return loader.load_and_split()
    elif document_format == "text":
        # Process the text directly
        return observation

    else:
        raise ValueError(f"Unsupported document format: {document_format}")


