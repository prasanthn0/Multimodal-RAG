import streamlit as st
from rag.rag_image import RAGPipelineUsingImage
from rag.rag_text import RAGPipelineUsingText
from rag.ingest import ingest_file
from src.vectordbs.chromadb import ChromaDB
from src.config import settings
import time
import os
from concurrent.futures import ThreadPoolExecutor

st.title("RAG Approach Comparison")
st.subheader("Upload a PDF Document")

# State variables to track if ingestion has been done and the currently ingested file
if 'ingested' not in st.session_state:
    st.session_state.ingested = False
    st.session_state.current_file_name = None

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a new file is uploaded or a different file than the previously ingested one
if uploaded_file:
    file_path = os.path.join(settings.DATA_STORAGE, uploaded_file.name)

    # Reset the ingestion state if a new file is uploaded
    if uploaded_file.name != st.session_state.current_file_name:
        st.session_state.ingested = False
        st.session_state.current_file_name = uploaded_file.name

    # Ingest the file only once
    if not st.session_state.ingested:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner('Reading the file...'):
            message = ingest_file(file_path)

        st.success(message)
        st.session_state.ingested = True

# Input box for the query
query = st.text_input("Enter your query")

if query and st.session_state.ingested:
    try:
        def run_text_mode():
            collection_name = settings.TEXT_COLLECTION_NAME
            vector_db_client = ChromaDB(collection_name=collection_name)
            start_time = time.time()
            result, total_tokens, input_tokens = RAGPipelineUsingText(vector_db_client).get_query(query)
            elapsed_time = time.time() - start_time
            return result, total_tokens, input_tokens, elapsed_time

        def run_image_mode():
            collection_name = settings.IMAGE_COLLECTION_NAME
            vector_db_client = ChromaDB(collection_name=collection_name)
            start_time = time.time()
            result, total_tokens, input_tokens = RAGPipelineUsingImage(vector_db_client).get_query(query)
            elapsed_time = time.time() - start_time
            return result, total_tokens, input_tokens, elapsed_time

        with ThreadPoolExecutor() as executor:
            with st.spinner('Processing...'):
                text_future = executor.submit(run_text_mode)
                image_future = executor.submit(run_image_mode)

                # Retrieve results
                text_mode_answer, text_mode_total_tokens, text_mode_input_tokens, text_mode_time = text_future.result()
                image_mode_answer, image_mode_total_tokens, image_mode_input_tokens, image_mode_time = image_future.result()

        st.write(f"LLM used: {settings.CHAT_MODEL}")
        # Display answers side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.header("Text Mode")
            st.markdown(f"**Response:** {text_mode_answer}")
            st.write(f"Response Time: {text_mode_time:.2f} seconds")
            st.write(f"Total tokens used: {text_mode_total_tokens}")
            st.write(f"Input tokens used: {text_mode_input_tokens}")

        with col2:
            st.header("Image Mode")
            st.markdown(f"**Response:** {image_mode_answer}")
            st.write(f"Response Time: {image_mode_time:.2f} seconds")
            st.write(f"Total tokens used: {image_mode_total_tokens}")
            st.write(f"Input tokens used: {image_mode_input_tokens}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
