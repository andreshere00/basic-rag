# Basic RAG (Retrieval Augmented Generation)
A repository that contains the code to execute a basic RAG system.

## Features

- **Document Handling**:
  - Supports reading and processing PDF, Excel, and TXT files.
  - Extracts text from PDF files using direct text extraction or OCR for scanned or non-text accessible PDFs.
  - Handles multi-sheet Excel files, converting each sheet into a Markdown table.

- **Text Splitting**:
  - Implements multiple text splitting strategies: fixed-length, paragraph-based, sentence-based, and word-based.
  - Configurable chunk sizes to optimize text for embedding and input to language models.

- **Model Integration**:
  - Supports embedding generation using OpenAI's `text-embedding-ada-002` and models from Hugging Face's `SentenceTransformer`.
  - Dynamic model loading and caching for efficient repeated use.

- **Embedding and Vector Storage**:
  - Generates high-dimensional embeddings and stores them in a Qdrant vector database.
  - Supports various distance metrics (cosine, euclidean, dot product) for effective vector similarity searches.

- **Retrieval-Augmented Generation (RAG)**:
  - Complete RAG pipeline including retrieval of relevant document chunks, context augmentation, and response generation.
  - Flexible configuration of top-k document retrieval and distance metrics for custom RAG implementations.

- **Debugging and Performance Monitoring**:
  - Includes decorators for logging execution times of key processes, aiding in performance optimization.

## Class Structure

The system is organized into the following classes:

1. **`DocumentReader`**: Handles the reading and processing of PDF, Excel, and TXT files.
2. **`Splitter`**: Splits text into chunks based on different methods (fixed, paragraph, sentence, word).
3. **`Embedder`**: Manages model loading and embedding generation using OpenAI or Hugging Face models.
4. **`RAG`**: Implements the retrieval, augmentation, and generation phases of the RAG process, along with Qdrant integration.
5. **`Main`**: Orchestrates the entire process, from document reading to query answering.

## Usage

### Example Execution

To execute the RAG system with a specific query and document, use the following code:

```
if __name__ == "__main__":
    query = "What is about the text?"
    file_path = "./examples/example.pdf"
    model_name = "text-embedding-ada-002"
    length = 8192
    method = "fixed"

    main_process = Main()
    main_process.execute(query, file_path, model_name, length, method)
```

### Running the System

Clone the repository:
```
git clone https://github.com/yourusername/your-repo-name.git
```
Install the required dependencies (unavailable at the moment):
```
pip install -r requirements.txt
```

Place your document (e.g., `example.pdf`) in the `./examples/ directory`.

Update the `query`, `file_path`, `model_name`, `length`, and `method` parameters in the example execution script according to your needs.

Run the script:
```
python rag_v2.py
```
