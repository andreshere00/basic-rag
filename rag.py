## LIBRARIES
# Basics & Debugging
import time
import uuid
import re
import json
import sys
import math
# NLP & Modeeling
import openai
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
# Data ETL
import io
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_text_to_fp
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LTTextBoxHorizontal, LTPage, LAParams
from pdfminer.converter import TextConverter
from PyPDF4 import PdfFileReader, PdfFileWriter
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import fitz
# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

## COMPUTING TIME

def compute_time(func):
    """
    Collocator for computing time between processes.
    
    Args:
    - func: a function
    
    Returns:
    - wrapper: the methods that the collocator adds.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"DEBUG: {func.__name__} execution time: {execution_time:.4f} seconds")
        return result
    return wrapper

## READING DOCUMENT

def read_pdf(file_path):
    """
    Function to read text from a PDF file using pdfminer, and OCR if necessary.
    
    Args:
    - file_path: directory where the file is located, including the file name.
    
    Returns:
    - text (str): the text extracted from the pdf.
    """
    # Initialize pdfminer resources
    resMgr = PDFResourceManager()
    retData = io.StringIO()
    laparams = LAParams()
    TxtConverter = TextConverter(resMgr, retData, laparams=laparams)
    interpreter = PDFPageInterpreter(resMgr, TxtConverter)

    text = ""
    with open(file_path, 'rb') as inFile:
        for page in PDFPage.get_pages(inFile):
            interpreter.process_page(page)
            text += retData.getvalue()
            retData.truncate(0)
            retData.seek(0)
    
    # Check if text is empty and use OCR if necessary
    if not text.strip():
        print(f"DEBUG: Text could not be extracted from PDF, using OCR.")
        text = perform_ocr(file_path)
    
    return text

def perform_ocr(file_path):
    """
    Perform OCR on a PDF file and rotate to the best angle.
    
    Args:
    - file_path: directory where the file is located, including the file name.
    
    Returns:
    - text (str): the text extracted from the pdf using OCR.
    """
    doc = fitz.open(file_path)
    text = ""
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        
        # Convert to PIL image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Perform OCR on the original and rotated versions
        degrees = [0, 90, 180, 270]
        ocr_texts = []
        for d in degrees:
            ocr_texts.append(pytesseract.image_to_string(img.rotate(d, expand=True)))
        
        # Choose the rotation with the most text
        best_ocr_text = max(ocr_texts, key=len)
        
        text += best_ocr_text
    
    return text

def read_txt(file_path):
    """
    Function to read text from a txt file.
    
    Args:
    - file_path (str): directory where is located the file among the file name.
    
    Returns:
    - text (str): the text extracted from the txt file.
    """
    with open("examples/long_text.txt", "r") as file:
        text = file.read()
        
    #print(f"DEBUG: Text extracted: {text}")
    
    return text

def read_excel(file_path):
    """
    Reads an Excel file and stores each sheet in a list of lists with strings, separated by a comma.
    Adds the sheet name as a title for each sheet.
    
    Args:
    - file_path: directory where the file is located along with the file name.
    
    Returns:
    - excel (list[str]): A list of lists where each inner list contains the sheet name and the markdown string representing the table.
    """
    try:
        excel_sheets = pd.read_excel(file_path, sheet_name=None)
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return []

    excel = []
    max_tokens_per_subpage = 8192 // 8

    for sheet_name, sheet in excel_sheets.items():
        try:
            # Explicitly cast to string dtype before replacing NaN values
            sheet = sheet.astype(str)
            sheet.fillna('EMPTY', inplace=True)
            sheet_list = sheet.values.tolist()
            
            # Flatten the list of lists to count the tokens
            flat_list = [item for sublist in sheet_list for item in sublist]
            total_tokens = len(flat_list)
            print(f"DEBUG: total_tokens for sheet '{sheet_name}': {total_tokens}")

            # Split the sheet into subpages if necessary
            if total_tokens > max_tokens_per_subpage:
                num_subpages = math.ceil(total_tokens / max_tokens_per_subpage)
                for i in range(num_subpages):
                    start = i * max_tokens_per_subpage
                    end = min((i + 1) * max_tokens_per_subpage, total_tokens)
                    subpage_list = flat_list[start:end]
                    print(f"DEBUG: Processing subpage {i+1} of {num_subpages} for sheet '{sheet_name}' with token range ({start}:{end})")
                    
                    # Convert the flat subpage_list back into a list of lists
                    subpage_list_2d = [subpage_list[j:j + len(sheet.columns)] for j in range(0, len(subpage_list), len(sheet.columns))]
                    
                    subpage_name = f"{sheet_name}_part{str(i+1)}"
                    if subpage_list_2d:
                        markdown_table = create_markdown_table(subpage_name, subpage_list_2d)
                        excel.append(markdown_table)
            else:
                if sheet_list:
                    markdown_table = create_markdown_table(sheet_name, sheet_list)
                    excel.append(markdown_table)
        except Exception as e:
            print(f"Error processing sheet '{sheet_name}': {e}")
            
    print(f"DEBUG: text extracted from excel: {excel}")

    return excel

def create_markdown_table(sheet_name, sheet_list):
    """
    Converts a list of lists into a markdown table string.
    
    Args:
    - sheet_name: The name of the sheet.
    - sheet_list: List of lists representing the sheet data.
    
    Returns:
    - markdown (str): A string formatted as a markdown table.
    """
    try:
        markdown = f"### Sheet: {sheet_name}\n\n"
        
        # Check if sheet_list is empty
        if not sheet_list:
            markdown += "No data available.\n"
            return markdown
        
        # Create the header
        headers = sheet_list[0]
        markdown += "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        
        # Add the rows
        for row in sheet_list[1:]:
            markdown += "| " + " | ".join(row) + " |\n"
        
        return markdown
    except Exception as e:
        print(f"Error creating markdown table for sheet '{sheet_name}': {e}")
        return f"Error creating markdown table for sheet '{sheet_name}'."

@compute_time
def read_manager(file_path):
    """
    Manager which assigns to each document a particular reader.
    
    Args:
    - file_path: directory where the file is located along with the file name.
    
    Returns: 
    - file (list['str']): the text extracted from the file.
    - file_type (str): the extension of the file. 
    """
    file_type = file_path.rpartition('.')[-1]

    print(f"INFO: Loading document...")
    if file_type == "pdf":
        file = read_pdf(file_path)
    elif file_type == "xlsx":
        file = read_excel(file_path)
    elif file_type == "txt":
        file = read_txt(file_path)
    else:
        raise Exception(f"ERROR: Unsupported file extension: {file_type}")
    
    print(f"INFO: Document loaded.")
    return file, file_type

## SPLITTING

@compute_time
def splitter_manager(text, length, method='fixed'):
    """
    Function to chunk text into pieces of mxaximum length.

    Args:
    - text: text obtained from the documents.
    - method: method to split the text ('fixed', 'paragraph', 'sentences', 'words').
    
    Returns:
    - chunks (list[list['str']]: list of lists of strings.)
    """
    chunks = []
    
    if method == 'fixed': 
        length = length / 8

    if method == 'fixed' or method == 'max_length':
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # Add 1 for the space
            if current_length + word_length <= length:
                current_chunk.append(word)
                current_length += word_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
    elif method == 'paragraph':
        paragraphs = text.split('\n')
        for paragraph in paragraphs:
            if paragraph:
                chunks.append(paragraph.strip())
                
    elif method == 'sentences':
        sentences = sent_tokenize(text)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence) + 1  # Add 1 for the space
            if current_length + sentence_length <= length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))

    elif method == 'words':
        words = word_tokenize(text)
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # Add 1 for the space
            if current_length + word_length <= length:
                current_chunk.append(word)
                current_length += word_length
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length

        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    print(f"DEBUG: number of chunks: {len(chunks)}")
    # print(f"DEBUG: chunks obtained: {chunks}")
    
    return chunks

## MODEL LOADING

# Diccionario para almacenar los modelos cargados
loaded_models = {}

def model_loader(model_name):
    """
    Function to load a LLM.
    
    Args:
    - model_name (str): could be either `text-embedding-ada-002` or `dunzhang/stella_en_1.5B_v5`. 
    Other models could not be supported.
    
    Returns:
    - model: the model itself.
    """
    if model_name in loaded_models:
        print(f"INFO: Model '{model_name}' is already loaded.")
        return loaded_models[model_name]
    
    init_time = time.time()
    #print(f"DEBUG: model name: {model_name}")
    print(f"INFO: loading model...")
    
    if model_name != "text-embedding-ada-002":
        model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
    else:
        model = openai.OpenAI()
    
    print(f"INFO: model loaded.")
    print(f"DEBUG: model_loader execution time: {time.time() - init_time}")
    
    loaded_models[model_name] = model
    return model

## EMBEDDING GENERATION

def get_embedding_openai(text, model, model_name):
    """
    Function to get embeddings using OpenAI API.
    
    Args:
    - text (str): text to embed.
    - model: model which will perform the embedding process.
    - model_name: could be either `text-embedding-ada-002` or `dunzhang/stella_en_1.5B_v5`. 
    Other models could not be supported.
    
    Returns:
    - embedding_vector (list[float]): a vector which saves the word representations in a high-dimensional space.
    """
    result = model.embeddings.create(input=text, model=model_name)
    embedding_vector = []
    for i in range(len(text)):
        embedding_vector.append(result.data[i].embedding)
        #print(f"DEBUG: chunk {i+1}/{len(text)} embedded.")
    return embedding_vector

def get_embedding_stella(text, model):
    """
    Function to get embeddings using SentenceTransformer model.
    
    Args:
    - text (list['str']): the text to encode.
    - model: the model itself.
    
    Returns:
    - embedding_vector (list[float]): a vector which saves the word representations in a high-dimensional space.
    """
    embedding_vector = []
    for t in text:
        the_vec = model.encode(t)
        embedding_vector.append(the_vec)
    return embedding_vector

# Additional methods for generating embeddings with Stella:

# @compute_time
# def get_embedding_stella_1(text, model):
#     print(f'DEBUG: Embedding with the first method...')
#     init_time = time.time()
#     passages = [
#        text[i] for i in range(len(text))
#     ]
#     embedding_vector = model.encode(passages) 
#     print(f'DEBUG: Embedded with the first method. Embedding generation time: {time.time() - init_time}')
#     return embedding_vector

# @compute_time
# def get_embedding_stella_2(text, model):
#     print(f'DEBUG: Embedding with the second method...')
#     init_time = time.time()
#     embedding_vector = model.encode(text)
#     print(f'DEBUG: Embedded with the second method. Embedding generation time: {time.time() - init_time}')
#     return embedding_vector

# @compute_time
# def get_embedding_stella_3(text, model):
#     print(f'DEBUG: Embedding with the third method...')
#     init_time = time.time()
#     embedding_vector = []
#     for t in text:
#       the_vec = model.encode(t)
#       embedding_vector.append(the_vec)
#     print(f'DEBUG: Embedded with the third method. Embedding generation time: {time.time() - init_time}')
#     return embedding_vector

@compute_time
def embedder_manager(text, model, model_name):
    """
    Function to encode chunks which checks for which model to use.
    
    Args:
    - text: content.
    - model: model which will perform the embedding process.
    - model_name: could be either `text-embedding-ada-002` or `dunzhang/stella_en_1.5B_v5`. 
    Other models could not be supported.
    
    Returns:
    - embedding_vector (list[float]): a vector which saves the word representations in a high-dimensional space.
    """
    if model_name == "text-embedding-ada-002":
        embedding_vector = get_embedding_openai(text, model, model_name)
    else:
        embedding_vector = get_embedding_stella(text, model)
    
    return embedding_vector

## RAG

### Qdrant
@compute_time
def upload_to_qdrant(embeddings, text, file_type, workspace_id, document_id, file_path, model_name, distance_metric="cosine", collection_name="your_collection_name"):
    """
    Uploads embeddings to Qdrant with specified metadata.

    Args:
    - embeddings (list): List of embeddings to be uploaded.
    - chunks (list): List of text chunks to be uploaded.
    - file_type (str): Type of the file (pdf, xlsx, txt).
    - workspace_id (str): Workspace ID.
    - document_id (str): Document ID.
    - file_path (str): File path to extract the document name.
    - model_name (str): Name of the model used to generate the embeddings.
    - collection_name (str): Name of the Qdrant collection where the embeddings will be uploaded. Default is 'your_collection_name'.
    
    Returns:
    - collection_name (str): name of the collection where points are located.
    - points: the points themselves.
    """

    if distance_metric == "euclidean":
        distance_metric = Distance.EUCLID
    elif distance_metric == "cosine":
        distance_metric = Distance.COSINE
    elif distance_metric == "dot_product":
        distance_metric = Distance.DOT
    else:
        raise Exception(f"ERROR: unsupported distance provided.")
    
    client = QdrantClient(host='localhost', port=6333)
    # Ensure the collection exists
    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        )

    document_name = file_path.replace('examples/', '')

    # Define the points with metadata
    points = [
        PointStruct(
            id=index,  # Assigns the current index to the id of the PointStruct
            vector=embedding,
            payload={
                "file_type": file_type,
                "workspace_id": workspace_id,
                "document_id": document_id,
                "document_name": document_name,
                "model_name": model_name,
                "text": text[index] # if file_type != 'xlsx' else text.get(index, "")
            }
        )
        for index, embedding in enumerate(embeddings)  # enumerate provides the index
    ]

    # Upload the points to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    return collection_name, points
 
### RAG
@compute_time   
def rag_system(query, model, model_name, length, method, top_k=3, distance_metric="cosine", collection_name="your_collection_name"):
    """
    Function which accomplish all the RAG system. This function has three methods:
    - retrieval: for returning the context-aware documents related to the query.
    - augmentation: returns the enriched query with the context information extracted from the retrieval phase.
    - generation: returns the final output which will answer to the query.
    
    Args:
    - query (str): query that is asked to the generative model which will match with the RAG's documents. Default = ""
    - top_k (int): max documents to retrieve. Default = 3
    - distance_metric (str): distance metric that documents will be compared to match with the query. Options are:
        - "cosine" [Default]
        - "dot_product" 
        - "euclidean"
    - model: embedding model.
    - model_name (str): model which embed the documents and query. Should be the same model that embedded the original documents.
    
    Returns: 
    - output (str): the final answer.
    """
    
    @compute_time 
    def retrieval(query, model, model_name, length, method, top_k=3, collection_name="your_collection_name"):
        """
        Function which retrieves all the query-related documents based on the top-k algorithm.

        Args:
        - query (str): The query which will match with the documents.
        - model: embedding model.
        - model_name (str): model which embed the documents and query. Should be the same model that embedded the original documents.
        - top_k (int): Number of top results to retrieve. Default is 10.
        - distance_metric (Distance): The distance metric for matching. Default is COSINE.
        - collection_name (str): The name of the Qdrant collection to search within. Default is 'your_collection_name'.

        Returns:
        - documents (list['str']): A list of matching documents.
        """

        client = QdrantClient(host='localhost', port=6333)
        splitted_query = splitter_manager(query, length, method)
        query_embedding = embedder_manager(splitted_query, model, model_name)
        query_vector = query_embedding[0]
        
        #print(f"DEBUG: Query vector: {query_vector}")

        # Search in the collection
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        
        print(f"DEBUG: search result obtained: {search_result}")

        # Extract and return the documents from the search result
        documents = [(result.payload['text'], result.score) for result in search_result]
        # Sort the documents based on relevance scores (assuming higher score is more relevant)
        documents_sorted = sorted(documents, key=lambda x: x[1], reverse=True)
        documents = [result.payload['text'] for result in search_result]
        #print(f"DEBUG: Documents obtained: {documents}")

        # Calculate the total number of tokens in all documents
        total_tokens = sum(len(word.split()) for doc in documents for word in doc)
        #print(f"DEBUG: number of tokens before: {total_tokens}")
        
        while total_tokens > (128000*3)/4:
            documents.pop()
            total_tokens = sum(len(word.split()) for doc in documents for word in doc)
            
        #print(f"DEBUG: number of tokens after: {total_tokens}")
        # print(f"DEBUG: documents retrieved: {documents}")
        
        return documents

    @compute_time 
    def augmentation(query, documents):
        """
        Function which enrich the query's contextual information based on the documents content.
        
        Args:
        - query (str): The query which will match with the documents.
        - documents (str): Context-aware extracted information from the documents on the retrieval phase.
        
        Returns:
        - enriched_prompt (str): Query with contextual-enriched information.
        """
        
        enriched_prompt = f"""
        ### INSTRUCTIONS
        Answer to the user's query based on the contextual information provided. 
        If no information is provided, answer with I don't know.
        
        ### Contextual information:
        {documents}
        
        ### Query:
        {query}
        
        ### Answer:
        """
        
        print(f'DEBUG: enriched prompt obtained: {enriched_prompt}')
        
        return enriched_prompt
    
    @compute_time 
    def generation(enriched_prompt):
        """
        Generates the output based on the query and the contextual information provided by the documents.
        
        Args:
        - enriched_prompt (str): Query with the contextual-relevant retrieved information.
        - model: embedding model.
        - model_name (str): model which embed the documents and query. Should be the same model that embedded the original documents.
        
        Returns:
        - output (str): response obtained by the generative model through the RAG process.
        """
        
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"{enriched_prompt}"}
        ]
        )
        
        output = response.choices[0].message.content
        
        return output
    
    documents = retrieval(query = query, 
                          model = model, 
                          model_name = model_name, 
                          method=method,
                          length = length,
                          top_k=1, 
                          collection_name=collection_name)
    
    enriched_prompt = augmentation(query=query, 
                                   documents=documents)
    
    output = generation(enriched_prompt = enriched_prompt)
    
    return output

## MAIN

@compute_time
def main(query, file_path, model_name, length, method = 'fixed'):
    """
    Main function to execute the whole splitting, embedding and RAG process.
    
    Args:
    - query (str): the question that you want to ask to the generative LM.
    - file_path (str): document to embed.
    - model_name (str): model to be used. It could be ' text-embedding-ada-002' or any of the Huggingface available models.
    - length (int): maximum context window for the model, or length that you want to use to split text.
    - method (str): it could be either fixed (given by the length value) of max_length (given by the specific maximum context window size of the model).
    
    Returns:
    - output (str): response obtained by the generative model through the RAG process.
    """
    
    print(f"INFO: {"="*80} \nINFO: Starting Splitting, Embedding and RAG process. \nINFO: {"="*80}")
    print(f"DEBUG: Executed with the following parameters: \n - query = '{query}', \n - file_path = {file_path}, \n - model = {model_name}, \n - splitting_method = {method}.")
    
    print("INFO: Step 1 - Read and split documents into chunks.")
    text, file_type = read_manager(file_path)
    if file_type != "xlsx":
        text = splitter_manager(text, length, method = 'fixed')

    print("INFO: Step 2 - Load model.")
    model = model_loader(model_name)
    #print(f"DEBUG: {text}")
    
    print("INFO: Step 3 - Generate embeddings.")
    embeddings = embedder_manager(text, model, model_name)
    #print(f"DEBUG: Embeddings - {embeddings[0]} \n Longitud - {len(embeddings[0])}")
    
    print(f"INFO: Splitting and Embedding process for documents has been completed.")
    print("-"*80)
    
    print("INFO: Step 4 - Upload documents.")
    timestamp = time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime(time.time()))
    collection_name = f"{file_type}-{timestamp}"
    
    upload_to_qdrant(embeddings=embeddings, 
                     text = text,
                     file_type=file_type, 
                     workspace_id=uuid.uuid4(), 
                     document_id=uuid.uuid4(), 
                     file_path=file_path, 
                     model_name=model_name,
                     distance_metric="cosine", 
                     collection_name=collection_name)
    
    print("INFO: Step 5 - RAG.")
    output = rag_system(query=query, 
                        model=model, 
                        model_name=model_name, 
                        length=length,
                        top_k=10, 
                        distance_metric="cosine", 
                        collection_name=collection_name,
                        method=method)
    
    print(f'INFO: Generated answer: ```\n {output} \n```')
    
    return output

# TESTS
with open('test_cases.json', 'r') as file:
    test_cases = json.load(file)

filename = f"logs/results-{time.strftime('%d_%m_%Y-%H_%M_%S', time.localtime(time.time()))}.log"
with open(filename, 'w') as file:
    sys.stdout = file
    for test in test_cases:
        # Generate the prompt
        main(test["query"], test["file_path"], test["model"], test["length"], test["method"])

# Restore standard output to console
sys.stdout = sys.__stdout__