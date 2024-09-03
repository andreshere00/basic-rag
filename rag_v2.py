import io
import time
import uuid
import fitz
import pandas as pd
import pytesseract
import openai
from PIL import Image
from pdfminer.high_level import extract_text_to_fp
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams, LTTextBoxHorizontal
from pdfminer.converter import TextConverter
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class DocumentReader:
    def read_pdf(self, file_path):
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

        if not text.strip():
            print(f"DEBUG: Text could not be extracted from PDF, using OCR.")
            text = self.perform_ocr(file_path)

        return text

    def perform_ocr(self, file_path):
        doc = fitz.open(file_path)
        text = ""

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            degrees = [0, 90, 180, 270]
            ocr_texts = []
            for d in degrees:
                ocr_texts.append(pytesseract.image_to_string(img.rotate(d, expand=True)))

            best_ocr_text = max(ocr_texts, key=len)
            text += best_ocr_text

        return text

    def read_txt(self, file_path):
        with open(file_path, "r") as file:
            text = file.read()
        return text

    def read_excel(self, file_path):
        try:
            excel_sheets = pd.read_excel(file_path, sheet_name=None)
        except Exception as e:
            print(f"Error reading the Excel file: {e}")
            return []

        excel = []
        max_tokens_per_subpage = 8192 // 8

        for sheet_name, sheet in excel_sheets.items():
            try:
                sheet = sheet.astype(str)
                sheet.fillna('EMPTY', inplace=True)
                sheet_list = sheet.values.tolist()

                flat_list = [item for sublist in sheet_list for item in sublist]
                total_tokens = len(flat_list)

                if total_tokens > max_tokens_per_subpage:
                    num_subpages = math.ceil(total_tokens / max_tokens_per_subpage)
                    for i in range(num_subpages):
                        start = i * max_tokens_per_subpage
                        end = min((i + 1) * max_tokens_per_subpage, total_tokens)
                        subpage_list = flat_list[start:end]

                        subpage_list_2d = [subpage_list[j:j + len(sheet.columns)] for j in range(0, len(subpage_list), len(sheet.columns))]

                        subpage_name = f"{sheet_name}_part{str(i+1)}"
                        if subpage_list_2d:
                            markdown_table = self.create_markdown_table(subpage_name, subpage_list_2d)
                            excel.append(markdown_table)
                else:
                    if sheet_list:
                        markdown_table = self.create_markdown_table(sheet_name, sheet_list)
                        excel.append(markdown_table)
            except Exception as e:
                print(f"Error processing sheet '{sheet_name}': {e}")

        return excel

    def create_markdown_table(self, sheet_name, sheet_list):
        try:
            markdown = f"### Sheet: {sheet_name}\n\n"

            if not sheet_list:
                markdown += "No data available.\n"
                return markdown

            headers = sheet_list[0]
            markdown += "| " + " | ".join(headers) + " |\n"
            markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

            for row in sheet_list[1:]:
                markdown += "| " + " | ".join(row) + " |\n"

            return markdown
        except Exception as e:
            print(f"Error creating markdown table for sheet '{sheet_name}': {e}")
            return f"Error creating markdown table for sheet '{sheet_name}'."

    def read_manager(self, file_path):
        file_type = file_path.rpartition('.')[-1]

        print(f"INFO: Loading document...")
        if file_type == "pdf":
            file = self.read_pdf(file_path)
        elif file_type == "xlsx":
            file = self.read_excel(file_path)
        elif file_type == "txt":
            file = self.read_txt(file_path)
        else:
            raise Exception(f"ERROR: Unsupported file extension: {file_type}")

        print(f"INFO: Document loaded.")
        return file, file_type
    
class Splitter:
    def splitter_manager(self, text, length, method='fixed'):
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
        return chunks

class Embedder:
    def __init__(self):
        self.loaded_models = {}

    def model_loader(self, model_name):
        if model_name in self.loaded_models:
            print(f"INFO: Model '{model_name}' is already loaded.")
            return self.loaded_models[model_name]

        print(f"INFO: loading model...")

        if model_name != "text-embedding-ada-002":
            model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
        else:
            model = openai.OpenAI()

        print(f"INFO: model loaded.")
        self.loaded_models[model_name] = model
        return model

    def get_embedding_openai(self, text, model, model_name):
        result = model.embeddings.create(input=text, model=model_name)
        embedding_vector = [result.data[i].embedding for i in range(len(text))]
        return embedding_vector

    def get_embedding_stella(self, text, model):
        embedding_vector = [model.encode(t) for t in text]
        return embedding_vector

    def embedder_manager(self, text, model, model_name):
        if model_name == "text-embedding-ada-002":
            embedding_vector = self.get_embedding_openai(text, model, model_name)
        else:
            embedding_vector = self.get_embedding_stella(text, model)
        return embedding_vector

class RAG:
    def __init__(self):
        self.client = QdrantClient(host='localhost', port=6333)

    def upload_to_qdrant(self, embeddings, text, file_type, workspace_id, document_id, file_path, model_name, distance_metric="cosine", collection_name="your_collection_name"):
        if distance_metric == "euclidean":
            distance_metric = Distance.EUCLID
        elif distance_metric == "cosine":
            distance_metric = Distance.COSINE
        elif distance_metric == "dot_product":
            distance_metric = Distance.DOT
        else:
            raise Exception(f"ERROR: unsupported distance provided.")

        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
            )

        document_name = file_path.replace('examples/', '')

        points = [
            PointStruct(
                id=index,
                vector=embedding,
                payload={
                    "file_type": file_type,
                    "workspace_id": workspace_id,
                    "document_id": document_id,
                    "document_name": document_name,
                    "model_name": model_name,
                    "text": text[index]
                }
            )
            for index, embedding in enumerate(embeddings)
        ]

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

        return collection_name, points

    def rag_system(self, query, model, model_name, length, method, top_k=3, collection_name="your_collection_name"):
        documents = self.retrieval(query=query, model=model, model_name=model_name, length=length, method=method, top_k=top_k, collection_name=collection_name)
        enriched_prompt = self.augmentation(query=query, documents=documents)
        output = self.generation(enriched_prompt=enriched_prompt)
        return output

    def retrieval(self, query, model, model_name, length, method, top_k=3, collection_name="your_collection_name"):
        splitter = Splitter()
        query_embedding = Embedder().embedder_manager(splitter.splitter_manager(query, length, method), model, model_name)[0]

        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k,
        )

        documents = [result.payload['text'] for result in search_result]
        return documents

    def augmentation(self, query, documents):
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
        return enriched_prompt

    def generation(self, enriched_prompt):
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"{enriched_prompt}"}
            ]
        )
        output = response.choices[0].message.content
        return output

class Main:
    def __init__(self):
        self.document_reader = DocumentReader()
        self.splitter = Splitter()
        self.embedder = Embedder()
        self.rag = RAG()

    def execute(self, query, file_path, model_name, length, method='fixed'):
        print(f"INFO: Starting Splitting, Embedding and RAG process.")
        text, file_type = self.document_reader.read_manager(file_path)
        if file_type != "xlsx":
            text = self.splitter.splitter_manager(text, length, method=method)

        model = self.embedder.model_loader(model_name)
        embeddings = self.embedder.embedder_manager(text, model, model_name)

        timestamp = time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime(time.time()))
        collection_name = f"{file_type}-{timestamp}"

        self.rag.upload_to_qdrant(embeddings=embeddings, 
                                  text=text, 
                                  file_type=file_type, 
                                  workspace_id=uuid.uuid4(), 
                                  document_id=uuid.uuid4(), 
                                  file_path=file_path, 
                                  model_name=model_name,
                                  distance_metric="cosine", 
                                  collection_name=collection_name)

        output = self.rag.rag_system(query=query, 
                                     model=model, 
                                     model_name=model_name, 
                                     length=length,
                                     top_k=10, 
                                     distance_metric="cosine", 
                                     collection_name=collection_name,
                                     method=method)

        print(f'INFO: Generated answer: ```\n {output} \n```')
        return output
