from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Load raw PDF(s)
DATA_PATH="data/"
def load_pdf_files(data):
    # with the help of loader we can extract all the .pdf files inside /data and with PyPDFLoader, we can load PDF files
    loader = DirectoryLoader(data,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print("Length of PDF pages: ", len(documents))

#Step2: Create Chunks using CharacterSplitter
def createChunks(extractedData):
    # gave best results in chunk size 500 and overlap 50, to keep the context of the text we use chunk overlap
    # we are using RecursiveCharacterTextSplitter to split the text into chunks
    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    text_chunks = textSplitter.split_documents(extractedData)
    return text_chunks

text_chunks = createChunks(extractedData=documents)
print("Length of Text Chunks: ", len(text_chunks))
# print("Sample Text Chunk: ", text_chunks[445])

# Step 3: Create Vector Embeddings
def get_embedding_model():
    # HuggingFaceEmbeddings to create vector embeddings
    # we are using 'sentence-transformers/paraphrase-MiniLM-L6-v2' model this model maps sentneces and paras to a 
    # 384 dense vector space so that it can be used for semantic serch, similarity and clustering
    # text_chunks --> vector embeddings to numerical format 
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')
    return embedding_model
embedding_model = get_embedding_model()

# Step 4: Store embeddings in FAISS
# FAISS (Facebook AI Similarity Search) is a VectorDB which stores vector embddings locally on sytem which makes it
# easier and faster to search in VectorSpace. If we not use FAISS, we have to calculate embeddings everytime we search
# which is computationally expensive. FAISS stores embeddings in a file which can be used later for searching.

DB_FAISS_PATH = "vectorStore/db_faiss"
# make chunks from these text_chunks, create vector embeddings and store them in FAISS
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)