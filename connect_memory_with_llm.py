import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Setup LLM (Mistral with HuggingFace)
HFTOKEN = os.environ.get('HFTOKEN')

HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def loadLLM(huggingFace_repo_id):
    # Load LLM from HuggingFace
    # repoID = huggingFace_repo_id temp= the degree with which model can generate diverse outputs
    # model_kwargs = {} are the parameters of the model (keyword arguments)
    llm = HuggingFaceEndpoint(
                                repo_id = HUGGING_FACE_REPO_ID, 
                                 temperature = 0.6,
                                 model_kwargs = {"token": HFTOKEN,
                                                 "max_length": "1000"}
                            )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

DB_FAISS_PATH="vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont try to provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly.
"""

def set_custom_prompt(CUSTOM_PROMPT_TEMPLATE):
    # Create a custom prompt template. i/p from user will be given.
    # PromptTemplate() is a part of langchain_core.prompts used to create/bind custom prompt templates
    prompt = PromptTemplate(template = CUSTOM_PROMPT_TEMPLATE, input_variables = ['context', 'question'])
    return prompt

# Loading of Database. But first we need to load LLM model with which the embeddings got created and stored in FAISS
# to create embedding for the user input question.
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-MiniLM-L6-v2')

db  = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
# allow_dangerous_deserialization=True this is newly launched by langchain for safety and security purpose when we trust the data

# Create QA Chain. Here we r using RetrievalQA chain.

# QAChain (Question Answering Chain) refers to a structured workflow or sequence of steps specifically designed to handle 
# question-answering tasks. The main purpose of it is to retrieve, process, and generate accurate answers based on a 
# given question, often leveraging external knowledge sources or large language models (LLMs)
# It involves several steps, such as: 
# Input: A user submits a question, like "What is the capital of France?"
# Step 1 (Retrieval): The system might search a knowledge base or external source for relevant documents or data related to countries and capitals.
# Step 2 (Preprocessing): The retrieved documents might be filtered or summarized to extract the key detail (in this case, the capital of France).
# Step 3 (Answer Generation): A generative model processes the retrieved information and forms an answer, such as "The capital of France is Paris."
# Step 4 (Output): The answer is delivered to the user.

qa_chain = RetrievalQA.from_chain_type(
    llm=loadLLM(HUGGING_FACE_REPO_ID), 
    chain_type = "stuff", 
    retriever = db.as_retriever(search_kwargs = {'k':3}),
    return_source_documents = True,
    chain_type_kwargs = {'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

# chain_type = 'stuff' is basically simplest form of chain type which is used directly combine and pass all retrieved 
# documents or pieces of information to the LLM for answering the question. No Preprocessing of Retrieved Data is done or
# just passed as it is (stuffed) to LLM. Generally used when retrieved information is already well-structured, small and relevant.

#db.as_retriever(search_kwargs = {}) dictates how many similar documents accoding to question similarity it will retrieve
#how many topmost similar of semantic closest (matching) docs it will bring back in response to the question.

# return_source_documents = from where it will return. The source documents of it which page or page no., embedding metadata etc.
   # basically whenever the embeddings are stored in FAISS, it also stores the metadata of the embeddings which can be used later for searching

#Now invoke the QA Chain with single question
user_input = input("Write yout Query Here: ")
response = qa_chain.invoke({'query': user_input})
print("Result:", response["result"])
print("Source Documents:", response["source_documents"])
# query -> custom prompt template -> embeddings of query wil be created -> search in FAISS -> retrieve top 3 similar docs 
# -> pass to LLM 

# What are the possible causes of Seborrheic Dermatitis?