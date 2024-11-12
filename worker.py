import os
import torch
import pandas as pd

from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import DataFrameLoader

import bs4
from langchain_community.document_loaders import WebBaseLoader
#from langchain.chains import LLMChain, SimpleSequentialChain

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversational_rag_chain = None
chat_history = []
llm = None
embeddings = None
temperature = 0.7

# Function to initialize the language model and its embeddings
def init_llm():
    global llm, embeddings
    llm =  ChatOpenAI(model="gpt-4o-mini", temperature = temperature)
    embeddings = OpenAIEmbeddings()


def process_local_documents(country):


    documents = []

    documents = process_heatlh_risks_documents(documents,country)
    documents = process_diet_guidelines_documents(documents,country)
    documents = process_agriculture_documents(documents,country)

    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def process_pdfs_loop(documents,prefixe):
    for file in os.listdir():
        if file.endswith(".pdf") and file.startswith(prefixe):
            loader = PyPDFLoader(os.path.join(os.getcwd(), file))
            documents.extend(loader.load())
    return documents

def process_urls_loop(documents,url_filename):

    url_tuple = ()
    with open(url_filename) as url_file:
        for line in url_file:
            url_tuple.append(line)

    bs4_strainer = bs4.SoupStrainer()
    loader = WebBaseLoader(web_paths=url_tuple,
    bs_kwargs={"parse_only": bs4_strainer},
    )
    documents.extend(loader.load())

    return documents

# Function to process a PDF document
def process_heatlh_risks_documents(documents,country):

    hces_norm_filtered = pd.read_csv('hces_norm_filtered.csv')
    hces_norm_df_filtered_area = hces_norm_filtered[hces_norm_filtered['Area'] == country]

    if not hces_norm_df_filtered_area.empty:
        loader = DataFrameLoader(hces_norm_df_filtered_area, page_content_column="Food Group Indicator")
        documents.extend(loader.load())
    else:
        print("No HCES data found for the specified country.")

    fs_norm_filtered = pd.read_csv('fs_norm_filtered.csv')
    fs_norm_df_filtered_area = fs_norm_filtered[fs_norm_filtered['Area'] == country]

    if not fs_norm_df_filtered_area.empty:
        loader = DataFrameLoader(fs_norm_df_filtered_area, page_content_column="Item")
        documents.extend(loader.load())
    else:
        print("No FS data found for the specified country.")

    documents = process_pdfs_loop(documents,prefixe = 'heatlh_risks')
    documents = process_urls_loop(documents,url_filename = 'health_risks_urls.txt' )

    return documents

def process_diet_guidelines_documents(documents,country):

    

    documents = process_pdfs_loop(documents,prefixe = 'diet_guidelines')
    documents = process_urls_loop(documents,url_filename = 'diet_guidelines_urls.txt')

    return documents

def process_agriculture_documents(documents,country):

    documents = []

    documents = process_pdfs_loop(documents,prefixe = 'agriculture')
    documents = process_urls_loop(documents,url_filename = 'agriculture_urls.txt' )

    return documents


def process_document(documents,user_informations):
    global conversational_rag_chain
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    vectorstore = Chroma.from_documents(texts, embedding=embeddings)

    retriever = vectorstore.as_retriever()

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = ("You are a Nutrition assistant for question-answering tasks." +
    f"You are speaking to a  user of {user_informations['gender']} gender, of {user_informations['age']}years of age,"+
    f"with a size of {user_informations['size']}  cm and a weight of {user_informations['weight']}  kg from the country {user_informations['country']}." +
    "you need to help this person with their diet." + 
    """If you don't know the answer, just say that you don't know. \

    {context}""")
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)


    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    ### Statefully manage chat history ###
    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )



# Function to process a user prompt
def process_prompt(prompt):
    global conversational_rag_chain
    global chat_history
        
    # Query the model
    output = conversational_rag_chain.invoke(
    {"input": prompt},
    config={
        "configurable": {"session_id": "abc123"}
    },  
)["answer"]
    
    answer = output
    
    chat_history.append((prompt, answer))
    
    
    return answer

# Initialize the language model
init_llm()
