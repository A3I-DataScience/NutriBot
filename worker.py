import os
import torch
import pandas as pd
import datetime

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
temperature = 0.2

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

    url_list = []
    with open(url_filename) as url_file:
        for line in url_file:
            if line.startswith('https://'):
                url_list.append(line)
    if len(url_list) > 0:
        bs4_strainer = bs4.SoupStrainer()
        loader = WebBaseLoader(web_paths=url_list,
        bs_kwargs={"parse_only": bs4_strainer},
        )
        documents.extend(loader.load())

    return documents

# Function to process a PDF document
def process_heatlh_risks_documents(documents,country):

    fs_norm_filtered = pd.read_csv('fs_norm_filtered.csv')
    fs_norm_df_filtered_area = fs_norm_filtered[fs_norm_filtered['Area'] == country][['Area','Item','Sentence']]
    fs_norm_df_filtered_area.fillna('Data not available', inplace=True)

    if not fs_norm_df_filtered_area.empty:
        loader = DataFrameLoader(fs_norm_df_filtered_area, page_content_column="Item")
        documents.extend(loader.load())
    else:
        print(f"No FS data found for the country {country}.")

    documents = process_pdfs_loop(documents,prefixe = 'heatlh_risks')
    documents = process_urls_loop(documents,url_filename = 'health_risks_urls.txt' )

    return documents

def process_diet_guidelines_documents(documents,country):

    

    documents = process_pdfs_loop(documents,prefixe = 'diet_guidelines')
    documents = process_urls_loop(documents,url_filename = 'diet_guidelines_urls.txt')

    return documents

def process_agriculture_documents(documents,country):

    documents = []

    production_norm_filtered = pd.read_csv('production_norm_filtered.csv')
    production_norm_filtered = production_norm_filtered[production_norm_filtered['Area'] == country][['Area','Item','Sentence']]

    if not production_norm_filtered.empty:
        loader = DataFrameLoader(production_norm_filtered, page_content_column="Sentence")
        documents.extend(loader.load())
    else:
        print(f"No Production data found for the country {country}.")

    documents = process_pdfs_loop(documents,prefixe = 'agriculture')
    documents = process_urls_loop(documents,url_filename = 'agriculture_urls.txt' )
    fao_url = f'https://www.fao.org/nutrition/education/food-dietary-guidelines/regions/countries/{country.lower()}/en/'

    bs4_strainer = bs4.SoupStrainer()
    loader = WebBaseLoader(web_paths=[fao_url],
                            bs_kwargs={"parse_only": bs4_strainer},)
    documents.extend(loader.load())
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
    today = datetime.date.today().strftime('%Y-%m-%d')
    ### Answer question ###
    qa_system_prompt = ("You are a Nutrition assistant for question-answering tasks." +
                        f"The date is {today}. " +
    f"You are speaking to a  user of {user_informations['gender']} gender, of {user_informations['age']}years of age,"+
    f"with a size of {user_informations['size']}  cm and a weight of {user_informations['weight']}  kg from the country {user_informations['country']}." +
    "you need to help this person with their diet." + 
    "you will initially ask one after the other, 3 questions to the end user about their health"+
    "if someone doesn't answer one of your question, you will re-ask it up to 3 times."+
    "then you will ask them if they have particular allergies, intolerences or food preferences."+
    "After that you will produce a 1 week meal plan in a csv format that is optimised for the user health and "+
    "also optimised to maximise the consumption of locally produced food and of seasonal products."+
    "You will then ask the user if their is something you should correct in this plan."+
    "If necessary you will correct this plan and re-submit it to the user."+
    "Finally you will produce a csv file containing the final meal plan."+
    "If you are asked questions about anything else but health, nutrition, agriculture, food or diet, you will answer that you don't know."+
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
)
    
    #print(f"{output["answer"]} and  context: \n \n {output["context"]}") 
    
    answer = output["answer"]
    
    chat_history.append((prompt, answer))
    file_name = f'Meal-Plan_{datetime.datetime.now()}.csv'
    if '```' in answer:
        with open(file_name, "w") as csv_file:
            csv_file.write(answer.split('```')[1])


    
    return answer

# Initialize the language model
init_llm()
