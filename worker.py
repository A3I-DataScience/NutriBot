import datetime
import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
from uuid import uuid4

import bs4
import chromadb
import pandas as pd
import torch
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document as LangchainDocument
from langchain.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data.country_list import country_list

# from langchain.chains import LLMChain, SimpleSequentialChain

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversational_rag_chain = None

llm = None
embeddings = None
temperature = 0.2
chunk_size = 1000
embedding_model_name = "text-embedding-3-small"
temperature = 0.2
chat_history = []


# Function to initialize the language model and its embeddings
def init_llm():
    global llm, embeddings, chunk_size, embedding_model_name
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    embeddings = OpenAIEmbeddings(chunk_size=chunk_size, model=embedding_model_name)
    init_chroma_vector_store(embeddings)


def init_chroma_vector_store(embeddings):
    global vector_store_from_client

    persistent_client = chromadb.PersistentClient(path="./data/chroma_db/chroma_langchain_db")

    try:
        persistent_client.get_collection("nutribot_collection")
    except:
        persistent_client.get_or_create_collection("nutribot_collection")
        documents = process_local_documents()
        texts = split_documents(documents=documents, chunk_size=chunk_size)

        # Create an embeddings database using Chroma from the split text chunks.
        embeddings = OpenAIEmbeddings(chunk_size=chunk_size, model=embedding_model_name)

        Chroma.from_documents(
            texts,
            embedding=embeddings,
            collection_name="nutribot_collection",
            persist_directory="./data/chroma_db/chroma_langchain_db",
        )

    vector_store_from_client = Chroma(
        client=persistent_client,
        collection_name="nutribot_collection",
        embedding_function=embeddings,
    )


def create_sub_vector_store(country):
    global vector_store_from_client

    persistent_client = chromadb.PersistentClient(path="./data/chroma_db/chroma_langchain_db")

    # Create a new sub-vector store with the filtered documents
    user_collection = vector_store_from_client.get(
        where={"$or": [{"Area": {"$eq": country}}, {"Type": {"$eq": "commun"}}]}
    )
    country_documents = [Document(page_content=doc) for doc in user_collection["documents"]]

    vectordb = Chroma.from_documents(
        documents=country_documents,
        embedding=embeddings,
        persist_directory="./data/chroma_db/chroma_langchain_db",  # type: ignore
    )
    return vectordb


def process_local_documents():

    documents = []

    documents = process_heatlh_risks_documents(documents)
    documents = process_diet_guidelines_documents(documents)
    documents = process_agriculture_documents(documents)
    documents = process_recipes_documents(documents)

    return documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def process_pdfs_loop(documents: List[Document], folder: Path) -> List[Document]:
    for file in folder.iterdir():
        if file.suffix != ".pdf":
            continue
        loader = PyPDFLoader(file)
        metadata = {"Type": "commun"}
        loaded_documents = loader.load()

        # Add metadata to each document
        for doc in loaded_documents:
            # If doc is a string, convert it into a Document object
            if isinstance(doc, str):
                doc = Document(page_content=doc)

            # Add the metadata to the document
            doc.metadata.update(metadata)  # Assumes 'metadata' is a dictionary

            # Append the document to the list
            documents.append(doc)
        documents.extend(loaded_documents)
    return documents


def process_urls_loop(documents: List[Document], url_file: Path) -> List[Document]:
    url_list = []
    with open(url_file) as file:
        for line in file:
            if line.startswith("https://"):
                url_list.append(line)
    if url_list:
        bs4_strainer = bs4.SoupStrainer()
        loader = WebBaseLoader(
            web_paths=url_list,
            bs_kwargs={"parse_only": bs4_strainer},
        )
        metadata = {"Type": "commun"}
        loaded_documents = loader.load()

        # Add metadata to each document
        for doc in loaded_documents:
            # If doc is a string, convert it into a Document object
            if isinstance(doc, str):
                doc = Document(page_content=doc)

            # Add the metadata to the document
            doc.metadata.update(metadata)  # Assumes 'metadata' is a dictionary

            # Append the document to the list
            documents.append(doc)
        documents.extend(loader.load())

    return documents


def process_heatlh_risks_documents(documents: List[Document]) -> List[Document]:

    for country in country_list:
        health_risks_folder = Path("data/health_risks")
        fs_norm_filtered = pd.read_csv(health_risks_folder / "fs_norm_filtered.csv")
        if country is not None:
            fs_norm_df_filtered_area = fs_norm_filtered[fs_norm_filtered["Area"] == country][
                ["Area", "Item", "Sentence"]
            ]
        else:
            fs_norm_df_filtered_area = fs_norm_filtered[["Area", "Item", "Sentence"]]
        fs_norm_df_filtered_area.fillna("Data not available", inplace=True)

        if not fs_norm_df_filtered_area.empty:
            loader = DataFrameLoader(fs_norm_df_filtered_area, page_content_column="Item")
            documents.extend(loader.load())
        else:
            print(f"No FS data found for the country {country}.")

        documents = process_pdfs_loop(documents=documents, folder=health_risks_folder)
        documents = process_urls_loop(
            documents=documents, url_file=health_risks_folder / "health_risks_urls.txt"
        )
        # Add metadata manually after loading the documents

    return documents


def process_diet_guidelines_documents(documents: List[Document]) -> List[Document]:

    diet_guidelines_folder = Path("data/diet_guidelines")
    documents = process_pdfs_loop(documents=documents, folder=diet_guidelines_folder)
    documents = process_urls_loop(
        documents=documents, url_file=diet_guidelines_folder / "diet_guidelines_urls.txt"
    )

    return documents


def process_agriculture_documents(documents: List[Document]) -> List[Document]:
    for country in country_list:
        agriculture_folder = Path("data/agriculture")
        production_norm_filtered = pd.read_csv(agriculture_folder / "production_norm_filtered.csv")
        if country is not None:
            production_norm_filtered = production_norm_filtered[
                production_norm_filtered["Area"] == country
            ][["Area", "Item", "Sentence"]]
        else:
            production_norm_filtered = production_norm_filtered[["Area", "Item", "Sentence"]]

        if not production_norm_filtered.empty:
            loader = DataFrameLoader(production_norm_filtered, page_content_column="Sentence")
            documents.extend(loader.load())
        else:
            print(f"No Production data found for the country {country}.")

        documents = process_pdfs_loop(documents=documents, folder=agriculture_folder)
        documents = process_urls_loop(
            documents=documents, url_file=agriculture_folder / "agriculture_urls.txt"
        )
        if country is not None:

            fao_url = f"https://www.fao.org/nutrition/education/food-dietary-guidelines/regions/countries/{country.lower()}/en/"

            bs4_strainer = bs4.SoupStrainer()
            loader = WebBaseLoader(
                web_paths=[fao_url],
                bs_kwargs={"parse_only": bs4_strainer},
            )
            documents.extend(loader.load())
    return documents


def process_recipes_documents(documents: List[Document]) -> List[Document]:

    recipes_folder = Path("data/recipes")
    documents = process_pdfs_loop(documents=documents, folder=recipes_folder)
    documents = process_urls_loop(documents=documents, url_file=recipes_folder / "recipes_urls.txt")

    recipes_df = pd.read_csv(recipes_folder / "recipes_1.csv").dropna()
    if not recipes_df.empty:
        loader = DataFrameLoader(recipes_df, page_content_column="Sentence")
        documents.extend(loader.load())
    else:
        print(f"No recipes found.")

    return documents


def split_documents(
    chunk_size: int,
    documents: List[LangchainDocument],
) -> List[LangchainDocument]:
    """
    Split documents into chunks of size `chunk_size` characters and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)

    texts = []
    for doc in documents:
        texts += text_splitter.split_documents([doc])

    return texts


def load_embeddings(texts, embedding_model_name, chunk_size):
    """
    Load embeddings into a Chroma vectorstore.

    """
    global vector_store_from_client
    embeddings = OpenAIEmbeddings(chunk_size=chunk_size, model=embedding_model_name)

    # vectorstore = Chroma.from_documents(texts, embeddings, persist_directory="./data/chroma_db/chroma_langchain_db")
    vector_store_from_client.add_documents(texts)

    return vector_store_from_client


def process_new_profile(user_informations):
    global conversational_rag_chain

    # Split the document into chunks
    vector_store = create_sub_vector_store(user_informations["country"])
    retriever = vector_store.as_retriever()

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

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    today = datetime.date.today().strftime("%Y-%m-%d")
    ### Answer question ###
    qa_system_prompt = (
        "You are a Nutrition assistant for question-answering tasks."
        + f"The date is {today}. "
        + f"You are speaking to a  user of {user_informations['gender']} gender, of {user_informations['age']}years of age,"
        + f"with a size of {user_informations['size']}  cm and a weight of {user_informations['weight']}  kg from the country {user_informations['country']}."
        + "you need to help this person with their diet."
        + "Using the information contained in the context,"
        + "you will initially ask one after the other, 3 questions to the end user about their health"
        + "if someone doesn't answer one of your question, you will re-ask it up to 3 times."
        + "also ask them about their social habits like drinking or smoking and the frequency"
        "then you will ask them if they have particular allergies, intolerences or food preferences."
        + "After that, using the information contained in the context"
        + "you will identify 25 ingredients produced in the country of the user and available in this season"
        + " you will ask the user if these ingredients are ok for them to eat."
        + "After that, Using the information contained in the context,"
        + "you will produce a 1 week meal plan with snacks in between meals  in a csv format between triple quote marks that is optimised for the user health and "
        + " that is based on the previous ingredients"
        + "the 1 week meal plan should contain the amount of calories, serving size, fats, carbohydrates, sugars, proteins, Percent Daily Value, calcium, iron, potassium, and fiber for each meal"
        + "mention the total of calories each day along with suggested calroie intake for the user based on their BMI"
        + "you will not use expressions such as 'season vegetables' or 'season fruits' but instead you will use the names"
        + " of the fruits and vegetables to eat in this season and in this country"
        + "also suggest some exercises to go with the meal plan as an additional response"
        + "also optimised to maximise the consumption of locally produced food and of seasonal products."
        + "You will then ask the user if their is something you should correct in this plan."
        + "If necessary you will correct this plan and re-submit it to the user."
        + "Finally you will produce a csv file containing the final meal plan."
        + "If you are asked questions about anything else but health, nutrition, agriculture, food or diet, you will answer that you don't know."
        + """If you don't know the answer, just say that you don't know. \

    {context}"""
    )
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


# def process_document(documents, user_informations):

#     global conversational_rag_chain


#     # Split the document into chunks

#     texts = split_documents(documents =  documents, chunk_size = chunk_size)

#     # Create an embeddings database using Chroma from the split text chunks.
#     vectorstore = load_embeddings(texts, embedding_model_name,chunk_size)

#     retriever = vectorstore.as_retriever()

#     ### Contextualize question ###
#     contextualize_q_system_prompt = """Given a chat history and the latest user question \
#     which might reference context in the chat history, formulate a standalone question \
#     which can be understood without the chat history. Do NOT answer the question, \
#     just reformulate it if needed and otherwise return it as is."""
#     contextualize_q_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", contextualize_q_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )

#     history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
#     today = datetime.date.today().strftime("%Y-%m-%d")
#     ### Answer question ###
#     qa_system_prompt = (
#         "You are a Nutrition assistant for question-answering tasks."
#         + f"The date is {today}. "
#         + f"You are speaking to a  user of {user_informations['gender']} gender, of {user_informations['age']}years of age,"
#         + f"with a size of {user_informations['size']}  cm and a weight of {user_informations['weight']}  kg from the country {user_informations['country']}."
#         + "you need to help this person with their diet."
#         +"Using the information contained in the context,"
#         + "you will initially ask one after the other, 3 questions to the end user about their health"
#         + "if someone doesn't answer one of your question, you will re-ask it up to 3 times."
#         + "then you will ask them if they have particular allergies, intolerences or food preferences."
#         +"After that, using the information contained in the context"
#         +"you will identify 25 ingredients produced in the country of the user and available in this season"
#         +" you will ask the user if these ingredients are ok for them to eat."
#         + "After that, Using the information contained in the context,"
#         + "you will produce a 1 week meal plan in a csv format between triple quote marks that is optimised for the user health and "
#         +" that is based on the previous ingredients"
#         + "the 1 week meal plan should contain the calories of each meal along with amount of nutrients"
#         + "you will not use expressions such as 'season vegetables' or 'season fruits' but instead you will use the names"
#         + " of the fruits and vegetables to eat in this season and in this country"
#         + "also suggest some exercises to go with the meal plan"
#         + "also optimised to maximise the consumption of locally produced food and of seasonal products."
#         + "You will then ask the user if their is something you should correct in this plan."
#         + "If necessary you will correct this plan and re-submit it to the user."
#         + "Finally you will produce a csv file containing the final meal plan."
#         + "If you are asked questions about anything else but health, nutrition, agriculture, food or diet, you will answer that you don't know."
#         + """If you don't know the answer, just say that you don't know. \

#     {context}"""
#     )
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", qa_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     ### Statefully manage chat history ###
#     store = {}

#     def get_session_history(session_id: str) -> BaseChatMessageHistory:
#         if session_id not in store:
#             store[session_id] = ChatMessageHistory()
#         return store[session_id]

#     conversational_rag_chain = RunnableWithMessageHistory(
#         rag_chain,
#         get_session_history,
#         input_messages_key="input",
#         history_messages_key="chat_history",
#         output_messages_key="answer",
#     )


# Function to process a user prompt
def process_prompt(prompt, first_message):
    global conversational_rag_chain
    global chat_history

    # Query the model
    output = conversational_rag_chain.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": "abc123"}},
    )

    # print(f"{output["answer"]} and  context: \n \n {output["context"]}")

    answer = output["answer"]

    chat_history.append((prompt, answer))
    file_name = f"Meal-Plan_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
    if "```" in answer:
        with open(file_name, "w") as csv_file:
            csv_file.write(answer.split("```")[1])

    return answer


def reset_chat_history():
    global chat_history
    chat_history = []
    ChatMessageHistory().clear()


init_llm()
