# model under development
import os
import streamlit as st

from langchain.document_loaders.csv_loader import CSVLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_text_splitters import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
#from langchain_core.prompts import ChatPromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain, LLMChain

#from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

# below 3 libraries are for loading remote models
#from transformers import LlamaForCausalLM
#from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
import torch


# adding separator
def add_vertical_space(spaces=1):
    for _ in range(spaces):
        st.sidebar.markdown("---")

# main method
def main():
    # page title
    st.set_page_config(page_title="Chatbot for NHPC", layout="wide")
    st.title("Chatbot for NHPC")
    st.write("##### 🚧 Under development 🚧")

    # faiss db directory
    DB_FAISS_PATH = "vectorstore/db_faiss"
    TEMP_DIR = "temp"

    # embedding model path
    #EMBEDDING_MODEL_PATH = "embeddings/MiniLM-L6-v2"
    
    # creating faiss db direcoty if it doesnot exist already
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # uploading csv file
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'], help="Upload a CSV file")

    # adding vertical space
    add_vertical_space(1)

    # creating faiss vectorstore
    if uploaded_file is not None:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.write(f"Uploaded file: {uploaded_file.name}")
        st.write("Processing CSV file...")
        st.sidebar.markdown('##### The model may sometime generate excessive or incorrect response.')
        # calling CSVLoader for loading CSV file
        loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})

        # loading the CSV file data
        data = loader.load()

        # creating embeddings using huggingface
        #embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # loading remote embedding model
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        # creating chunks from CSV file
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="interquartile")
        #text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        text_chunks = text_splitter.split_documents(data)

        # chunks message to output
        st.write(f"Total text chunks: {len(text_chunks)}")
        st.write("---")
        
        # creating vectorstore from the text chunks
        docsearch = FAISS.from_documents(text_chunks, embeddings)

        # saving the vector store to local directory
        docsearch.save_local(DB_FAISS_PATH)

        # loading remote llama model
        #llm = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        #llm = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it")

        token = os.environ["HF_TOKEN"]
        llm = AutoModelForCausalLM.from_pretrained(
        "google/gemma-7b-it",
        # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        torch_dtype=torch.float16,
        token=token,
        )

        
        # custom prompt
        custom_template="""
You are a smart personal assistant and your task is to provide the answer of the given question based only on the given context. \n
If you can't find the answer in the context, just say that "I don't know, please look up the policy." and don't try to make up an answer. \n\n
Please, give the answer in plain english and don't repeat your answer and don't mention that you found the answer form the context and don't mention that the answer can be found in the context. \n
        
Question: "{question}" \n\n
        
Context: "{context}" \n\n
        
Helpful Answer:
"""
        
        QA_PROMPT = PromptTemplate(template=custom_template,input_variables=["question", "context"])

        # main llm chain
        qa = ConversationalRetrievalChain.from_llm(llm,
                                                   #chain_type = "stuff",
                                                   chain_type = "stuff",
                                                   #verbose=True,
                                                   #retriever=docsearch.as_retriever()
                                                   retriever=docsearch.as_retriever(search_kwargs = {"k" : 4, "search_type" : "similarity"}),
                                                   combine_docs_chain_kwargs={"prompt": QA_PROMPT}
                                                   #retriever=docsearch.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.1})
                                                   #memory=memory
                                                   )
        
        # taking question from user
        #st.write("### Enter your query:")
        query = st.chat_input("Ask a question to the chatbot.")
        
        if query:
            st.write("#### Query: "+query)
            with st.spinner("Processing your question..."):
                chat_history = []
                result = qa({"question": query, "chat_history": chat_history})
                #st.write("---")
                #st.write("### Response:")
                #st.write("#### Query: "+query)
                st.write(f"> {result['answer']}")

        os.remove(file_path)

if __name__ == "__main__":
    main()
