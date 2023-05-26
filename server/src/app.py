
import openai
import json
import numpy as np
import textwrap
import re
import os


from flask import Flask, render_template, request, make_response 
from time import time,sleep
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from werkzeug.utils import secure_filename

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import  Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain.document_loaders import GoogleDriveLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader

from langchain.docstore.document import Document
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os


os.environ['OPENAI_API_KEY'] = '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'
credentials_file = 'credentials.json'


arr = []
PINECONE_API_KEY = '220b469f-3e56-44fb-9291-d389c129183c'
PINECONE_API_ENV = 'us-west1-gcp'
api_key = os.getenv("PINECONE_API_KEY") or "YOUR-API-KEY"
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
OPENAI_API_KEY = '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'
llm = OpenAI(model_name="gpt-3.5-turbo", n=2, best_of=2, openai_api_key=OPENAI_API_KEY)
conversation = ConversationChain(
    llm=llm, 
    verbose=True, 
    memory=ConversationBufferMemory()
)
template = """You are a chatbot having a conversation with a human.

Given the following extracted parts of a long document, a question, and the chat history, create a final answer using only those elements.  If the text from the documentation is complex, please explain it in additional detail. Also, you should add additional context about what document, section and page number you found the answer on as well as the URL if known.  If the answer to the question is not contained in the document, the chat history or the question its self just say "I did not find that in the documentation, could you please rephrase the question?"

{context}

{chat_history}

Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], 
    template=template
)
memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input") 


# openai = OpenAI(
#     model_name="gpt-3.5-turbo",
#     openai_api_key="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"
# )
# List all indexes currently present for your key


app = Flask(__name__)


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


# openai.api_key = open_file('openaiapikey.txt')
COMPLETIONS_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-ada-002"
openai.api_key = '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'


@app.route("/")
def search_form():
    return render_template("search_form.html")


# @app.route("/search")
# def search_form_function():
#     return 'This works'

@app.route('/myfunction', methods=['POST'])
def myfunction():
 
    return 'Function called successfully!'

@app.route("/search")
def search():
    
    query = request.args.get("query")
    
    
    textFile = "/texts.txt"
    llm = OpenAI(model_name="gpt-3.5-turbo",temperature=0, openai_api_key=OPENAI_API_KEY)
    # chain = load_qa_chain(llm, chain_type="stuff")
    
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "va"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_texts([t for t in textFile], embeddings, index_name=index_name)
    docs = docsearch.similarity_search(query, include_metadata=True)
    print("docs=",docs)
    # my_string = ', '.join(docs)

     
    print(memory)  
    chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
   
    
    response = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    answer = response['output_text']
    return render_template("search_page.html", results=answer, query=query)

# @app.route("/search")
# def search():
#     OPENAI_API_KEY = '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'
#     openai_api_key = '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'
#     query = request.args.get("query")
#     textFile = "/texts.txt"
    
#     pinecone.init(
#         api_key=PINECONE_API_KEY,  # find at app.pinecone.io
#         environment=PINECONE_API_ENV  # next to api key in console
#     )
#     index_name = "westranch"
#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     docsearch = Pinecone.from_texts([t for t in textFile], embeddings, index_name=index_name)
#     docs = docsearch.similarity_search(query, include_metadata=True)

#     template = """You are a chatbot having a conversation with a human.

#     Given the following extracted parts of a long document and a question, create a final answer.  Make sure that the answer you provide comes explicitely from the documentation.  If you do not know the answer to the queston reply with "I cannot find that in the documentation, could you please rephrase your question?".  

#     {context}

#     {chat_history}
#     Human: {human_input}
#     Chatbot:"""

#     prompt = PromptTemplate(
#         input_variables=["chat_history", "human_input", "context"], 
#         template=template
#     )
#     memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
#     chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt)
#     chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
    
#     print(memory)
    
#     return render_template("search_page.html", results=chain, query=query, conversation = chain.memory.buffer)
@app.route('/upload')
def upload_file():
   
           # # do something here
        # loader = GoogleDriveLoader(folder_id="1I9R_NbR1U27z5c2DQ2XoVoRRL1kuIKFT")
        # loader = DirectoryLoader('/docs', glob="**/*.md", loader_cls=TextLoader)
        loader = UnstructuredPDFLoader("Lender_Handbook_VA_Pamphlet_Complete.pdf")
        
        data = loader.load()
        len(data)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        texts = text_splitter.split_documents(data)
        str_dict = str(texts)
        with open("texts.json", mode="w") as file:
            file.write(str_dict) 
        with open("texts.txt", mode="w") as file:
            file.write(str_dict) 

       
        pinecone.init(
                api_key=PINECONE_API_KEY,  # find at app.pinecone.io
                environment=PINECONE_API_ENV  # next to api key in console
            )
        index_name = "va"
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)  
        return 'File processed correctly!'

   

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
    
   
        


