
import pickle
import openai
import langchain
import unstructured
import sys
from pdf2image import convert_from_path
from flask import Flask, render_template, request, make_response 
from time import time,sleep
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from werkzeug.utils import secure_filename
import pandas as pd

OPENAI_API_KEY = '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'
PINECONE_API_KEY = '220b469f-3e56-44fb-9291-d389c129183c'
PINECONE_API_ENV = 'us-west1-gcp'


from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

loader = UnstructuredPDFLoader("Lender_Handbook_VA_Pamphlet_Complete.pdf")
# # loader = OnlinePDFLoader("https://firebasestorage.googleapis.com/v0/b/valeadform.appspot.com/o/Resources%2FLender_Handbook_VA_Pamphlet_Complete.pdf?alt=media&token=2caeb6a4-478e-4cbb-ac55-29df96bc6319")

@app.route("/load")
def load():
    data = loader.load()

# print (f'You have {len(data)} document(s) in your data')
# print (f'There are {len(data[0].page_content)} characters in your document')


    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)

# print (f'Now you have {len(texts)} documents')

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    index_name = "va"

    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    # query = "What is the funding fee percentage for subsequent use on an IRRRL?"
    # docs = docsearch.similarity_search(query, include_metadata=True)


    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    query = "What is the funding fee percentage for subsequent use on an IRRRL?"
    docs = docsearch.similarity_search(query, include_metadata=True)

    chain.run(input_documents=docs, question=query)
    print(chain.run(input_documents=docs, question=query))


app = Flask(__name__)

@app.route("/searchLang")
def search():
    query = "What is the funding fee percentage for subsequent use on an IRRRL?"
    answer = chain.run(input_documents=docs, question=query)
    return render_template("search_results_langchain.html", results=answer, query=query)







if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))