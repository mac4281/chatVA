from flask import Flask, render_template, request
import openai
import pandas as pd
import numpy as np
from config import OPEN_API_KEY
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import time

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = OPEN_API_KEY

app = Flask(__name__)

items = ["item1", "item2", "item3"]

@app.route("/")
def search_form():
    return render_template("search_form.html")

@app.route("/search")
def search():
    data_received = True
    query = request.args.get("query")


    search_term_vector = get_embedding(query, engine='text-embedding-ada-002')


    df = pd.read_csv('encompass_embeddings.csv')

    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

    sorted_by_similarity = df.sort_values("similarities", ascending=False).head(1)
    results = sorted_by_similarity['Text'].values.tolist()
    context = results[0]
     
    prompt = """Answer the question as truthfully as possible using the provided text and explain the answer in a funny way. If the answer is not contained within the text below, say "I don't know. Context = """+context+ "Q:"+query+"A:''"
   

    newResults = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )["choices"][0]["text"].strip(" \n")
  

    while not data_received:
    # code to call the API goes here
        if newResults:
            data_received = False
        else:
            for _ in range(3):
                print(".", end="", flush=True)
            time.sleep(0.5)
    return render_template("search_results.html", results=newResults, query=query)


if __name__ == "__main__":
    app.run(debug=True)
