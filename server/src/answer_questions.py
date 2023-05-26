import openai
import json
import numpy as np
import textwrap
import re
from flask import Flask, render_template, request, make_response 
from time import time,sleep
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
from werkzeug.utils import secure_filename
import pandas as pd


app = Flask(__name__)

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

arr = []
openai.api_key = open_file('openaiapikey.txt')
COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"

openai.api_key = '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'
@app.route("/")
def search_form():
    return render_template("search_form.html")


@app.route("/search")
def search():
    
    # db = firebase.database()
    # users = db.child("files").child("embeddings").get()
    # print(users.val())
    # data_received = True
    query = request.args.get("query")
    arr.append('\nHuman: '+ query)
    str1 = " "
    newArr = str1.join(arr)

    search_term_vector = get_embedding(query, engine='text-embedding-ada-002')

    df = pd.read_csv("https://firebasestorage.googleapis.com/v0/b/doohanigains-bar.appspot.com/o/friendswood.csv?alt=media&token=df393754-440d-4504-8a22-479d882c3698")

    # df = pd.read_csv("https://firebasestorage.googleapis.com/v0/b/doohanigains-bar.appspot.com/o/encompass_embeddings.csv?alt=media&token=953dae6b-086e-485e-beb1-d467b272fde9")
    # print (df)
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)

    df['similarities'] = df['embedding'].apply(lambda x: cosine_similarity(x, search_term_vector))

    sorted_by_similarity = df.sort_values("similarities", ascending=False).head()
    results = sorted_by_similarity['text'].values.tolist()
    context = results[0]
     
    prompt = """Answer the question as truthfully as possible and explain the response in a fun way. If the answer is not contained within the text below, say "I'm sorry, I could not find that inthe documentation.". Context = """+context+ "Q:"+query+"A:''"
    # prompt = """If the word is contained in the list say "This word is in the list". If it is not in the list, list the top 3 words contained in the context results list. Context = """+context+ "Q:"+query+"A:''"

    newResults = openai.Completion.create(
        prompt=prompt,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        model=COMPLETIONS_MODEL
    )["choices"][0]["text"].strip(" \n")
    
    # arr.append('\nAI:'+ newResults)
    # print(arr)
    return render_template("search_results.html", results=context, query=query)


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search_index(text, data, count=1):
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])
        #print(score)
        scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    
    return ordered[0:count]


def gpt3_completion(prompt, engine='text-davinci-002', temp=0.0, top_p=1.0, tokens=250, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 1
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            with open('gpt3_logs/%s' % filename, 'w') as outfile:
                outfile.write('PROMPT:\n\n' + prompt + '\n\n==========\n\nRESPONSE:\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    with open('index.json', 'r') as infile:
        data = json.load(infile)
    #print(data)
    while True:
        query = input("Enter your question here: ")
        #print(query)
        results = search_index(query, data)
        #print(results)
        #exit(0)
        answers = list()
        # answer the same question for all returned chunks
        for result in results:
            prompt = open_file('prompt_answer.txt').replace('<<PASSAGE>>', result['content']).replace('<<QUERY>>', query)
            answer = gpt3_completion(prompt)
            print('\n\n', answer)
            answers.append(answer)
        # summarize the answers together
        all_answers = '\n\n'.join(answers)
        chunks = textwrap.wrap(all_answers, 3000)
        final = list()
        for chunk in chunks:
            prompt = open_file('prompt_summary.txt').replace('<<SUMMARY>>', chunk)
            # summary = gpt3_completion(prompt)
            # final.append(summary)
        print('\n\n=========\n\n', '\n\n'.join(final))


        # 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