from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Loading the Data
loader = UnstructuredPDFLoader("CompaniesAct2013.pdf")
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

#Initializing the API keys
OPENAI_API_KEY = #your openai's API key
PINECONE_API_KEY = #your pinecone's API key
PINECONE_API_ENV = 'us-west4-gcp'
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

#Initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "act" # put in the name of your pinecone index here

docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chain = load_qa_chain(llm, chain_type="stuff")

#Creating API end point
from flask import Flask , jsonify
app = Flask(__name__)
@app.route('/')
def hello_world():
    return 'Hello world'

@app.route('/ask/<n>')
def req_resp(n):
    query = n
    docs = docsearch.similarity_search(query)
    resp = chain.run(input_documents=docs, question=query)
    result = {
        "query" : query,
        "response" : resp,
        "User Id" : '01',
        "bot" : "lawphoria" 
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5656)
