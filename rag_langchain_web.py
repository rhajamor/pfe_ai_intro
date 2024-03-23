import os
from flask import Flask, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
# from logging.config import dictConfig

# dictConfig({
#     'version': 1,
#     'formatters': {'default': {
#         'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
#     }},
#     'handlers': {'wsgi': {
#         'class': 'logging.StreamHandler',
#         'stream': 'ext://flask.logging.wsgi_errors_stream',
#         'formatter': 'default'
#     }},
#     'root': {
#         'level': 'DEBUG',
#         'handlers': ['wsgi']
#     }
# })
# Flask app initialization
app = Flask(__name__)

# Load environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

# Global variables initialization
docs_dir = "./handbook/"
persist_dir = "./handbook_faiss"
embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

if os.path.exists(persist_dir):
    vectorstore = FAISS.load_local(persist_dir, embedding)
else:
    app.logger.info(f"Building FAISS index from documents in {docs_dir}")
    loader = DirectoryLoader(docs_dir,
        loader_cls=Docx2txtLoader,
        recursive=True,
        silent_errors=True,
        show_progress=True,
        glob="**/*.docx"  # which files get loaded
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )
    frags = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(frags, embedding)
    vectorstore.save_local(persist_dir)

app.logger.debug(f"OPENAI_API_KEY: {os.environ['OPENAI_API_KEY']}")

llm = ChatOpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    temperature=0.6
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)
memory.load_memory_variables({})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vectorstore.as_retriever()
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_input = data.get('message', '')

        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        memory.chat_memory.add_user_message(user_input)
        result = qa_chain({"question": user_input})
        response = result["answer"]
        memory.chat_memory.add_ai_message(response)
        return jsonify(response)
    except Exception as e:
         # Log the exception to your Flask server's log
        app.logger.error(f"An error occurred: {str(e)}")
        
        # Return a JSON response containing the error message and a 500 Internal Server Error status code
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
