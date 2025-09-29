from flask import Flask, render_template, request
from dotenv import load_dotenv
load_dotenv()
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.prompt import *
import os


app = Flask(__name__)




load_dotenv()

#Create Memory buffer
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

pinecone_api_key=os.environ.get('pinecone_api_key')
openai_api_key=os.environ.get('openai_api_key')

os.environ["pinecone_api_key"] =pinecone_api_key
os.environ["openai_api_key"] =openai_api_key

embeddings = download_hugging_face_embeddings()


index_name = "chatbot"
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(api_key = openai_api_key,temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("system", "Previous conversation: {chat_history}"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "")
    print("Received from frontend:", msg)
    if not msg:
        return "No input received."
    #Get Chat history from memory
    memory_variable = memory.load_memory_variables({})
    chat_history = memory_variable.get("chat_history", "No Previous Conversation")

    try:
        response = rag_chain.invoke({"input": msg, "chat_history": chat_history})
        print("Response:", response)
        answer = response.get("output") or response.get("answer") or str(response)
    # Save interaction to memory    
        memory.save_context(
            {"input": msg},
            {"answer": answer}
        )
        print('memory ::', memory)
    except Exception as e:
        print("Error:", e)
        answer = "There was an error processing your request."

    return str(answer)


if __name__ == '__main__':
    app.run(debug=True)
