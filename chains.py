import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from tqdm import tqdm
import tempfile


def set_environment_variables():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

def doc_loader(uploaded_file):
   # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    return documents

def doc_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs


def initialize_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def init_pincone_index():
    set_environment_variables()
    index_name = "langchain-rag-1"
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)

def create_vectorstore(docs):
    index = init_pincone_index()
    embeddings = initialize_embeddings()
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)
    vector_store.delete(delete_all=True)   #delete all vectors in the index to add new ones
    vectors = []
    for doc in tqdm(docs):
        vector = embeddings.embed_query(doc.page_content)
        doc_id = str(hash(doc.page_content))
        metadata = {'text': doc.page_content}
        index.upsert(vectors=[(doc_id, vector, metadata)]) #add new vectors to the index
    retriever = vector_store.as_retriever()
    return retriever

def initialize_llm():
    return ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp')

def create_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="refine",
        retriever=retriever
    )

def create_response(query, retriever):
    set_environment_variables()
    llm = initialize_llm()
    qa_chain = create_chain(llm, retriever)
    response = qa_chain.invoke(query)
    return response['result']
