# Q & A App built using langchain, streamlit, and pinecone
# dated: 30-12-2024
# this app is a simple question and answer app that takes a pdf file and 
# a question as input and returns the answer to the question based on the content of the pdf file.
# The app uses stremlit for UI, langchain for text processing, and pinecone for vector storage and retrieval.
# The app is divided into three parts: document loading, document splitting, and vector store creation.

import streamlit as st
from chains import doc_loader, doc_splitter, create_vectorstore, create_response

def main():
    
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Q & A App </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: blue;'>RAG App using Langchain, Streamlit, and Pinecone</h3>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: left;'>This app takes a PDF file and a question as input and returns the answer to the question based on the content of the PDF file.</p>", unsafe_allow_html=True)
    # Initialize session state for vector store status
    if "vector_retriever" not in st.session_state:
        st.session_state["vector_retriever"] = None
    
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if st.button("Load Documents"):
        with st.spinner("Processing... Please wait."):    
            # ####### Document Loading ########
            if uploaded_file is not None:
                loaded_docs = doc_loader(uploaded_file)
                
                # ####### Document Splitting ########
                if loaded_docs:
                    documents = doc_splitter(loaded_docs)
                    
                    #  ##### Vector Store Creation #####
                    st.session_state["vector_retriever"] = create_vectorstore(documents)
                    st.success("Document loaded and vector db created successfully.")
                else:
                    st.warning("No documents loaded.")
            else:
                st.warning("Please upload a PDF file.")
                      
    question = st.text_input("Ask a question:")
    
    if st.button("Submit"):
        if st.session_state["vector_retriever"]:
            if question:
                with st.spinner("Waiting for response..."):
                    response = create_response(question, st.session_state["vector_retriever"])
                    st.write("### Response")
                    st.write(response)
            else:
                st.warning("Please enter a question.")
        else:
            st.warning("Please load a document and create the vector store first.")
    

if __name__ == "__main__":
    main()




