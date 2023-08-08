import streamlit as st
import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
os.environ["OPENAI_API_KEY"] = "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"

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Find the Right Talent for Your Business")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      

      query = "Candidate name, email, number, education and experience all details with double semicolon seperated(;;)"
      qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=knowledge_base.as_retriever())
      
      
      # def split_string_with_commas_or_semicolons(string):
      #   separators = r"\W+|\;|\,"
      #   paragraphs = re.split(separators, string)
      #   return paragraphs
      
      # paragraphs = split_string_with_commas_or_semicolons(answer)
      # for i in paragraphs:
      #    st.write(i)

      # show user input
      with st.sidebar:
          user_question = st.text_input("Ask a question about your PDF:")
          if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            

            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)
              
            st.write(response)
         
