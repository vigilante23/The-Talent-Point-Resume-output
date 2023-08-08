import streamlit as st
import os
import re
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(page_title="CHECK DETAILS FROM YOUR RESUME")
st.header("Find the Right Talent for Your Business")

pdf = st.file_uploader("Upload your PDF", type="pdf")

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

embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

st.write(knowledge_base)
        
         
