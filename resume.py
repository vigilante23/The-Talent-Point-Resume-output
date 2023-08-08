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

st.write('hello')
              
        
         
