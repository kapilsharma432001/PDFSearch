from flask import Flask, request, jsonify
import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from streamlit_chat import message

st.set_page_config(layout="wide")

app = Flask(__name__)
device = torch.device('cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map=device,
    torch_dtype=torch.float32
)


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa


def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer, generated_text


@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    answer, metadata = process_answer(question)
    return jsonify({'answer': answer})


if __name__ == "__main__":
    app.run(debug=True)
