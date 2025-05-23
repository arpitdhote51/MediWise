import os
import streamlit as st
import huggingface_hub
from huggingface_hub import InferenceClient

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
from huggingface_hub import InferenceClient


HF_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]



#if not HF_TOKEN:
   # raise ValueError("Hugging Face Token not found in environment!")

client = InferenceClient(token=HF_TOKEN)



DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def get_context_from_vectorstore(prompt):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def build_chat_prompt(context, question):
    prompt_template = """
You are a medical assistant AI. Based only on the information provided in the context below, answer the user's question by giving structured information in the following format:

1. **Disease Name**: (if identifiable from the question/context)
2. **Causes**: Briefly list or explain the causes.
3. **Precautions**: Suggest preventive steps if available.
4. **Medicines**: List only if specifically mentioned in the context.
5. **Other Relevant Details**: Add any helpful description or symptoms provided in the context.

Only use the data present in the context. If any detail is missing, state "Not mentioned in context".

Do not generate or assume information outside the given context.

---

Context:
{context}

Question:
{question}

Answer:

Please Note: This is an AI-generated response based on limited information. Always consult a certified medical practitioner for diagnosis and treatment.
"""
    return prompt_template.format(context=context, question=question)


def get_mistral_response(prompt, HF_TOKEN):
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=HF_TOKEN
    
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.5)
    return response.choices[0].message["content"]


def main():
    def display_project_info():
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #2C3E50;'>🩺 MediWise: AI Medical Assistant</h1>
            <h3 style='color: #16A085;'>A Project by <span style='color: #2980B9;'>ARPIT DHOTE</span></h3>
            <hr style='margin-top: 20px; margin-bottom: 20px;'>
        </div>
    """, unsafe_allow_html=True)
    display_project_info()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HF_TOKEN= os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            st.error("Hugging Face Token not found in environment!")
            return

        try:
            context, source_documents = get_context_from_vectorstore(prompt)
            full_prompt = build_chat_prompt(context, prompt)
            response = get_mistral_response(full_prompt, HF_TOKEN
        )

            st.chat_message('assistant').markdown(response)
            sources_text = "\n\n**Source Documents:**\n" + "\n\n".join([doc.metadata.get("source", "") for doc in source_documents])
            st.markdown(sources_text)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
