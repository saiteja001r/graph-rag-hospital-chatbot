import dotenv
from dotenv import load_dotenv

import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_intro.tools import get_current_wait_time
from langchain_core.tools import tool






load_dotenv()

dotenv.load_dotenv()
api_key = os.getenv("groq")
REVIEWS_CHROMA_PATH = r"C:\Users\dhira\Desktop\ALL_Projects\Bunny_proj\Graph_RAG\chroma_data"

model = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=api_key)

review_template_str = """your job is to use patient
reviews to answer questions about thier experience at 
a hospital. Use the following context to answer the questions.
Be as detailed as possible. but don't make up any information
that's  not from the context. if you don't know the answer, say
you don't know.

{context}
"""


review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=review_template_str
    )
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"], template="{question}"
    )
)

message = [review_system_prompt, review_human_prompt]
review_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"],
    messages=message
)


output_parser = StrOutputParser()


context = "I had a great stay!"
question = "Did anyone have a positive experience?"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

reviews_retriever  = reviews_vector_db.as_retriever(k=10)



review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_prompt_template
    | model
    | StrOutputParser()
)



