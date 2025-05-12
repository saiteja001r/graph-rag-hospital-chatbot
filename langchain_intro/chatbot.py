import dotenv
from dotenv import load_dotenv
import random
import time
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
from langsmith import Client
from langgraph.prebuilt import create_react_agent

dotenv.load_dotenv()
api_key = os.getenv("groq")
ls_api_key = os.getenv("LANGSMITH_API_KEY")
nim = os.getenv("NIM")


load_dotenv()


client = Client(api_key=ls_api_key)


REVIEWS_CHROMA_PATH = r"C:\Users\dhira\Desktop\ALL_Projects\Bunny_proj\Graph_RAG\chroma_data"

model = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=api_key)
agent_model = init_chat_model("meta/llama3-70b-instruct", model_provider="nvidia",api_key=nim)


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


# tools = [
#     tool(
#     name="reviews",
#     func=review_chain.invoke,
#     description="""Useful when you need to answer questions
#         about patient reviews or experiences at the hospital.
#         Not useful for answering questions about specific visit
#         details such as payer, billing, treatment, diagnosis,
#         chief complaint, hospital, or physician information.
#         Pass the entire question as input to the tool. For instance,
#         if the question is "What do patients think about the triage system?",
#         the input should be "What do patients think about the triage system?"
#         """,
#     ),
#     tool(
#         name="waits",
#         func=get_current_wait_time,
#         description="""Use when asked about current wait times
#         at a specific hospital. This tool can only get the current
#         wait time at a hospital and does not have any information about
#         aggregate or historical wait times. This tool returns wait times in
#         minutes. Do not pass the word "hospital" as input,
#         only the hospital name itself. For instance, if the question is
#         "What is the wait time at hospital A?", the input should be "A".
#         """,
#     ),
# ]



# @tool(return_direct=True)
def get_current_wait_time(hospital: str) -> int | str:
    """Dummy function to generate fake wait times"""

    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist."
    time.sleep(1)
    return random.randint(1, 10000)

# @tool(return_direct=True)
def reviews(context: str, question: str) -> str:
    """Answer questions about patient reviews."""
    return review_chain.invoke({"context": context, "question": question})



# tools = [
#     reviews,
#     get_current_wait_time,
# ]


# model = agent_model.bind_tools(tools)



# hospital_agent_prompt  = client.pull_prompt("hwchase17/openai-functions-agent", include_model=True)




agent = create_react_agent(
    # disable parallel tool calls
    model = agent_model,
    tools=[get_current_wait_time, reviews],
    prompt="""you are a helpful assistant that answers questions about hospitals. 
    you can answer questions about patient reviews
    {input}
    """,
)




# agent.invoke(
#     {"messages": [{"role": "user", "content": "What is the current wait time at hospital C?"}]}
# )