from sqlalchemy import create_engine
from core.models import Conversation, ChatMessage
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_community.agent_toolkits import create_sql_agent

load_dotenv

memory = ConversationBufferMemory()
llm = ChatOpenAI(model="gpt-40-mini", temperature=0)
system = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! 

You have access to the following tables: {table_names}

If the question does not seem related to the database, just return "I don't know" as the answer."""

#API for title generator
API_URL = "https://api-inference.huggingface.co/models/czearing/article-title-generator"
headers = {"Authorization": f"Bearer YOUR_API_TOKEN"}

def generate_title():
    return "default"


# retriving the last 4 conversations from the db for memory eficiency while providing context to model
def retrieve_conversation(title, user):
    # number of conversations
    num_recent_conversations = 4

    # Retrieve the most recent conversation history from the database
    conversation_obj = Conversation.objects.get(title=title, user=user)
    conversation_id = getattr(conversation_obj, 'id')
    
    # Retrieve recent conversation messages
    conversation_context = ChatMessage.objects.filter(
        conversation_id=conversation_id
    ).order_by('-timestamp')[:num_recent_conversations:-1]
    
    # Storing the retrived data from db to model memory 
    lst = []
    for msg in conversation_context:
        input_msg = getattr(msg, 'user_response')
        output_msg = getattr(msg, 'ai_response')
        lst.append({"input": input_msg, "output": output_msg})
    
    for x in lst:
        inputs = {"input": x["input"]}
        outputs = {"output": x["output"]}
        memory.save_context(inputs, outputs)
    
   
    retrieved_chat_history = ChatMessageHistory(
        messages=memory.chat_memory.messages
    )

    return retrieved_chat_history


# Function to store the conversation to DB
def store_message(user_response, ai_response, conversation_id):
    ChatMessage.objects.create(
        user_response=user_response,
        ai_response=ai_response,
        conversation_id=conversation_id,
    )

# Function to create a Conversation in DB
def store_title(title, user):
    Conversation.objects.create(
        title=title,
        user=user
    )

def create_sql_engine(user):
    directory = 'app/database'
    file_name = f'{user.id}.db'
    file_path = os.path.join(directory, file_name)
    engine = create_engine(f"sqlite:///{file_path}")
    return engine

def create_llm_agent(engine):
    
    db = SQLDatabase(engine)
    prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "{input}"), MessagesPlaceholder("agent_scratchpad")]
    )
    agent = create_sql_agent(
        llm=llm,
        db=db,
        prompt=prompt,
        agent_type="openai-tools",
        verbose=True,
    )
    return agent

def create_response(user_id, prompt):