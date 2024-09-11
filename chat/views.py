import os
from dotenv import load_dotenv
import sqlite3
from langchain_community.utilities import SQLDatabase
from rest_framework import generics, authentication, permissions
from rest_framework.response import Response
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework import status
from core.models import ChatMessage, Conversation
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from .serializers import ChatMessageSerializer, ConversationSerializer

llm = OpenAI()

memory = ConversationBufferMemory()



@api_view(['POST'])
@authentication_classes([authentication.TokenAuthentication])
@permission_classes([permissions.IsAuthenticated])
def chat(request):
