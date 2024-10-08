from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework import status
from core.models import ChatMessage, Conversation
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from .serializers import ChatMessageSerializer, ConversationSerializer
from chat.utils import *
from django.http import JsonResponse    
from chat.utils import create_response


@api_view(['POST', 'GET', 'DELETE'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def chat(request):
    #get chat history
    if request.method == 'GET':
        provided_title = request.GET.get('title')
        user = request.user
        if provided_title:
            try:
                conversation_title = Conversation.objects.get(
                    title=provided_title, user=user)
                conversation_id = getattr(conversation_title, 'id')
                ChatObj = ChatMessage.objects.filter(
                    conversation_id=conversation_id).order_by('timestamp')
                Chat = ChatMessageSerializer(ChatObj, many=True)
                return JsonResponse(Chat.data, safe=False)
            except:
                return JsonResponse({"message": "Does not exist this conversation"}, status=404)
        else:
            return JsonResponse({'error': 'Title not provided'}, status=400)

    #create new chat or continue old conversation by providing title
    elif request.method == 'POST':
        prompt = request.data.get('prompt')
        user = request.user
        provided_title = request.data.get('title')
        if provided_title:
            # Create a ChatMessageHistory instance
            title = provided_title
            retrieved_chat_history = retrieve_conversation(
                provided_title, user)
            print(retrieved_chat_history)
        else:
            memory.clear()
            retrieved_chat_history = ChatMessageHistory(messages=[])
            # Generate a default title if not provided
            title = generate_title()
            store_title(title, user)

        response = create_response(user, retrieved_chat_history, prompt)
        print(retrieved_chat_history)
        conversation_title = Conversation.objects.get(title=title, user=user)
        conversation_id = getattr(conversation_title, 'id')
        store_message(prompt, response, conversation_id)

        return JsonResponse({
            'ai_response': response,
            'title':title
        }, status=201)
    elif request.method == "DELETE":
        user=request.user
        title= request.query_params.get('title')
        try:
            obj=Conversation.objects.get(user=user, title=title)
            obj.delete()
            return JsonResponse({"message": "Deleted succesfully"}, safe=False)
        except:
            return JsonResponse({"message": "Does not exist this conversation"}, status=404)


# Retriving all conversations of a user ( Titles only )

@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])  
def get_title(request):
    user=request.user
    titles= Conversation.objects.filter(user=user)
    serialized= ConversationSerializer(titles, many=True)
    return JsonResponse(serialized.data, safe=False)

# Delete a conversation by providing title of conversation
 
# @api_view(['DELETE'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([IsAuthenticated]) 
# def delete_conversation(request):
#     user=request.user
#     title= request.data.get('title')
#     obj=Conversation.objects.get(user=user, title=title)
#     obj.delete()
#     return JsonResponse("Deleted succesfully", safe=False)

  
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_data(request):
    provided_title = request.data.get('title')
    user = request.user
    if provided_title:
        conversation_title = Conversation.objects.get(
            title=provided_title, user=user)
        conversation_id = getattr(conversation_title, 'id')
        ChatObj = ChatMessage.objects.filter(
            conversation_id=conversation_id).order_by('timestamp')
        Chat = ChatMessageSerializer(ChatObj, many=True)
        return JsonResponse(Chat.data, safe=False)
    else:
        return JsonResponse({'error': 'Title not provided'}, status=400)