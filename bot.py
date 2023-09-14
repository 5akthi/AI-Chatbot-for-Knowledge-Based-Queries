from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import ChannelAccount
from langchain_chatbot import retrieve_answer

class MyBot(ActivityHandler):
    async def on_message_activity(self, turn_context: TurnContext):
        query = turn_context.activity.text
        response = retrieve_answer(query)
        await turn_context.send_activity(response)
        
    async def on_members_added_activity(
        self,
        members_added: ChannelAccount,
        turn_context: TurnContext
    ):
        for member_added in members_added:
            if member_added.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello! I am AI chatbot for Knowledge based queries, how can I help you ?")