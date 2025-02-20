import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory  

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class OpenGptLangChain:
    def __init__(self, memory_limit=5):
        load_dotenv()
        self.ogpt_key = os.getenv("kry")
        self.memory = ChatMessageHistory()
        self.memory_limit = memory_limit 

    def send(self, mess):
        llm = ChatOpenAI(
            openai_api_key=self.ogpt_key,
            base_url="https://openrouter.ai/api/v1",
            model_name="openai/gpt-4o-mini"
        )

        self.memory.add_message(HumanMessage(content=mess))

        if len(self.memory.messages) > self.memory_limit * 2:
            self.memory.messages = self.memory.messages[-self.memory_limit * 2:]

        try:
            messages = self.memory.messages
            response = llm.invoke(messages)
            self.memory.add_message(AIMessage(content=response.content)) 

            logging.info(f"User: {mess}")
            logging.info(f"AI: {response.content}")

            print(response.content)
        except Exception as e:
            logging.error(f"Ошибка при запросе к AI: {e}")
            print("Что-то пошло не так. Попробуйте ещё раз.")

gpt = OpenGptLangChain(memory_limit=5)
gpt.send("В яких ситуаціях кіт може їсти піццу в космосі?")
gpt.send("А якщо він захоче ще й каву?")
gpt.send("Чи може він грати на гітарі?")
gpt.send("А що, якщо він хоче бути рок-зіркою?")
gpt.send("Які пісні він буде співати?")
gpt.send("А які інопланетні фанати в нього можуть бути?")
