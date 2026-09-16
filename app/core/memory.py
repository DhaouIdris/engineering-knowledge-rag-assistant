from langchain.memory import ConversationBufferWindowMemory

def get_memory(k: int = 5) -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )