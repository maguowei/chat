import ollama
import gradio as gr


def predict(message, history):
    history_messages = []
    for human, assistant in history:
        history_messages.append({"role": "user", "content": human})
        history_messages.append({"role": "assistant", "content": assistant})
    history_messages.append({"role": "user", "content": message})

    response = ollama.chat(model='codegemma', messages=history_messages, stream=True)

    partial_message = ""
    for chunk in response:
        if chunk['message']['content'] is not None:
            partial_message = partial_message + chunk['message']['content']
            yield partial_message


gr.ChatInterface(predict, title="CodeGemma").launch()