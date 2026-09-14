import gradio as gr

from model import get_model_output


def chat(message, _history):
    prompt = message["text"] or "Describe the mushroom in this image."
    return get_model_output(prompt, message["files"]).text


if __name__ == "__main__":
    gr.ChatInterface(chat, multimodal=True).launch()
