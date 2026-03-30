import gradio as gr
from main import ask

def chat(q):
    return ask(q)

gr.Interface(fn=chat, inputs="text", outputs="text").launch()