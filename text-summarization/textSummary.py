import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

# for local
# model_path = "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
# text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# for remote
text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)


def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")
# demo.launch()

demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text summarize", lines=6)],
                    outputs=[gr.Textbox(label="Summarized text", lines=4)],
                    title="@GenAI Project 1: Text Summarizer",
                    description="This application will be used to summarize the text."
                    )
demo.launch()