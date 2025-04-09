import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"

text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

text = '''Elon Reeve Musk (/ˈiːlɒn/ EE-lon; born June 28, 1971) is a businessman known for his leadership of Tesla, SpaceX, and X (formerly Twitter). Since 2025, he has been a senior advisor to United States President Donald Trump and the de facto head of the Department of Government Efficiency (DOGE). Musk is the wealthiest person in the world; as of March 2025, Forbes estimates his net worth to be US$345 billion. He was named Time magazine's Person of the Year in 2021.'''

print(text_summary(text))