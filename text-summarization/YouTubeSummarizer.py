from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import torch
import gradio as gr

# Use a pipeline as a high-level helper
from transformers import pipeline

# for local
model_path = "../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/a4f8f3ea906ed274767e9906dbaede7531d660ff"
text_summary = pipeline("summarization", model=model_path, torch_dtype=torch.bfloat16)

# for remote
# text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)


def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

def extract_video_id(url):
    """
    Extracts video ID from a YouTube URL
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ['www.youtube.com', 'youtube.com']:
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        elif query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        elif query.path[:3] == '/v/':
            return query.path.split('/')[2]
    return None

def get_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = ' '.join([item['text'] for item in transcript])
        return summary(full_text)
    except Exception as e:
        return f"Error retrieving transcript: {e}"

gr.close_all()

demo = gr.Interface(fn=get_transcript,
                    inputs=[gr.Textbox(label="Input YouTube URL to summarize", lines=6)],
                    outputs=[gr.Textbox(label="Summarized text", lines=4)],
                    title="@GenAI Project 2: YouTube Script Summarizer",
                    description="This application will be used to summarize the YouTube video script."
                    )
demo.launch()