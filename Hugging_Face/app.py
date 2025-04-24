from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import cohere
from langchain.prompts import PromptTemplate
import os
import soundfile as sf

# Load environment variables
load_dotenv(find_dotenv())

# Initialize Cohere client
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_api_key)

# 1. Image to Text
def img2text(url):
    image_to_text = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base")
    text = image_to_text(url)[0]["generated_text"]
    print("[Image Caption]:", text)
    return text

# 2. Text to Story using Cohere
def generate_story(scenario):
    template = '''
    You are a story teller.
    You can generate a short story based on a simple narrative.
    The story should be no more than half a page.

    context: {scenario}
    STORY:
    '''

    prompt = PromptTemplate(template=template, input_variables=['scenario'])

    # Use Cohere for text generation
    response = cohere_client.generate(
        model="command-r-plus",  # Specify the model name
        prompt=prompt.format(scenario=scenario),
        max_tokens=300
    )

    story = response.generations[0].text.strip()
    print("[Generated Story]:", story)
    return story

# 3. Text to Speech using Hugging Face's FastSpeech2
def text2speech(message):
    # Load the TTS pipeline (make sure you have GPU available, otherwise set device=-1 for CPU)
    tts_pipeline = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech", device=0)  # Use GPU (device=0) or CPU (device=-1)

    # Generate speech from text
    audio = tts_pipeline(message)
    
    # Save the generated speech to a file (e.g., MP3 or WAV)
    sf.write("simple.wav", audio["speech"], 22050)  # Use appropriate sampling rate (22050 Hz here)

    print("[Audio Saved]: simple.wav")

# Main Pipeline
if __name__ == "__main__":
    image_url = "https://tinypng.com/images/social/website.jpg"
    caption = img2text(image_url)
    story = generate_story(caption)
    text2speech(story)
