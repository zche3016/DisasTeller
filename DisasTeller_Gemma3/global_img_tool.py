import os
import base64
import io
from PIL import Image

from crewai_tools import tool
# from openai import OpenAI
import ollama  # Make sure to install via `pip install ollama`
# from dotenv import load_dotenv
#
# load_dotenv()
#
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

folder_path = r"./Disaster_image/Global_image"

def encode_image(image_path, target_size):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
    with io.BytesIO() as buffer:
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_images_in_folder(folder_path,target_size=(256,256)):
    encoded_image = []
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.jpeg') or f.endswith('.jpg') or f.endswith('.png')]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        encoded_image.append(encode_image(image_path, target_size))
    return encoded_image


encoded_images = encode_images_in_folder(folder_path, target_size=(720,720)) #encoded_images[i]

base_text = "Analyze the map for potential dangerous areas. "
# Prepare images in the format expected by the Ollama Vision model
images_data = [img for img in encoded_images]

@tool
def global_img_interpreter(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The processed result from the LLM.
    """

    full_text = base_text + text
    print("Global information:", full_text)

    # Construct the message dictionary for Ollama
    message = {
        "role": "user",
        "content": full_text,
        "images": images_data
    }

    try:
        response = ollama.chat(
            model="gemma3:27b",  # Use the vision model to support images
            messages=[message]
        )
        return response["message"]["content"]

    except Exception as e:
        return f"An error occurred: {str(e)}"











