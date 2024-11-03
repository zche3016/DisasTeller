import os
import base64
import io
from PIL import Image

from crewai_tools import tool
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

content = [
    {"type": "text",
        "text": "Analyze the map for potential dangerous areas."

    }
]

for image in encoded_images:
    content.append({"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image}"}})


@tool
def global_img_interpreter(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The input from the LLM.
    """
    content[0]["text"] = content[0]["text"] + text
    print("Global information:", content[0]["text"])

    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # or the model you are using
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {str(e)}"




