import os
import json
import re
from openai import OpenAI
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from crewai_tools import tool
import ollama  # Make sure to install via `pip install ollama`

def encode_image(image_path, target_size):
    with Image.open(image_path) as img:
        img_resized = img.resize(target_size)
    with io.BytesIO() as buffer:
        img_resized.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def encode_images_in_folder(folder_path,target_size=(256,256)):
    encoded_image = []
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.endswith('.png') or f.endswith('.jpeg') or f.endswith('.jpg') ]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        encoded_image.append(encode_image(image_path, target_size))
    return encoded_image

folder_path = r"./Disaster_image/Global_image"
encoded_images = encode_images_in_folder(folder_path, target_size=(1080, 1522))


def annotate_image(image_path, annotations, output_path, size=None):
    image = Image.open(image_path)
    if size is not None:
        image = image.resize(size)
    draw = ImageDraw.Draw(image)

    font_size = 70
    font_color = (255, 255, 0)
    try:
        font_path = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf"
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    for annotation in annotations:
        position, text = annotation['position'], annotation['text']
        draw.text(position, text, font=font, fill=font_color)

    image.save(output_path)

@tool
def global_map_annotation(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The input from the LLM.
    """

    base_instruction = (
        "these images are at the same place that experienced disaster. "
        "There are different disaster grades such as G1~G10 for different locations in these images. "
        "Firstly, please find the following locations in the image according to location names; "
        "Secondly, generate a JSON structure for all the relevant disaster locations with position and grading information "
        "to annotate these labels in the second image like this: "
        "{annotations: [{position: [820, 380], text: G1}, {position: [660, 620], text: G2}]}, "
        "The disaster locations with relevant grading are following: "
    )
    full_instruction = base_instruction + text
    print(full_instruction)

    images_data = [img for img in encoded_images]

    message = {
        "role": "user",
        "content": full_instruction,
        "images": images_data
    }

    try:
        response = ollama.chat(
            model="gemma3:27b",
            messages=[message]
        )
        llm_response = response["message"]["content"]
        print(llm_response)
        json_match = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)

        if json_match:
            json_str = json_match[-1].strip()
            json_str = re.sub(r'//.*', '', json_str)
            try:
                content_dict = json.loads(json_str)
                print(content_dict['annotations'])

            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON: {e}")
        else:
            print("No JSON structure found in the content.")

        image_path = r"./Disaster_image/Global_image/2.jpg"
        output_path = r"./Disaster_image/Global_image/4_anotated.png"
        annotate_image(image_path, content_dict['annotations'], output_path, size=(1080, 1522)) # size closer to original, more accurate

        return text

    except Exception as e:
        return f"An error occurred: {str(e)}"