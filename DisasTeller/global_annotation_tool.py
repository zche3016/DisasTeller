import os
import json
import re
from openai import OpenAI
import base64
import io
from PIL import Image, ImageDraw, ImageFont
from crewai_tools import tool
from dotenv import load_dotenv
load_dotenv()
# Set OpenAI API key from environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

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
encoded_images = encode_images_in_folder(folder_path, target_size=(1080, 1522)) #encoded_images[i] #1000x1000


def annotate_image(image_path, annotations, output_path, size=None):
    image = Image.open(image_path)
    if size is not None:
        image = image.resize(size)
    draw = ImageDraw.Draw(image)

    # Define font
    font_size = 150
    font_color = (255, 255, 0)  # Red color
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # Draw annotations on the image
    for annotation in annotations:
        position, text = annotation['position'], annotation['text']
        draw.text(position, text, font=font, fill=font_color)

    # Save the annotated image
    image.save(output_path)



client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),

)

content = []

for image in encoded_images:
    content.append({"type": "image_url",
                 "image_url": {"url":f"data:image/jpeg;base64,{image}"}})


content.append({"type": "text",
        "text": "these images are the same place that experienced disaster. "
                "There are different disaster grade such as G1~G10 for the different locations in these images."
                "Firstly, please find the following locations in the first image according to location names; Secondly,"
                "generate a json structure for all the relevant disaster locations with position and grading information "
                "to annotate these labels in the second image like this:"
                "{annotations: [{position: [820, 380], text: G1}, {position: [660, 620], text: G2}},"
                "The disaster locations with relevant grading are following:"
    })


@tool
def global_map_annotation(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The input from the LLM.
    """
    
    content[-1]["text"] = content[-1]["text"] + text
    print(content[-1]["text"])

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-4o-2024-08-06",  ## model="gpt-4o", model="gpt-4-turbo"
            max_tokens=500,
        )


        llm_response = response.choices[0].message.content
        print(llm_response)

        json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)


        if json_match:
            json_str = json_match.group(1).strip()

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