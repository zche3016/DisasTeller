import os
import base64
import io
from PIL import Image

from crewai_tools import tool
# from openai import OpenAI
# from dotenv import load_dotenv
import ollama  # Make sure to install via `pip install ollama`


folder_path = r"./Disaster_image/Local_image"
# folder_path = r"C:\Users\Ziv\Python_Projects\Project4_LLM\otherimage"

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


@tool
def local_img_interpreter(text: str) -> str:
    """
    Process the text with the LLM.

    Args:
        text (str): The input text.

    Returns:
        str: The processed result from the LLM.
    """

    base_text = ("Describe the earthquake in Wajima with completely exact location name in image sticker, Ensure that the output"
                 "only have these two items with exact 80 tokens output: location name:xxx, "
                "disaster description:xxx.")
    full_text = base_text + text
    print("Local information:", full_text)

    all_descriptions = []

    for idx, img in enumerate(encoded_images):
        message = {
            "role": "user",
            "content": full_text,
            "images": [img]
        }

        try:
            response = ollama.chat(
                model="gemma3:27b",
                messages=[message],
                options={'temperature': 0.8, 'top_p': 0.8},
            )
            description = response["message"]["content"]
            start_idx = description.find("location name:")
            if start_idx != -1:
                extracted_content = description[start_idx:]
                print(extracted_content)
            else:
                print("No 'location name:' ")
            all_descriptions.append(f"Image {idx + 1} description:\n{extracted_content}")
            print(f"Image {idx + 1} processed.")
        except Exception as e:
            error_msg = f"Image {idx + 1} description: An error occurred: {str(e)}"
            all_descriptions.append(error_msg)
            print(error_msg)

    combined_response = "\n\n".join(all_descriptions)
    print("Combined Response:\n", combined_response)
    return combined_response














