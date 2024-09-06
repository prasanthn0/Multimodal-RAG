import openai 
from src.config import settings
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from src.prompts import PredefinedPrompts
from typing import List
import base64
import re
import time
import io
import base64

def ask_gpt(prompt: str, base64_images: List[str] = None):
    """
    Function to call ChatGPT's chat completion API with support for multiple images.

    Args:
        prompt (str): The text prompt for the GPT model.
        base64_images (List[str], optional): A list of Base64 encoded strings of images.

    Returns:
        str: The response message from ChatGPT.
    """
    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    # Construct messages without images
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # If there are images, append each as a separate content item
    if base64_images:
        image_messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image}"
                }
            }
            for image in base64_images
        ]
        # Add images to the user message
        messages[0]["content"].extend(image_messages)

    start_time = time.time()

    # Call the chat completion API
    response = client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=messages,
        max_tokens=4096
    )

    print(response)
    # Calculate the time taken
    time_taken = time.time() - start_time

    # Extract token usage details
    tokens_used = response.usage.total_tokens
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens

    # Estimate cost based on token usage (assuming $0.002 per 1k tokens)
    estimated_cost = (tokens_used / 1000) * 0.002

    # Extract the response message
    message_content = response.choices[-1].message.content

    return {
        "response": message_content,
        "time_taken": time_taken,
        "tokens_used": tokens_used,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_cost": estimated_cost
    }


def create_embeddings(text):
    """
    function to create text embeddings using openAI
    """
    response = openai.Embedding.create(
            input=text,
            model= settings.EMBEDDINGS_MODEL  
        )
    return response['data'][0]['embedding']


def describe_image(b64_image):
    # Single API call to extract text and generate a description for the image
    response = ask_gpt(PredefinedPrompts.image_description_template, [b64_image])
    
    # Access the text and description from the response
    try:
        # Extract the content from the first choice in the response
        completion_message = response["response"]
        completion_message =completion_message.replace('\n','')
        completion_message =completion_message.replace('```','')
        completion_message =completion_message.replace('json','')
        try:
            completion_message=eval(completion_message.strip())
        except:
            None
    except (IndexError, AttributeError) as e:
        print(f"Error processing API response: {e}")

    return completion_message

def get_summary(data):
    try:
        template = PredefinedPrompts.summary_template.format(source_text = data)
        response = ask_gpt(
                    prompt=template
                )
        return response["response"]
    except Exception as e:
        print(e)
        return ""

def convert_image_to_base64(image_path: str) -> str:
    """
    Converts an image file to its Base64 encoded string representation.

    Args:
        image_path (str): The file path of the image to be encoded.

    Returns:
        str: The Base64 encoded string of the image.
    """
    try:
        # Open the image file in binary mode
        with open(image_path, 'rb') as image_file:
            # Read the image file and encode it to Base64
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_image
    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

def execute_parallel(func,data_list, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=len(data_list))  # Adjust max_workers as needed

    # Submit tasks for each dataframe
    futures = [executor.submit(func, data, *args, **kwargs) for data in data_list]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)
    results = []
    
    # Close the executor to free resources
    executor.shutdown()

    for future in futures:
        if future.exception() is None:
            results.append(future.result())
    return results

def clean_text(text):
    # Remove newline characters and replace them with spaces
    text = text.replace('\n', ' ')
    # Remove extra spaces, tabs, and non-printable characters
    text = re.sub(r'\s+', ' ', text).strip()
    # Fix common formatting issues, such as removing unnecessary bullet points or numbers
    text = re.sub(r'(\d+\.\s*)', '', text)  # Removes numbering like "1.", "2.", etc.
    text = re.sub(r'-\s*\d+\s*Points', '', text)  # Removes points scoring if not needed

    return text

def convert_pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG") 
    b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return b64_image
