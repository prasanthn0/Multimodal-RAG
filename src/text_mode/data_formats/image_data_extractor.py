# import comtypes.client
import fitz  # PyMUPDF
import io
from PIL import Image
from src.utils import execute_parallel, describe_image, convert_pil_to_base64

def extract_images_and_text_from_pdf(file_path):
    pdf_path = file_path
    doc = fitz.open(pdf_path)
    images = []
    text = ""
    print("Opened PDF file:", pdf_path)

    for page_index, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        print(f"Found {len(image_list)} images on page {page_index}.")

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image:
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                base64_image = convert_pil_to_base64(image)
                images.append(base64_image)

        text += page.get_text()

    doc.close()
    return images, text


def process_images(file_path):
    images, text = extract_images_and_text_from_pdf(file_path)
    image_descriptions = execute_parallel(describe_image,images)
    return image_descriptions
