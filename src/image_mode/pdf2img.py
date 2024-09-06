from pdf2image import convert_from_path
import os

def convert_pdf_to_images(pdf_path: str):
    """
    Converts each page of a PDF into images and saves them in a dynamically created folder
    with the format 'images-<pdf_name>'.

    Args:
        pdf_path (str): The file path of the PDF to be converted.
    """
    # Extract the directory and file name from the given path
    directory, file_name = os.path.split(pdf_path)
    # Create an output folder with the format 'images-<pdf_name_without_extension>'
    output_dir = os.path.join(directory, f'images-{os.path.splitext(file_name)[0]}')
    os.makedirs(output_dir, exist_ok=True)

    # Convert each page of the PDF into an image
    images = convert_from_path(pdf_path)

    # Save each page as an image in the output folder
    for i, page in enumerate(images):
        image_path = os.path.join(output_dir, f'page_{i + 1}.png')
        page.save(image_path, 'PNG')
        print(f'Saved: {image_path}')

    return str(output_dir)

        

if __name__ == "__main__":
    pdf_path = r'data/JA-207652.pdf'
    convert_pdf_to_images(pdf_path)

