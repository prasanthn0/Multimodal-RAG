class PredefinedPrompts:
    """
    A collection of predefined prompts
    """
    rag_prompt_template = """Strictly Use the following pieces of context/images to answer the question at the end.
    Donot assume and add any further information.
    If you don't know the answer, don't try to make up an answer.
    Respond with only the exact answer; do not include any extra words, sentences, or symbols.

    {context}

    Question: {question}
    """

    keywords_template = """For the following piece of source text, generate a list of keywords separated by commas without any other addtional information.
    The keywords that are generated will be crucial in answering questions that are asked on the spource text
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the list of keywords precise
    and relevant , donot try to create keywords outside of context :

    {source_text}

    """
    
    summary_template = """For the given table, generate a summary.
    The summary that are generated will be crucial in answering questions that are asked on the source text
    Keep the summary precise  and relevant , donot try to create summary outside of context :

    {source_text}

    """
    
    image_description_template = """For the given image, which is a part of a PDF, extract every visible text element on the image and provide a detailed text description. Ensure that every bit of information is included verbatim, without summarization. Separate the content into relevant sections, such as headers, dates, various factual details, terms, and specific instructions. 

    Output should be in the following format:
    {
        "extracted_text": "Complete extracted text from the image as visible, including all details.",
        "image_description": "Detailed description including all elements from the extracted text. Clearly categorize information into sections, ensuring no details are missed."
    }
    Ensure no information is omitted or generalized in the output.
    """
