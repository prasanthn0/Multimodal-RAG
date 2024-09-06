1. Set Up Environment:

    Create a virtual environment: python -m venv env

    Activate the environment:
        Windows: .\env\Scripts\activate
        macOS/Linux: source env/bin/activate

2. Install Requirements:
    Install dependencies: pip install -r requirements.txt

3. Configure API Key:
    Open the config file in the src folder (src/config.py).
    Add your OpenAI API key: OPENAI_API_KEY = 'your-openai-api-key-here’

4. Launch the App:
    Run the app: streamlit run app.py

5. Using the App:
    Upload a PDF and wait for ingestion.
    Enter a query to compare results from text and image modes.

Flowchart:
![alt text](image.png)


