# MP3 Transcription and Chatbot Application

Welcome to the MP3 Transcription and Chatbot Application! This project combines advanced transcription services, vector databases, and Large Language Models (LLMs) to create a responsive and intelligent chatbot. The application allows users to upload MP3 files, transcribe the audio, and interact with the transcript through a conversational interface.

## Objective

The goal of this project is to develop a domain-specific application that leverages the strengths of LLMs for understanding and processing natural language queries and the efficiency of a vector database for data storage and retrieval. The focus is on creating a chatbot that provides personalized responses based on MP3 transcriptions.

## Features

- **User-Friendly Interface:** Upload MP3 files and interact with the transcript via a conversational chatbot.
- **Advanced Transcription:** Utilizes AssemblyAI's API to transcribe audio files accurately.
- **Efficient Data Storage:** Stores transcribed data in a Pinecone vector database for quick and efficient retrieval.
- **Intelligent Responses:** Uses OpenAI's GPT model to generate contextually relevant responses based on the transcriptions.
- **Seamless Integration:** The backend logic is managed using LangChain to ensure smooth interaction flow and data retrieval.

## Installation

Follow these steps to set up the application locally:

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/your-username/mp3-transcription-chatbot.git
    cd mp3-transcription-chatbot
    ```

2. **Set Up a Virtual Environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the Dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set Up Environment Variables:**
    Create a `.env` file in the root directory and add the following:
    ```sh
    PINECONE_API_KEY=your_pinecone_api_key
    OPENAI_API_KEY=your_openai_api_key
    ASSEMBLYAI_API_KEY=your_assemblyai_api_key
    ```

5. **Run the Application:**
    ```sh
    streamlit run app.py
    ```

## Usage

1. **Upload an MP3 File:**
   - Open the application in your browser.
   - Upload an MP3 file using the provided interface.

2. **View the Transcript:**
   - Once the file is uploaded, it will be transcribed automatically.
   - The transcript will be displayed in a text area within the application.

3. **Interact with the Chatbot:**
   - Ask questions about the transcript using the chat input box.
   - The chatbot will provide relevant responses based on the transcribed content.

## Demo Video

For a detailed walkthrough and demo of the application, watch the [YouTube video](https://youtu.be/b9eJE_mo6bk).

[![Demo Video](https://img.youtube.com/vi/b9eJE_mo6bk/0.jpg)](https://youtu.be/b9eJE_mo6bk)

## Implementation Snippets

### Setting Up Environment Variables

```python
os.environ['PINECONE_API_KEY'] = "your_pinecone_api_key"
os.environ['OPENAI_API_KEY'] = "your_openai_api_key"
os.environ['ASSEMBLYAI_API_KEY'] = "your_assemblyai_api_key"
'''

##Transcription of MP3 Files
```python
def transcribe_audio(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    if transcript.error:
        st.error(f"Transcription error: {transcript.error}")
        return ""
    return transcript.text

