
# MP3 Transcription and Recommendation Chatbot Application

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
    source venv/bin/activate  # On Windows, use `venv\Scripts ctivate`
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
```

### Transcription of MP3 Files

```python
def transcribe_audio(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    if transcript.error:
        st.error(f"Transcription error: {transcript.error}")
        return ""
    return transcript.text
```

### Storing Transcripts in Pinecone

```python
def push_transcription_to_pinecone(transcript):
    index.delete(delete_all=True, namespace="mp3-transcription-namespace")
    chunks = text_splitter.split_text(transcript)
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_documents([chunk])[0]
        vectors.append({
            "id": str(i),
            "values": vector,
            "metadata": {"text": chunk, "source": f"chunk_{i}"}
        })
    index.upsert(vectors=vectors, namespace="mp3-transcription-namespace")
```

### Query Processing with LangChain

```python
def get_chatbot_response(user_query, transcript):
    if not check_query_relevance(user_query, transcript):
        return "I'm sorry, but the transcript doesn't contain any information related to your question."

    llm = OpenAI(temperature=0, max_tokens=500)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    result = qa_chain({"query": user_query})
    answer = result['result']
    source_documents = result.get('source_documents', [])
    
    if "I don't have any information" in answer:
        return "I'm sorry, but I don't have any information about that in the provided transcript."
    
    sources = [doc.metadata.get('source', 'Unknown') for doc in source_documents]
    response = f"{answer}

Sources: {', '.join(sources) if sources else 'No specific sources found'}"
    return response
```

## Challenges and Solutions

### Handling Large Transcripts

**Challenge:** Managing and processing large transcripts efficiently.

**Solution:** The transcripts are split into manageable chunks using a text splitter to ensure efficient processing and storage.

### Ensuring Relevant Responses

**Challenge:** Providing relevant responses to user queries.

**Solution:** Implemented a preprocessing step to check query relevance before generating responses.

## Future Work

Future enhancements could include:

- Extending the application to support additional file formats.
- Improving the user interface for a more seamless user experience.
- Adding more advanced natural language processing capabilities to handle more complex queries.
