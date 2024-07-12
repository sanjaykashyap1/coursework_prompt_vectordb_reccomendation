import os
import json
import streamlit as st
import assemblyai as aai
import tempfile
import re
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# Get API keys from environment variables
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
assemblyai_api_key = os.getenv('ASSEMBLYAI_API_KEY')

# Set up API keys
aai.settings.api_key = assemblyai_api_key

# Initialize Pinecone
pc = PineconeClient(api_key=pinecone_api_key)

# Define index name and check if it exists
index_name = "mp3-transcription-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

@st.cache_data
def transcribe_audio(file_path):
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    if transcript.error:
        st.error(f"Transcription error: {transcript.error}")
        return ""
    return transcript.text

def push_transcription_to_pinecone(transcript):
    # Clear the existing index before adding new data
    index.delete(delete_all=True, namespace="mp3-transcription-namespace")
    
    # Split the transcript into chunks
    chunks = text_splitter.split_text(transcript)
    vectors = []
    for i, chunk in enumerate(chunks):
        vector = embeddings.embed_documents([chunk])[0]
        vectors.append({
            "id": str(i),
            "values": vector,
            "metadata": {"text": chunk, "source": f"chunk_{i}"}
        })
    # Upsert vectors to Pinecone with namespace
    namespace = "mp3-transcription-namespace"
    index.upsert(vectors=vectors, namespace=namespace)

# Initialize LangChain components
vectorstore = Pinecone(index=index, embedding=embeddings, text_key='text', namespace="mp3-transcription-namespace")

# Create a retriever with increased k value
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Update the PROMPT
PROMPT = PromptTemplate(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer or if the information is not explicitly stated in the given context, just say "I'm sorry, but I don't have any information about that in the provided transcript." Do not try to make up an answer or use any external knowledge.

{context}

Question: {question}
Answer: """,
    input_variables=["context", "question"]
)

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
    
    # Check if the answer is relevant
    if "I don't have any information" in answer or "The transcript doesn't contain information" in answer:
        return "I'm sorry, but I don't have any information about that in the provided transcript."
    
    return answer

    
    # Extract sources
    sources = []
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown')
        if source not in sources:
            sources.append(source)
    
    response = f"{answer}\n\nSources: {', '.join(sources) if sources else 'No specific sources found'}"
    return response

def preprocess_query(query):
    # Remove punctuation and convert to lowercase
    query = re.sub(r'[^\w\s]', '', query.lower())
    return query

def check_query_relevance(query, transcript):
    preprocessed_query = preprocess_query(query)
    preprocessed_transcript = preprocess_query(transcript)
    query_terms = preprocessed_query.split()
    return any(term in preprocessed_transcript for term in query_terms)


def main():
    st.title("MP3 Transcription and Chatbot")

    if 'transcript' not in st.session_state:
        st.session_state.transcript = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.subheader("Upload an MP3 file for transcription:")
    uploaded_file = st.file_uploader("Choose an MP3 file", type="mp3")
    
    if uploaded_file is not None and st.session_state.transcript is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.read())
            audio_file_path = tmp_file.name
            transcript = transcribe_audio(audio_file_path)
            st.session_state.transcript = transcript
            st.success("Transcription completed successfully.")
            
            # Push the transcription to Pinecone
            push_transcription_to_pinecone(transcript)
    
    if st.session_state.transcript:
        st.subheader("Transcript")
        st.text_area("", st.session_state.transcript, height=250, key="transcript_display")

        st.subheader("Chat about the transcript:")
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask a question about the transcript"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = get_chatbot_response(prompt, st.session_state.transcript)
                message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)