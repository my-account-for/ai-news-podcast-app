import streamlit as st
import os
import json
import datetime
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment # Keep if you plan to assemble audio, otherwise optional for base TTS
import tempfile # For handling service account key file

# --- Google Cloud Imports ---
# Option A: Using Vertex AI (requires service account key or application default credentials)
from google.oauth2 import service_account # For loading service account from secrets
from vertexai import init as vertex_init
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import texttospeech

# Option B: If using google-generativeai SDK (simpler for API key auth from AI Studio)
# import google.generativeai as genai


# --- Configuration & Secret Handling ---
# Load secrets for local development (secrets.toml) or Streamlit Cloud deployment
NEWS_API_KEY_FROM_SECRETS = st.secrets.get("NEWS_API_KEY", os.environ.get("NEWS_API_KEY"))
GCP_PROJECT_ID_FROM_SECRETS = st.secrets.get("GCP_PROJECT_ID", os.environ.get("GCP_PROJECT_ID"))
GCP_LOCATION_FROM_SECRETS = st.secrets.get("GCP_LOCATION", os.environ.get("GCP_LOCATION"))
GCP_SERVICE_ACCOUNT_JSON_STR = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON_CONTENT")

# --- Initialize Google Cloud Services ---
# This should ideally run only once. @st.cache_resource is good for this.
@st.cache_resource
def initialize_gcp_clients():
    clients = {"tts": None, "gemini": None}
    try:
        if GCP_SERVICE_ACCOUNT_JSON_STR and GCP_PROJECT_ID_FROM_SECRETS and GCP_LOCATION_FROM_SECRETS:
            # Create a temporary file for the service account JSON
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_json_file:
                tmp_json_file.write(GCP_SERVICE_ACCOUNT_JSON_STR)
                service_account_json_path = tmp_json_file.name
            
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_json_path
            
            credentials = service_account.Credentials.from_service_account_file(service_account_json_path)
            
            # Initialize Vertex AI
            vertex_init(project=GCP_PROJECT_ID_FROM_SECRETS, location=GCP_LOCATION_FROM_SECRETS, credentials=credentials)
            clients["gemini"] = GenerativeModel("gemini-1.0-pro") # Or your preferred Gemini model

            # Initialize Text-to-Speech client
            clients["tts"] = texttospeech.TextToSpeechClient(credentials=credentials)
            
            # Clean up the temporary file after use (optional, OS might handle it)
            # os.remove(service_account_json_path) # Be careful if other parts of the app still need it
            st.session_state.gcp_initialized = True
            print("GCP Clients Initialized with Service Account.")
            return clients

        # Fallback or alternative: Use Application Default Credentials if GOOGLE_APPLICATION_CREDENTIALS env var is set directly
        # This is common when running locally after `gcloud auth application-default login`
        # or on some cloud environments.
        elif os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and GCP_PROJECT_ID_FROM_SECRETS and GCP_LOCATION_FROM_SECRETS:
            vertex_init(project=GCP_PROJECT_ID_FROM_SECRETS, location=GCP_LOCATION_FROM_SECRETS)
            clients["gemini"] = GenerativeModel("gemini-1.0-pro")
            clients["tts"] = texttospeech.TextToSpeechClient()
            st.session_state.gcp_initialized = True
            print("GCP Clients Initialized with Application Default Credentials.")
            return clients
        else:
            st.error("GCP credentials or project/location info not found in secrets. Please configure them.")
            st.session_state.gcp_initialized = False
            return clients

    except Exception as e:
        st.error(f"Error initializing GCP clients: {e}")
        st.session_state.gcp_initialized = False
        return clients

gcp_clients = initialize_gcp_clients()
gemini_model = gcp_clients.get("gemini")
tts_client = gcp_clients.get("tts")

# --- News Fetching Functions (from previous script, adapted) ---
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news_newsapi(api_key, topics, companies, num_articles=5):
    if not api_key:
        st.error("News API key not configured.")
        return []
    query_parts = []
    if topics:
        query_parts.append(f"({' OR '.join(topics)})")
    if companies:
        query_parts.append(f"({' OR '.join(companies)})")
    
    if not query_parts:
        st.warning("Please provide at least one topic or company.")
        return []

    query_string = " AND ".join(query_parts)
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')
    
    params = {
        'q': query_string,
        'apiKey': api_key,
        'from': yesterday,
        'sortBy': 'relevancy', # relevancy or publishedAt
        'language': 'en',
        'pageSize': num_articles 
    }
    results = []
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get('articles', [])
        for article in articles:
            results.append({
                'title': article['title'],
                'link': article['url'],
                'snippet': article.get('description', '') or article.get('content', '')[:200],
                'source_name': article.get('source', {}).get('name', 'Unknown Source'),
                'published_at': article.get('publishedAt', '')
            })
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news from NewsAPI: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred with NewsAPI: {e}")
    return results

def get_article_text(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'MyNewsPodcastAppStreamlit/1.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p') 
        article_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
        return article_text[:3000] # Limit length for Gemini context
    except Exception as e:
        st.warning(f"Could not fetch/parse article {url}: {e}")
        return ""

# --- Gemini Script Generation (adapted) ---
def generate_podcast_script_gemini(gemini_model_instance, news_items_with_content, topics, companies):
    if not gemini_model_instance:
        st.error("Gemini model not initialized.")
        return "Error: Gemini model not available."

    prompt_parts = [
        "You are a podcast host creating a concise and engaging news summary. Your tone should be informative yet conversational.",
        "Create a podcast script summarizing the key news from the last 24 hours for the following criteria.",
        f"Topics: {', '.join(topics) if topics else 'General News'}",
        f"Companies: {', '.join(companies) if companies else 'Not specified'}\n",
        "The podcast should have a brief, friendly intro (e.g., 'Welcome to your daily news briefing! Here's what's making headlines...'), then cover 2-4 key news items, and a brief outro (e.g., 'That's all for today. Stay informed!').",
        "For each news item, mention the source if available, provide a concise summary (2-4 sentences), and highlight the key takeaway.",
        "Do not use markdown formatting like ** or ## in the script. Just plain text for TTS.",
        "Here is the news content (title, source, snippet, and some extracted text if available):\n"
    ]

    if not news_items_with_content:
        return "I couldn't find enough relevant news for your topics/companies to generate a script."

    for i, item in enumerate(news_items_with_content):
        prompt_parts.append(f"\n--- News Item {i+1} ---")
        prompt_parts.append(f"Title: {item['title']}")
        prompt_parts.append(f"Source: {item.get('source_name', 'N/A')}")
        if item.get('snippet'):
             prompt_parts.append(f"Snippet: {item['snippet']}")
        if item.get('full_text_content'):
            prompt_parts.append(f"Content Highlights: {item['full_text_content'][:1000]}") # Limit length
        elif item.get('link'):
             prompt_parts.append(f"URL: {item['link']}")
    
    prompt_parts.append("\n--- End of News Items ---")
    prompt_parts.append("\nPodcast Script:")
    
    full_prompt = "\n".join(prompt_parts)
    
    try:
        response = gemini_model_instance.generate_content(full_prompt)
        # print(f"DEBUG Gemini Response: {response}") # For debugging
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else: # Handle cases where the response might be empty or malformed
            st.warning(f"Gemini response was not in the expected format or was empty. Raw: {response}")
            return "Error: Gemini did not return a valid script. Check logs."
    except Exception as e:
        st.error(f"Error generating script with Gemini: {e}")
        # print(f"Gemini Request Prompt for Debugging:\n{full_prompt}")
        return "Error generating podcast script."

# --- Text-to-Speech (adapted) ---
def text_to_speech_gcp(tts_client_instance, text, output_filename="podcast_audio.mp3", voice_name="en-US-News-K"):
    if not tts_client_instance:
        st.error("TTS client not initialized.")
        return None
    if not text or "Error" in text:
        st.warning("No valid script to synthesize.")
        return None

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name # e.g., "en-US-Wavenet-D", "en-US-News-K", "en-US-Studio-M"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0 # Adjust as needed
    )
    try:
        response = tts_client_instance.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
        return output_filename
    except Exception as e:
        st.error(f"Error during Text-to-Speech synthesis: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="AI News Podcast Generator", layout="wide")
st.title("🎙️ AI-Powered Daily News Podcast Generator")
st.caption("Get a custom podcast for your topics and companies from the last 24 hours.")

# Initialize session state variables
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = ""
if 'gcp_initialized' not in st.session_state:
    st.session_state.gcp_initialized = False # Will be set by initialize_gcp_clients

# Re-check initialization status
if not st.session_state.gcp_initialized and not (GCP_SERVICE_ACCOUNT_JSON_STR and GCP_PROJECT_ID_FROM_SECRETS and GCP_LOCATION_FROM_SECRETS):
    st.warning("⚠️ Google Cloud services may not be initialized. Please check your secrets configuration if issues persist.")
elif not st.session_state.gcp_initialized: # Attempted but failed
     st.error("⚠️ Failed to initialize Google Cloud services. Podcast generation will not work.")

with st.sidebar:
    st.header("Podcast Configuration")
    raw_topics = st.text_area("Enter Topics (comma-separated)", "artificial intelligence, space exploration, renewable energy")
    raw_companies = st.text_area("Enter Companies (comma-separated)", "SpaceX, OpenAI, Google, Tesla")
    
    num_articles_to_fetch = st.slider("Max news articles to fetch for processing:", 3, 15, 5)
    num_articles_for_script = st.slider("Max articles to include in script:", 1, 5, 3)
    
    # Voice selection (add more as desired from https://cloud.google.com/text-to-speech/docs/voices)
    tts_voice = st.selectbox(
        "Choose Podcast Voice:",
        options=[
            "en-US-News-K", "en-US-News-L", # News voices
            "en-US-Wavenet-D", "en-US-Wavenet-F", # Standard Wavenet
            "en-US-Studio-M", "en-US-Studio-O" # Premium Studio voices
        ],
        index=0 # Default to News-K
    )

    generate_button = st.button("🚀 Generate Podcast", type="primary", disabled=not st.session_state.gcp_initialized)

if generate_button:
    st.session_state.podcast_script = "" # Clear previous script
    st.session_state.audio_file_path = "" # Clear previous audio path

    user_topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
    user_companies = [c.strip() for c in raw_companies.split(',') if c.strip()]

    if not user_topics and not user_companies:
        st.warning("Please enter at least one topic or company.")
    elif not NEWS_API_KEY_FROM_SECRETS:
        st.error("News API Key is not configured. Please set it in your secrets.")
    elif not gemini_model or not tts_client:
        st.error("GCP services (Gemini or TTS) are not initialized. Cannot generate podcast.")
    else:
        with st.spinner("Step 1/4: Fetching latest news... 📰"):
            news_items = fetch_news_newsapi(NEWS_API_KEY_FROM_SECRETS, user_topics, user_companies, num_articles_to_fetch)
        
        if not news_items:
            st.error("No news items found for your criteria.")
        else:
            st.success(f"Found {len(news_items)} news articles.")
            
            processed_news_for_gemini = []
            with st.spinner(f"Step 2/4: Processing article content (max {num_articles_for_script} articles)... 📝"):
                # Select top N articles for script generation based on num_articles_for_script
                articles_to_process = news_items[:num_articles_for_script]
                for item in articles_to_process:
                    st.write(f"  - Processing: {item['title'][:60]}...") # Show progress
                    full_text = get_article_text(item['link'])
                    if full_text:
                        item['full_text_content'] = full_text
                    processed_news_for_gemini.append(item)
            
            if not processed_news_for_gemini:
                st.warning("Could not process content for any articles.")
            else:
                st.success(f"Content processed for {len(processed_news_for_gemini)} articles.")
                with st.spinner("Step 3/4: Generating podcast script with AI... 🤖✍️"):
                    script = generate_podcast_script_gemini(gemini_model, processed_news_for_gemini, user_topics, user_companies)
                    st.session_state.podcast_script = script

                if "Error" in st.session_state.podcast_script or not st.session_state.podcast_script.strip():
                    st.error(f"Failed to generate script: {st.session_state.podcast_script}")
                else:
                    st.subheader("🎧 Generated Podcast Script:")
                    st.text_area("Script", st.session_state.podcast_script, height=300)
                    
                    with st.spinner("Step 4/4: Synthesizing audio... 🔊"):
                        # Ensure the audio filename is unique or overwritten safely
                        audio_output_filename = "generated_podcast.mp3"
                        audio_path = text_to_speech_gcp(tts_client, st.session_state.podcast_script, audio_output_filename, voice_name=tts_voice)
                        if audio_path:
                            st.session_state.audio_file_path = audio_path
                            st.success("Podcast audio generated successfully!")
                        else:
                            st.error("Failed to synthesize audio.")

# Display audio player if an audio file was generated in the current or previous run
if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
    st.subheader("▶️ Listen to your Podcast:")
    audio_file = open(st.session_state.audio_file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    
    with open(st.session_state.audio_file_path, "rb") as file:
        st.download_button(
            label="Download Podcast MP3",
            data=file,
            file_name="my_custom_news_podcast.mp3",
            mime="audio/mp3"
        )
elif st.session_state.podcast_script and not st.session_state.audio_file_path and generate_button:
    # This case means script was generated, but audio failed or was not attempted due to script error
    if "Error" not in st.session_state.podcast_script :
         st.warning("Script was generated, but there was an issue creating the audio.")

st.markdown("---")
st.markdown("Powered by NewsAPI, Google Gemini, and Google Cloud Text-to-Speech. Built with Streamlit.")
