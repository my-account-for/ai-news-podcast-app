import streamlit as st
import json
import datetime
import requests
from bs4 import BeautifulSoup
# from pydub import AudioSegment # Keep if you assemble audio later, optional for base TTS

# --- Google Cloud Imports ---
from google.oauth2 import service_account
from vertexai import init as vertex_init
from vertexai.preview.generative_models import GenerativeModel
from google.cloud import texttospeech
import google.auth # For checking if ADC is a fallback, though we prioritize secrets

# --- THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="AI News Podcast Generator", layout="wide")

# --- Configuration from Streamlit Secrets ---
# All configurations are expected to be in Streamlit secrets
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
GCP_PROJECT_ID = st.secrets.get("GCP_PROJECT_ID")
GCP_LOCATION = st.secrets.get("GCP_LOCATION")
GCP_SERVICE_ACCOUNT_JSON_STR = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON_CONTENT") # Ensure this key matches your secret name

# --- Initialize Google Cloud Services ---
@st.cache_resource # Caches the initialized clients
def initialize_gcp_clients():
    clients = {"tts": None, "gemini": None}
    
    # Primary method: Service Account JSON from Streamlit Secrets
    if GCP_SERVICE_ACCOUNT_JSON_STR and GCP_PROJECT_ID and GCP_LOCATION:
        try:
            sa_info = json.loads(GCP_SERVICE_ACCOUNT_JSON_STR)
            credentials = service_account.Credentials.from_service_account_info(sa_info)
            
            vertex_init(project=GCP_PROJECT_ID, location=GCP_LOCATION, credentials=credentials)
            clients["gemini"] = GenerativeModel("gemini-1.0-pro") # Or your preferred model
            
            clients["tts"] = texttospeech.TextToSpeechClient(credentials=credentials)
            
            st.session_state.gcp_initialized_method = "Service Account JSON (from Secrets)"
            st.session_state.gcp_initialized_status = True
            print(f"GCP Clients Initialized via Service Account JSON for project {GCP_PROJECT_ID}.")
            return clients
        except json.JSONDecodeError:
            # This st.error will only be shown if the function is called,
            # which happens after st.set_page_config
            st.error("Critical: Failed to parse GCP_SERVICE_ACCOUNT_JSON_CONTENT from Streamlit secrets. Please check its format.")
            st.session_state.gcp_initialized_status = False
            return clients
        except Exception as e:
            st.error(f"Critical: Error initializing GCP clients with Service Account JSON: {e}")
            st.session_state.gcp_initialized_status = False
            return clients
    
    # If SA JSON is not provided or failed, but project/location are,
    # we could try ADC, but for pure "Streamlit Secrets" approach, we'll consider this a config error.
    elif GCP_PROJECT_ID and GCP_LOCATION:
        # This part is a potential fallback, but if the intention is "all in secrets",
        # then the absence of GCP_SERVICE_ACCOUNT_JSON_STR is the primary issue.
        # For simplicity and strict adherence to "all in secrets", we can make this path less prominent
        # or assume it's a misconfiguration if SA JSON is expected.
        try:
            google.auth.default() # Check if ADC are available (e.g. local dev with gcloud login)
            vertex_init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
            clients["gemini"] = GenerativeModel("gemini-1.0-pro")
            clients["tts"] = texttospeech.TextToSpeechClient()
            st.session_state.gcp_initialized_method = "Application Default Credentials (ADC)"
            st.session_state.gcp_initialized_status = True
            print(f"GCP Clients Initialized via ADC for project {GCP_PROJECT_ID} (SA JSON not found/used).")
            return clients
        except Exception as e:
            st.warning(f"Service Account JSON not found or invalid. Attempted ADC, but failed or not configured: {e}")
            # Fall through to the final error message
            pass


    # If neither method worked or essential configs are missing:
    if not st.session_state.get("gcp_initialized_status"): # Check if status was ever set to true
        missing_configs = []
        if not GCP_SERVICE_ACCOUNT_JSON_STR: missing_configs.append("GCP_SERVICE_ACCOUNT_JSON_CONTENT")
        if not GCP_PROJECT_ID: missing_configs.append("GCP_PROJECT_ID")
        if not GCP_LOCATION: missing_configs.append("GCP_LOCATION")
        
        if missing_configs:
            st.error(f"Critical: GCP configuration missing in Streamlit secrets: {', '.join(missing_configs)}. Podcast generation disabled.")
        else: # Should ideally not happen if the logic above is correct, but as a fallback
            st.error("Critical: GCP services could not be initialized. Please check Streamlit secrets and app logs. Podcast generation disabled.")
        st.session_state.gcp_initialized_status = False
    return clients

# Initialize session state for GCP status tracking
if 'gcp_initialized_status' not in st.session_state:
    st.session_state.gcp_initialized_status = False # Default to false
if 'gcp_initialized_method' not in st.session_state:
    st.session_state.gcp_initialized_method = "None"

# --- Actually call the initialization function ---
# This call will populate st.session_state.gcp_initialized_status and method
gcp_clients = initialize_gcp_clients()
gemini_model = gcp_clients.get("gemini")
tts_client = gcp_clients.get("tts")


# --- Streamlit UI ---
st.title("🎙️ AI-Powered Daily News Podcast Generator")
st.caption("Get a custom podcast for your topics and companies from the last 24 hours.")

# UI feedback based on initialization status
if st.session_state.gcp_initialized_status:
    st.sidebar.success(f"GCP Connected ({st.session_state.gcp_initialized_method})")
else:
    st.sidebar.error("GCP Not Connected. Podcast generation disabled.")
    # If critical secrets are missing, the error messages above will also be prominent.

# Initialize other session state variables
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = ""

# --- News Fetching Functions ---
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_news_newsapi(topics, companies, num_articles=5):
    if not NEWS_API_KEY:
        st.error("News API key (NEWS_API_KEY) not configured in Streamlit secrets.")
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
    # Get news from up to 24 hours ago
    yesterday = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    params = {
        'q': query_string,
        'apiKey': NEWS_API_KEY,
        'from': yesterday,
        'sortBy': 'relevancy', 
        'language': 'en',
        'pageSize': num_articles 
    }
    results = []
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=10)
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        articles = response.json().get('articles', [])
        for article in articles:
            results.append({
                'title': article['title'],
                'link': article['url'],
                'snippet': article.get('description', '') or article.get('content', '')[:200] if article.get('content') else '',
                'source_name': article.get('source', {}).get('name', 'Unknown Source'),
                'published_at': article.get('publishedAt', '')
            })
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news from NewsAPI: {e}")
    except Exception as e: # Catch other potential errors like JSONDecodeError if API response is weird
        st.error(f"An unexpected error occurred with NewsAPI: {e}")
    return results

def get_article_text(url):
    try:
        # Added a common user-agent to potentially avoid some simple blocks
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'}) 
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find common main content containers first
        main_content_selectors = ['article', 'main', '.main-content', '#main-content', 
                                  '.story-content', '.article-body', '#content']
        content_text = ""
        for selector in main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                paragraphs = main_element.find_all('p')
                content_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
                break # Found good content, stop searching
        
        if not content_text: # Fallback to all <p> tags if specific containers not found
            paragraphs = soup.find_all('p') 
            content_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
            
        return content_text[:4000] # Limit length for Gemini context (increased slightly)
    except Exception as e:
        st.warning(f"Could not fetch/parse article {url}: {e}")
        return ""

# --- Gemini Script Generation ---
def generate_podcast_script_gemini(news_items_with_content, topics, companies):
    if not gemini_model: # Check if the global model instance is available
        st.error("Gemini model not initialized. Check GCP configuration in secrets.")
        return "Error: Gemini model not available."

    prompt_parts = [
        "You are an engaging podcast host. Your task is to create a concise news summary podcast script.",
        "The tone should be informative, professional, yet conversational and easy to listen to.",
        "Summarize the key news from the provided articles based on the following criteria:",
        f"Primary Topics of Interest: {', '.join(topics) if topics else 'General Tech & Business News'}",
        f"Companies to Focus On: {', '.join(companies) if companies else 'Not specified, focus on general relevance'}\n",
        "Podcast Structure:",
        "1. Intro: A brief, friendly welcome (e.g., 'Welcome to your AI-powered news brief! Here’s what’s trending...').",
        "2. News Segments: Cover 2-4 distinct news items. For each:",
        "   - Clearly state the headline or main point.",
        "   - Mention the source if available (e.g., 'According to TechCrunch...').",
        "   - Provide a succinct summary (2-4 sentences) explaining the 'what' and 'why it matters'.",
        "   - Highlight a key takeaway or implication.",
        "3. Outro: A brief closing (e.g., 'That’s your news update for today. Stay informed and tune in next time!').",
        "IMPORTANT: The output script must be plain text suitable for Text-to-Speech. Do NOT use markdown (like **, ##, or lists). Use natural language.",
        "Here is the news content (title, source, snippet, and some extracted text if available):\n"
    ]

    if not news_items_with_content:
        return "I couldn't find enough relevant news for your topics/companies to generate a script after processing."

    for i, item in enumerate(news_items_with_content):
        prompt_parts.append(f"\n--- News Item {i+1} ---")
        prompt_parts.append(f"Title: {item['title']}")
        prompt_parts.append(f"Source: {item.get('source_name', 'N/A')}")
        if item.get('snippet'):
             prompt_parts.append(f"Snippet: {item['snippet']}")
        if item.get('full_text_content'): # Prioritize full text if available
            prompt_parts.append(f"Extracted Content: {item['full_text_content'][:1500]}") # Limit for prompt
        elif item.get('link'): # Fallback to link if no full text
             prompt_parts.append(f"URL: {item['link']} (content extraction might have failed or was skipped)")
    
    prompt_parts.append("\n--- End of News Items ---")
    prompt_parts.append("\nNow, generate the podcast script based on these instructions and news items:")
    
    full_prompt = "\n".join(prompt_parts)
    
    try:
        # Make sure gemini_model is the actual GenerativeModel instance
        response = gemini_model.generate_content(full_prompt)
        if response.candidates and response.candidates[0].content.parts:
            script_text = response.candidates[0].content.parts[0].text
            # Basic cleanup: remove potential markdown-like list indicators if Gemini adds them
            script_text = script_text.replace("*- ", "").replace("* ", "").replace("- ", "")
            return script_text
        else: 
            st.warning(f"Gemini response was not in the expected format or was empty. Raw response: {response}")
            return "Error: Gemini did not return a valid script. Please check the logs and try adjusting the prompt or input articles."
    except Exception as e:
        st.error(f"Error generating script with Gemini: {e}")
        # For debugging, you might want to print the full_prompt if running locally
        # print(f"DEBUG: Gemini Request Prompt:\n{full_prompt}")
        return "Error: An exception occurred while generating the podcast script."

# --- Text-to-Speech ---
def text_to_speech_gcp(text, output_filename="podcast_audio.mp3", voice_name="en-US-News-K"):
    if not tts_client: # Check if the global tts_client instance is available
        st.error("Text-to-Speech client not initialized. Check GCP configuration in secrets.")
        return None
    if not text or "Error" in text or not text.strip():
        st.warning("No valid script content to synthesize into audio.")
        return None

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name 
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.0, # Adjust between 0.25 and 4.0
        pitch=0.0 # Adjust between -20.0 and 20.0
    )
    try:
        response = tts_client.synthesize_speech(
            request={"input": input_text, "voice": voice, "audio_config": audio_config}
        )
        # Save the audio to a file (Streamlit Cloud has an ephemeral filesystem)
        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
        return output_filename # Return the path to the generated file
    except Exception as e:
        st.error(f"Error during Text-to-Speech synthesis: {e}")
        return None
        
# --- Streamlit UI Elements (Sidebar and Main Area) ---
with st.sidebar:
    st.header("🎙️ Podcast Configuration")
    default_topics = "artificial intelligence trends, large language models, generative AI applications, tech ethics"
    default_companies = "OpenAI, Google, Microsoft, Nvidia, Anthropic"
    
    raw_topics = st.text_area("Enter Topics (comma-separated)", default_topics, height=100)
    raw_companies = st.text_area("Enter Companies (comma-separated)", default_companies, height=100)
    
    num_articles_to_fetch = st.slider("Max news articles to fetch for processing:", 3, 20, 7, help="How many articles to initially get from NewsAPI.")
    num_articles_for_script = st.slider("Max articles to include in script:", 1, 5, 3, help="How many of the fetched articles will be summarized in the podcast.")
    
    tts_voice_options = {
        "News K (US Male)": "en-US-News-K",
        "News L (US Male)": "en-US-News-L",
        "News M (US Female)": "en-US-News-M", # Added a female news voice
        "Wavenet D (US Male)": "en-US-Wavenet-D",
        "Wavenet F (US Female)": "en-US-Wavenet-F",
        "Studio M (US Male - Premium)": "en-US-Studio-M",
        "Studio O (US Female - Premium)": "en-US-Studio-O"
    }
    selected_voice_friendly_name = st.selectbox(
        "Choose Podcast Voice:",
        options=list(tts_voice_options.keys()),
        index=0 # Default to News K
    )
    tts_voice_code = tts_voice_options[selected_voice_friendly_name]

    # The button is disabled if GCP services are not initialized (status is False)
    generate_button = st.button("🚀 Generate Podcast", type="primary", disabled=not st.session_state.gcp_initialized_status)

# --- Main Application Logic when Button is Clicked ---
if generate_button:
    st.session_state.podcast_script = "" 
    st.session_state.audio_file_path = "" 

    user_topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
    user_companies = [c.strip() for c in raw_companies.split(',') if c.strip()]

    # --- Input Validations ---
    if not user_topics and not user_companies:
        st.warning("Please enter at least one topic or company to focus the news search.")
    elif not NEWS_API_KEY:
        st.error("News API Key is not configured in Streamlit secrets. Please add `NEWS_API_KEY`.")
    # The gcp_initialized_status check on the button itself handles GCP client issues,
    # but we can double-check the model variables just in case.
    elif not gemini_model or not tts_client:
        st.error("GCP services (Gemini or TTS) are not properly initialized. This usually means a problem with GCP secrets (Project ID, Location, or Service Account JSON). Please check the error messages at the top or in the sidebar.")
    else:
        # --- Step 1: Fetch News ---
        with st.spinner("Step 1/4: Fetching latest news... 📰"):
            news_items = fetch_news_newsapi(user_topics, user_companies, num_articles_to_fetch)
        
        if not news_items:
            st.error("No news items found for your criteria. Try broadening your topics/companies or check NewsAPI.")
        else:
            st.success(f"Found {len(news_items)} potentially relevant news articles.")
            
            # --- Step 2: Process Article Content ---
            processed_news_for_gemini = []
            articles_to_process_for_script = news_items[:num_articles_for_script] # Select top N for detailed processing

            with st.spinner(f"Step 2/4: Extracting content from {len(articles_to_process_for_script)} articles... 📝"):
                # Display sub-progress for article fetching
                progress_bar_articles = st.progress(0)
                for i, item in enumerate(articles_to_process_for_script):
                    st.write(f"   L Processing: {item['title'][:70]}...") 
                    full_text = get_article_text(item['link'])
                    if full_text:
                        item['full_text_content'] = full_text # Add extracted content to the item
                    processed_news_for_gemini.append(item)
                    progress_bar_articles.progress((i + 1) / len(articles_to_process_for_script))
            
            if not processed_news_for_gemini:
                st.warning("Could not process content for any of the selected articles. The script might be less detailed.")
            else:
                st.success(f"Content processed for {len(processed_news_for_gemini)} articles.")

            # --- Step 3: Generate Podcast Script ---
            with st.spinner("Step 3/4: Generating podcast script with Gemini AI... 🤖✍️"):
                script = generate_podcast_script_gemini(processed_news_for_gemini, user_topics, user_companies)
                st.session_state.podcast_script = script

            if "Error" in st.session_state.podcast_script or not st.session_state.podcast_script.strip():
                st.error(f"Failed to generate script: {st.session_state.podcast_script}")
            else:
                st.subheader("🎧 Generated Podcast Script:")
                st.text_area("Script Preview:", st.session_state.podcast_script, height=300, key="script_display")
                
                # --- Step 4: Synthesize Audio ---
                with st.spinner("Step 4/4: Synthesizing audio with Google Cloud TTS... 🔊"):
                    audio_output_filename = "generated_podcast_news.mp3" # Use a consistent filename
                    audio_path = text_to_speech_gcp(st.session_state.podcast_script, audio_output_filename, voice_name=tts_voice_code)
                    if audio_path:
                        st.session_state.audio_file_path = audio_path
                        st.success("Podcast audio generated successfully!")
                    else:
                        st.error("Failed to synthesize audio. Check TTS client initialization and script content.")

# --- Display Audio Player and Download Button (if audio exists) ---
# This part runs on every rerun if audio_file_path is set and the file exists
if st.session_state.audio_file_path: # Check if path is set
    try:
        with open(st.session_state.audio_file_path, 'rb') as audio_f:
            audio_bytes = audio_f.read()
        st.subheader("▶️ Listen to your Podcast:")
        st.audio(audio_bytes, format='audio/mp3')
        
        # Provide download button
        st.download_button(
            label="Download Podcast MP3",
            data=audio_bytes, # Use the bytes directly
            file_name="ai_news_podcast.mp3", # Consistent download filename
            mime="audio/mp3"
        )
    except FileNotFoundError:
        st.warning("Audio file was generated previously but seems to be missing now. Please regenerate.")
        st.session_state.audio_file_path = "" # Clear the path
    except Exception as e:
        st.error(f"Error displaying or downloading audio: {e}")
        st.session_state.audio_file_path = ""


# Footer
st.markdown("---")
st.markdown("Powered by NewsAPI, Google Gemini, and Google Cloud Text-to-Speech. Built with Streamlit.")
