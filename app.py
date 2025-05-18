import streamlit as st
import datetime
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import google.generativeai as genai
from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch # For grounding
import os
import json # Still useful for debugging API responses sometimes

# --- THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="AI News Podcast (Gemini Search & OpenAI TTS)", layout="wide")

# --- Initialize Gemini Client (using API Key from AI Studio) ---
gemini_model_instance = None
if 'gemini_initialized_status' not in st.session_state:
    st.session_state.gemini_initialized_status = False

if not st.session_state.gemini_initialized_status:
    GEMINI_API_KEY_FROM_SECRETS = st.secrets.get("GEMINI_API_KEY")
    if GEMINI_API_KEY_FROM_SECRETS:
        try:
            genai.configure(api_key=GEMINI_API_KEY_FROM_SECRETS)
            # Choose a model that supports the search tool well.
            # "gemini-1.5-pro-latest" or "gemini-1.5-flash-latest" are good candidates.
            MODEL_NAME_FOR_GEMINI = "gemini-1.5-flash-latest" # Or gemini-1.5-pro-latest
            gemini_model_instance = genai.GenerativeModel(MODEL_NAME_FOR_GEMINI)
            st.session_state.gemini_initialized_status = True
            print(f"Gemini Client & Model ({MODEL_NAME_FOR_GEMINI}) Initialized with API Key.")
        except Exception as e:
            st.session_state.gemini_init_error = f"Failed to initialize Gemini client/model with API Key: {type(e).__name__} - {e}"
            print(st.session_state.gemini_init_error)
            st.session_state.gemini_initialized_status = False
    else:
        st.session_state.gemini_init_error = "GEMINI_API_KEY not found in Streamlit secrets."
        print(st.session_state.gemini_init_error)
        st.session_state.gemini_initialized_status = False

# --- Initialize OpenAI Client (for TTS) ---
openai_client = None
if 'openai_initialized_status' not in st.session_state:
    st.session_state.openai_initialized_status = False

if not st.session_state.openai_initialized_status:
    OPENAI_API_KEY_FROM_SECRETS = st.secrets.get("OPENAI_API_KEY")
    if OPENAI_API_KEY_FROM_SECRETS:
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY_FROM_SECRETS)
            st.session_state.openai_initialized_status = True
            print("OpenAI Client (for TTS) Initialized.")
        except Exception as e:
            st.session_state.openai_tts_init_error = f"Failed to initialize OpenAI client for TTS: {type(e).__name__} - {e}"
            print(st.session_state.openai_tts_init_error)
    else:
        st.session_state.openai_tts_init_error = "OPENAI_API_KEY for TTS not found in secrets."
        print(st.session_state.openai_tts_init_error)

# --- UI Status Indication ---
st.title("🎙️ AI News Podcast Generator")
st.caption("News via Gemini with Google Search, Speech by OpenAI TTS")

with st.sidebar:
    st.header("🚦 Initialization Status")
    if st.session_state.get("gemini_initialized_status", False) and gemini_model_instance:
        st.success("Gemini Client & Model Ready.")
    else:
        st.error("Gemini Client FAILED.")
        if 'gemini_init_error' in st.session_state:
            st.caption(f"Error: {st.session_state.gemini_init_error}")

    if st.session_state.get("openai_initialized_status", False) and openai_client:
        st.success("OpenAI Client (for TTS) Ready.")
    else:
        st.error("OpenAI Client (for TTS) FAILED.")
        if 'openai_tts_init_error' in st.session_state:
            st.caption(f"Error: {st.session_state.openai_tts_init_error}")

# --- Session State for app data ---
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = ""
if 'search_suggestions_html' not in st.session_state:
    st.session_state.search_suggestions_html = None
if 'cited_articles_for_display' not in st.session_state:
    st.session_state.cited_articles_for_display = [] # For displaying sources

# --- News Fetching and Script Generation with Gemini Search ---
@st.cache_data(ttl=1800)
def get_news_script_via_gemini_search(model_instance_to_use, topics_list, companies_list, num_articles_target=3):
    if not model_instance_to_use: # Check if the model instance was passed correctly
        return "Error: Gemini model instance not provided to search function.", None, []

    topic_str = ", ".join(topics_list) if topics_list else "current global events"
    company_str = ", ".join(companies_list) if companies_list else "major relevant companies"

    prompt_content = f"""
    You are an expert news summarizer and podcast script writer.
    Your task is to generate a concise and engaging podcast script based on the latest news (within the last 24-48 hours)
    related to topics: "{topic_str}" and companies: "{company_str}".
    Aim to cover about {num_articles_target} key news items.

    The podcast script should have:
    1. A brief, friendly introduction.
    2. For each news item: a clear headline, a 2-3 sentence summary explaining its significance, and mention the primary source if apparent from your search.
    3. A brief, engaging outro.

    Please ensure the entire output is plain text, suitable for direct Text-to-Speech conversion.
    Do not use markdown formatting like **, ##, or lists. Use natural paragraph breaks.

    Begin the podcast script now:
    """

    google_search_tool_instance = Tool(google_search=GoogleSearch()) # Create tool instance
    # Ensure the model you chose supports tools configuration this way.
    # Some models might need it in safety_settings or other config.
    # The docs for "Search as a tool" show it in GenerateContentConfig.
    config_for_generation = GenerateContentConfig(
        tools=[google_search_tool_instance],
        temperature=0.6
    )

    try:
        print(f"DEBUG: Sending prompt to Gemini for search & script: Topics='{topic_str}', Companies='{company_str}'")
        response = model_instance_to_use.generate_content(
            contents=prompt_content,
            generation_config=config_for_generation
        )
        print(f"DEBUG: Gemini Response received. Candidate count: {len(response.candidates) if hasattr(response, 'candidates') else 'N/A'}")

        if not response.candidates:
            # Check for block reason if no candidates
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                block_reason_msg = f"Content generation blocked. Reason: {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
                print(f"DEBUG: {block_reason_msg}")
                return f"Error: {block_reason_msg}", None, []
            return "Error: Gemini returned no candidates.", None, []

        candidate = response.candidates[0]
        script_text = ""
        # Iterate through parts of the content, as it can be multi-part
        for part in candidate.content.parts:
            if hasattr(part, 'text'): # Ensure part has text attribute
                script_text += part.text
        
        search_suggestions_html_output = None
        cited_articles_output = []

        if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
            print("DEBUG: Grounding metadata found in Gemini response.")
            if hasattr(candidate.grounding_metadata, 'search_entry_point') and \
               candidate.grounding_metadata.search_entry_point and \
               hasattr(candidate.grounding_metadata.search_entry_point, 'rendered_content'):
                search_suggestions_html_output = candidate.grounding_metadata.search_entry_point.rendered_content
                print(f"DEBUG: Search Suggestions HTML (first 100 chars): {search_suggestions_html_output[:100] if search_suggestions_html_output else 'None'}")
            
            if hasattr(candidate.grounding_metadata, 'grounding_chunks') and candidate.grounding_metadata.grounding_chunks:
                for chunk in candidate.grounding_metadata.grounding_chunks:
                    if hasattr(chunk, 'web') and chunk.web: # Check if it's a web chunk
                        title = getattr(chunk.web, 'title', "Unknown Title")
                        uri = getattr(chunk.web, 'uri', "#") # This is a redirect URI
                        cited_articles_output.append({
                            'title': title,
                            'link': uri, 
                            'source_name': title # The title often indicates source
                        })
                print(f"DEBUG: Extracted {len(cited_articles_output)} cited articles from grounding chunks.")
        else:
            print("DEBUG: No grounding metadata found in Gemini response. Model may have answered from its own knowledge or search was not triggered/successful.")

        if not script_text.strip() and not cited_articles_output : # If both script is empty AND no sources, likely a bigger issue.
             return "Gemini generated an empty script and found no search results, possibly due to query constraints or content filters.", search_suggestions_html_output, cited_articles_output
        elif not script_text.strip() and cited_articles_output:
            script_text = "The model found some search results but did not generate a script. Please check the cited sources."


        return script_text.strip(), search_suggestions_html_output, cited_articles_output

    except Exception as e:
        error_msg = f"Error during Gemini search & script generation: {type(e).__name__} - {e}"
        print(f"CRITICAL DEBUG: {error_msg}")
        # Try to see if there's more detail in the response object itself if it exists
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            print(f"DEBUG: Gemini Prompt Feedback: {response.prompt_feedback}")
        return f"Error: {error_msg}", None, []


# --- OpenAI Text-to-Speech (Same as before, uses openai_client) ---
def text_to_speech_openai(client_instance_for_tts, text_to_speak, output_filename="podcast_audio_openai.mp3", voice_model_for_tts="alloy"):
    if not client_instance_for_tts:
        st.error("OpenAI client (for TTS) instance is not available.")
        return None
    if not text_to_speak or "Error" in text_to_speak or not text_to_speak.strip():
        st.warning("No valid script content to synthesize into audio for OpenAI TTS.")
        return None
    try:
        response_tts = client_instance_for_tts.audio.speech.create(
            model="tts-1", voice=voice_model_for_tts, input=text_to_speak, response_format="mp3"
        )
        response_tts.stream_to_file(output_filename)
        return output_filename
    except Exception as e:
        st.error(f"Error during OpenAI Text-to-Speech synthesis: {type(e).__name__} - {e}")
        return None

# --- Streamlit UI Elements (Sidebar for Config) ---
with st.sidebar:
    st.header("⚙️ Podcast Configuration")
    default_topics = "latest AI breakthroughs, space technology advancements"
    default_companies = "Nvidia, OpenAI, SpaceX"
    
    raw_topics = st.text_area("Enter Topics (comma-separated)", default_topics, height=100)
    raw_companies = st.text_area("Enter Companies (comma-separated)", default_companies, height=100)
    
    num_articles_target_for_script = st.slider("Target news items in script:", 1, 5, 3)
    
    openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_openai_voice = st.selectbox("Choose OpenAI TTS Voice:", openai_tts_voices, index=0)

    # Disable button if clients aren't initialized
    generate_button_disabled = not (
        st.session_state.get("gemini_initialized_status", False) and \
        st.session_state.get("openai_initialized_status", False) and \
        gemini_model_instance and openai_client # Check actual client instances
    )
    generate_button = st.button("🚀 Generate Podcast", type="primary", disabled=generate_button_disabled)

# --- Main Application Logic ---
if generate_button:
    st.session_state.podcast_script = ""
    st.session_state.audio_file_path = ""
    st.session_state.search_suggestions_html = None
    st.session_state.cited_articles_for_display = []

    user_topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
    user_companies = [c.strip() for c in raw_companies.split(',') if c.strip()]

    # No specific topic/company check here, let Gemini try with broad terms if empty
    # if not user_topics and not user_companies:
    #     st.warning("No specific topics/companies entered. Gemini will try to find general news.")

    with st.spinner("Step 1: Asking Gemini to research news & write script... 🤖📰✍️"):
        script, suggestions_html, cited_articles = get_news_script_via_gemini_search(
            gemini_model_instance, user_topics, user_companies, num_articles_target_for_script
        )
        st.session_state.podcast_script = script
        st.session_state.search_suggestions_html = suggestions_html
        st.session_state.cited_articles_for_display = cited_articles

    if "Error" in st.session_state.podcast_script or not st.session_state.podcast_script.strip():
        # The error message from get_news_script_via_gemini_search is already in podcast_script
        st.error(f"Script Generation Failed: {st.session_state.podcast_script}")
    else:
        st.subheader("🎧 Generated Podcast Script:")
        st.text_area("Script Preview:", st.session_state.podcast_script, height=250, key="script_display_final")

        if st.session_state.search_suggestions_html:
            st.subheader("🔎 Google Search Suggestions (from Gemini):")
            st.markdown(st.session_state.search_suggestions_html, unsafe_allow_html=True)
        
        if st.session_state.cited_articles_for_display:
            st.subheader("📚 Sources potentially consulted by Gemini:")
            for i, article_cite in enumerate(st.session_state.cited_articles_for_display):
                st.markdown(f"- {article_cite.get('title', 'Unknown Source')}")
            st.caption("Note: Links are via Google's grounding service.")

        with st.spinner("Step 2: Synthesizing audio with OpenAI TTS... 🔊"):
            audio_output_filename = "podcast_output_audio.mp3"
            audio_path = text_to_speech_openai(openai_client, st.session_state.podcast_script, audio_output_filename, voice_model=selected_openai_voice)
            
            if audio_path and os.path.exists(audio_path):
                st.session_state.audio_file_path = audio_path
                st.success("Podcast audio generated successfully!")
            else:
                st.error("Failed to synthesize audio with OpenAI TTS.")

# --- Display Audio Player and Download Button ---
if st.session_state.audio_file_path:
    try:
        if os.path.exists(st.session_state.audio_file_path):
            with open(st.session_state.audio_file_path, 'rb') as audio_f:
                audio_bytes = audio_f.read()
            st.subheader("▶️ Listen to your Podcast:")
            st.audio(audio_bytes, format='audio/mp3')
            st.download_button(
                label="Download Podcast MP3",
                data=audio_bytes,
                file_name="ai_news_podcast_gemini_search.mp3",
                mime="audio/mp3"
            )
        else:
            st.warning("Audio file seems to be missing. Please regenerate.")
            st.session_state.audio_file_path = ""
    except Exception as e:
        st.error(f"Error displaying or downloading audio: {e}")
        st.session_state.audio_file_path = ""

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Gemini API with Google Search, and OpenAI TTS.")
