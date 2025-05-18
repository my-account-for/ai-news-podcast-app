import streamlit as st
import datetime
import requests
from bs4 import BeautifulSoup
from openai import OpenAI # OpenAI Python library
import os

# --- THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="AI News Podcast (OpenAI)", layout="wide")

# --- Attempt to Initialize OpenAI Client and Store in Session State ---

# We use @st.cache_resource to ensure the client is created only once
# and persists across reruns, unless the API key changes.
@st.cache_resource
def get_openai_client():
    """
    Initializes and returns the OpenAI client if the API key is available.
    Returns None if initialization fails or key is missing.
    Manages 'openai_init_error' in session state.
    """
    st.session_state.pop('openai_init_error', None) # Clear previous error
    api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        st.session_state.openai_init_error = "OPENAI_API_KEY not found in Streamlit secrets. Please configure it."
        print(st.session_state.openai_init_error) # Log to server
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        # Perform a minimal test call to verify the key and connectivity (optional but good)
        # client.models.list() # This call will raise an AuthenticationError if the key is bad
        print("OpenAI client object created successfully.")
        return client
    except Exception as e:
        error_message = f"Failed to initialize OpenAI client or verify API key: {type(e).__name__} - {e}"
        st.session_state.openai_init_error = error_message
        print(error_message) # Log to server
        return None

# Get the client (or None if it failed)
openai_client = get_openai_client()

# Update session state based on whether the client was successfully obtained
if openai_client:
    st.session_state.openai_initialized = True
else:
    st.session_state.openai_initialized = False

# --- Configuration from Streamlit Secrets (NewsAPI) ---
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")

# --- UI Initialization & Feedback ---
st.title("🎙️ AI News Podcast Generator")
st.caption("Powered by NewsAPI and OpenAI (GPT & TTS)")

# Display initialization status prominently
if st.session_state.get("openai_initialized", False):
    st.sidebar.success("OpenAI Client Connected & Ready.")
else:
    st.sidebar.error("OpenAI Client NOT Initialized.")
    # Display the specific error message if available from initialization
    if 'openai_init_error' in st.session_state:
        st.error(f"Initialization Problem: {st.session_state.openai_init_error}")
    else:
        # This case should ideally be covered by openai_init_error now
        st.error("Problem during OpenAI client setup. Check logs and secrets.")


# --- Session State for app data (script, audio path) ---
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = ""


# --- News Fetching Functions (Same as before - ensure NEWS_API_KEY is handled) ---
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
@st.cache_data(ttl=3600)
def fetch_news_newsapi(topics, companies, num_articles=5):
    # ... (your existing fetch_news_newsapi function using NEWS_API_KEY)
    if not NEWS_API_KEY: # This check is important
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
        response.raise_for_status()
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
    except Exception as e:
        st.error(f"An unexpected error occurred with NewsAPI: {e}")
    return results

@st.cache_data(ttl=3600)
def get_article_text(url):
    # ... (your existing get_article_text function)
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 MyNewsPodcastApp/1.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        main_content_selectors = ['article', 'main', '.main-content', '#main-content',
                                  '.story-content', '.article-body', '#content', '.entry-content']
        content_text = ""
        for selector in main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                paragraphs = main_element.find_all('p')
                content_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
                if len(content_text.split()) > 50:
                    break
        if not content_text:
            paragraphs = soup.find_all('p')
            content_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
        return content_text[:4000]
    except Exception as e:
        st.warning(f"Could not fetch/parse article {url}: {e}")
        return ""

# --- OpenAI Script Generation (Pass the global openai_client) ---
def generate_podcast_script_openai(client_instance, news_items_for_script, topics, companies):
    if not client_instance: # Check the passed client instance
        st.error("OpenAI client instance is not available for script generation.")
        return "Error: OpenAI client instance missing."
    # ... (rest of your generate_podcast_script_openai function is the same)
    news_details_for_prompt = ""
    if not news_items_for_script:
        return "I couldn't find enough relevant news for your topics/companies to generate a script."

    for i, item in enumerate(news_items_for_script):
        news_details_for_prompt += f"\n--- News Item {i+1} ---\n"
        news_details_for_prompt += f"Title: {item['title']}\n"
        news_details_for_prompt += f"Source: {item.get('source_name', 'N/A')}\n"
        if item.get('snippet'):
            news_details_for_prompt += f"Summary: {item['snippet']}\n"
        if item.get('full_text_content'):
            news_details_for_prompt += f"Extracted Content Highlights: {item['full_text_content'][:1000]}\n"

    system_prompt = """You are an engaging podcast host. Your task is to create a concise news summary podcast script.
The tone should be informative, professional, yet conversational and easy to listen to.
The output script must be plain text suitable for Text-to-Speech. Do NOT use markdown (like **, ##, or lists).
Use natural language and paragraph breaks for readability.
Podcast Structure:
1. Intro: A brief, friendly welcome.
2. News Segments: Cover 2-4 distinct news items based on the provided content. For each:
   - Clearly state the headline or main point.
   - Mention the source if available.
   - Provide a succinct summary (2-4 sentences).
   - Highlight a key takeaway.
3. Outro: A brief closing.
"""
    user_prompt = f"""
Please generate a podcast script based on the following criteria and news items.
Primary Topics of Interest: {', '.join(topics) if topics else 'General Tech & Business News'}
Companies to Focus On: {', '.join(companies) if companies else 'Focus on general relevance'}

News items to summarize:
{news_details_for_prompt}

Podcast Script:
"""
    try:
        response = client_instance.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        script_text = response.choices[0].message.content.strip()
        return script_text
    except Exception as e:
        st.error(f"Error generating script with OpenAI: {type(e).__name__} - {e}")
        return "Error generating podcast script."

# --- OpenAI Text-to-Speech (Pass the global openai_client) ---
def text_to_speech_openai(client_instance, text, output_filename="podcast_audio_openai.mp3", voice_model="alloy"):
    if not client_instance: # Check the passed client instance
        st.error("OpenAI client instance is not available for TTS.")
        return None
    # ... (rest of your text_to_speech_openai function is the same)
    if not text or "Error" in text or not text.strip():
        st.warning("No valid script content to synthesize into audio.")
        return None
    try:
        response = client_instance.audio.speech.create(
            model="tts-1",
            voice=voice_model,
            input=text,
            response_format="mp3"
        )
        response.stream_to_file(output_filename)
        return output_filename
    except Exception as e:
        st.error(f"Error during OpenAI Text-to-Speech synthesis: {type(e).__name__} - {e}")
        return None

# --- Streamlit UI Elements (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Podcast Configuration")
    default_topics = "artificial intelligence, space exploration, renewable energy, quantum computing"
    default_companies = "OpenAI, SpaceX, Google, Microsoft, Nvidia"
    
    raw_topics = st.text_area("Enter Topics (comma-separated)", default_topics, height=100, help="Keywords for news articles")
    raw_companies = st.text_area("Enter Companies (comma-separated)", default_companies, height=100, help="Specific company names")
    
    num_articles_to_fetch = st.slider("Max news articles to fetch:", 3, 20, 7, help="How many articles NewsAPI should return.")
    num_articles_for_script = st.slider("Max articles for script content:", 1, 5, 3, help="How many articles will be used for script.")
    
    openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_openai_voice = st.selectbox("Choose OpenAI TTS Voice:", openai_tts_voices, index=0, help="Select the voice for the podcast.")

    # Disable button if OpenAI client isn't initialized
    # Use the st.session_state.openai_initialized flag which is set based on get_openai_client() result
    generate_button = st.button("🚀 Generate Podcast", type="primary",
                                disabled=not st.session_state.get("openai_initialized", False))

# --- Main Application Logic when Button is Clicked ---
if generate_button:
    st.session_state.podcast_script = ""
    st.session_state.audio_file_path = ""

    user_topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
    user_companies = [c.strip() for c in raw_companies.split(',') if c.strip()]

    if not user_topics and not user_companies:
        st.warning("Please enter at least one topic or company.")
    elif not NEWS_API_KEY:
        st.error("News API Key is not configured in Streamlit secrets.")
    elif not openai_client: # Check the global client directly, which should reflect the cached result
        st.error("OpenAI client is not initialized. Cannot generate podcast. Check sidebar for error details.")
    else:
        # ... (rest of your button logic, passing `openai_client` to your functions)
        with st.spinner("Step 1/4: Fetching latest news... 📰"):
            news_items_fetched = fetch_news_newsapi(user_topics, user_companies, num_articles_to_fetch)
        
        if not news_items_fetched:
            st.error("No news items found for your criteria.")
        else:
            st.success(f"Found {len(news_items_fetched)} potentially relevant news articles.")
            articles_to_use_in_script = news_items_fetched[:num_articles_for_script]
            processed_news_for_script_gen = []

            with st.spinner(f"Step 2/4: Extracting content from up to {len(articles_to_use_in_script)} articles... 📝"):
                progress_bar_articles = st.progress(0)
                for i, item in enumerate(articles_to_use_in_script):
                    st.write(f"   L Processing: {item['title'][:70]}...")
                    full_text = get_article_text(item['link'])
                    if full_text:
                        item['full_text_content'] = full_text
                    processed_news_for_script_gen.append(item)
                    progress_bar_articles.progress((i + 1) / len(articles_to_use_in_script))
            
            if not processed_news_for_script_gen:
                st.warning("Could not process any articles for the script.")
                processed_news_for_script_gen = articles_to_use_in_script
            else:
                st.success(f"Content processing complete for {len(processed_news_for_script_gen)} articles.")

            with st.spinner("Step 3/4: Generating podcast script with OpenAI... 🤖✍️"):
                script = generate_podcast_script_openai(openai_client, processed_news_for_script_gen, user_topics, user_companies) # Pass client
                st.session_state.podcast_script = script

            if "Error" in st.session_state.podcast_script or not st.session_state.podcast_script.strip():
                st.error(f"Failed to generate script: {st.session_state.podcast_script}")
            else:
                st.subheader("🎧 Generated Podcast Script:")
                st.text_area("Script Preview:", st.session_state.podcast_script, height=300, key="script_display_area")
                
                with st.spinner("Step 4/4: Synthesizing audio with OpenAI TTS... 🔊"):
                    audio_output_filename = "podcast_output.mp3"
                    audio_path = text_to_speech_openai(openai_client, st.session_state.podcast_script, audio_output_filename, voice_model=selected_openai_voice) # Pass client
                    
                    if audio_path and os.path.exists(audio_path):
                        st.session_state.audio_file_path = audio_path
                        st.success("Podcast audio generated successfully!")
                    else:
                        st.error("Failed to synthesize audio with OpenAI TTS.")

# --- Display Audio Player and Download Button ---
if st.session_state.audio_file_path: # Check if path is set
    try:
        if os.path.exists(st.session_state.audio_file_path):
            with open(st.session_state.audio_file_path, 'rb') as audio_f:
                audio_bytes = audio_f.read()
            st.subheader("▶️ Listen to your Podcast:")
            st.audio(audio_bytes, format='audio/mp3')
            
            st.download_button(
                label="Download Podcast MP3",
                data=audio_bytes,
                file_name="ai_news_podcast.mp3",
                mime="audio/mp3"
            )
        else:
            st.warning("Previously generated audio file not found. Please regenerate.")
            st.session_state.audio_file_path = ""
    except Exception as e:
        st.error(f"Error displaying or downloading audio: {e}")
        st.session_state.audio_file_path = ""

# Footer
st.markdown("---")
st.caption("Built with Streamlit.")
