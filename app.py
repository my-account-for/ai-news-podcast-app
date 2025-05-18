import streamlit as st
import datetime
import requests
from bs4 import BeautifulSoup
from openai import OpenAI # OpenAI Python library
import os # For saving the audio file temporarily

# --- THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="AI News Podcast (OpenAI)", layout="wide")

# --- Configuration from Streamlit Secrets ---
# These keys will be read from Streamlit's secret management when deployed.
# For local testing, you might set them as environment variables or temporarily hardcode
# them (but remove before committing/deploying!).
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# --- Initialize OpenAI Client ---
# This client will be used for both text generation (script) and TTS
openai_client = None
if 'openai_initialized' not in st.session_state: # Initialize only once
    st.session_state.openai_initialized = False

if not st.session_state.openai_initialized and OPENAI_API_KEY: # Check if already initialized
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        st.session_state.openai_initialized = True
        print("OpenAI Client Initialized.") # For server logs
    except Exception as e:
        # This error will show up in the app UI if initialization fails
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.session_state.openai_initialized = False
elif not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in Streamlit secrets. Please configure it.")
    st.session_state.openai_initialized = False


# --- News Fetching Functions ---
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

@st.cache_data(ttl=3600) # Cache news data for 1 hour
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
        'sortBy': 'relevancy', # or 'publishedAt', 'popularity'
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
    except Exception as e:
        st.error(f"An unexpected error occurred with NewsAPI: {e}")
    return results

@st.cache_data(ttl=3600) # Cache article text for 1 hour
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0 MyNewsPodcastApp/1.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Try to find common main content containers first
        main_content_selectors = ['article', 'main', '.main-content', '#main-content',
                                  '.story-content', '.article-body', '#content', '.entry-content']
        content_text = ""
        for selector in main_content_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                paragraphs = main_element.find_all('p')
                content_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
                if len(content_text.split()) > 50: # Heuristic: if it has more than 50 words
                    break # Found good content, stop searching
        
        if not content_text: # Fallback to all <p> tags if specific containers not found or text too short
            paragraphs = soup.find_all('p')
            content_text = ' '.join([para.get_text(strip=True) for para in paragraphs])
            
        return content_text[:4000] # Limit length for API context
    except Exception as e:
        st.warning(f"Could not fetch/parse article {url}: {e}")
        return ""

# --- OpenAI Script Generation ---
def generate_podcast_script_openai(client, news_items_for_script, topics, companies):
    if not client or not st.session_state.get("openai_initialized", False):
        st.error("OpenAI client not initialized. Cannot generate script.")
        return "Error: OpenAI client not available."

    news_details_for_prompt = ""
    if not news_items_for_script:
        return "I couldn't find enough relevant news for your topics/companies to generate a script."

    for i, item in enumerate(news_items_for_script):
        news_details_for_prompt += f"\n--- News Item {i+1} ---\n"
        news_details_for_prompt += f"Title: {item['title']}\n"
        news_details_for_prompt += f"Source: {item.get('source_name', 'N/A')}\n"
        if item.get('snippet'): # Use snippet if full text is not available or too short
            news_details_for_prompt += f"Summary: {item['snippet']}\n"
        if item.get('full_text_content'):
            news_details_for_prompt += f"Extracted Content Highlights: {item['full_text_content'][:1000]}\n" # Limit length
    
    system_prompt = """You are an engaging podcast host. Your task is to create a concise news summary podcast script.
The tone should be informative, professional, yet conversational and easy to listen to.
The output script must be plain text suitable for Text-to-Speech. Do NOT use markdown (like **, ##, or lists).
Use natural language and paragraph breaks for readability.
Podcast Structure:
1. Intro: A brief, friendly welcome (e.g., 'Welcome to your AI-powered news brief! Here’s what’s trending...').
2. News Segments: Cover 2-4 distinct news items based on the provided content. For each:
   - Clearly state the headline or main point.
   - Mention the source if available (e.g., 'According to TechCrunch...').
   - Provide a succinct summary (2-4 sentences) explaining the 'what' and 'why it matters'.
   - Highlight a key takeaway or implication.
3. Outro: A brief closing (e.g., 'That’s your news update for today. Stay informed and tune in next time!').
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125", # A capable and cost-effective model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7 # A balance between creativity and factualness
        )
        script_text = response.choices[0].message.content.strip()
        return script_text
    except Exception as e:
        st.error(f"Error generating script with OpenAI: {e}")
        return "Error generating podcast script."

# --- OpenAI Text-to-Speech ---
def text_to_speech_openai(client, text, output_filename="podcast_audio_openai.mp3", voice_model="alloy"):
    if not client or not st.session_state.get("openai_initialized", False):
        st.error("OpenAI client not initialized. Cannot synthesize audio.")
        return None
    if not text or "Error" in text or not text.strip():
        st.warning("No valid script content to synthesize into audio.")
        return None

    try:
        # Available voices: alloy, echo, fable, onyx, nova, shimmer
        # Available models: tts-1, tts-1-hd
        response = client.audio.speech.create(
            model="tts-1", # or "tts-1-hd" for higher quality
            voice=voice_model,
            input=text,
            response_format="mp3"
        )
        # Stream the audio to a file. Streamlit Cloud has an ephemeral filesystem.
        response.stream_to_file(output_filename)
        return output_filename
    except Exception as e:
        st.error(f"Error during OpenAI Text-to-Speech synthesis: {e}")
        return None

# --- Streamlit UI ---
st.title("🎙️ AI News Podcast Generator")
st.caption("Powered by NewsAPI and OpenAI (GPT & TTS)")

# UI feedback for OpenAI client initialization
if st.session_state.get("openai_initialized", False):
    st.sidebar.success("OpenAI Client Connected.")
else:
    # Error messages for missing API key are handled during client init
    st.sidebar.error("OpenAI Client Not Connected. Check API Key in secrets.")

# Initialize session state variables for script and audio path
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = ""

with st.sidebar:
    st.header("⚙️ Podcast Configuration")
    default_topics = "artificial intelligence, space exploration, renewable energy, quantum computing"
    default_companies = "OpenAI, SpaceX, Google, Microsoft, Nvidia"
    
    raw_topics = st.text_area("Enter Topics (comma-separated)", default_topics, height=100,
                              help="Keywords for news articles (e.g., AI, climate change)")
    raw_companies = st.text_area("Enter Companies (comma-separated)", default_companies, height=100,
                                 help="Specific company names to look for (e.g., Apple, Tesla)")
    
    num_articles_to_fetch = st.slider("Max news articles to fetch:", 3, 20, 7,
                                      help="How many articles NewsAPI should return.")
    num_articles_for_script = st.slider("Max articles for script content:", 1, 5, 3,
                                        help="How many of the fetched articles will be used to create the script content.")
    
    openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_openai_voice = st.selectbox("Choose OpenAI TTS Voice:", openai_tts_voices, index=0,
                                         help="Select the voice for the podcast.")

    # Disable button if OpenAI client isn't initialized
    generate_button = st.button("🚀 Generate Podcast", type="primary",
                                disabled=not st.session_state.get("openai_initialized", False))

# --- Main Application Logic when Button is Clicked ---
if generate_button:
    # Clear previous results
    st.session_state.podcast_script = ""
    st.session_state.audio_file_path = ""

    user_topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
    user_companies = [c.strip() for c in raw_companies.split(',') if c.strip()]

    # --- Input Validations ---
    if not user_topics and not user_companies:
        st.warning("Please enter at least one topic or company.")
    elif not NEWS_API_KEY: # Check if NewsAPI key is present
        st.error("News API Key is not configured in Streamlit secrets. Please add `NEWS_API_KEY`.")
    elif not openai_client or not st.session_state.get("openai_initialized", False): # Double check OpenAI client
        st.error("OpenAI client is not initialized. Check `OPENAI_API_KEY` in secrets.")
    else:
        # --- Step 1: Fetch News ---
        with st.spinner("Step 1/4: Fetching latest news... 📰"):
            news_items_fetched = fetch_news_newsapi(user_topics, user_companies, num_articles_to_fetch)
        
        if not news_items_fetched:
            st.error("No news items found for your criteria. Try broadening your topics/companies or check NewsAPI.")
        else:
            st.success(f"Found {len(news_items_fetched)} potentially relevant news articles.")
            
            # --- Step 2: Process Article Content for Script ---
            # Select only the number of articles specified by the user for the script
            articles_to_use_in_script = news_items_fetched[:num_articles_for_script]
            processed_news_for_script_gen = [] # This will hold items with 'full_text_content' if fetched

            with st.spinner(f"Step 2/4: Extracting content from up to {len(articles_to_use_in_script)} articles... 📝"):
                progress_bar_articles = st.progress(0)
                for i, item in enumerate(articles_to_use_in_script):
                    st.write(f"   L Processing: {item['title'][:70]}...") # Show sub-progress in main area
                    full_text = get_article_text(item['link'])
                    if full_text:
                        item['full_text_content'] = full_text # Add extracted content
                    processed_news_for_script_gen.append(item) # Add item whether text was fetched or not
                    progress_bar_articles.progress((i + 1) / len(articles_to_use_in_script))
            
            if not processed_news_for_script_gen:
                st.warning("Could not process any articles for the script. The script might be less detailed or based only on snippets.")
                # Fallback: use the originally fetched items if no processing happened (should not occur if articles_to_use_in_script is not empty)
                processed_news_for_script_gen = articles_to_use_in_script
            else:
                st.success(f"Content processing complete for {len(processed_news_for_script_gen)} articles to be used in script.")

            # --- Step 3: Generate Podcast Script ---
            with st.spinner("Step 3/4: Generating podcast script with OpenAI... 🤖✍️"):
                # Pass the initialized openai_client
                script = generate_podcast_script_openai(openai_client, processed_news_for_script_gen, user_topics, user_companies)
                st.session_state.podcast_script = script

            if "Error" in st.session_state.podcast_script or not st.session_state.podcast_script.strip():
                st.error(f"Failed to generate script: {st.session_state.podcast_script}")
            else:
                st.subheader("🎧 Generated Podcast Script:")
                st.text_area("Script Preview:", st.session_state.podcast_script, height=300, key="script_display_area")
                
                # --- Step 4: Synthesize Audio ---
                with st.spinner("Step 4/4: Synthesizing audio with OpenAI TTS... 🔊"):
                    audio_output_filename = "podcast_output.mp3" # Consistent filename
                    # Pass the initialized openai_client
                    audio_path = text_to_speech_openai(openai_client, st.session_state.podcast_script, audio_output_filename, voice_model=selected_openai_voice)
                    
                    if audio_path and os.path.exists(audio_path): # Check if file was created
                        st.session_state.audio_file_path = audio_path
                        st.success("Podcast audio generated successfully!")
                    else:
                        st.error("Failed to synthesize audio with OpenAI TTS.")

# --- Display Audio Player and Download Button (if audio exists) ---
# This part runs on every rerun if audio_file_path is set and the file exists
if st.session_state.audio_file_path:
    try:
        # Ensure the file actually exists before trying to open it
        if os.path.exists(st.session_state.audio_file_path):
            with open(st.session_state.audio_file_path, 'rb') as audio_f:
                audio_bytes = audio_f.read()
            st.subheader("▶️ Listen to your Podcast:")
            st.audio(audio_bytes, format='audio/mp3')
            
            st.download_button(
                label="Download Podcast MP3",
                data=audio_bytes,
                file_name="ai_news_podcast.mp3", # Consistent download filename for user
                mime="audio/mp3"
            )
        else:
            st.warning("Previously generated audio file not found. It might have been cleared due to app restart. Please regenerate.")
            st.session_state.audio_file_path = "" # Clear the invalid path
    except Exception as e:
        st.error(f"Error displaying or downloading audio: {e}")
        st.session_state.audio_file_path = "" # Clear path on error

# Footer
st.markdown("---")
st.caption("Built with Streamlit.")
