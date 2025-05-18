import streamlit as st
import datetime
import requests
from bs4 import BeautifulSoup
from openai import OpenAI # OpenAI Python library
import os # For saving the audio file temporarily
import json # For decoding potential JSON errors from NewsAPI

# --- THIS MUST BE THE VERY FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="AI News Podcast (OpenAI)", layout="wide")

# --- Attempt to Initialize OpenAI Client and Store in Session State ---
@st.cache_resource
def get_openai_client():
    st.session_state.pop('openai_init_error', None)
    api_key = st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        st.session_state.openai_init_error = "OPENAI_API_KEY not found in Streamlit secrets. Please configure it."
        print(st.session_state.openai_init_error)
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        # Optional: Test call to verify key. Comment out if causing issues or cost concerns.
        # client.models.list() 
        print("OpenAI client object created successfully.")
        return client
    except Exception as e:
        error_message = f"Failed to initialize OpenAI client or verify API key: {type(e).__name__} - {e}"
        st.session_state.openai_init_error = error_message
        print(error_message)
        return None

openai_client = get_openai_client()

if openai_client:
    st.session_state.openai_initialized = True
else:
    st.session_state.openai_initialized = False

# --- Configuration from Streamlit Secrets (NewsAPI) ---
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY")

# --- UI Initialization & Feedback ---
st.title("🎙️ AI News Podcast Generator")
st.caption("Powered by NewsAPI and OpenAI (GPT & TTS)")

if st.session_state.get("openai_initialized", False):
    st.sidebar.success("OpenAI Client Connected & Ready.")
else:
    st.sidebar.error("OpenAI Client NOT Initialized.")
    if 'openai_init_error' in st.session_state:
        st.error(f"OpenAI Initialization Problem: {st.session_state.openai_init_error}")
    else:
        st.error("Problem during OpenAI client setup. Check logs and secrets.")

# --- Session State for app data (script, audio path) ---
if 'podcast_script' not in st.session_state:
    st.session_state.podcast_script = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = ""


# --- News Fetching Functions ---
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"

# MODIFIED fetch_news_newsapi FOR TESTING - IGNORES UI INPUTS
@st.cache_data(ttl=3600) # Caching is fine for this test
def fetch_news_newsapi(topics, companies, num_articles=5): # UI parameters are ignored by this test version
    if not NEWS_API_KEY:
        st.error("News API key (NEWS_API_KEY) not configured in Streamlit secrets.")
        print("DEBUG: NEWS_API_KEY is MISSING in secrets")
        return []

    print(f"DEBUG: Using NewsAPI Key starting with: {NEWS_API_KEY[:5] if NEWS_API_KEY else 'None (This should not happen if key check passed)'}")

    # --- FORCED SIMPLE QUERY FOR TESTING ---
    test_query = "world" # A very broad term. Try "apple" or "google" if "world" yields nothing.
    
    from_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')

    params = {
        'q': test_query,
        'apiKey': NEWS_API_KEY,
        'from': from_date,
        'language': 'en',
        'pageSize': 5
        # You can also try the 'top-headlines' endpoint as a test if 'everything' is problematic
        # NEWS_API_ENDPOINT = "https://newsapi.org/v2/top-headlines"
        # params = {'country': 'us', 'apiKey': NEWS_API_KEY, 'pageSize': 5} 
    }
    
    print(f"DEBUG: Test NewsAPI Request URL: {NEWS_API_ENDPOINT}") # Will show /everything or /top-headlines if you change it
    print(f"DEBUG: Test NewsAPI Request Params: {params}")
    
    results = []
    try:
        response = requests.get(NEWS_API_ENDPOINT, params=params, timeout=15)
        print(f"DEBUG: Test NewsAPI Response Status Code: {response.status_code}")
        print(f"DEBUG: Test NewsAPI Response Headers: {response.headers}")
        # Try to print the full response text for detailed error messages from NewsAPI
        # Be cautious if the response could be extremely large, but for error diagnosis it's helpful.
        response_text = response.text
        print(f"DEBUG: Test NewsAPI Response Text (first 1000 chars): {response_text[:1000]}")
        
        response.raise_for_status() # This will raise an error for 4xx or 5xx status codes
        
        data = response.json() # This line will fail if response_text is not valid JSON
        articles = data.get('articles', [])
        print(f"DEBUG: Number of articles received: {len(articles)}")

        if not articles and data.get('status') == 'ok': # API call was ok, but no articles found
            print(f"DEBUG: NewsAPI reported status OK but returned 0 articles for query '{test_query}'.")
            print(f"DEBUG: Full JSON response from NewsAPI: {data}") # See totalResults, etc.
        elif data.get('status') == 'error': # NewsAPI itself reported an error
             print(f"DEBUG: NewsAPI reported an error. Code: {data.get('code')}, Message: {data.get('message')}")


        for article in articles:
            results.append({
                'title': article['title'],
                'link': article['url'],
                'snippet': article.get('description', '') or article.get('content', '')[:200] if article.get('content') else '',
                'source_name': article.get('source', {}).get('name', 'Unknown Source'),
                'published_at': article.get('publishedAt', '')
            })
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred with NewsAPI: {http_err}") # Error shown in UI
        print(f"CRITICAL DEBUG: HTTP error occurred with NewsAPI: {http_err} - Response text was: {response_text}") # Logged
    except requests.exceptions.RequestException as req_err:
        st.error(f"Error during NewsAPI request: {req_err}")
        print(f"CRITICAL DEBUG: Error during NewsAPI request: {req_err}")
    except json.JSONDecodeError as json_err: # If response.text wasn't valid JSON
        st.error(f"Error decoding JSON response from NewsAPI. Status: {response.status_code}.")
        print(f"CRITICAL DEBUG: Error decoding JSON from NewsAPI: {json_err}. Response text was: {response_text}")
    except Exception as e:
        st.error(f"An unexpected error occurred with NewsAPI: {type(e).__name__}")
        print(f"CRITICAL DEBUG: An unexpected error occurred with NewsAPI: {type(e).__name__} - {e}")
    
    if not results:
        print("DEBUG: fetch_news_newsapi is returning an empty list from the test query.")
    return results


@st.cache_data(ttl=3600)
def get_article_text(url):
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

# --- OpenAI Script Generation ---
def generate_podcast_script_openai(client_instance, news_items_for_script, topics, companies):
    if not client_instance:
        st.error("OpenAI client instance is not available for script generation.")
        return "Error: OpenAI client instance missing."

    news_details_for_prompt = ""
    if not news_items_for_script: # Check if any items made it to this stage
        st.warning("No news items provided to generate_podcast_script_openai. Script will be generic or fail.")
        # Fallback: create a message indicating no specific news was processed
        news_details_for_prompt = "No specific news articles were processed. Please generate a generic news intro and outro."

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
1. Intro: A brief, friendly welcome (e.g., 'Welcome to your AI-powered news brief! Here’s what’s trending...').
2. News Segments: Cover 2-4 distinct news items based on the provided content. If no specific news items are provided, give a general news update or a placeholder message. For each news item:
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
{news_details_for_prompt if news_details_for_prompt else "No specific news content was available to process. Please provide a generic, short news podcast greeting and sign-off."}

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

# --- OpenAI Text-to-Speech ---
def text_to_speech_openai(client_instance, text, output_filename="podcast_audio_openai.mp3", voice_model="alloy"):
    if not client_instance:
        st.error("OpenAI client instance is not available for TTS.")
        return None
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
    
    num_articles_to_fetch = st.slider("Max news articles to fetch (ignored in test mode):", 3, 20, 7, help="How many articles NewsAPI should return.")
    num_articles_for_script = st.slider("Max articles for script content:", 1, 5, 3, help="How many articles will be used for script.")
    
    openai_tts_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    selected_openai_voice = st.selectbox("Choose OpenAI TTS Voice:", openai_tts_voices, index=0, help="Select the voice for the podcast.")

    generate_button = st.button("🚀 Generate Podcast", type="primary",
                                disabled=not st.session_state.get("openai_initialized", False))

# --- Main Application Logic when Button is Clicked ---
if generate_button:
    st.session_state.podcast_script = ""
    st.session_state.audio_file_path = ""

    # UI inputs are technically read but ignored by the TEST version of fetch_news_newsapi
    user_topics = [t.strip() for t in raw_topics.split(',') if t.strip()]
    user_companies = [c.strip() for c in raw_companies.split(',') if c.strip()]

    if not NEWS_API_KEY: # Check if NewsAPI key is present before proceeding
        st.error("News API Key is not configured in Streamlit secrets. Cannot fetch news.")
    elif not openai_client: # Check the global client directly
        st.error("OpenAI client is not initialized. Cannot generate podcast. Check sidebar for error details.")
    else:
        with st.spinner("Step 1/4: Fetching latest news (TEST MODE)... 📰"):
            # The topics/companies from UI are passed but will be IGNORED by the test function
            news_items_fetched = fetch_news_newsapi(user_topics, user_companies, num_articles_to_fetch)
        
        if not news_items_fetched:
            st.error("TEST MODE: No news items found using the hardcoded test query ('world'). Check NewsAPI key, plan, or logs for details.")
        else:
            st.success(f"TEST MODE: Found {len(news_items_fetched)} news articles using hardcoded test query.")
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
            
            if not processed_news_for_script_gen: # Should contain items even if full_text_content is missing
                st.warning("No content could be extracted from fetched articles. Script will use titles/snippets.")
                processed_news_for_script_gen = articles_to_use_in_script # Use items with at least title/snippet
            else:
                st.success(f"Content processing complete for {len(processed_news_for_script_gen)} articles.")

            with st.spinner("Step 3/4: Generating podcast script with OpenAI... 🤖✍️"):
                script = generate_podcast_script_openai(openai_client, processed_news_for_script_gen, user_topics, user_companies)
                st.session_state.podcast_script = script

            if "Error" in st.session_state.podcast_script or not st.session_state.podcast_script.strip():
                st.error(f"Failed to generate script: {st.session_state.podcast_script}")
            else:
                st.subheader("🎧 Generated Podcast Script:")
                st.text_area("Script Preview:", st.session_state.podcast_script, height=300, key="script_display_area")
                
                with st.spinner("Step 4/4: Synthesizing audio with OpenAI TTS... 🔊"):
                    audio_output_filename = "podcast_output.mp3"
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
