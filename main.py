import os
import json
import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
from typing import List
from asyncio import Lock
import httpx
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(level=logging.INFO)

# Directories for saving the model and trusted URLs
MODEL_DIR = './models/'
TRUSTED_URLS_FILE = './trusted_urls.json'

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search & Completion API",
    description="Suggests completions and categories based on semantic similarity across multiple domains.",
    version="1.2.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with specific origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if the model exists locally, otherwise download and save it
def load_model(model_name: str):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logging.info(f"Downloading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logging.info(f"Model {model_name} saved to {model_path}")
    else:
        logging.info(f"Loading model {model_name} from local path...")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer

# Load the GPT-Neo model from disk
model_name = "EleutherAI/gpt-neo-2.7B"
model, tokenizer = load_model(model_name)

# Initialize the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load pre-trained language detection model
lang_detector = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load trusted URLs from local JSON file
def load_trusted_urls():
    if os.path.exists(TRUSTED_URLS_FILE):
        with open(TRUSTED_URLS_FILE, 'r') as file:
            trusted_urls = json.load(file)
            logging.info(f"Loaded {len(trusted_urls)} trusted URLs from file.")
            return trusted_urls
    else:
        logging.warning(f"No trusted URLs file found at {TRUSTED_URLS_FILE}. Using default URLs.")
        return [
            # Random Image APIs
            "https://dog.ceo/api/breeds/image/random",  # Random dog images
            "https://api.thecatapi.com/v1/images/search",  # Random cat images
            "https://picsum.photos/200/300",  # Random placeholder images

            # Joke and Fun APIs
            "https://v2.jokeapi.dev/joke/Programming?type=single",  # Programming jokes
            "https://api.chucknorris.io/jokes/random",  # Chuck Norris jokes
            "https://icanhazdadjoke.com/",  # Dad jokes

            # Placeholder Data APIs
            "https://jsonplaceholder.typicode.com/posts",  # Placeholder posts
            "https://jsonplaceholder.typicode.com/users",  # Placeholder users
            "https://jsonplaceholder.typicode.com/comments",  # Placeholder comments
            "https://jsonplaceholder.typicode.com/albums",  # Placeholder albums

            # Random User Data APIs
            "https://randomuser.me/api",  # Random user data
            "https://random-data-api.com/api/name/random_name",  # Random name data
            "https://random-data-api.com/api/address/random_address",  # Random address data
            "https://random-data-api.com/api/company/random_company",  # Random company data

            # Fun Random Activities
            "https://www.boredapi.com/api/activity/",  # Random activities to prevent boredom
            "https://api.nasa.gov/planetary/apod?api_key=DEMO_KEY",  # NASA's Astronomy Picture of the Day (APOD)

            # Programming Data APIs
            "https://pokeapi.co/api/v2/pokemon/ditto",  # Pokémon API
            "https://api.github.com/users/octocat",  # Example GitHub user data
            "https://dev.to/api/articles",  # Developer community articles
            "https://www.codeshare.io/api/articles",  # API for code sharing articles

            # Health & Fitness APIs
            "https://api.openweathermap.org/data/2.5/weather?q=London&appid=b1b15e88fa797225412429c1c50c206c",  # Weather data
            "https://api.covid19api.com/summary",  # COVID-19 data
            "https://api.openaq.org/v1/measurements",  # Air quality data

            # Wikipedia and Other Knowledge APIs
            "https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch=dog",  # Wikipedia search API
            "https://en.wikipedia.org/w/api.php?action=parse&format=json&page=Python_(programming_language)",  # Wikipedia page details
            "https://www.googleapis.com/books/v1/volumes?q=python",  # Google Books API (search Python books)
            "https://datahub.io/core/finance-vix/r/vix-daily.csv",  # Financial data (CSV format)
            
            # Financial APIs
            "https://api.coingecko.com/api/v3/coins/bitcoin",  # Bitcoin information from CoinGecko
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",  # Bitcoin price in USD
            "https://api.cryptonator.com/api/full/btc-usd",  # Bitcoin to USD exchange rate

            # Text Generation and NLP APIs
            "https://api.textgears.com/check.php?text=Hello&key=YOUR_API_KEY",  # Grammar checking
            "https://api.languagetool.org/v2/check",  # Language tool for text grammar and style checking

            # More General Data APIs
            "https://api.publicapis.org/entries",  # Public APIs directory (great for discovering APIs)
            "https://api.openai.com/v1/engines/davinci-codex/completions",  # OpenAI GPT API (for advanced AI tasks, requires API key)

            # Historical Data & Information APIs
            "https://history.muffinlabs.com/date",  # Historical data API
            "https://api.themoviedb.org/3/movie/popular?api_key=YOUR_API_KEY",  # Movie database (requires API key)

            # News APIs
            "https://newsapi.org/v2/top-headlines?country=us&apiKey=YOUR_API_KEY",  # News API (requires API key)
            "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",  # NY Times RSS feed
            "https://news.ycombinator.com/rss",  # Hacker News RSS feed

            # Other Data APIs
            "https://api.qrserver.com/v1/create-qr-code/?data=example",  # QR code generation
            "https://api.nationalize.io/?name=michael",  # Nationality prediction based on name
            "https://api.genderize.io/?name=john",  # Gender prediction based on name
            "https://api.agify.io/?name=lucas",  # Age prediction based on name

            # Country and Location APIs
            "https://restcountries.com/v3.1/all",  # List of all countries
            "https://ipinfo.io/json",  # Get location info based on IP address
            "https://geocode.xyz/51.5074,-0.1278?json=1",  # Geolocation info for coordinates (London)

            # Famous Quotes and Motivational Quotes APIs
            "https://quotes.rest/qod?language=en",  # Quote of the day
            "https://api.quotable.io/random",  # Random quote
            "https://zenquotes.io/api/random",  # Zen quotes API
        ]

# Save trusted URLs to a local JSON file
def save_trusted_urls(trusted_urls):
    with open(TRUSTED_URLS_FILE, 'w') as file:
        json.dump(trusted_urls, file, indent=4)
        logging.info(f"Saved {len(trusted_urls)} trusted URLs to file.")

# Function to fetch data from a list of trustworthy URLs
async def fetch_data_from_trusted_urls(timeout=10):
    trusted_urls = load_trusted_urls()  # Load the URLs from the local file
    async with httpx.AsyncClient() as client:
        for url in trusted_urls:
            try:
                logging.info(f"Trying to fetch data from: {url}")
                response = await client.get(url, timeout=timeout)
                response.raise_for_status()  # This will raise an exception for 4xx/5xx responses
                if "html" in response.headers["Content-Type"]:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    articles = [article.get_text() for article in soup.find_all('p')]
                    if articles:
                        logging.info(f"Successfully fetched data from: {url}")
                        return url, articles
                else:
                    return url, response.json()
            except httpx.RequestError as e:
                logging.error(f"Error fetching data from {url}: {e}")
                continue

    logging.error("All URLs failed to fetch data.")
    return None, []  # Return None if no URL succeeds

# Function to detect language of the input prompt
def detect_language(prompt: str):
    try:
        return lang_detector(prompt, candidate_labels=["en", "es", "fr", "de", "it", "pt"])
    except Exception as e:
        logging.error(f"Error during language detection: {e}")
        return {"labels": [], "scores": []}

# Function to generate text completions based on the prompt
def generate_text_completions(prompt: str):
    try:
        generated_texts = text_generator(prompt, max_length=len(prompt.split()) + 5, num_return_sequences=3)
        return [gen['generated_text'].strip() for gen in generated_texts]
    except Exception as e:
        logging.error(f"Error generating text completions: {e}")
        return []

# Queue and lock for pausing and resuming tasks
queue_lock = Lock()
current_queue_length = 0
average_processing_time = 5  # Placeholder for the average time a task takes

# Define CompletionResponse Pydantic model
class CompletionResponse(BaseModel):
    input: str
    completions: List[dict]
    status: str
    language_info: dict
    waiting_time: int
    number_in_line: int
    url_that_worked: str

# FastAPI endpoint to generate suggestions
@app.get("/complete", response_model=CompletionResponse)
async def complete(prompt: str = Query(..., min_length=2)):
    try:
        global current_queue_length
        async with queue_lock:
            waiting_time = current_queue_length * average_processing_time
            current_queue_length += 1

        # Detect language and get language info
        language_info = detect_language(prompt)

        # Generate word suggestions based on the prompt using pre-trained model
        suggestions = generate_text_completions(prompt)

        if not suggestions:
            return CompletionResponse(
                input=prompt,
                completions=[{"suggested_word": "No suggestions found.", "chance": 0}],
                status="failure",
                language_info=language_info,
                waiting_time=waiting_time,
                number_in_line=current_queue_length,
                url_that_worked="None"
            )

        # Prepare completions data
        completions = [{"suggested_word": suggestion, "chance": 100} for suggestion in suggestions]

        # Fetch data from trusted URLs and get the URL that worked
        url_that_worked, _ = await fetch_data_from_trusted_urls()

        return CompletionResponse(
            input=prompt,
            completions=[{"sentence": f"{completion['suggested_word']}", "chance": completion['chance']} for completion in completions],
            status="success",
            language_info=language_info,
            waiting_time=waiting_time,
            number_in_line=current_queue_length,
            url_that_worked=url_that_worked if url_that_worked else "None"
        )

    except Exception as e:
        logging.error(f"Error during completion: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    finally:
        # Ensure queue length is updated after the request
        async with queue_lock:
            current_queue_length -= 1

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
