import feedparser
import newspaper
import datetime
import json
import time
import redis # For Bloom Filter
from redisbloom.client import Client as RedisBloomClient # Correct import for redisbloom
import os # Import os to read environment variables
import logging # Added logging
import hashlib # Added for article ID generation

# Assuming app.config is accessible (for settings)
import sys
# Adjust path based on location of ingestion/collector.py relative to project root
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR) # Assuming ingestion is direct child of project root
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# Attempt to import settings (it's okay if this fails if running collector.py directly outside app)
try:
    from app.config import settings
except ImportError:
    # Define placeholder settings if app.config is not available (e.g., running collector.py standalone)
    class MockSettings:
        redis_host: str = "localhost"
        redis_port: int = 6379
        redis_db: int = 0
        redis_syndication_key: str = "SEEN_ARTICLE_HASHES"
        # Define other settings defaults needed by collector.py if any

    settings = MockSettings()
    print("Warning: Could not import app.config. Using mock settings in ingestion/collector.py.")

logger = logging.getLogger(__name__)

# Configuration for RSS feeds
# Use settings for RSS feed URLs if they are configurable, otherwise keep here
RSS_FEEDS = settings.rss_feeds if hasattr(settings, 'rss_feeds') and settings.rss_feeds else {
    "Yahoo Finance": "https://finance.yahoo.com/rss/",
    "CNBC Economy": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "CNBC Top News": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "Financial Times Home": "https://www.ft.com/rss/home",
    # Add Reuters and Investing.com later as per notes if specific feeds are found
    # "Reuters": "URL_FOR_REUTERS_RSS",
    # "Investing.com": "URL_FOR_INVESTING_COM_RSS",
}

# Configuration for Redis Bloom Filter - Use settings values
# BLOOM_FILTER_KEY = "news_article_bloom_filter" # Use from settings if available
BLOOM_FILTER_KEY = settings.redis_bloom_filter_key if hasattr(settings, 'redis_bloom_filter_key') else "news_article_bloom_filter"
BLOOM_FILTER_CAPACITY = settings.redis_bloom_filter_capacity if hasattr(settings, 'redis_bloom_filter_capacity') else 1_000_000
BLOOM_FILTER_ERROR_RATE = settings.redis_bloom_filter_error_rate if hasattr(settings, 'redis_bloom_filter_error_rate') else 0.001

# Removed global redis_client and redis_bloom_client

# Removed initialize_redis_and_bloom_filter function and its call
# Redis client will be passed from main.py

# --- Helper Functions ---

def normalize_timestamp_to_utc_ms(ts_struct):
    """Converts a time.struct_time object to UTC milliseconds timestamp."""
    if ts_struct:
        try:
            # Attempt to create a datetime object, assuming it might be naive
            dt = datetime.datetime(*ts_struct[:6])
            # If no timezone info, assume UTC (common for feedparser for published_parsed)
            # For more robustness, one might need to check feed-specific timezone handling
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            else:
                # Ensure it's UTC
                dt = dt.astimezone(datetime.timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception as e:
            logger.warning(f"Error normalizing timestamp {ts_struct}: {e}")
            return None
    return None

def parse_article_with_newspaper(url: str) -> Optional[Dict[str, Any]]:
    """Fetches and parses a single article using newspaper3k."""
    try:
        article = newspaper.Article(url)
        article.download()
        if article.is_downloaded:
            article.parse()
            
            # Newspaper's publish_date is often naive, convert to UTC
            publish_dt = article.publish_date
            publish_ts_utc_ms = None
            if publish_dt:
                try:
                    if publish_dt.tzinfo is None:
                        publish_dt = publish_dt.replace(tzinfo=datetime.timezone.utc) # Assume UTC if naive
                    else:
                        publish_dt = publish_dt.astimezone(datetime.timezone.utc)
                    publish_ts_utc_ms = int(publish_dt.timestamp() * 1000)
                except Exception as e:
                    logger.warning(f"Error processing newspaper3k publish date {publish_dt}: {e}")

            return {
                "url": article.url,
                "title": article.title,
                "body": article.text,
                "publish_ts": publish_ts_utc_ms, # Store as UTC milliseconds
                "source_html_page": url # To distinguish from RSS source
            }
        else:
            logger.warning(f"Newspaper3k failed to download article: {url}")
            return None
    except newspaper.article.ArticleException as e:
        logger.warning(f"Could not process article {url} with newspaper3k: {e}")
        return None
    except Exception as e:
         logger.error(f"An unexpected error occurred during newspaper3k parsing of {url}: {e}", exc_info=True)
         return None

# --- Main Ingestion Logic ---

def fetch_from_rss(
    feed_name: str,
    feed_url: str,
    redis_client: Optional[redis.Redis] = None, # Accept Redis client
    bloom_filter_key: str = BLOOM_FILTER_KEY,
    bloom_filter_capacity: int = BLOOM_FILTER_CAPACITY,
    bloom_filter_error_rate: float = BLOOM_FILTER_ERROR_RATE
) -> List[Dict[str, Any]]:
    """Fetches articles from a given RSS feed with Redis Bloom filter deduplication."""
    logger.info(f"Fetching from RSS feed: {feed_name} ({feed_url})")
    articles = []
    redis_bloom_client = None

    # Initialize Bloom Filter client using the provided redis_client
    if redis_client:
        try:
            # Check if the Bloom Filter exists, if not, create it
            # This requires the RedisBloom module installed on the Redis server
            if not redis_client.exists(bloom_filter_key):
                logger.info(f"Creating Bloom Filter '{bloom_filter_key}' with capacity {bloom_filter_capacity}, error rate {bloom_filter_error_rate}")
                # Need to use RedisBloomClient specifically for bf.create
                try:
                    temp_bloom_client = RedisBloomClient(host=redis_client.connection_pool.host, port=redis_client.connection_pool.port, db=redis_client.connection_pool.connection_kwargs.get('db', 0))
                    temp_bloom_client.bfCreate(bloom_filter_key, bloom_filter_error_rate, bloom_filter_capacity) # Note bfCreate args order
                    logger.info(f"Bloom Filter '{bloom_filter_key}' created successfully.")
                    redis_bloom_client = temp_bloom_client
                except Exception as e:
                    logger.error(f"Failed to create or connect to RedisBloom for filter '{bloom_filter_key}': {e}. Deduplication will be disabled.", exc_info=True)
                    redis_bloom_client = None
            else:
                 logger.debug(f"Bloom Filter '{bloom_filter_key}' already exists.")
                 # If it exists, just get a RedisBloomClient instance for it
                 try:
                     redis_bloom_client = RedisBloomClient(host=redis_client.connection_pool.host, port=redis_client.connection_pool.port, db=redis_client.connection_pool.connection_kwargs.get('db', 0))
                 except Exception as e:
                      logger.error(f"Failed to connect to RedisBloom for filter '{bloom_filter_key}': {e}. Deduplication will be disabled.", exc_info=True)
                      redis_bloom_client = None

        except Exception as e:
            logger.error(f"Error initializing Bloom Filter client using provided Redis client: {e}. Deduplication disabled.", exc_info=True)
            redis_bloom_client = None
    else:
        logger.warning("Redis client not provided to fetch_from_rss. Deduplication will be disabled.")

    try:
        parsed_feed = feedparser.parse(feed_url)
    except Exception as e:
        logger.error(f"Error parsing RSS feed {feed_url}: {e}", exc_info=True)
        return []

    if parsed_feed.bozo: # Check for parse errors
         logger.warning(f"Bozo bit set for feed {feed_url}. Parse error: {parsed_feed.bozo_exception}")


    for entry in parsed_feed.entries:
        title = entry.get("title", "No Title")
        link = entry.get("link")
        published_parsed = entry.get("published_parsed")
        summary = entry.get("summary", "")

        # Convert published_parsed (time.struct_time) to UTC milliseconds
        publish_ts_utc_ms = normalize_timestamp_to_utc_ms(published_parsed)

        # --- Deduplication with Redis Bloom Filter ---
        # Create a unique key for the article: canonical_url + publish_ts (if available)
        # Use link + string representation of publish_ts_utc_ms for uniqueness.
        article_unique_key = None
        if link and publish_ts_utc_ms:
            article_unique_key = f"{link}|{publish_ts_utc_ms}"
        elif link: # Fallback if no timestamp from RSS, less precise deduplication
            article_unique_key = link 
        # No key if no link

        is_duplicate = False
        if redis_bloom_client and article_unique_key:
            try:
                # bfExists returns 1 if exists, 0 otherwise
                if redis_bloom_client.bfExists(bloom_filter_key, article_unique_key):
                     logger.info(f"  Skipping (Redis Bloom Filter duplicate): '{title}' ({article_unique_key})")
                     is_duplicate = True
            except Exception as e:
                 logger.error(f"  Error checking Bloom Filter for '{title}': {e}. Skipping deduplication check for this item.", exc_info=True)
                 # Do not set is_duplicate = True if check fails

        if is_duplicate:
             continue # Skip this article if identified as duplicate
        # --- End Deduplication Check ---

        article_data = {
            "id": f"rss_{feed_name}_{hashlib.sha1((link or title).encode()).hexdigest()[:8]}_{publish_ts_utc_ms}", # Generate unique ID
            "url": link,
            "headline": title,
            "body": summary, # Start with summary, attempt to scrape full body
            "published_ts": publish_ts_utc_ms,
            "source": feed_name
        }
        
        # Attempt to fetch full article content if URL seems valid
        if link:
            logger.debug(f"Attempting to scrape full content for: {title} from {link}")
            scraped_content = parse_article_with_newspaper(link)
            if scraped_content:
                article_data["body"] = scraped_content["body"]
                # Prefer publish_ts and URL from newspaper3k if available and more precise/canonical
                if scraped_content["publish_ts"]:
                     article_data["published_ts"] = scraped_content["publish_ts"]
                article_data["url"] = scraped_content["url"]
                
                # Re-calculate unique key based on scraped data for Bloom Filter addition
                article_unique_key = None
                if article_data["url"] and article_data["published_ts"]:
                    article_unique_key = f"{article_data['url']}|{article_data['published_ts']}"
                elif article_data["url"]:
                    article_unique_key = article_data["url"]
            else:
                 logger.warning(f"Failed to scrape full content for {link}. Using RSS summary/body: {len(article_data['body'])} chars.")
                 # If scraping fails, fall back to RSS summary


        articles.append(article_data)
        logger.info(f"Collected: '{title}' from {feed_name}")
        
        # Add to Bloom Filter if successfully ingested and filter is available
        if redis_bloom_client and article_unique_key and not is_duplicate: # Only add if not already found as duplicate
            try:
                # bfAdd returns 1 if added, 0 if already exists (should be caught by bfExists but double check)
                if redis_bloom_client.bfAdd(bloom_filter_key, article_unique_key):
                    logger.debug(f"  Added to Bloom Filter: {article_unique_key}")
            except Exception as e:
                logger.error(f"  Error adding to Bloom Filter '{article_unique_key}': {e}", exc_info=True)
        
        # Implement rate limiting if needed (e.g., time.sleep(1))
        # time.sleep(0.1) # Basic rate limit per article

    return articles

def run_ingestion_cycle(redis_client: Optional[redis.Redis] = None) -> List[Dict[str, Any]]:
    """Runs a single cycle of ingesting news from all configured sources."""
    logger.info("--- Starting New Ingestion Cycle ---")
    all_collected_articles = []
    for feed_name, feed_url in RSS_FEEDS.items():
        try:
            # Pass the Redis client to fetch_from_rss
            articles_from_feed = fetch_from_rss(
                feed_name, 
                feed_url, 
                redis_client=redis_client, 
                bloom_filter_key=settings.redis_bloom_filter_key if hasattr(settings, 'redis_bloom_filter_key') else BLOOM_FILTER_KEY,
                bloom_filter_capacity=settings.redis_bloom_filter_capacity if hasattr(settings, 'redis_bloom_filter_capacity') else BLOOM_FILTER_CAPACITY,
                bloom_filter_error_rate=settings.redis_bloom_filter_error_rate if hasattr(settings, 'redis_bloom_filter_error_rate') else BLOOM_FILTER_ERROR_RATE
            )
            all_collected_articles.extend(articles_from_feed)
            logger.info(f"Ingested {len(articles_from_feed)} articles from {feed_name}")
        except Exception as e:
            logger.error(f"Error fetching from {feed_name}: {e}", exc_info=True)
            # Implement more robust error handling/logging

    # Removed placeholder for saving to file - storage should be handled downstream/separately

    logger.info(f"--- Ingestion Cycle Completed. Total articles collected: {len(all_collected_articles)} ---")
    return all_collected_articles

if __name__ == "__main__":
    # This __main__ block is for testing ingestion.collector directly.
    # It will attempt to initialize Redis itself if run standalone.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running ingestion/collector.py direct test...")

    # Initialize Redis for standalone testing (if not already done by implicit import)
    # This replicates the original module-level initialization idea for standalone use
    temp_redis_client = None
    try:
        temp_redis_client = redis.Redis(
            host=settings.redis_host, # Use settings from app.config or MockSettings
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True # Ensure keys/values are strings
        )
        temp_redis_client.ping()
        logger.info(f"__main__: Successfully connected to Redis at {settings.redis_host}:{settings.redis_port} for standalone test.")
    except (ImportError, redis.exceptions.ConnectionError) as e:
        logger.warning(f"__main__: Redis connection failed ({e}). Standalone deduplication will be disabled.")
        temp_redis_client = None
    except Exception as e:
         logger.error(f"__main__: An unexpected error occurred during standalone Redis connection: {e}", exc_info=True)
         temp_redis_client = None

    # Run one ingestion cycle with the standalone Redis client
    collected_articles = run_ingestion_cycle(redis_client=temp_redis_client)

    logger.info(f"__main__: Collected {len(collected_articles)} articles in standalone test.")

    # In a real scenario, these articles would be sent to the processing pipeline or saved.
    # Example: Print titles of collected articles
    # logger.info("__main__: Collected Article Titles:")
    # for article in collected_articles:
    #     logger.info(f" - {article.get('headline', 'N/A')}")

    logger.info("ingestion/collector.py direct test finished.") 