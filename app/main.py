# app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn # For running the app
import time
import os
import logging # Added for logging

# Assuming services, utils, models are in the 'app' directory or subdirectories
import sys
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.append(_APP_DIR)
# Ensure project root is also in path for app.config import if main.py is run directly for tests/dev
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from app.config import settings # Import centralized settings
from app.services.news_processor import NewsProcessorService, FinBERTSentiment, RAKETopicTagger
from app.causality.extractor import NLP as SpacyNLP # Reuse the loaded spaCy model instance, or load using settings.spacy_model_name
from app.graph.propagator import EventPropagator
from app.kafka_utils import create_kafka_producer # KafkaProducer type hint is implicitly handled by create_kafka_producer return type
from ingestion.collector import run_ingestion_cycle # Added for pipeline cycle

# Setup logging
logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import redis, fall back to a mock for local dev if not installed/running
try:
    import redis
    redis_client_instance = redis.Redis(
        host=settings.redis_host, 
        port=settings.redis_port, 
        db=settings.redis_db, 
        decode_responses=True
    )
    redis_client_instance.ping()
    logger.info(f"Successfully connected to Redis at {settings.redis_host}:{settings.redis_port}")
except (ImportError, redis.exceptions.ConnectionError) as e:
    logger.warning(f"Redis connection failed or redis package not installed ({e}). Using MockRedis for main.py.")
    class MockRedisClient:
        def __init__(self):
            self._store = {}
            logger.info("MockRedisClient initialized for main.py")
        def sismember(self, key, value): return value in self._store.get(key, set())
        def sadd(self, key, value):
            self._store.setdefault(key, set()).add(value)
            return 1
        def ping(self): logger.debug("MockRedis pinged"); return True # Mock ping
    redis_client_instance = MockRedisClient()

# --- Global Resources --- 
app_state: Dict[str, Any] = {}

# Using FastAPI's modern lifespan context manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application startup via lifespan manager...")
    
    # 1. Initialize spaCy NLP Model
    # SpacyNLP is loaded when causality.extractor is imported. 
    # We should ensure causality.extractor also uses settings.spacy_model_name if it re-loads.
    # For now, assume SpacyNLP from extractor is the one to use.
    if SpacyNLP is None:
        logger.critical(f"spaCy NLP model (SpacyNLP from causality.extractor) not loaded! Trying to load {settings.spacy_model_name}")
        try:
            import spacy
            app_state["spacy_nlp"] = spacy.load(settings.spacy_model_name)
            logger.info(f"Successfully loaded spaCy model '{settings.spacy_model_name}' in lifespan.")
        except OSError as e:
            logger.critical(f"Could not load spaCy model '{settings.spacy_model_name}': {e}. Service will be impaired.")
            # Fallback to a dummy if absolutely necessary
            class MockSpacyDoc: 
                def __init__(self, text): self.text = text; self.ents = []; self.sents = [self]
                def __len__(self): return len(self.text.split())
            class MockSpacy: 
                def __call__(self, text): return MockSpacyDoc(text)
            app_state["spacy_nlp"] = MockSpacy()
    else:
        app_state["spacy_nlp"] = SpacyNLP
        logger.info("Reused spaCy NLP model from causality.extractor and stored in app_state.")

    # 2. Initialize FinBERT and RAKE models (using mocks for this example)
    app_state["finbert_model"] = FinBERTSentiment() # Replace with real model loading using config paths
    app_state["rake_model"] = RAKETopicTagger()       # Replace with real model loading using config paths
    logger.info("Mock FinBERT and RAKE models initialized.")

    # 3. Initialize Redis Client (already initialized above)
    app_state["redis_client"] = redis_client_instance
    logger.info(f"Redis client ('{type(redis_client_instance).__name__}') stored in app_state.")

    # 4. Initialize Kafka Producer
    # create_kafka_producer should ideally also use settings from app.config
    kafka_producer_instance = create_kafka_producer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        client_id=settings.kafka_client_id,
        security_protocol=settings.kafka_security_protocol,
        sasl_mechanism=settings.kafka_sasl_mechanism,
        sasl_plain_username=settings.kafka_sasl_plain_username,
        sasl_plain_password=settings.kafka_sasl_plain_password
    )
    if kafka_producer_instance:
        app_state["kafka_producer"] = kafka_producer_instance
        logger.info("Kafka producer initialized and stored in app_state.")
    else:
        app_state["kafka_producer"] = None
        logger.warning("Kafka producer could not be initialized! Kafka sending will fail.")

    # 5. Initialize Event Propagator
    try:
        # EventPropagator should ideally take the map path from settings
        app_state["event_propagator"] = EventPropagator(entity_map_path=os.path.join(_PROJECT_ROOT, settings.entity_relation_map_path))
        logger.info(f"EventPropagator initialized with map '{settings.entity_relation_map_path}' and stored in app_state.")
    except Exception as e:
        logger.critical(f"EventPropagator could not be initialized ({e}). Relation propagation will fail.")
        class DummyPropagator: 
            def propagate_and_send_relations(self, *args, **kwargs): return 0
        app_state["event_propagator"] = DummyPropagator()

    # 6. Initialize NewsProcessorService with all dependencies
    if app_state.get("spacy_nlp") and app_state.get("event_propagator") and app_state.get("finbert_model") and app_state.get("rake_model") and app_state.get("redis_client") :
        app_state["news_processor_service"] = NewsProcessorService(
            spacy_nlp=app_state["spacy_nlp"],
            finbert_model=app_state["finbert_model"],
            rake_model=app_state["rake_model"],
            redis_client=app_state["redis_client"],
            kafka_producer=app_state.get("kafka_producer"),
            event_propagator=app_state["event_propagator"],
            min_source_trust=settings.default_min_source_trust,
            min_tokens_density=settings.default_min_tokens_density,
            min_org_entities_density=settings.default_min_org_entities_density,
            causality_topic=settings.kafka_causality_topic,
            relations_topic=settings.kafka_relations_topic,
            clickbait_patterns_path=settings.clickbait_patterns_path,
            source_profiles_path=settings.source_profiles_path
        )
        logger.info("NewsProcessorService initialized with dependencies using centralized settings.")
    else:
        logger.critical("NewsProcessorService could NOT be initialized due to missing critical components.")
        app_state["news_processor_service"] = None
    
    yield # Application runs here

    logger.info("FastAPI application shutdown via lifespan manager...")
    kafka_producer = app_state.get("kafka_producer")
    if kafka_producer:
        logger.info("Flushing and closing Kafka producer...")
        try:
            kafka_producer.flush(timeout=10)
            kafka_producer.close(timeout=10)
            logger.info("Kafka producer closed.")
        except Exception as e:
            logger.error(f"Error closing Kafka producer: {e}")

# --- FastAPI App Definition ---
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Processes financial news articles, extracts features, and sends to Kafka.",
    lifespan=lifespan # Use the new lifespan context manager
)

# --- Pydantic Models for API --- 
class RawArticleInput(BaseModel):
    id: str = Field(..., description="Unique identifier for the article", example="news_xyz_123")
    headline: str = Field(..., description="Article headline", example="Big Corp Announces Quarterly Earnings!")
    body: str = Field(..., description="Full content of the article.", example="The company reported a 10% increase...")
    source: str = Field(..., description="Source of the article (e.g., domain name or news agency ID)", example="reuters.com")
    published_ts: Optional[int] = Field(None, description="Publication timestamp (Unix milliseconds)", example=int(time.time()*1000))

class ProcessingStatus(BaseModel):
    article_id: str
    status: str
    message: Optional[str] = None
    processed_data: Optional[Dict[str, Any]] = None

# --- API Endpoints --- 

@app.get("/status", summary="Get service status", response_model=Dict[str, Any])
async def get_status():
    kafka_ok = False
    if app_state.get("kafka_producer"):
        try:
            # For confluent_kafka, bootstrap_connected() is a good check.
            # For kafka-python, it doesn't have a direct equivalent after init.
            # We can assume if it initialized, it's attempting to connect.
            # A more robust check would involve sending a test message or checking cluster metadata.
            if hasattr(app_state["kafka_producer"], 'bootstrap_connected'):
                 kafka_ok = app_state["kafka_producer"].bootstrap_connected()
            else: # Assuming kafka-python, which tries to connect on first operation or already did.
                 kafka_ok = True # Placeholder, consider a better check for kafka-python liveness
        except Exception:
            kafka_ok = False

    return {
        "service_name": settings.app_name,
        "version": settings.app_version,
        "status": "healthy" if app_state.get("news_processor_service") else "degraded",
        "timestamp": time.time(),
        "active_spacy_model": app_state.get("spacy_nlp").__class__.__name__ if app_state.get("spacy_nlp") else None,
        "redis_status": "connected" if app_state.get("redis_client") and app_state.get("redis_client").ping() else "disconnected",
        "kafka_producer_status": "connected" if kafka_ok else "disconnected_or_unavailable",
        "event_propagator_status": "initialized" if app_state.get("event_propagator") and not isinstance(app_state.get("event_propagator"), type) else "not_initialized_or_dummy"
    }

@app.post("/process-article", response_model=ProcessingStatus, summary="Process a single news article")
async def process_single_article(article_input: RawArticleInput, background_tasks: BackgroundTasks):
    news_service = app_state.get("news_processor_service")
    if not news_service:
        logger.error("Attempt to process article when NewsProcessorService is not available.")
        raise HTTPException(status_code=503, detail="NewsProcessorService not available. Critical components may have failed to initialize.")

    raw_article_dict = article_input.model_dump()
    if raw_article_dict.get("published_ts") is None:
        raw_article_dict["published_ts"] = int(time.time() * 1000)
    
    logger.info(f"Received article for processing: {article_input.id} - '{article_input.headline[:50]}...'")
    
    try:
        enriched_data = news_service.process_article(raw_article_dict)
        if enriched_data:
            logger.info(f"Article {article_input.id} processed successfully.")
            return ProcessingStatus(
                article_id=article_input.id, 
                status="processed_successfully",
                # processed_data=enriched_data # Decide if you want to return full data
            )
        else:
            logger.info(f"Article {article_input.id} rejected by pipeline.")
            return ProcessingStatus(
                article_id=article_input.id, 
                status="rejected_by_pipeline",
                message="Article did not pass processing and filtering stages."
            )
    except Exception as e:
        logger.exception(f"Error processing article {article_input.id}: {e}") # Use logger.exception for stack trace
        raise HTTPException(status_code=500, detail=f"Internal server error while processing article: {str(e)}")

@app.post("/run-pipeline-cycle", summary="Triggers ingestion from RSS feeds and processes the articles", response_model=List[ProcessingStatus])
async def run_pipeline_cycle_endpoint(background_tasks: BackgroundTasks): # Renamed for clarity, removed 'articles' param
    news_service = app_state.get("news_processor_service")
    redis_client = app_state.get("redis_client")

    if not news_service:
        logger.error("Attempt to run pipeline cycle when NewsProcessorService is not available.")
        raise HTTPException(status_code=503, detail="NewsProcessorService not available.")
    if not redis_client:
        logger.error("Attempt to run pipeline cycle when Redis client is not available.")
        raise HTTPException(status_code=503, detail="Redis client not available.")

    logger.info("--- Triggering Ingestion and Processing Cycle ---")
    
    try:
        # 1. Run ingestion cycle to fetch articles from RSS feeds
        # run_ingestion_cycle expects a redis.Redis client, ensure app_state["redis_client"] is the actual client.
        logger.info("Starting ingestion cycle...")
        # Ensure redis_client_instance used here is the actual Redis client, not MockRedis if Redis is down.
        # The run_ingestion_cycle function in collector.py should handle MockRedis gracefully if that's what it gets.
        collected_articles = run_ingestion_cycle(redis_client=redis_client)
        logger.info(f"Ingestion cycle completed. Collected {len(collected_articles)} articles.")

        if not collected_articles:
            logger.info("No articles collected in this cycle.")
            return [ProcessingStatus(article_id="N/A", status="no_articles_collected", message="No new articles found during ingestion.")]

        processing_results = []
        for article_data in collected_articles:
            # Ensure article_data structure matches RawArticleInput or adapt it
            # Current collector.py `article_data` fields:
            # "id", "url", "headline", "body", "published_ts", "source"
            # `RawArticleInput` expects: "id", "headline", "body", "source", "published_ts"
            # The `url` field in `article_data` is not directly in `RawArticleInput` but might be part of processing.
            # The `id` generated by collector.py should be suitable.
            
            # Adapt the collected article_data to what news_service.process_article expects.
            # news_service.process_article takes a dictionary.
            # Let's assume it's compatible for now, but this is a key integration point.
            
            # For background processing of each article:
            # background_tasks.add_task(news_service.process_article, article_data)
            # For now, let's process synchronously to return status.

            article_id_for_status = article_data.get("id", "unknown_id")
            article_headline_for_status = article_data.get("headline", "N/A")

            try:
                logger.debug(f"Processing article: {article_id_for_status} - '{article_headline_for_status[:50]}...'")
                enriched_data = news_service.process_article(article_data) # Process the dictionary directly
                if enriched_data:
                    logger.info(f"Article {article_id_for_status} processed successfully by pipeline.")
                    processing_results.append(ProcessingStatus(
                        article_id=article_id_for_status,
                        status="processed_successfully"
                        # processed_data=enriched_data # Optional: include processed data
                    ))
                else:
                    logger.info(f"Article {article_id_for_status} rejected by pipeline.")
                    processing_results.append(ProcessingStatus(
                        article_id=article_id_for_status,
                        status="rejected_by_pipeline",
                        message="Article did not pass filtering or processing stages."
                    ))
            except Exception as e:
                logger.error(f"Error processing article {article_id_for_status} during pipeline cycle: {e}", exc_info=True)
                processing_results.append(ProcessingStatus(
                    article_id=article_id_for_status,
                    status="processing_error",
                    message=str(e)
                ))
        
        logger.info(f"--- Ingestion and Processing Cycle Completed. Processed {len(processing_results)} articles. ---")
        return processing_results

    except Exception as e:
        logger.exception(f"A general error occurred during the pipeline cycle: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during pipeline cycle: {str(e)}")


# --- Main Execution Block (for local development) ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server for {settings.app_name} v{settings.app_version} on http://{settings.server_host}:{settings.server_port}")
    uvicorn.run(app, host=settings.server_host, port=settings.server_port, log_level=settings.log_level.lower()) 