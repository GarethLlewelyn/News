# app/services/news_processor.py
import time
from typing import Any, Dict, Optional, List, Tuple
import logging

# Import individual filter functions and other necessary components
# Assuming they are in app.filtering, app.causality, app.graph, app.kafka_utils
import sys
import os
_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_DIR not in sys.path:
    sys.path.append(_APP_DIR)
# Ensure project root is also in path for app.config import
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from app.config import settings
from app.filtering.advanced import (
    is_syndicated,
    is_clickbait_headline,
    meets_entity_density_requirements,
    get_source_trust_score,
    load_json_config
)
from app.causality.extractor import extract_and_send_causality, NLP as SpacyNLP_from_extractor
from app.graph.propagator import EventPropagator
# from kafka_utils import KafkaProducer # Type hint, actual instance passed in

logger = logging.getLogger(__name__)

# Placeholder for actual FinBERT, RAKE models/functions
# In a real app, these would be proper classes or functions.
class FinBERTSentiment:
    def __init__(self, model_path: Optional[str] = None):
        # In a real app, model_path would be used to load the model
        # For now, it's a placeholder.
        self.model_path = model_path
        logger.info(f"MockFinBERTSentiment initialized. (Path: {model_path})")
    def predict(self, text: str) -> Dict[str, Any]:
        if "fell sharply" in text or "concerns" in text or "drop" in text or "negative" in text:
            return {"positive": 0.1, "negative": 0.8, "neutral": 0.1, "sentiment_label": "negative"}
        if "positive market sentiment" in text or "surge" in text or "strong sales" in text:
            return {"positive": 0.85, "negative": 0.05, "neutral": 0.1, "sentiment_label": "positive"}
        return {"positive": 0.3, "negative": 0.2, "neutral": 0.5, "sentiment_label": "neutral"}

class RAKETopicTagger:
    def __init__(self, config_path: Optional[str] = None):
        # Placeholder for RAKE model/config loading
        self.config_path = config_path
        logger.info(f"MockRAKETopicTagger initialized. (Config: {config_path})")
    def extract_topics(self, text: str) -> List[str]:
        if "earnings" in text: return ["earnings", "finance"]
        if "trade deal" in text: return ["trade", "politics"]
        if "iPhone sales" in text: return ["product launch", "sales", "technology"]
        return ["general news"]

class NewsProcessorService:
    def __init__(
        self,
        spacy_nlp: Any, 
        finbert_model: FinBERTSentiment,
        rake_model: RAKETopicTagger,
        redis_client: Any,
        kafka_producer: Any, 
        event_propagator: EventPropagator,
        # Configuration now comes from settings object, passed during service instantiation in main.py
        min_source_trust: float,
        min_tokens_density: int,
        min_org_entities_density: int,
        causality_topic: str,
        relations_topic: str,
        clickbait_patterns_path: str, # Path from settings
        source_profiles_path: str    # Path from settings
    ):
        self.spacy_nlp = spacy_nlp
        self.finbert_model = finbert_model
        self.rake_model = rake_model
        self.redis_client = redis_client
        self.kafka_producer = kafka_producer
        self.event_propagator = event_propagator
        
        self.min_source_trust = min_source_trust
        self.min_tokens_density = min_tokens_density
        self.min_org_entities_density = min_org_entities_density
        self.causality_topic = causality_topic
        self.relations_topic = relations_topic

        # Load configurations for filters using paths from settings
        # These are loaded once when the service is initialized.
        self.clickbait_patterns = load_json_config(os.path.join(_PROJECT_ROOT, clickbait_patterns_path), "Clickbait Patterns")
        if not self.clickbait_patterns: # Fallback if loading fails
            logger.warning(f"Failed to load clickbait patterns from {clickbait_patterns_path}. Using empty list.")
            self.clickbait_patterns = [] 

        self.source_profiles = load_json_config(os.path.join(_PROJECT_ROOT, source_profiles_path), "Source Profiles")
        if not self.source_profiles:
             logger.warning(f"Failed to load source profiles from {source_profiles_path}. Using empty dict.")
             self.source_profiles = {}

        logger.info("NewsProcessorService initialized.")
        if self.spacy_nlp is None or isinstance(self.spacy_nlp, type): # Check if it's a mock class type
            logger.warning("NewsProcessorService initialized with a non-functional or missing spaCy NLP model!")
        if self.event_propagator is None or isinstance(self.event_propagator, type):
             logger.warning("NewsProcessorService initialized with a non-functional or missing EventPropagator!")

    def _initial_filtering(self, article_data: Dict[str, Any]) -> Tuple[bool, str]:
        """ Stage 1: Filters that don't require deep NLP (Syndication, Clickbait, Source Trust)."""
        headline = article_data.get('headline')
        content = article_data.get('body') or article_data.get('content')
        source = article_data.get('source')

        if not all([headline, content, source]):
            logger.warning(f"Article {article_data.get('id', 'N/A')} missing essential fields for initial filtering.")
            return False, "Missing essential fields (headline, content, source) for initial filtering."

        # 1. Syndication Detection
        if is_syndicated(content, redis_client=self.redis_client, update_seen=True, syndication_key=settings.redis_syndication_key):
            logger.info(f"Article {article_data.get('id', 'N/A')} identified as syndicated content.")
            return False, "Syndicated content"

        # 2. Clickbait Detection (using loaded patterns)
        if is_clickbait_headline(headline, clickbait_patterns_config=self.clickbait_patterns):
            logger.info(f"Article {article_data.get('id', 'N/A')} identified as clickbait.")
            return False, "Clickbait headline"
        
        # 3. Source Trust (using loaded profiles)
        trust_score = get_source_trust_score(source, source_profiles_config=self.source_profiles, default_trust_score=self.min_source_trust) # Pass default from service setting
        if trust_score < self.min_source_trust:
            logger.info(f"Article {article_data.get('id', 'N/A')} from {source} failed source trust: {trust_score:.2f} < {self.min_source_trust}")
            return False, f"Source trust score {trust_score:.2f} below threshold {self.min_source_trust}"
        
        return True, "Passed initial filters"

    def _perform_nlp(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 2: Core NLP processing (spaCy NER, FinBERT sentiment, RAKE topics)."""
        text_content = article_data.get('body') or article_data.get('content', '')
        processed_data = article_data.copy()
        article_id = article_data.get('id', 'N/A')

        doc = None
        try:
            if self.spacy_nlp and not isinstance(self.spacy_nlp, type):
                doc = self.spacy_nlp(text_content)
        except Exception as e:
            logger.error(f"spaCy processing failed for article {article_id}: {e}", exc_info=True)
            # Continue without spaCy features, or re-raise depending on desired strictness
        
        entities = []
        org_entity_count = 0
        token_count = len(text_content.split()) # Basic fallback if doc is None

        if doc:
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            org_entity_count = sum(1 for ent in doc.ents if ent.label_ == "ORG")
            token_count = len(doc)
        else:
            logger.warning(f"spaCy Doc is None for article {article_id}. NLP features will be limited.")
        
        processed_data['spacy_doc'] = doc 
        processed_data['entities_ner'] = entities
        processed_data['org_entity_count'] = org_entity_count
        processed_data['token_count'] = token_count

        try:
            sentiment_result = self.finbert_model.predict(text_content)
            processed_data['sentiment'] = sentiment_result 
        except Exception as e:
            logger.error(f"FinBERT sentiment analysis failed for article {article_id}: {e}", exc_info=True)
            processed_data['sentiment'] = {"error": str(e)} # Store error

        try:
            topics = self.rake_model.extract_topics(text_content)
            processed_data['topics'] = topics
        except Exception as e:
            logger.error(f"RAKE topic tagging failed for article {article_id}: {e}", exc_info=True)
            processed_data['topics'] = [f"error: {str(e)}"] # Store error
        
        return processed_data

    def _density_filtering(self, article_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Stage 3: Filters that require NLP output (Entity Density)."""
        token_count = article_data.get('token_count', 0)
        org_entity_count = article_data.get('org_entity_count', 0)
        article_id = article_data.get('id', 'N/A')

        if not meets_entity_density_requirements(token_count, org_entity_count, 
                                                 self.min_tokens_density, self.min_org_entities_density):
            logger.info(f"Article {article_id} failed entity density (tokens: {token_count}, orgs: {org_entity_count}). Min tokens: {self.min_tokens_density}, min orgs: {self.min_org_entities_density}")
            return False, f"Fails entity density (tokens: {token_count}, orgs: {org_entity_count})"
        
        return True, "Passed density filter"

    def _extract_and_propagate(self, processed_article_data: Dict[str, Any]):
        """Stage 4: Causality extraction and Event propagation, sending to Kafka."""
        text_content = processed_article_data.get('body') or processed_article_data.get('content', '')
        article_id = processed_article_data.get('id', f"article_{int(time.time())}")
        primary_entity = None # Determine primary entity if possible (e.g., most frequent ORG)
        
        # For simplicity, pick the first ORG entity as primary if available for propagation trigger
        # A more sophisticated method would be needed here.
        if processed_article_data.get('entities_ner'):
            for entity_text, label in processed_article_data['entities_ner']:
                if label == "ORG":
                    primary_entity = entity_text # This should be a normalized ID eventually
                    break
        
        # Causality Extraction and Sending
        # extract_and_send_causality expects the spacy_nlp to be loaded (it is, via self.spacy_nlp which uses causality.extractor.NLP)
        # It handles its own Kafka sending if producer is provided.
        num_causal = 0
        if self.kafka_producer: # Only attempt if Kafka is available
            try:
                num_causal = extract_and_send_causality(
                    producer=self.kafka_producer,
                    topic=self.causality_topic,
                    text=text_content, 
                    spacy_doc_input=processed_article_data.get('spacy_doc'),
                    message_key_prefix=article_id
                )
                if num_causal > 0:
                    logger.info(f"Sent {num_causal} causal relations for article {article_id} to topic '{self.causality_topic}'.")
            except Exception as e:
                logger.error(f"Causality extraction/sending failed for article {article_id}: {e}", exc_info=True)
        else:
            logger.warning(f"Kafka producer not available. Skipping causality sending for article {article_id}.")

        # Event Propagation and Sending
        # Requires a primary entity and its sentiment
        num_propagated = 0
        if primary_entity and processed_article_data.get('sentiment') and not processed_article_data['sentiment'].get('error'):
            if self.kafka_producer and self.event_propagator and not isinstance(self.event_propagator, type):
                try:
                    sentiment_score = processed_article_data['sentiment'].get('positive', 0.0) - \
                                      processed_article_data['sentiment'].get('negative', 0.0)
                    
                    original_ts = processed_article_data.get('published_ts', int(time.time() * 1000))
                    
                    num_propagated = self.event_propagator.propagate_and_send_relations(
                        producer=self.kafka_producer,
                        topic=self.relations_topic,
                        subject_id=primary_entity, # This needs to match keys in your entity_relation_map.json
                        original_sentiment=sentiment_score,
                        original_ts=original_ts,
                        message_key_prefix=article_id
                    )
                    if num_propagated > 0:
                        logger.info(f"Sent {num_propagated} propagated relations for {primary_entity} (article {article_id}) to topic '{self.relations_topic}'.")
                except Exception as e:
                    logger.error(f"Event propagation/sending failed for article {article_id}, entity {primary_entity}: {e}", exc_info=True)
            elif not self.kafka_producer:
                 logger.warning(f"Kafka producer not available. Skipping event propagation for article {article_id}.")
            elif not self.event_propagator or isinstance(self.event_propagator, type):
                 logger.warning(f"Event propagator not available/functional. Skipping event propagation for article {article_id}.")
        elif not primary_entity:
            logger.debug(f"No primary ORG entity found for event propagation in article {article_id}.")
        elif processed_article_data.get('sentiment', {}).get('error'):
            logger.warning(f"Sentiment analysis failed for article {article_id}. Skipping event propagation.")

    def process_article(self, raw_article_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a single raw article through the full pipeline.
        Returns the enriched article data if successfully processed, None otherwise.
        """
        article_id = raw_article_data.get('id', 'N/A')
        headline_preview = raw_article_data.get('headline', 'N/A')[:50]
        logger.info(f"--- Starting processing for article: {article_id} ('{headline_preview}...') ---")

        # Stage 1: Initial Filtering (Syndication, Clickbait, Source Trust)
        passed_initial, reason_initial = self._initial_filtering(raw_article_data)
        if not passed_initial:
            logger.info(f"Article {article_id} REJECTED at initial filtering: {reason_initial}")
            return None
        logger.debug(f"Article {article_id} PASSED initial filters.")
        
        # Stage 2: Core NLP (NER, Sentiment, Topics)
        # This populates 'token_count' and 'org_entity_count' needed for density filter.
        processed_data = self._perform_nlp(raw_article_data)
        logger.debug(f"Article {article_id} NLP processed. Tokens: {processed_data.get('token_count')}, ORGs: {processed_data.get('org_entity_count')}")
        logger.debug(f"  Sentiment: {processed_data.get('sentiment', {}).get('sentiment_label')}")
        logger.debug(f"  Topics: {processed_data.get('topics')}")

        # Stage 3: Density Filtering (requires counts from NLP)
        passed_density, reason_density = self._density_filtering(processed_data)
        if not passed_density:
            logger.info(f"Article {article_id} REJECTED at density filtering: {reason_density}")
            return None # Or just don't proceed to causality/propagation
        logger.debug(f"Article {article_id} PASSED density filter.")

        # Stage 4: Causality Extraction & Event Propagation (includes Kafka sending)
        self._extract_and_propagate(processed_data)
        
        # Stage 5: Final Storage (e.g., to .jsonl as per original plan)
        # For this example, we just return the enriched data. Actual storage would happen here.
        processed_data.pop('spacy_doc', None) # Remove non-serializable spacy_doc before final output/storage
        logger.info(f"Article processing COMPLETED for: {article_id} ('{headline_preview}...')")
        return processed_data


if __name__ == '__main__':
    # This __main__ is for basic testing of the service logic directly.
    # It does not run the FastAPI app.
    # For full app testing, run app/main.py

    # Setup basic logging for this test script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running NewsProcessorService direct test example...")

    # --- Mock Dependencies (simplified for direct testing) ---
    # 1. spaCy NLP Model (reuse from extractor or a simple mock)
    mock_spacy_nlp = SpacyNLP_from_extractor
    if mock_spacy_nlp is None:
        logger.warning("__main__: spaCy model not loaded. NLP features will be limited.")
        class MockSpacyDoc: 
            def __init__(self, text): self.text = text; self.ents = []; self.sents = [self]
            def __len__(self): return len(self.text.split())
        class MockSpacy: 
            def __call__(self, text): return MockSpacyDoc(text)
        mock_spacy_nlp = MockSpacy()

    # 2. FinBERT & RAKE (using mock classes defined above)
    mock_finbert = FinBERTSentiment() # Add mock model_path if settings were used here
    mock_rake = RAKETopicTagger()

    # 3. Redis Client (mock)
    class MockRedis:
        _store = {}
        def sismember(self, key, value): return value in self._store.get(key, set())
        def sadd(self, key, value):
            self._store.setdefault(key, set()).add(value)
            return 1
    mock_redis_client = MockRedis()

    # 4. Kafka Producer (mock)
    class MockKafkaProducer:
        def send(self, topic, value, key=None): logger.info(f"MockKafka: send to {topic}, key={key}, value={value}"); return True
        def flush(self, timeout=None): logger.info("MockKafka: flush called"); return True
        def close(self, timeout=None): logger.info("MockKafka: close called")
    mock_kafka_p = MockKafkaProducer()

    # 5. Event Propagator (mock or real with accessible map)
    try:
        # Use real settings path for entity_relation_map for more realistic test
        propagator_map_path = os.path.join(_PROJECT_ROOT, settings.entity_relation_map_path)
        if not os.path.exists(propagator_map_path):
            logger.warning(f"Entity relation map not found at {propagator_map_path} for test. Creating dummy map.")
            os.makedirs(os.path.dirname(propagator_map_path), exist_ok=True)
            with open(propagator_map_path, 'w') as f:
                json.dump({"DEFAULT_ENTITY": {"related_to": []}}, f) # Dummy map
        propagator = EventPropagator(entity_map_path=propagator_map_path)
    except Exception as e:
        logger.error(f"Error initializing EventPropagator for test: {e}. Using dummy.")
        class DummyPropagator: 
            def propagate_and_send_relations(self, *args, **kwargs): logger.info("DummyPropagator: propagate_and_send_relations called."); return 0
        propagator = DummyPropagator()

    # Initialize the service using settings values
    news_service = NewsProcessorService(
        spacy_nlp=mock_spacy_nlp,
        finbert_model=mock_finbert,
        rake_model=mock_rake,
        redis_client=mock_redis_client,
        kafka_producer=mock_kafka_p,
        event_propagator=propagator,
        min_source_trust=settings.default_min_source_trust,
        min_tokens_density=settings.default_min_tokens_density,
        min_org_entities_density=settings.default_min_org_entities_density,
        causality_topic=settings.kafka_causality_topic + "_test",
        relations_topic=settings.kafka_relations_topic + "_test",
        clickbait_patterns_path=settings.clickbait_patterns_path, # These paths are now used by service __init__
        source_profiles_path=settings.source_profiles_path
    )

    sample_articles_raw = [
        {
            "id": "art001_svc_test",
            "headline": "MSFT Corp Stock Fell Sharply Due to Lower Earnings",
            "body": "Shares of MSFT Corp fell sharply due to lower than expected earnings and concerns about future growth. The company addressed these concerns in a recent statement.",
            "source": "reuters.com",
            "published_ts": int(time.time() * 1000) - 100000
        },
        {
            "id": "art002_svc_test",
            "headline": "You WON\'T BELIEVE What Telsa Inc Did Next! Stock Price Soars!",
            "body": "Telsa Inc announced a new revolutionary battery, causing its stock price to surge. The positive market sentiment was also because of a new trade deal.",
            "source": "someblog.com", 
            "published_ts": int(time.time() * 1000) - 50000
        },
        {
            "id": "art003_svc_test",
            "headline": "Short news", 
            "body": "Very little text.", 
            "source": "cnbc.com",
            "published_ts": int(time.time() * 1000)
        }
    ]

    processed_results_count = 0
    for article_json in sample_articles_raw:
        result = news_service.process_article(article_json)
        if result:
            processed_results_count += 1
            # logger.info(f"Test article {article_json.get('id')} processed result: {result}")
        logger.info("-" * 50)
    
    logger.info(f"Total articles processed successfully in test: {processed_results_count} out of {len(sample_articles_raw)}") 