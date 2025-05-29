import hashlib
import re
import json
import os
from typing import Any, Dict, List, Set, Tuple, Optional
from urllib.parse import urlparse
import logging
import sys

# Assuming app.config is accessible
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_APP_DIR)) #This is assuming app/filtering/advanced.py (app is two levels up)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _APP_DIR not in sys.path: # Ensure app itself is also there
    sys.path.append(os.path.dirname(_APP_DIR)) # Add parent of app dir for `from app.config...`

from app.config import settings # Import centralized settings

logger = logging.getLogger(__name__)

# --- Configuration Loading ---
def load_json_config(config_path: str, config_name: str) -> Dict:
    """Loads a JSON configuration file."""
    # config_path is now expected to be an absolute path or relative to where this script is run.
    # If using settings, it might be relative to project root, so construct absolute path before calling.
    if not os.path.exists(config_path):
        logger.warning(f"{config_name} configuration file not found at {config_path}. Returning empty config.")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {config_path} for {config_name}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to load {config_name} from {config_path}: {e}", exc_info=True)
    return {}

# Load configurations at module level using paths from settings
# Construct absolute paths from project root
CLICKBAIT_PATTERNS_CONFIG_PATH = os.path.join(_PROJECT_ROOT, settings.clickbait_patterns_path)
SOURCE_PROFILES_CONFIG_PATH = os.path.join(_PROJECT_ROOT, settings.source_profiles_path)

CLICKBAIT_PATTERNS_CONFIG = load_json_config(CLICKBAIT_PATTERNS_CONFIG_PATH, "Clickbait Patterns")
SOURCE_PROFILES_CONFIG = load_json_config(SOURCE_PROFILES_CONFIG_PATH, "Source Profiles")

DEFAULT_CLICKBAIT_PATTERNS = CLICKBAIT_PATTERNS_CONFIG.get("patterns", [
    r"you won\'t believe",
    r"will shock you",
    r"(secret|trick|hack) to",
    r"number \d+ will",
    r"^\d+ (reasons|ways|tips|things)",
    r"this one weird",
    r"what happens next",
    r"can\'t handle this",
    r"is this the (worst|best)",
    r"(incredible|amazing|unbelievable)[?!]*$"
])
DEFAULT_SOURCE_TRUST_SCORE = SOURCE_PROFILES_CONFIG.get("defaultTrustScore", 0.5)

# Redis related constant for syndication checking
REDIS_SYNDICATION_SET_KEY = "financial_news:syndicated_article_hashes"

# In-memory store for article body hashes for syndication detection
# In a production system, this would likely be a Redis set or similar.
SEEN_ARTICLE_HASHES: Set[str] = set()

def get_article_hash(content: str) -> str:
    """Generates a SHA-256 hash for the given article content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def is_syndicated(
    content: str, 
    redis_client: Any, 
    update_seen: bool = True, 
    syndication_key: Optional[str] = None
) -> bool:
    """
    Checks if the article content hash has been seen before (syndication).
    Uses settings.redis_syndication_key if syndication_key is not provided.
    """
    if not redis_client:
        logger.warning("Redis client not available for syndication check. Assuming not syndicated.")
        return False
    
    key_to_use = syndication_key if syndication_key is not None else settings.redis_syndication_key
    article_hash = get_article_hash(content)
    
    try:
        if redis_client.sismember(key_to_use, article_hash):
            logger.debug(f"Syndication check: Hash {article_hash} found in Redis key '{key_to_use}'.")
            return True
        if update_seen:
            redis_client.sadd(key_to_use, article_hash)
            logger.debug(f"Syndication check: Hash {article_hash} added to Redis key '{key_to_use}'.")
    except Exception as e: # Catch generic Redis errors
        logger.error(f"Redis error during syndication check for key '{key_to_use}': {e}", exc_info=True)
        return False # Fail safe: treat as not syndicated if Redis fails
    return False

def is_clickbait_headline(headline: str, clickbait_patterns_config: Optional[List[str]] = None) -> bool:
    """Checks if a headline matches common clickbait patterns."""
    patterns_to_use = clickbait_patterns_config if clickbait_patterns_config is not None else DEFAULT_CLICKBAIT_PATTERNS
    if not patterns_to_use:
        logger.debug("No clickbait patterns loaded or provided. Skipping check.")
        return False
    for pattern in patterns_to_use:
        try:
            if re.search(pattern, headline, re.IGNORECASE):
                logger.debug(f"Clickbait pattern '{pattern}' matched headline: '{headline[:50]}...'")
                return True
        except re.error as e:
            logger.error(f"Regex error in clickbait pattern '{pattern}': {e}")
            continue # Skip invalid patterns
    return False

def meets_entity_density_requirements(
    token_count: int, 
    org_entity_count: int, 
    min_tokens: int, 
    min_org_entities: int
) -> bool:
    """Checks if the article meets minimum token and ORG entity density."""
    if token_count < min_tokens:
        logger.debug(f"Entity density: Failed token count ({token_count} < {min_tokens})")
        return False
    if org_entity_count < min_org_entities:
        logger.debug(f"Entity density: Failed ORG entity count ({org_entity_count} < {min_org_entities})")
        return False
    return True

def _get_domain_from_url(url_string: str) -> Optional[str]:
    """Extracts the domain (e.g., example.com) from a URL."""
    if not url_string or not isinstance(url_string, str):
        return None
    try:
        parsed_url = urlparse(url_string)
        netloc = parsed_url.netloc
        if not netloc and parsed_url.path: # Handle cases like "example.com/path" without scheme
            netloc = urlparse(f"http://{url_string}").netloc
        
        if not netloc: return None

        # Remove www. and port numbers
        domain_parts = netloc.split('.')
        if domain_parts[0] == 'www':
            domain_parts = domain_parts[1:]
        
        # Reconstruct domain, handle potential port
        domain_no_port = '.'.join(domain_parts).split(':')[0]

        # Handle cases like .co.uk, .com.au (simple check, might need more robust TLD list for perfection)
        if len(domain_parts) > 2 and domain_parts[-2] in ['co', 'com', 'org', 'net', 'gov', 'edu']:
            return '.'.join(domain_parts[-3:]).split(':')[0]
        elif len(domain_parts) >= 2:
            return '.'.join(domain_parts[-2:]).split(':')[0]
        else: # Should not happen if netloc was valid
            return domain_no_port # or netloc.split(':')[0]

    except Exception as e:
        logger.error(f"Error parsing domain from URL '{url_string}': {e}", exc_info=True)
        return None

def get_source_trust_score(
    source_url_or_id: str, 
    source_profiles_config: Optional[Dict] = None,
    default_trust_score: Optional[float] = None
) -> float:
    """Gets the trust score for a source. Uses loaded SOURCE_PROFILES_CONFIG by default."""
    profiles_to_use = source_profiles_config if source_profiles_config is not None else SOURCE_PROFILES_CONFIG
    effective_default_score = default_trust_score if default_trust_score is not None else DEFAULT_SOURCE_TRUST_SCORE

    if not profiles_to_use:
        logger.debug("No source profiles loaded or provided. Returning default trust score.")
        return effective_default_score

    domain = _get_domain_from_url(source_url_or_id) 
    if not domain: # If it's not a URL, maybe it's a direct ID like 'reuters'
        domain = source_url_or_id.lower()

    if domain in profiles_to_use:
        score = profiles_to_use[domain].get("trustScore", effective_default_score)
        logger.debug(f"Source '{domain}' found in profiles. Trust score: {score}")
        return score
    
    # Try matching subdomains: e.g., if "news.example.com" not found, check "example.com"
    parts = domain.split('.')
    if len(parts) > 2:
        parent_domain = '.'.join(parts[1:])
        if parent_domain in profiles_to_use:
            score = profiles_to_use[parent_domain].get("trustScore", effective_default_score)
            logger.debug(f"Source '{domain}' (matched parent '{parent_domain}') found in profiles. Trust score: {score}")
            return score
            
    logger.debug(f"Source '{domain}' not found in profiles. Returning default trust score: {effective_default_score}")
    return effective_default_score

def apply_advanced_filters(
    article_data: Dict[str, Any],
    redis_client: Any, # redis.Redis or mock
    check_syndication: bool = True,
    check_clickbait: bool = True,
    check_source_trust: bool = True,
    check_entity_density: bool = True,
    min_source_trust: Optional[float] = None,
    min_tokens_density: Optional[int] = None,
    min_org_entities_density: Optional[int] = None,
    # Pass loaded configs if not using module-level ones, or for testing
    syndication_key_override: Optional[str] = None, 
    clickbait_patterns_override: Optional[List[str]] = None,
    source_profiles_override: Optional[Dict] = None 
) -> Tuple[bool, str]:
    """
    Applies a series of advanced filters to an article.
    Uses thresholds from settings if not provided.
    Uses loaded configs (CLICKBAIT_PATTERNS_CONFIG, SOURCE_PROFILES_CONFIG) by default.
    """
    article_id = article_data.get('id', 'N/A')
    logger.debug(f"Applying advanced filters for article {article_id}")

    headline = article_data.get('headline')
    body_content = article_data.get('body') or article_data.get('content')
    source = article_data.get('source')

    if not all([headline, body_content, source]):
        logger.warning(f"Article {article_id} missing required fields for filtering.")
        return False, "Missing required fields for filtering (headline, body/content, or source)"

    # Resolve thresholds: use argument if provided, else use settings default
    effective_min_source_trust = min_source_trust if min_source_trust is not None else settings.default_min_source_trust
    effective_min_tokens = min_tokens_density if min_tokens_density is not None else settings.default_min_tokens_density
    effective_min_orgs = min_org_entities_density if min_org_entities_density is not None else settings.default_min_org_entities_density

    if check_syndication:
        synd_key = syndication_key_override if syndication_key_override is not None else settings.redis_syndication_key
        if is_syndicated(body_content, redis_client, syndication_key=synd_key):
            logger.info(f"Filter: Article {article_id} rejected as syndicated.")
            return False, "Syndicated content"

    if check_clickbait:
        cb_patterns = clickbait_patterns_override if clickbait_patterns_override is not None else CLICKBAIT_PATTERNS_CONFIG.get("patterns", DEFAULT_CLICKBAIT_PATTERNS)
        if is_clickbait_headline(headline, clickbait_patterns_config=cb_patterns):
            logger.info(f"Filter: Article {article_id} rejected as clickbait.")
            return False, "Clickbait headline"

    if check_source_trust:
        src_profiles = source_profiles_override if source_profiles_override is not None else SOURCE_PROFILES_CONFIG
        default_st_score = src_profiles.get("defaultTrustScore", effective_min_source_trust) # Use config's default if available

        trust_score = get_source_trust_score(source, source_profiles_config=src_profiles, default_trust_score=default_st_score)
        if trust_score < effective_min_source_trust:
            logger.info(f"Filter: Article {article_id} from {source} rejected due to low trust score ({trust_score:.2f} < {effective_min_source_trust:.2f}).")
            return False, f"Source trust score {trust_score:.2f} below threshold {effective_min_source_trust:.2f}"

    if check_entity_density:
        # These counts should ideally come from an NLP pre-processing step if available
        token_count = article_data.get('token_count', 0) 
        org_entity_count = article_data.get('org_entity_count', 0)
        if not meets_entity_density_requirements(token_count, org_entity_count, effective_min_tokens, effective_min_orgs):
            logger.info(f"Filter: Article {article_id} rejected due to entity density (tokens:{token_count}, orgs:{org_entity_count}). Mins: tk={effective_min_tokens}, org={effective_min_orgs}")
            return False, f"Failed entity density requirements (tokens: {token_count}, orgs: {org_entity_count})"

    logger.debug(f"Article {article_id} passed all enabled advanced filters.")
    return True, "Passed all enabled advanced filters"

if __name__ == '__main__':
    # Setup basic logging for this test script
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running advanced_filters.py direct test...")

    # Mock Redis client for direct testing
    class MockRedis:
        _store = {}
        def sismember(self, key, value): 
            logger.debug(f"MockRedis.sismember called: key={key}, value={value}, result={value in self._store.get(key, set())}")
            return value in self._store.get(key, set())
        def sadd(self, key, value):
            logger.debug(f"MockRedis.sadd called: key={key}, value={value}")
            self._store.setdefault(key, set()).add(value)
            return 1
    mock_redis = MockRedis()

    # Ensure config files exist for test or create dummies
    if not os.path.exists(CLICKBAIT_PATTERNS_CONFIG_PATH):
        logger.warning(f"Test: Clickbait patterns file not found at {CLICKBAIT_PATTERNS_CONFIG_PATH}. Creating dummy.")
        os.makedirs(os.path.dirname(CLICKBAIT_PATTERNS_CONFIG_PATH), exist_ok=True)
        with open(CLICKBAIT_PATTERNS_CONFIG_PATH, 'w') as f:
            json.dump({"patterns": ["test clickbait pattern"]}, f)
        # Reload for test if it was empty initially
        CLICKBAIT_PATTERNS_CONFIG = load_json_config(CLICKBAIT_PATTERNS_CONFIG_PATH, "Clickbait Patterns")
        DEFAULT_CLICKBAIT_PATTERNS = CLICKBAIT_PATTERNS_CONFIG.get("patterns", [])

    if not os.path.exists(SOURCE_PROFILES_CONFIG_PATH):
        logger.warning(f"Test: Source profiles file not found at {SOURCE_PROFILES_CONFIG_PATH}. Creating dummy.")
        os.makedirs(os.path.dirname(SOURCE_PROFILES_CONFIG_PATH), exist_ok=True)
        with open(SOURCE_PROFILES_CONFIG_PATH, 'w') as f:
            json.dump({"defaultTrustScore": 0.6, "goodsite.com": {"trustScore": 0.9}}, f)
        SOURCE_PROFILES_CONFIG = load_json_config(SOURCE_PROFILES_CONFIG_PATH, "Source Profiles")
        DEFAULT_SOURCE_TRUST_SCORE = SOURCE_PROFILES_CONFIG.get("defaultTrustScore", 0.5)


    sample_articles = [
        {
            "id": "adv_test_001",
            "headline": "Normal news about something important",
            "body": "This is a full article body for testing purposes. It mentions AlphaCorp.",
            "source": "https://www.goodsite.com/article1",
            "token_count": 20, "org_entity_count": 1
        },
        {
            "id": "adv_test_002",
            "headline": "YOU WON\'T BELIEVE THIS SECRET!",
            "body": "This is a clickbait article. It offers a secret.",
            "source": "https://www.badsite.com/page2",
            "token_count": 15, "org_entity_count": 0
        },
        {
            "id": "adv_test_003", # Syndicated of 001
            "headline": "Normal news about something important",
            "body": "This is a full article body for testing purposes. It mentions AlphaCorp.",
            "source": "https://www.anothersite.com/syndicated",
            "token_count": 20, "org_entity_count": 1
        },
        {
            "id": "adv_test_004",
            "headline": "Very short",
            "body": "Too short. No orgs.",
            "source": "https://www.goodsite.com/shorty",
            "token_count": 5, "org_entity_count": 0 # Fails density
        },
        {
            "id": "adv_test_005",
            "headline": "Article from untrusted source",
            "body": "Content from a source not explicitly listed and may fall to default trust.",
            "source": "https://unknownrandomsite.net/article",
            "token_count": 30, "org_entity_count": 1
        }
    ]

    logger.info(f"\nUsing clickbait patterns: {DEFAULT_CLICKBAIT_PATTERNS}")
    logger.info(f"Using source profiles: {SOURCE_PROFILES_CONFIG}")
    logger.info(f"Using default trust score: {DEFAULT_SOURCE_TRUST_SCORE}")
    logger.info(f"Using Redis syndication key: {settings.redis_syndication_key}")
    logger.info(f"Using min source trust: {settings.default_min_source_trust}")
    logger.info(f"Using min tokens: {settings.default_min_tokens_density}, min orgs: {settings.default_min_org_entities_density}\n")


    for article in sample_articles:
        logger.info(f"--- Processing article: {article['id']} ---")
        passed, reason = apply_advanced_filters(
            article_data=article,
            redis_client=mock_redis,
            # Using defaults from settings for thresholds which are now used inside apply_advanced_filters
        )
        if passed:
            logger.info(f"Article '{article['id']}' PASSED filters. Reason: {reason}")
        else:
            logger.warning(f"Article '{article['id']}' FAILED filters. Reason: {reason}")
        print("") # Blank line for readability

    # Test syndication again for article 001 to see if it's caught
    logger.info("--- Reprocessing article adv_test_001 (should now be syndicated) ---")
    passed_again, reason_again = apply_advanced_filters(sample_articles[0], mock_redis)
    if not passed_again and reason_again == "Syndicated content":
        logger.info("Correctly identified adv_test_001 as syndicated on second pass.")
    else:
        logger.error(f"Syndication test failed for adv_test_001 on second pass. Passed: {passed_again}, Reason: {reason_again}")

    logger.info("\nadvanced_filters.py direct test finished.") 