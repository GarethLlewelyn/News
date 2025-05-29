import pytest
from unittest.mock import patch, MagicMock, mock_open
import hashlib
import json

# Adjust import path
import sys
import os
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app'))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from filtering.advanced import (
    get_article_hash,
    is_syndicated,
    is_clickbait_headline,
    meets_entity_density_requirements,
    get_source_trust_score,
    apply_advanced_filters,
    load_json_config,
    _get_domain_from_url # Also test private helper if complex
)

# --- Fixtures ---

@pytest.fixture
def mock_redis_client():
    client = MagicMock()
    client.sismember = MagicMock(return_value=False)
    client.sadd = MagicMock(return_value=1)
    return client

@pytest.fixture
def sample_article_data():
    return {
        "headline": "This is a Normal Headline about Something Interesting",
        "body": "This is the main body of the article. It contains several words and some entities like AlphaCorp and BetaTech.",
        "content": "This is the main content of the article. It contains several words and some entities like AlphaCorp and BetaTech.", # Fallback for body
        "source": "https://www.reputablesource.com/news/article1",
        "token_count": 20, # Assume pre-calculated for some tests
        "org_entity_count": 2 # Assume pre-calculated for some tests
    }

# --- Tests for individual functions ---

def test_get_article_hash():
    content1 = "This is some article content."
    content2 = "This is some different article content."
    hash1 = get_article_hash(content1)
    hash2 = get_article_hash(content2)
    hash1_again = get_article_hash(content1)

    assert isinstance(hash1, str)
    assert len(hash1) == 64 # SHA-256
    assert hash1 != hash2
    assert hash1 == hash1_again
    assert get_article_hash("") == hashlib.sha256("".encode('utf-8')).hexdigest()

def test_is_syndicated_new_article(mock_redis_client):
    content = "Unique article content."
    mock_redis_client.sismember.return_value = False # Not seen before
    
    assert not is_syndicated(content, mock_redis_client, update_seen=True)
    mock_redis_client.sismember.assert_called_once()
    mock_redis_client.sadd.assert_called_once() # Should be added

def test_is_syndicated_seen_article(mock_redis_client):
    content = "Previously seen article content."
    article_hash = get_article_hash(content)
    mock_redis_client.sismember.return_value = True # Seen before
    
    assert is_syndicated(content, mock_redis_client, update_seen=True)
    mock_redis_client.sismember.assert_called_with("SEEN_ARTICLE_HASHES", article_hash)
    mock_redis_client.sadd.assert_not_called() # Should not be added again if update_seen is true but already member

def test_is_syndicated_no_update_seen(mock_redis_client):
    content = "Another unique article."
    mock_redis_client.sismember.return_value = False
    
    assert not is_syndicated(content, mock_redis_client, update_seen=False)
    mock_redis_client.sismember.assert_called_once()
    mock_redis_client.sadd.assert_not_called() # Should not be added

def test_is_syndicated_redis_error(mock_redis_client):
    content = "Content that causes redis error."
    mock_redis_client.sismember.side_effect = Exception("Redis connection failed")
    
    # Default behavior is to return False (treat as not syndicated) on error
    assert not is_syndicated(content, mock_redis_client)
    mock_redis_client.sadd.assert_not_called() # Should not attempt add if sismember failed

    mock_redis_client.sadd.side_effect = Exception("Redis connection failed during add")
    mock_redis_client.sismember.side_effect = None # reset sismember
    mock_redis_client.sismember.return_value = False
    # Still should be not syndicated, error during add is logged but doesn't make it syndicated
    assert not is_syndicated(content, mock_redis_client, update_seen=True)


@pytest.mark.parametrize("headline, expected", [
    ("You WON\'T BELIEVE What Happens Next!", True),
    ("10 Reasons Why Clickbait is Bad For You - Number 7 Will SHOCK You!", True),
    ("Normal News Headline About Market Trends", False),
    ("Tech Company Announces New Product", False),
    ("Is This The Worst Mistake Ever Made?", True), # Interrogative, common in clickbait
    ("Incredible! This Changes Everything!", True),
    ("", False) # Empty headline
])
def test_is_clickbait_headline(headline, expected):
    # Using default patterns loaded by the module
    assert is_clickbait_headline(headline) == expected

def test_is_clickbait_headline_custom_patterns():
    custom_patterns = [r"custom pattern", r"another one \d+"]
    assert is_clickbait_headline("This matches a custom pattern here", custom_patterns)
    assert not is_clickbait_headline("Normal headline", custom_patterns)

def test_meets_entity_density_requirements():
    assert meets_entity_density_requirements(token_count=100, org_entity_count=3, min_tokens=50, min_org_entities=1)
    assert not meets_entity_density_requirements(token_count=30, org_entity_count=3, min_tokens=50, min_org_entities=1) # Too few tokens
    assert not meets_entity_density_requirements(token_count=100, org_entity_count=0, min_tokens=50, min_org_entities=1) # Too few orgs
    assert not meets_entity_density_requirements(token_count=10, org_entity_count=0, min_tokens=50, min_org_entities=1) # Both too few
    assert meets_entity_density_requirements(token_count=50, org_entity_count=1, min_tokens=50, min_org_entities=1) # Exact match

def test_get_domain_from_url():
    assert _get_domain_from_url("https://www.example.com/path/to/page?q=1") == "example.com"
    assert _get_domain_from_url("http://subdomain.example.co.uk/other") == "example.co.uk"
    assert _get_domain_from_url("ftp://example.com") == "example.com"
    assert _get_domain_from_url("example.com/path") == "example.com" # No scheme
    assert _get_domain_from_url("localhost:8000") == "localhost"
    assert _get_domain_from_url("http://192.168.1.1/test") == "192.168.1.1"
    assert _get_domain_from_url("Just a string") is None # Invalid URL
    assert _get_domain_from_url("") is None # Empty string

def test_get_source_trust_score_default_profiles():
    # Assumes default source_profiles.json is loaded by the module
    # These tests depend on the content of that default file.
    # For more robust tests, mock load_json_config.
    
    # Example: If "reuters.com" has score 0.9 and "untrustedsite.com" has 0.2
    # and default is 0.5
    with patch('filtering.advanced.SOURCE_PROFILES_CONFIG', {
        "reuters.com": {"trustScore": 0.9, "category": "High"},
        "example.com": {"trustScore": 0.7},
        "someblog.com": {"trustScore": 0.3, "category": "Low"},
        "defaultTrustScore": 0.5
    }):
        assert get_source_trust_score("https://www.reuters.com/article") == 0.9
        assert get_source_trust_score("http://news.example.com/story") == 0.7
        assert get_source_trust_score("https://someblog.com/post1") == 0.3
        assert get_source_trust_score("https://unknownsite.org") == 0.5 # Default
        assert get_source_trust_score("invalid-url") == 0.5 # Default for unparseable

def test_get_source_trust_score_custom_profiles():
    custom_profiles = {
        "myfavesite.com": {"trustScore": 0.95},
        "banned.com": {"trustScore": 0.01},
        "defaultTrustScore": 0.4
    }
    assert get_source_trust_score("http://myfavesite.com", custom_profiles) == 0.95
    assert get_source_trust_score("https://www.banned.com/page.html", custom_profiles) == 0.01
    assert get_source_trust_score("http://another.com", custom_profiles) == 0.4

def test_load_json_config_success():
    mock_file_content = '{"key": "value", "number": 123}'
    expected_data = {"key": "value", "number": 123}
    
    with patch("builtins.open", mock_open(read_data=mock_file_content)):
        with patch("os.path.exists", return_value=True):
            data = load_json_config("dummy_path.json", "Test")
            assert data == expected_data

def test_load_json_config_file_not_found():
    with patch("os.path.exists", return_value=False):
        data = load_json_config("nonexistent.json", "Test")
        assert data == {} # Should return empty dict

def test_load_json_config_json_decode_error():
    # Malformed JSON content (e.g., trailing comma, missing quotes)
    malformed_json_content = '{"key": "value", "number": 123,}' 
    with patch("builtins.open", mock_open(read_data=malformed_json_content)) as mock_file:
        with patch("os.path.exists", return_value=True):
            data = load_json_config("bad_json.json", "Test")
            assert data == {}
            mock_file.assert_called_once_with("bad_json.json", "r", encoding="utf-8")

# --- Tests for apply_advanced_filters (orchestrator) ---

@patch('filtering.advanced.is_syndicated')
@patch('filtering.advanced.is_clickbait_headline')
@patch('filtering.advanced.get_source_trust_score')
@patch('filtering.advanced.meets_entity_density_requirements')
def test_apply_advanced_filters_all_pass(
    mock_density, mock_trust, mock_clickbait, mock_syndicated, 
    sample_article_data, mock_redis_client
):
    mock_syndicated.return_value = False
    mock_clickbait.return_value = False
    mock_trust.return_value = 0.8 # Above threshold
    mock_density.return_value = True

    result, reason = apply_advanced_filters(
        sample_article_data, mock_redis_client, 
        min_source_trust=0.5, min_tokens_density=50, min_org_entities_density=1
    )
    assert result is True
    assert reason == "Passed all enabled advanced filters"

def test_apply_advanced_filters_syndicated(sample_article_data, mock_redis_client):
    with patch('filtering.advanced.is_syndicated', return_value=True) as mock_synd:
        result, reason = apply_advanced_filters(sample_article_data, mock_redis_client)
        assert result is False
        assert reason == "Syndicated content"
        mock_synd.assert_called_once()

def test_apply_advanced_filters_clickbait(sample_article_data, mock_redis_client):
    with patch('filtering.advanced.is_syndicated', return_value=False):
        with patch('filtering.advanced.is_clickbait_headline', return_value=True) as mock_click:
            result, reason = apply_advanced_filters(sample_article_data, mock_redis_client)
            assert result is False
            assert reason == "Clickbait headline"
            mock_click.assert_called_once()

def test_apply_advanced_filters_low_trust(sample_article_data, mock_redis_client):
    with patch('filtering.advanced.is_syndicated', return_value=False):
        with patch('filtering.advanced.is_clickbait_headline', return_value=False):
            with patch('filtering.advanced.get_source_trust_score', return_value=0.2) as mock_trust:
                result, reason = apply_advanced_filters(sample_article_data, mock_redis_client, min_source_trust=0.5)
                assert result is False
                assert "Source trust score 0.20 below threshold 0.50" in reason
                mock_trust.assert_called_once()

def test_apply_advanced_filters_low_density(sample_article_data, mock_redis_client):
    with patch('filtering.advanced.is_syndicated', return_value=False):
        with patch('filtering.advanced.is_clickbait_headline', return_value=False):
            with patch('filtering.advanced.get_source_trust_score', return_value=0.8):
                with patch('filtering.advanced.meets_entity_density_requirements', return_value=False) as mock_density:
                    article = sample_article_data.copy()
                    article['token_count'] = 10 # ensure these are used
                    article['org_entity_count'] = 0
                    result, reason = apply_advanced_filters(
                        article, mock_redis_client, 
                        min_tokens_density=50, min_org_entities_density=1
                    )
                    assert result is False
                    assert "Failed entity density requirements" in reason
                    mock_density.assert_called_with(10, 0, 50, 1)

def test_apply_advanced_filters_missing_fields(mock_redis_client):
    article_missing_headline = {"body": "body", "source": "source.com"}
    result, reason = apply_advanced_filters(article_missing_headline, mock_redis_client)
    assert result is False
    assert reason == "Missing required fields for filtering (headline, body/content, or source)"
    
    article_missing_body = {"headline": "headline", "source": "source.com"}
    result, reason = apply_advanced_filters(article_missing_body, mock_redis_client)
    assert result is False
    assert reason == "Missing required fields for filtering (headline, body/content, or source)"

def test_apply_advanced_filters_density_fields_missing(sample_article_data, mock_redis_client):
    article_no_counts = sample_article_data.copy()
    del article_no_counts['token_count']
    del article_no_counts['org_entity_count']

    with patch('filtering.advanced.is_syndicated', return_value=False):
        with patch('filtering.advanced.is_clickbait_headline', return_value=False):
            with patch('filtering.advanced.get_source_trust_score', return_value=0.8):
                # Density check should effectively be skipped or use defaults if fields are missing
                # The function apply_advanced_filters itself extracts these, so if they are missing from input,
                # they will be 0.
                with patch('filtering.advanced.meets_entity_density_requirements') as mock_density:
                    mock_density.return_value = False # Make it fail if called with 0s
                    result, reason = apply_advanced_filters(article_no_counts, mock_redis_client)
                    assert result is False # Fails because density check uses 0,0
                    assert "Failed entity density requirements" in reason
                    mock_density.assert_called_with(0,0, 30, 0) # Default min values in apply_advanced_filters

def test_apply_advanced_filters_individual_toggles(sample_article_data, mock_redis_client):
    # Test that a filter is skipped if its toggle is False
    with patch('filtering.advanced.is_syndicated', return_value=True) as mock_synd: # Should fail if called
        result, _ = apply_advanced_filters(sample_article_data, mock_redis_client, check_syndication=False)
        assert result is True # Assuming other filters pass or are also disabled by default in this call
        mock_synd.assert_not_called()

    with patch('filtering.advanced.is_clickbait_headline', return_value=True) as mock_click:
        result, _ = apply_advanced_filters(sample_article_data, mock_redis_client, check_clickbait=False)
        assert result is True 
        mock_click.assert_not_called()

    with patch('filtering.advanced.get_source_trust_score', return_value=0.1) as mock_trust: # Should fail
        result, _ = apply_advanced_filters(sample_article_data, mock_redis_client, check_source_trust=False)
        assert result is True
        mock_trust.assert_not_called()

    with patch('filtering.advanced.meets_entity_density_requirements', return_value=False) as mock_density: # Should fail
        result, _ = apply_advanced_filters(sample_article_data, mock_redis_client, check_entity_density=False)
        assert result is True
        mock_density.assert_not_called()

# Example test for ensuring default configs are loaded if files are missing/corrupt
# This requires a bit more setup to control what load_json_config returns in the module.
@patch('filtering.advanced.load_json_config')
def test_is_clickbait_with_default_fallback_patterns(mock_load_config):
    mock_load_config.return_value = {} # Simulate config file loading failed or was empty

    # Re-import or reload the module or specific function to use this mocked load_json_config
    # This is tricky. A better way is to make load_json_config a parameter or use dependency injection.
    # For now, let's assume is_clickbait_headline has its own internal default if config is empty.
    # The current `is_clickbait_headline` uses `CLICKBAIT_PATTERNS_CONFIG` which is loaded at module level.
    # To test this properly, you'd need to reload the `filtering.advanced` module after patching.
    # Pytest doesn't easily support module reloading between tests.
    
    # Alternative: Test the behavior assuming the internal default patterns are used.
    # This depends on knowing what those internal defaults are if config load fails.
    # from filtering.advanced import DEFAULT_CLICKBAIT_PATTERNS # If such a default exists and is accessible
    
    # For this example, we'll assume that if config is empty, it uses some hardcoded basic pattern.
    # Or, we can test that it doesn't crash and perhaps returns False for everything.
    # This test is more conceptual without module reloading.
    
    # If filtering.advanced.CLICKBAIT_PATTERNS_CONFIG is patched directly:
    with patch('filtering.advanced.CLICKBAIT_PATTERNS_CONFIG', []): # No patterns
         assert not is_clickbait_headline("YOU WONT BELIEVE THIS")
    
    with patch('filtering.advanced.CLICKBAIT_PATTERNS_CONFIG', [r"test only pattern"]):
        assert is_clickbait_headline("this is a test only pattern here")
        assert not is_clickbait_headline("normal headline")


</rewritten_file> 