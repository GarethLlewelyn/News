import json
import hashlib
# Attempt to import NlpProcessor, assuming it's in a reachable path (e.g., PYTHONPATH)
# Or adjust import path based on actual project structure, e.g., from nlp.processor import NlpProcessor

# For now, to make this module potentially runnable standalone for testing without fully setting up PYTHONPATH,
# we'll use a try-except for the import and allow a placeholder if NlpProcessor can't be imported.
# In a full application context, this import should be direct.
try:
    from nlp.processor import NlpProcessor # Assumes nlp module is in the same root or PYTHONPATH
except ImportError:
    print("Warning: NlpProcessor could not be imported. NLP-based filters will be skipped.")
    print("Ensure 'nlp' module is in PYTHONPATH or adjust import statement in filtering/filters.py")
    NlpProcessor = None # Placeholder

# --- Configuration ---
# Path to the source whitelist JSON file. This file should be created manually or by another process.
# Example format: {"allowed_sources": ["Yahoo Finance", "Reuters", "TrustedSiteX"]}
SOURCE_WHITELIST_FILE = "source_whitelist.json"
MIN_HEADLINE_TOKENS = 5 # Example: minimum number of tokens for a headline to be considered
REQUIRE_ORG_ENTITY = True # Whether to require at least one ORG entity

# --- Filter Implementations ---

def load_source_whitelist():
    """Loads the list of whitelisted sources from a JSON file."""
    try:
        with open(SOURCE_WHITELIST_FILE, 'r', encoding='utf-8') as f:
            whitelist_data = json.load(f)
            return set(whitelist_data.get("allowed_sources", []))
    except FileNotFoundError:
        print(f"Warning: Source whitelist file not found at {SOURCE_WHITELIST_FILE}. Allowing all sources.")
        return None # Indicates all sources are allowed if file not found
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {SOURCE_WHITELIST_FILE}. Allowing all sources.")
        return None

def generate_article_hash(article):
    """Generates a SHA256 hash of the article's title and body for deduplication."""
    content = (article.get("title", "") + article.get("body", "")).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


class ArticleFilter:
    def __init__(self, nlp_processor_instance=None):
        self.source_whitelist = load_source_whitelist()
        self.seen_article_hashes = set() # For in-memory deduplication by content hash
        
        if NlpProcessor and not nlp_processor_instance:
            print("ArticleFilter: Initializing its own NlpProcessor instance for NLP filters.")
            self.nlp_processor = NlpProcessor() # Initialize if NlpProcessor is available and no instance is passed
        elif nlp_processor_instance:
            self.nlp_processor = nlp_processor_instance
            print("ArticleFilter: Using provided NlpProcessor instance.")
        else:
            self.nlp_processor = None
            print("ArticleFilter: NlpProcessor not available or not provided. NLP-based filters will be skipped.")

    def _tokenize_text(self, text):
        """Basic whitespace tokenizer."""
        if not text: return []
        return text.split()

    def filter_article(self, article):
        """Applies a series of filters to an article. 
           Returns True if the article passes all filters, False otherwise.
        """

        # 1. Filter by source whitelist
        if self.source_whitelist is not None:
            if article.get("source") not in self.source_whitelist:
                print(f"Filtered out by source whitelist: {article.get('title')} from {article.get('source')}")
                return False

        # 2. Filter by content hash (suppress duplicates with same title/body)
        article_hash = generate_article_hash(article)
        if article_hash in self.seen_article_hashes:
            print(f"Filtered out as duplicate (content hash): {article.get('title')}")
            return False
        self.seen_article_hashes.add(article_hash)

        # --- NLP-based Filters (require NlpProcessor) ---
        if self.nlp_processor and self.nlp_processor.nlp_spacy: # Check if spaCy model loaded in processor
            article_title = article.get("title", "")
            article_body = article.get("body", "") # For NER check, might use body or title+body

            # 3. Filter by headline token count
            title_tokens = self._tokenize_text(article_title)
            if len(title_tokens) < MIN_HEADLINE_TOKENS:
                print(f"Filtered (headline too short, {len(title_tokens)} tokens): '{article_title}'")
                return False

            # 4. Filter by presence of ORG NER entities
            if REQUIRE_ORG_ENTITY:
                # We need to run at least the NER part of NLP processing
                # Use the body for NER check as title might be too brief
                # For efficiency, only run processing if other checks pass
                # This means NlpProcessor might be called multiple times if not careful
                # Better to have a cached nlp_results if an article object is passed around
                # For now, let's assume we process relevant text for this check.
                text_for_ner = article_body if article_body else article_title
                if text_for_ner:
                    # We only need NER for this filter, not full sentiment etc.
                    # A more optimized NlpProcessor could have methods for specific tasks.
                    # For now, using the main process_article and checking its ner_org output.
                    nlp_results = self.nlp_processor.process_article(text_for_ner, article_title)
                    if not nlp_results.get("ner_org"):
                        print(f"Filtered (no ORG entities found): '{article_title}'")
                        return False
                else:
                    # No text to analyze for ORG entities
                    print(f"Filtered (no text for ORG entity check): '{article_title}'")
                    return False
        elif REQUIRE_ORG_ENTITY or MIN_HEADLINE_TOKENS > 0:
            print(f"Warning: NlpProcessor not available, skipping NLP-based filters for '{article.get('title')}'.")

        return True # Article passed all current filters


# --- Example Usage (for testing this module independently) ---
if __name__ == "__main__":
    print("\n--- Testing ArticleFilter with NLP capabilities ---")
    # Create a dummy whitelist file for testing
    dummy_whitelist_data = {"allowed_sources": ["Test Source A", "Test Source B", "Yahoo Finance"]}
    with open(SOURCE_WHITELIST_FILE, 'w', encoding='utf-8') as f_wl:
        json.dump(dummy_whitelist_data, f_wl)

    # Initialize NLP Processor if available (this might download models)
    nlp_proc_instance = None
    if NlpProcessor:
        print("Initializing NlpProcessor for filter tests...")
        nlp_proc_instance = NlpProcessor() # This instance will be passed to ArticleFilter
        if not nlp_proc_instance.nlp_spacy or not nlp_proc_instance.sentiment_analyzer:
            print("WARNING: NLP models in NlpProcessor did not load correctly. NLP filters might be skipped or fail.")
    else:
        print("NlpProcessor class not available. NLP filters will be skipped in tests.")

    article_filter = ArticleFilter(nlp_processor_instance=nlp_proc_instance)

    # Sample articles for testing
    articles_to_test = [
        {"url": "http://example.com/1", "title": "Apple Inc. Announces New Product", "body": "Apple today announced a new iPhone.", "source": "Test Source A"}, # Should pass
        {"url": "http://example.com/2", "title": "Short", "body": "News.", "source": "Test Source A"}, # Filtered by token count
        {"url": "http://example.com/3", "title": "Vague News Story Kicks Off", "body": "Something happened somewhere.", "source": "Test Source A"}, # Filtered if no ORG (depends on NLP)
        {"url": "http://example.com/4", "title": "Article From Unwanted Source", "body": "Content from XYZ Corp.", "source": "Test Source C"}, # Filtered by source
        {"url": "http://example.com/5", "title": "Apple Inc. Announces New Product", "body": "Apple today announced a new iPhone.", "source": "Test Source A"}, # Filtered by hash
        {"url": "http://example.com/6", "title": "Microsoft Corp. News Today", "body": "Microsoft discussed Azure.", "source": "Test Source B"}, # Should pass
        {"url": "http://example.com/7", "title": "Good Title But No Body Content", "body": "", "source": "Yahoo Finance"}, # Might pass token, but fail ORG if title doesn't have it and body is empty
    ]

    print("\nRunning filter tests:")
    filtered_articles_count = 0
    for art_idx, art in enumerate(articles_to_test):
        print(f"\n--- Test Article {art_idx + 1} ---")
        print(f"Input: Title='{art.get('title')}', Source='{art.get('source')}', Body snippet='{art.get('body')[:30]}...'")
        if article_filter.filter_article(art):
            filtered_articles_count += 1
            print(f"RESULT: PASSED - '{art.get('title')}'")
        else:
            print(f"RESULT: FILTERED - '{art.get('title')}'")

    print(f"\nTotal articles passed after filtering: {filtered_articles_count}")
    # Expected count depends heavily on whether NLP models load and correctly identify ORG entities.
    # If NLP works: Article 1 (Apple), Article 6 (Microsoft) should pass. 
    # Article 3 might pass if "Vague News Story" gets an ORG by mistake or if ORG check is lenient.
    # Article 7 ORG check depends on title processing by NLP.

    # Clean up dummy file
    import os
    try:
        os.remove(SOURCE_WHITELIST_FILE)
        print(f"Cleaned up {SOURCE_WHITELIST_FILE}")
    except OSError as e:
        print(f"Error removing dummy whitelist file: {e}")

    print("\nNote: If NLP models (spaCy, FinBERT) were downloaded, it was a one-time process.")
    print("If NLP filters were skipped, ensure NlpProcessor is correctly imported and initialized.") 