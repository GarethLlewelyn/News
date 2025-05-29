import time
import json # For potentially printing rich data if needed, and for consistency

# Import modules from the project
from ingestion.collector import run_ingestion_cycle
from filtering.filters import ArticleFilter
from nlp.processor import NlpProcessor
from relevance.classifier import RelevanceClassifier
from storage.writer import save_bulk_processed_articles, save_processed_article

# --- Global Components Initialisation ---
# These are initialized once and reused to save on model loading times, etc.
NLP_PROCESSOR_INSTANCE = None
ARTICLE_FILTER_INSTANCE = None
RELEVANCE_CLASSIFIER_INSTANCE = None

def initialize_components():
    """Initializes global instances of processors and classifiers."""
    global NLP_PROCESSOR_INSTANCE, ARTICLE_FILTER_INSTANCE, RELEVANCE_CLASSIFIER_INSTANCE
    
    print("Initializing pipeline components...")
    start_time = time.time()

    if not NLP_PROCESSOR_INSTANCE:
        print("Initializing NlpProcessor...")
        NLP_PROCESSOR_INSTANCE = NlpProcessor()
        # Check if NLP models loaded successfully, critical for pipeline
        if not NLP_PROCESSOR_INSTANCE.nlp_spacy or not NLP_PROCESSOR_INSTANCE.sentiment_analyzer:
            print("CRITICAL ERROR: NLP models in NlpProcessor did not load. Pipeline cannot continue effectively.")
            # Depending on desired behavior, could raise an exception or exit
            # For now, it will allow proceeding but NLP-dependent steps will fail or be skipped.

    if not ARTICLE_FILTER_INSTANCE:
        print("Initializing ArticleFilter...")
        # ArticleFilter can take an NlpProcessor instance to use its capabilities
        ARTICLE_FILTER_INSTANCE = ArticleFilter(nlp_processor_instance=NLP_PROCESSOR_INSTANCE)

    if not RELEVANCE_CLASSIFIER_INSTANCE:
        print("Initializing RelevanceClassifier...")
        RELEVANCE_CLASSIFIER_INSTANCE = RelevanceClassifier()
    
    end_time = time.time()
    print(f"Pipeline components initialized in {end_time - start_time:.2f} seconds.")

# --- Main Pipeline Logic ---

def run_full_pipeline_cycle():
    """Runs one full cycle of the news processing pipeline."""
    if not NLP_PROCESSOR_INSTANCE or not ARTICLE_FILTER_INSTANCE or not RELEVANCE_CLASSIFIER_INSTANCE:
        print("Error: Pipeline components not initialized. Call initialize_components() first.")
        return

    print("\n--- Starting New Pipeline Cycle ---")
    pipeline_start_time = time.time()

    # 1. Ingestion
    print("\nStep 1: Ingesting articles...")
    raw_articles = run_ingestion_cycle() # From ingestion.collector
    print(f"Ingested {len(raw_articles)} raw articles.")
    if not raw_articles:
        print("No articles ingested. Ending cycle.")
        return

    # 2. Filtering, NLP, Relevance Classification, and Merging
    print("\nStep 2: Filtering, Processing (NLP), and Classifying Relevance...")
    processed_articles_for_storage = []
    articles_filtered_out = 0
    articles_processed_successfully = 0
    articles_failed_processing = 0

    for i, article in enumerate(raw_articles):
        print(f'\nProcessing article {i+1}/{len(raw_articles)}: "{article.get("title", "[No Title]")[:60]}..."')
        
        # Ensure basic article structure for subsequent steps
        if not isinstance(article, dict) or not article.get("url") or not article.get("title"):
            print(f"  Skipping malformed raw article: {article}")
            articles_filtered_out +=1 # or a different counter for malformed data
            continue

        # Temporarily get NLP data for logging BEFORE filtering (for a few articles)
        if i < 3: # Log for the first 3 articles
            try:
                temp_nlp_data = NLP_PROCESSOR_INSTANCE.process_article(article.get("body", ""), article.get("title", ""))
                print(f"  DEBUG NLP (pre-filter) for '{article.get('title', '')[:30]}...': ner_org: {temp_nlp_data.get('ner_org')}, all_ner: {temp_nlp_data.get('_debug_ner_all')}")
            except Exception as e:
                print(f"  DEBUG NLP (pre-filter) failed for '{article.get('title', '')[:30]}...': {e}")

        # Apply filters
        if not ARTICLE_FILTER_INSTANCE.filter_article(article):
            print(f'  Article filtered out: "{article.get("title", "[No Title]")}"')
            articles_filtered_out += 1
            continue
        print(f"  Article passed filters.")

        try:
            # NLP Processing
            article_text = article.get("body", "")
            article_title = article.get("title", "")
            print("    Running NLP processing...")
            nlp_data = NLP_PROCESSOR_INSTANCE.process_article(article_text, article_title)
            # nlp_data will be like: {"entity_id": ..., "sent_pos": ..., ...}
            print("    NLP processing complete.")

            # Relevance Classification
            print("    Running relevance classification...")
            relevance_data = RELEVANCE_CLASSIFIER_INSTANCE.classify_relevance(article_text, article_title)
            # relevance_data will be like: {"relevance_label": ..., "relevance_score": ...}
            print("    Relevance classification complete.")

            # Merge all data for the final processed article
            # Start with the original (and potentially filtered) article data
            final_article_data = article.copy() # Make a copy to avoid modifying the iterated list item directly
            final_article_data.update(nlp_data) # Add NLP results
            final_article_data.update(relevance_data) # Add relevance results
            
            # Ensure consistent naming for output schema (as per project plan)
            # The nlp_processor already tries to match the schema, but we can ensure here.
            # Example project output: {"entity_id": "AAPL", "sent_pos": 0.7, "sent_neg": 0.1, 
            #                         "topics": ["earnings", "forecast"], "relevance_label": "short_term", 
            #                         "source": "Reuters", "published_ts": 1717190000000}
            # We need to ensure original fields like 'url', 'title', 'body', 'publish_ts', 'source' are present.
            # `article.copy()` should preserve these. `nlp_data` adds its fields, `relevance_data` adds its.

            processed_articles_for_storage.append(final_article_data)
            articles_processed_successfully += 1
            print(f'  Successfully processed: "{article.get("title", "[No Title]")}"')

        except Exception as e:
            articles_failed_processing += 1
            print(f'  ERROR processing article "{article.get("title", "[No Title]")}": {e}')
            # Optionally, save errored articles or details to a separate log/file

    print(f"\nFiltering & Processing Summary:")
    print(f"  Articles passed filters and processed: {articles_processed_successfully}")
    print(f"  Articles filtered out: {articles_filtered_out}")
    print(f"  Articles failed during NLP/Relevance processing: {articles_failed_processing}")

    # 3. Storage
    if processed_articles_for_storage:
        print(f"\nStep 3: Saving {len(processed_articles_for_storage)} processed articles...")
        saved_count, failed_save_count = save_bulk_processed_articles(processed_articles_for_storage)
        print(f"Storage complete. Saved: {saved_count}, Failed to save: {failed_save_count}")
    else:
        print("\nStep 3: No processed articles to save.")

    pipeline_end_time = time.time()
    print(f"\n--- Pipeline Cycle Completed in {pipeline_end_time - pipeline_start_time:.2f} seconds ---")

# --- Main Execution ---
if __name__ == "__main__":
    print("=== Financial News AI Pipeline Start ===")
    
    # Initialize components (loads models, etc.)
    # This is crucial and should only be done once if the script is run as a service/daemon.
    # For a single run, it's fine here.
    initialize_components()

    # Run one full pipeline cycle for demonstration
    # In a production scenario, this might be scheduled (e.g., cron) or run in a continuous loop with delays.
    try:
        run_full_pipeline_cycle()
    except Exception as e:
        print(f"An unexpected error occurred during the pipeline run: {e}")
        # Add more robust error logging here in a production system

    print("\n=== Financial News AI Pipeline End ===")
    print("Note: NLP models (spaCy, FinBERT) and NLTK data might have been downloaded on first run.")
    print("Check the 'data/processed/' directory for output files.") 