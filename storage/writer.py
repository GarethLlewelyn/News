import json
import os
import datetime

# --- Configuration ---
PROCESSED_DATA_DIR = "data/processed/"
# Ensure the directory exists
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)
    print(f"Created directory: {PROCESSED_DATA_DIR}")

# --- Article Writer ---

def save_processed_article(article_data, filename_prefix="news_features"):
    """Saves a single processed article to a JSONL file.
    The file will be named based on the prefix and current date.
    Each article is appended as a new line in JSON format.

    Args:
        article_data (dict): The fully processed article data including original fields,
                             NLP outputs, and relevance classification.
        filename_prefix (str): Prefix for the output filename.
    """
    if not isinstance(article_data, dict):
        print("Error: article_data must be a dictionary.")
        return False

    # Generate filename based on current date to group articles by day
    # e.g., news_features_YYYY-MM-DD.jsonl
    date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    output_filename = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_{date_str}.jsonl")

    try:
        with open(output_filename, 'a', encoding='utf-8') as f:
            json.dump(article_data, f)
            f.write('\n')
        # print(f"Successfully saved article to {output_filename}: {article_data.get('title', '[No Title]')[:50]}...")
        return True
    except IOError as e:
        print(f"Error saving article to {output_filename}: {e}")
        return False
    except TypeError as e:
        print(f"Error serializing article data to JSON: {e}. Article data: {article_data}")
        return False

def save_bulk_processed_articles(articles_list, filename_prefix="news_features"):
    """Saves a list of processed articles to a JSONL file.
    Uses the save_processed_article for each article, ensuring they go to the same daily file.

    Args:
        articles_list (list): A list of processed article data dictionaries.
        filename_prefix (str): Prefix for the output filename.
    """
    if not isinstance(articles_list, list):
        print("Error: articles_list must be a list of dictionaries.")
        return False
    
    saved_count = 0
    failed_count = 0
    
    if not articles_list:
        print("No articles to save.")
        return saved_count, failed_count

    # All articles in this bulk save will go to the same dated file
    date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    output_filename = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_{date_str}.jsonl")
    
    print(f"Starting bulk save of {len(articles_list)} articles to {output_filename}...")
    try:
        with open(output_filename, 'a', encoding='utf-8') as f:
            for article_data in articles_list:
                if not isinstance(article_data, dict):
                    print(f"Skipping non-dictionary item in articles_list: {article_data}")
                    failed_count += 1
                    continue
                try:
                    json.dump(article_data, f)
                    f.write('\n')
                    saved_count += 1
                except TypeError as te:
                    print(f"Error serializing article data to JSON: {te}. Article: {article_data.get('title', '{}')}")
                    failed_count +=1 
        print(f"Bulk save completed. Saved: {saved_count}, Failed: {failed_count} to {output_filename}")
    except IOError as e:
        print(f"Error during bulk save to {output_filename}: {e}")
        # If the file open fails, all are considered failed for this batch
        failed_count = len(articles_list) - saved_count 

    return saved_count, failed_count

# --- Example Usage (for testing this module independently) ---
if __name__ == "__main__":
    print("\n--- Testing Storage Writer ---")

    # Ensure the processed data directory exists for the test
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    # Sample processed article data (combining outputs from all stages)
    sample_article_1 = {
        "url": "http://example.com/news/1",
        "title": "Big News Today! Markets React!",
        "body": "Detailed content about the big news and market reactions...",
        "publish_ts": int(datetime.datetime.utcnow().timestamp() * 1000),
        "source": "Test Source",
        "entity_id": "COMPANY_A",
        "sent_pos": 0.85,
        "sent_neg": 0.05,
        "ner": ["Company A", "New York"],
        "topics": ["earnings", "market reaction", "finance"],
        "relevance_label": "short_term",
        "relevance_score": 0.92
    }
    sample_article_2 = {
        "url": "http://example.com/news/2",
        "title": "Future Plans for 2028 Unveiled",
        "body": "Company B outlines its long-term strategy for the next five years, targeting expansion by 2028.",
        "publish_ts": int(datetime.datetime.utcnow().timestamp() * 1000) - (86400000*2), # 2 days ago
        "source": "Another Source",
        "entity_id": "COMPANY_B",
        "sent_pos": 0.60,
        "sent_neg": 0.10,
        "ner": ["Company B", "Global"],
        "topics": ["strategy", "expansion", "long-term"],
        "relevance_label": "long_term",
        "relevance_score": 0.88
    }
    sample_article_3_invalid = "This is not a dict"

    print("\nTesting single article save:")
    save_processed_article(sample_article_1, filename_prefix="test_features")
    save_processed_article(sample_article_2, filename_prefix="test_features")

    # Construct a filename to check for test output
    test_date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    expected_filename = os.path.join(PROCESSED_DATA_DIR, f"test_features_{test_date_str}.jsonl")
    print(f"Check for output in: {expected_filename}")
    
    # Test bulk save
    print("\nTesting bulk article save:")
    articles_for_bulk = [
        {**sample_article_1, "id": "bulk_1"}, 
        {**sample_article_2, "id": "bulk_2", "title": "Slightly different title for bulk"},
        sample_article_3_invalid, # type: ignore
        {**sample_article_1, "id": "bulk_3", "publish_ts": None} # Test with potential serialization issue
    ]
    saved, failed = save_bulk_processed_articles(articles_for_bulk, filename_prefix="test_bulk_features")
    print(f"Bulk save result - Saved: {saved}, Failed: {failed}")
    expected_bulk_filename = os.path.join(PROCESSED_DATA_DIR, f"test_bulk_features_{test_date_str}.jsonl")
    print(f"Check for bulk output in: {expected_bulk_filename}")

    print("\nStorage writer test complete.")
    print(f"Make sure to clean up test files in {PROCESSED_DATA_DIR} if necessary.") 