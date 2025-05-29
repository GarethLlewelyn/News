# import spacy
# from transformers import pipeline # For FinBERT
# import pytextrank # For TextRank

# --- Configuration ---
# SPACY_MODEL = "en_core_web_sm"
# FINBERT_MODEL = "ProsusAI/finbert" # Or other suitable FinBERT model

# --- NLP Processor Class ---

# class NlpProcessor:
#     def __init__(self):
#         # Load models (this can take time, so ideally done once)
#         print(f"Loading spaCy model: {SPACY_MODEL}...")
#         try:
#             self.nlp_spacy = spacy.load(SPACY_MODEL)
#         except OSError:
#             print(f"spaCy model {SPACY_MODEL} not found. Please download it: python -m spacy download {SPACY_MODEL}")
#             # Potentially exit or raise an exception if model is critical
#             self.nlp_spacy = None # Or handle this more gracefully

#         if self.nlp_spacy:
#             # Add pytextrank to spaCy pipeline if desired for topic tagging
#             # Make sure pytextrank is compatible and installed
#             # self.nlp_spacy.add_pipe("textrank")
#             print("spaCy model loaded.")

#         print(f"Loading FinBERT model: {FINBERT_MODEL}...")
#         try:
#             # device = 0 if torch.cuda.is_available() else -1 # For GPU if available
#             self.sentiment_analyzer = pipeline("sentiment-analysis", model=FINBERT_MODEL, tokenizer=FINBERT_MODEL)
#             print("FinBERT model loaded.")
#         except Exception as e:
#             print(f"Error loading FinBERT model {FINBERT_MODEL}: {e}")
#             self.sentiment_analyzer = None
        
#     def process_article(self, article_text):
#         """Processes article text to extract sentiment, NER, and topics."""
#         results = {
#             "sentiment": None, # e.g., {"label": "positive", "score": 0.9}
#             "ner_org": [],     # List of organization names
#             "ner_gpe": [],     # List of geopolitical entities
#             "topics": []       # List of topic keywords
#         }

#         if not article_text:
#             return results

#         # 1. Sentiment Analysis with FinBERT
#         if self.sentiment_analyzer:
#             try:
#                 sentiment_result = self.sentiment_analyzer(article_text)
#                 # FinBERT typically returns a list with a dict: [{'label': 'positive', 'score': 0.98...}]
#                 if sentiment_result and isinstance(sentiment_result, list):
#                     results["sentiment"] = sentiment_result[0]
#             except Exception as e:
#                 print(f"Error during sentiment analysis: {e}")
        
#         # 2. NER and Topic Tagging with spaCy
#         if self.nlp_spacy:
#             try:
#                 doc = self.nlp_spacy(article_text)
                
#                 # Extract NER (ORG and GPE)
#                 for ent in doc.ents:
#                     if ent.label_ == "ORG":
#                         results["ner_org"].append(ent.text)
#                     elif ent.label_ == "GPE":
#                         results["ner_gpe"].append(ent.text)
                
#                 # Extract Topics using TextRank (if pipe was added)
#                 # This depends on how pytextrank stores results in doc._.phrases
#                 # if hasattr(doc._, 'phrases') and doc._.phrases:
#                 #     results["topics"] = [phrase.text for phrase in doc._.phrases[:5]] # Top 5 phrases

#             except Exception as e:
#                 print(f"Error during spaCy processing (NER/Topics): {e}")

#         return results

# --- Main function for testing this module ---
# if __name__ == "__main__":
#     print("Initializing NLP Processor...")
#     processor = NlpProcessor()

#     sample_article_text = (
#         "Apple Inc. announced record profits for the last quarter, driven by strong iPhone sales in China. "
#         "Tim Cook, CEO of Apple, mentioned plans to expand into new markets in South East Asia. "
#         "However, some analysts from Google and Microsoft expressed concerns about future growth."
#     )
    
#     if processor.nlp_spacy and processor.sentiment_analyzer: # Check if models loaded
#         print(f"\nProcessing sample article text:\n'{sample_article_text}'")
#         processed_data = processor.process_article(sample_article_text)
#         print("\nNLP Processing Results:")
#         import json
#         print(json.dumps(processed_data, indent=2))

#         # Example output structure from sprint plan:
#         # { "entity_id": "AAPL", "sent_pos": 0.6, "sent_neg": 0.0, 
#         #   "ner": ["Apple", "Tim Cook"], "topics": ["earnings", "guidance"] }
#         # Need to adapt the output to match this, especially sent_pos/sent_neg from FinBERT label/score
#         # and how entity_id is derived.
#     else:
#         print("NLP models not loaded. Cannot run test.")

print("NLP module placeholder. Implementation to follow.") 

import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# import pytextrank # For TextRank - to be added if explicitly chosen over RAKE etc.
import torch # PyTorch is a dependency for transformers
import nltk # For RAKE or other NLTK based keyword extractors
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

# Ensure NLTK data is available (run this once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab') # Check for punkt_tab
except nltk.downloader.DownloadError:
    print("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab', quiet=True) # Download punkt_tab

# --- Configuration ---
SPACY_MODEL = "en_core_web_sm"
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# --- NLP Processor Class ---

class NlpProcessor:
    def __init__(self):
        # Load models (this can take time, so ideally done once)
        print(f"Loading spaCy model: {SPACY_MODEL}...")
        try:
            self.nlp_spacy = spacy.load(SPACY_MODEL)
            print("spaCy model loaded.")
        except OSError:
            print(f"spaCy model '{SPACY_MODEL}' not found. Please download it by running: python -m spacy download {SPACY_MODEL}")
            self.nlp_spacy = None

        # if self.nlp_spacy:
            # Add pytextrank to spaCy pipeline if desired for topic tagging
            # Make sure pytextrank is compatible and installed
            # Example: if 'textrank' not in self.nlp_spacy.pipe_names:
            # try:
            #     self.nlp_spacy.add_pipe("textrank")
            #     print("pytextrank pipe added to spaCy.")
            # except Exception as e:
            #     print(f"Could not add pytextrank to spaCy pipeline: {e}. Topics might not be generated via TextRank.")

        print(f"Loading FinBERT model: {FINBERT_MODEL_NAME}...")
        try:
            # Determine device (GPU if available, else CPU)
            self.device = 0 if torch.cuda.is_available() else -1 
            if self.device == 0:
                print("CUDA (GPU) available. FinBERT will run on GPU.")
            else:
                print("CUDA (GPU) not available. FinBERT will run on CPU.")

            # Load tokenizer and model for FinBERT
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            if self.device == 0: # Move model to GPU if available
                 self.finbert_model = self.finbert_model.to("cuda")

            # Create a pipeline instance using the loaded model and tokenizer
            # Note: Using device directly in pipeline constructor for transformers >= 4.0.0
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model=self.finbert_model, 
                tokenizer=self.finbert_tokenizer,
                device=self.device # Pass device index
            )
            print("FinBERT model loaded and pipeline created.")
        except Exception as e:
            print(f"Error loading FinBERT model '{FINBERT_MODEL_NAME}': {e}")
            self.sentiment_analyzer = None
            self.finbert_tokenizer = None
            self.finbert_model = None
    
    def _get_finbert_sentiment(self, text):
        """Helper to get sentiment from FinBERT and structure it."""
        if not self.sentiment_analyzer or not text:
            return {"sent_pos": None, "sent_neg": None, "sent_neu": None, "sentiment_label": "N/A"}

        try:
            # FinBERT expects single sentences or short paragraphs. 
            # For longer texts, consider splitting or summarizing first, or processing sentence by sentence.
            # Here, we pass the whole text, but be mindful of token limits (usually 512 for BERT models)
            # Truncate if necessary
            inputs = self.finbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=510)
            if self.device == 0: # Move inputs to GPU if model is on GPU
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad(): # Disable gradient calculations for inference
                outputs = self.finbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Labels for ProsusAI/finbert are typically: 0:positive, 1:negative, 2:neutral
            # Verify this based on model config if issues arise.
            # model.config.id2label should provide this mapping
            id2label = self.finbert_model.config.id2label
            
            scores_list = predictions[0].tolist() # Get scores for the first (and only) input
            
            # Create a mapping from label string to score
            label_to_score = {}
            for i, score_val in enumerate(scores_list):
                label = id2label.get(i, f"unknown_label_{i}")
                label_to_score[label.lower()] = score_val
            
            sentiment_scores = {
                "sent_pos": label_to_score.get("positive"),
                "sent_neg": label_to_score.get("negative"),
                "sent_neu": label_to_score.get("neutral"), # Corrected: use .get() for safety
                "sentiment_label": id2label[torch.argmax(predictions).item()] # Dominant label
            }
            # Adjust to the specific output format required by the project (pos/neg only)
            return {
                "sent_pos": sentiment_scores["sent_pos"],
                "sent_neg": sentiment_scores["sent_neg"],
                "dominant_sentiment": sentiment_scores["sentiment_label"]
            }

        except Exception as e:
            print(f"Error during FinBERT sentiment analysis: {e}")
            return {"sent_pos": None, "sent_neg": None, "dominant_sentiment": "Error"}

    def _extract_rake_keywords(self, text, max_keywords=5):
        """Extracts keywords using a simplified RAKE-like approach (nltk based for now)."""
        if not text:
            return []
        try:
            # Simple RAKE: score phrases by summing word scores (word_freq / word_degree)
            # This is a very basic interpretation. For a more robust RAKE, use a dedicated library
            # or implement the full algorithm (stopwords, co-occurrence graph, etc.)
            sentences = sent_tokenize(text.lower()) # NLTK sentence tokenization
            words_in_sentences = [word_tokenize(s) for s in sentences]
            all_words = [word for sublist in words_in_sentences for word in sublist if word.isalpha() and len(word) > 2]
            
            if not all_words:
                return []

            word_freq = Counter(all_words)
            # For simplicity, consider all words as candidates for now, not full RAKE phrases
            # A more advanced RAKE would identify candidate phrases (sequences of non-stop words)
            
            # Score words (simplified: just frequency for this basic version)
            # True RAKE scores phrases based on sum of member word scores (freq/degree)
            scored_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            return [kw for kw, score in scored_keywords[:max_keywords]]
        except Exception as e:
            print(f"Error during RAKE keyword extraction: {e}")
            return []

    def process_article(self, article_text, article_title=""):
        """Processes article text to extract sentiment, NER, and topics.
           article_title can be used for deriving entity_id if no prominent ORG found in body.
        """
        results = {
            "entity_id": None,   # e.g., "AAPL"
            "sent_pos": None,
            "sent_neg": None,
            "dominant_sentiment": None, # Added for clarity from FinBERT
            "ner_org": [],       # List of organization names
            "ner_gpe": [],       # List of geopolitical entities
            "ner_all": [],       # List of all named entities (text, label)
            "topics": []         # List of topic keywords
        }

        if not article_text and not article_title:
            return results
        
        text_to_process = article_text if article_text else ""

        # 1. Sentiment Analysis with FinBERT
        sentiment_data = self._get_finbert_sentiment(text_to_process)
        results["sent_pos"] = sentiment_data["sent_pos"]
        results["sent_neg"] = sentiment_data["sent_neg"]
        results["dominant_sentiment"] = sentiment_data["dominant_sentiment"]
        
        # 2. NER and Topic Tagging with spaCy
        if self.nlp_spacy:
            try:
                doc = self.nlp_spacy(text_to_process)
                
                # Extract NER (ORG and GPE specifically, and all for context)
                for ent in doc.ents:
                    results["ner_all"].append({"text": ent.text, "label": ent.label_})
                    if ent.label_ == "ORG":
                        results["ner_org"].append(ent.text)
                    elif ent.label_ == "GPE":
                        results["ner_gpe"].append(ent.text)
                
                # Determine entity_id: For now, use the first ORG found. 
                # This needs refinement, e.g., map to ticker, use title if no ORG in body.
                if results["ner_org"]:
                    results["entity_id"] = results["ner_org"][0] 
                # Fallback or more complex logic for entity_id can be added here
                # elif article_title: # Potentially try to extract from title if no ORG in body
                #    title_doc = self.nlp_spacy(article_title)
                #    for ent in title_doc.ents:
                #        if ent.label_ == "ORG":
                #            results["entity_id"] = ent.text
                #            break

                # Extract Topics 
                # Using basic RAKE-like keyword extraction for now as pytextrank is not added by default.
                # results["topics"] = self._extract_rake_keywords(text_to_process, max_keywords=5)
                # If pytextrank was added and works:
                # if hasattr(doc._, 'phrases') and doc._.phrases:
                #     results["topics"] = [phrase.text for phrase in doc._.phrases[:5]] # Top 5 phrases
                # else:
                #    results["topics"] = self._extract_rake_keywords(text_to_process, max_keywords=5)
                results["topics"] = self._extract_rake_keywords(text_to_process, max_keywords=5)

            except Exception as e:
                print(f"Error during spaCy processing (NER/Topics): {e}")
        else: # Fallback if spaCy is not available
            results["topics"] = self._extract_rake_keywords(text_to_process, max_keywords=5)

        # Adapt to the project's specified output schema
        # Project schema: {"entity_id": "AAPL", "sent_pos": 0.6, "sent_neg": 0.0, "ner": ["Apple", "Tim Cook"], "topics": ["earnings", "guidance"]}
        # Current output provides ner_org, ner_gpe. "ner" in schema seems to be a mix or primary entities.
        # For now, let's make "ner" in the output be ner_org + ner_gpe for closer match.
        final_output = {
            "entity_id": results["entity_id"],
            "sent_pos": results["sent_pos"],
            "sent_neg": results["sent_neg"],
            "ner_org": results["ner_org"], # Ensure ner_org is in the output
            "ner_gpe": results["ner_gpe"], # Ensure ner_gpe is in the output
            # "ner": list(set(results["ner_org"] + results["ner_gpe"])_[:5]), # Example: combined and limited
            "ner": results["ner_org"][:2] + results["ner_gpe"][:2], # simple concatenation for now (schema field)
            "topics": results["topics"],
            "_debug_sentiment_label": results["dominant_sentiment"], # For internal checking
            "_debug_ner_all": results["ner_all"] # For internal checking
        }
        return final_output

# --- Main function for testing this module ---
if __name__ == "__main__":
    print("Initializing NLP Processor...")
    # This might download models on first run if not cached by transformers/spacy
    processor = NlpProcessor()

    sample_article_text = (
        "Apple Inc. announced record profits for the last quarter, driven by strong iPhone sales in China. "
        "Tim Cook, CEO of Apple, mentioned plans to expand into new markets in South East Asia. "
        "The company also discussed its strategic roadmap for 2026. "
        "However, some analysts from Google and Microsoft Corp expressed concerns about future growth."
    )
    sample_title = "Apple Hits Record Highs"
    
    # Check if models actually loaded before proceeding with test
    if (processor.nlp_spacy or not SPACY_MODEL) and (processor.sentiment_analyzer or not FINBERT_MODEL_NAME):
        print(f"\nProcessing sample article text:\n'{sample_article_text}'")
        processed_data = processor.process_article(sample_article_text, article_title=sample_title)
        print("\nNLP Processing Results (schema matched as per project requirements):")
        import json
        print(json.dumps(processed_data, indent=2))
    else:
        print("\n--- NLP models did not load successfully. Cannot run test. ---")
        if not processor.nlp_spacy and SPACY_MODEL:
            print(f"Failed to load spaCy model: {SPACY_MODEL}. Please ensure it is downloaded: python -m spacy download {SPACY_MODEL}")
        if not processor.sentiment_analyzer and FINBERT_MODEL_NAME:
            print(f"Failed to load FinBERT model: {FINBERT_MODEL_NAME}. Check network or model name.") 