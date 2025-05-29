import spacy
from spacy.tokens import Doc, Span, Token
from typing import List, Dict, Tuple, Optional, Any
import time
import logging
import hashlib # Added for key generation

# Assuming app.config and app.kafka_utils are accessible
import sys
import os
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_APP_DIR)) # causality is under app/causality
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)
if _APP_DIR not in sys.path: # Ensure app dir itself is available
    sys.path.append(os.path.dirname(_APP_DIR))

from app.config import settings
from app.kafka_utils import send_to_kafka, KafkaProducer # create_kafka_producer no longer needed here

logger = logging.getLogger(__name__)

# Load spaCy model (consider a smaller model if performance is critical and full accuracy isn't needed for this step)
# For full dependency parsing, a model with a parser is needed, e.g., en_core_web_sm or en_core_web_md
# Ensure you have the model downloaded: python -m spacy download en_core_web_sm
NLP = None
try:
    NLP = spacy.load(settings.spacy_model_name)
    logger.info(f"spaCy model '{settings.spacy_model_name}' loaded successfully in causality.extractor.")
except OSError as e:
    logger.error(f"Failed to load spaCy model '{settings.spacy_model_name}' in causality.extractor: {e}. Causality extraction will be impaired.", exc_info=True)
except Exception as e:
    logger.error(f"An unexpected error occurred loading spaCy model '{settings.spacy_model_name}' in causality.extractor: {e}.", exc_info=True)

# Define causal connectors and patterns
# These are simplified examples; more sophisticated patterns might be needed.
CAUSAL_CONNECTORS = [
    "due to", "because of", "caused by", "resulted in", "led to", "consequence of", 
    "attributed to", "reason for", "thanks to", "owing to", "as a result of", 
    "driven by", "triggered by", "following", "amid", "on back of", "after reports of",
    "on news of", "stems from", "linked to", "associated with", "reflecting"
]

# Keywords that might indicate positive or negative sentiment around the cause/effect
POSITIVE_SENTIMENT_KEYWORDS = ["boost", "surge", "rally", "jump", "rise", "gain", "up", "optimism", "strong", "positive", "better-than-expected", "outperform"]
NEGATIVE_SENTIMENT_KEYWORDS = ["drop", "fall", "plunge", "decline", "slump", "down", "pessimism", "weak", "negative", "worse-than-expected", "underperform", "concerns", "fears", "warning"]

ENTITY_LABELS_OF_INTEREST = ["ORG", "PRODUCT", "EVENT", "LAW", "PERCENT", "MONEY", "QUANTITY", "NORP"]

def get_subtree_span(token: Token) -> Optional[Span]:
    if not isinstance(token, Token):
        logger.warning(f"get_subtree_span expected a Token, got {type(token)}")
        return None
    try:
        return token.doc[token.left_edge.i : token.right_edge.i + 1]
    except AttributeError as e:
        logger.error(f"Error getting subtree for token '{token.text}': {e}. Ensure token is part of a parsed doc.")
        return None

def find_org_entity_in_span(span: Span) -> Optional[Span]:
    if not isinstance(span, Span):
        logger.warning(f"find_org_entity_in_span expected a Span, got {type(span)}")
        return None
    for ent in span.ents:
        if ent.label_ == "ORG":
            return ent
    return None

def find_org_entities(doc: Doc) -> List[Span]:
    """Finds ORG entities in a spaCy Doc."""
    return [ent for ent in doc.ents if ent.label_ == "ORG"]

def extract_causal_relationships_from_sentence(sentence: Span) -> List[Dict[str, Any]]:
    global NLP
    if not NLP:
        logger.warning("spaCy NLP model not available in extract_causal_relationships_from_sentence. Cannot extract causality.")
        return []
    if not isinstance(sentence, Span) or not sentence.doc.has_annotation("DEP"):
        logger.warning(f"Input sentence is not a valid, parsed spaCy Span (type: {type(sentence)}, has_dep: {sentence.doc.has_annotation('DEP') if isinstance(sentence,Span) else 'N/A'}). Re-parsing.")
        # Attempt to re-parse the sentence text using the global NLP instance
        temp_doc = NLP(sentence.text)
        sentence = next(temp_doc.sents, None) # Get the first sentence from the re-parsed doc
        if not sentence or not sentence.doc.has_annotation("DEP"):
            logger.error("Failed to re-parse sentence or get dependencies. Cannot extract causality.")
            return []

    extracted_relations = []
    text_lower = sentence.text.lower()

    for connector in CAUSAL_CONNECTORS:
        if connector in text_lower:
            try:
                # Use spaCy Matcher for robust connector identification
                matcher = spacy.matcher.Matcher(NLP.vocab)
                pattern = []
                for part in connector.split(): # Handle multi-word connectors
                    pattern.append({"LOWER": part})
                matcher.add("CausalConnector", [pattern])
                matches = matcher(sentence) # Match on the sentence Span itself

                for match_id, start, end in matches:
                    connector_span = sentence[start:end]
                    matched_connector_token = connector_span[0] # Usually the first token is key for dependencies
                    
                    cause_phrase_span = None
                    effect_phrase_span = None
                    identified_cause_entity = None
                    identified_effect_entity = None

                    if matched_connector_token.dep_ in ["prep", "agent", "mark"] or (matched_connector_token.head.pos_ == "ADP" and matched_connector_token.dep_ == "pcomp") :
                        for child in matched_connector_token.children:
                            if child.dep_ == "pobj":
                                cause_phrase_span = get_subtree_span(child)
                                if cause_phrase_span: identified_cause_entity = find_org_entity_in_span(cause_phrase_span)
                                break
                        # Effect might be related to the head of the connector token or its subject
                        head_verb = matched_connector_token.head
                        if head_verb.pos_ in ["VERB", "AUX"]:
                            for subj_candidate in head_verb.children:
                                if subj_candidate.dep_ in ["nsubj", "nsubjpass"]:
                                    effect_phrase_span = get_subtree_span(subj_candidate)
                                    if effect_phrase_span: identified_effect_entity = find_org_entity_in_span(effect_phrase_span)
                                    break
                            if not effect_phrase_span: # If no clear subject, take the verb phrase
                                effect_phrase_span = get_subtree_span(head_verb)
                                if effect_phrase_span: identified_effect_entity = find_org_entity_in_span(effect_phrase_span)
                        elif head_verb.pos_ in ["NOUN", "PROPN"]:
                             effect_phrase_span = get_subtree_span(head_verb)
                             if effect_phrase_span: identified_effect_entity = find_org_entity_in_span(effect_phrase_span)
                    
                    # Handle cases where connector itself is a verb (e.g. "led to", "resulted in")
                    elif matched_connector_token.pos_ == "VERB":
                        # Effect is often the subject of the causal verb
                        for child in matched_connector_token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                effect_phrase_span = get_subtree_span(child)
                                if effect_phrase_span: identified_effect_entity = find_org_entity_in_span(effect_phrase_span)
                                break
                        # Cause is often the object (dobj or pobj of a following prep)
                        for child in matched_connector_token.children:
                            if child.dep_ == "dobj":
                                cause_phrase_span = get_subtree_span(child)
                                if cause_phrase_span: identified_cause_entity = find_org_entity_in_span(cause_phrase_span)
                                break
                            elif child.dep_ == "prep": # e.g. "resulted in X"
                                for p_child in child.children:
                                    if p_child.dep_ == "pobj":
                                        cause_phrase_span = get_subtree_span(p_child)
                                        if cause_phrase_span: identified_cause_entity = find_org_entity_in_span(cause_phrase_span)
                                        break
                                if cause_phrase_span: break

                    if cause_phrase_span and effect_phrase_span:
                        cause_sentiment = "neutral"
                        if any(kw in cause_phrase_span.text.lower() for kw in POSITIVE_SENTIMENT_KEYWORDS): cause_sentiment = "positive"
                        elif any(kw in cause_phrase_span.text.lower() for kw in NEGATIVE_SENTIMENT_KEYWORDS): cause_sentiment = "negative"
                        effect_sentiment = "neutral"
                        if any(kw in effect_phrase_span.text.lower() for kw in POSITIVE_SENTIMENT_KEYWORDS): effect_sentiment = "positive"
                        elif any(kw in effect_phrase_span.text.lower() for kw in NEGATIVE_SENTIMENT_KEYWORDS): effect_sentiment = "negative"

                        relation = {
                            "sentence": sentence.text,
                            "connector": connector_span.text,
                            "cause_phrase": cause_phrase_span.text,
                            "cause_entity": identified_cause_entity.text if identified_cause_entity else None,
                            "cause_entity_label": identified_cause_entity.label_ if identified_cause_entity else None,
                            "cause_sentiment": cause_sentiment,
                            "effect_phrase": effect_phrase_span.text,
                            "effect_entity": identified_effect_entity.text if identified_effect_entity else None,
                            "effect_entity_label": identified_effect_entity.label_ if identified_effect_entity else None,
                            "effect_sentiment": effect_sentiment
                        }
                        if relation not in extracted_relations: # Avoid duplicates from same match
                            extracted_relations.append(relation)
                            logger.debug(f"Extracted: EFFECT '{effect_phrase_span.text}' {connector_span.text} CAUSE '{cause_phrase_span.text}'")
            except Exception as e:
                logger.error(f"Error during causality extraction for connector '{connector}' in sentence '{sentence.text}': {e}", exc_info=True)
                continue # Try next connector
    
    if not extracted_relations and sentence.text:
        logger.debug(f"No causal relations found in sentence: '{sentence.text[:100]}...'")
    return extracted_relations

def extract_causality_from_text(text: str, spacy_doc_input: Optional[Doc] = None) -> List[Dict[str, Any]]:
    global NLP
    if not NLP and not spacy_doc_input:
        logger.error("spaCy NLP model not available and no pre-processed Doc provided. Cannot extract causality.")
        return []

    doc_to_process = None
    if spacy_doc_input:
        if not isinstance(spacy_doc_input, Doc):
            logger.warning(f"spacy_doc_input was not a spaCy Doc type ({type(spacy_doc_input)}). Processing text from scratch.")
            if NLP: doc_to_process = NLP(text)
        elif not spacy_doc_input.has_annotation("SENT_START") or not spacy_doc_input.has_annotation("DEP"):
            logger.warning(f"Provided spacy_doc_input lacks sentence ({spacy_doc_input.has_annotation('SENT_START')}) or dependency ({spacy_doc_input.has_annotation('DEP')}) annotations. Re-processing.")
            if NLP: doc_to_process = NLP(text) # Re-process to ensure all annotations
        else:
            doc_to_process = spacy_doc_input
            logger.debug("Using pre-processed and annotated spaCy Doc for causality extraction.")
    elif NLP:
        doc_to_process = NLP(text)
        logger.debug("Processed text with spaCy for causality extraction.")

    if not doc_to_process:
        logger.error("Failed to obtain a valid spaCy Doc. Cannot extract causality.")
        return []

    all_relations = []
    for sentence in doc_to_process.sents:
        relations = extract_causal_relationships_from_sentence(sentence)
        all_relations.extend(relations)
    
    logger.info(f"Extracted {len(all_relations)} potential causal relations from text (length {len(text)})." )
    return all_relations

def extract_and_send_causality(
    producer: Optional[KafkaProducer],
    topic: str,
    text: str,
    spacy_doc_input: Optional[Doc] = None,
    message_key_prefix: Optional[str] = None
) -> int:
    if not producer:
        logger.warning(f"Kafka producer is None. Cannot send causality messages for topic '{topic}'. Will attempt extraction only.")
        # Fallback to just extracting if no producer, so the count is still somewhat meaningful
        relations = extract_causality_from_text(text, spacy_doc_input=spacy_doc_input)
        return len(relations) if relations else 0

    relations = extract_causality_from_text(text, spacy_doc_input=spacy_doc_input)
    sent_count = 0
    if not relations:
        logger.info(f"No causal relations found in text to send to Kafka topic '{topic}'.")
        return 0

    for i, relation_data in enumerate(relations):
        relation_data["original_text_preview"] = text[:200] + "..." if len(text) > 200 else text
        
        kafka_key = None
        if message_key_prefix:
            kafka_key = f"{message_key_prefix}_causality_{i}"
        else:
            sent_hash = hashlib.sha1(relation_data.get("sentence", "").encode()).hexdigest()[:8]
            kafka_key = f"causal_{sent_hash}_{int(time.time())}" 

        if send_to_kafka(producer, topic, relation_data, key=kafka_key):
            sent_count += 1
        else:
            logger.error(f"Failed to send causal relation to Kafka topic '{topic}': {relation_data}")
            
    logger.info(f"Sent {sent_count}/{len(relations)} extracted causal relations to Kafka topic '{topic}'.")
    return sent_count

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running causality_extractor.py direct test...")

    if NLP is None:
        logger.critical("__main__: Global NLP model is None. Most causality extraction tests will be skipped or will fail.")

    sample_texts_for_main = [
        "Stock prices of AlphaCorp fell sharply due to rising inflation concerns.",
        "The launch of the new product resulted in a significant sales boost for BetaTech.",
        "Because of the trade agreement, Gamma Industries saw increased export volumes.",
        "Delta Corp's profits declined as a result of supply chain disruptions.",
        "No clear causal statement here.",
        "Market sentiment improved following the central bank's optimistic forecast, leading to a stock rally.",
        "AAPL shares surged higher on Tuesday after reports of strong iPhone sales."
    ]

    logger.info(f"\n--- Testing causality extraction (model: '{settings.spacy_model_name if NLP else 'None'}') ---")
    all_extracted_for_test = []
    for i, text_sample in enumerate(sample_texts_for_main):
        logger.info(f"Sample {i+1}: {text_sample}")
        if NLP:
            # Test with pre-processed Doc
            doc_instance = NLP(text_sample)
            relations = extract_causality_from_text(text_sample, spacy_doc_input=doc_instance)
            # Also test without pre-processed Doc to ensure internal NLP call works
            # relations_no_doc = extract_causality_from_text(text_sample)
            # assert len(relations) == len(relations_no_doc), "Mismatch with/without pre-processed doc"
        else:
            logger.warning("Skipping extraction test as NLP model is not loaded.")
            relations = [] # Can't extract without NLP
        
        all_extracted_for_test.extend(relations)
        if relations:
            for rel in relations:
                 logger.info(f"    Found: EFFECT '{rel.get('effect_phrase', 'N/A')}' ({rel.get('effect_entity', 'N/A')}) {rel.get('connector','N/A')} CAUSE '{rel.get('cause_phrase','N/A')}' ({rel.get('cause_entity','N/A')})")
        else:
            logger.info("    No causal relations found in this sample.")
        print("")

    logger.info(f"Total causal relations extracted from all samples: {len(all_extracted_for_test)}")

    logger.info("\n--- Testing Kafka sending (mocked Kafka producer) ---")
    class MockKafkaPMain:
        def send(self, topic, value, key):
            logger.info(f"MockKafka SEND to '{topic}': Key='{key}', Value={str(value)[:200]}...")
            return True
        def flush(self, timeout=None): logger.info("MockKafka FLUSH called")
        def close(self, timeout=None): logger.info("MockKafka CLOSE called")

    mock_producer_main = MockKafkaPMain()
    # Use settings for the topic name
    test_causality_topic_main = settings.kafka_causality_topic + "_extractor_main_test"

    total_sent_to_kafka = 0
    for i, text_sample in enumerate(sample_texts_for_main):
        if "No clear causal statement" in text_sample or not NLP: # Skip if no NLP
            logger.info(f"Skipping Kafka send test for non-causal/no-NLP sample: '{text_sample[:50]}...'")
            continue
        
        logger.info(f"Attempting to extract and send causality for: '{text_sample[:50]}...'")
        doc_for_sending = NLP(text_sample) # Create doc for this specific test send
        num_sent_kafka = extract_and_send_causality(
            producer=mock_producer_main, 
            topic=test_causality_topic_main, 
            text=text_sample, 
            spacy_doc_input=doc_for_sending,
            message_key_prefix=f"main_test_{i+1}"
        )
        total_sent_to_kafka += num_sent_kafka
        logger.info(f"Kafka send test: {num_sent_kafka} relations sent for sample {i+1}.")
        print("")
    
    logger.info(f"Total relations sent to Kafka in test: {total_sent_to_kafka}")
    logger.info("Causality extractor direct test finished.") 