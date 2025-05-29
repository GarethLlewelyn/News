import json
import logging
from typing import Optional, List, Dict, Any
import time

from kafka import KafkaProducer
from kafka.errors import KafkaError

# Assuming app.config is accessible
import sys
import os
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_APP_DIR) # app directory is child of project root
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)
if _APP_DIR not in sys.path: # Ensure app itself is also there if utils is directly under app
    sys.path.append(_APP_DIR)

from app.config import settings # Import centralized settings

logger = logging.getLogger(__name__)

def create_kafka_producer(
    bootstrap_servers: Optional[List[str]] = None,
    client_id: Optional[str] = None,
    retries: int = 5,
    security_protocol: Optional[str] = None,
    sasl_mechanism: Optional[str] = None,
    sasl_plain_username: Optional[str] = None,
    sasl_plain_password: Optional[str] = None,
    **kwargs: Any # For other KafkaProducer options
) -> Optional[KafkaProducer]:
    """
    Creates and returns a KafkaProducer instance.
    Uses values from centralized settings if specific parameters are not provided.
    """
    # Use settings as defaults if not provided
    effective_bootstrap_servers = bootstrap_servers if bootstrap_servers is not None else settings.kafka_bootstrap_servers
    effective_client_id = client_id if client_id is not None else settings.kafka_client_id
    effective_security_protocol = security_protocol if security_protocol is not None else settings.kafka_security_protocol
    effective_sasl_mechanism = sasl_mechanism if sasl_mechanism is not None else settings.kafka_sasl_mechanism
    effective_sasl_plain_username = sasl_plain_username if sasl_plain_username is not None else settings.kafka_sasl_plain_username
    effective_sasl_plain_password = sasl_plain_password if sasl_plain_password is not None else settings.kafka_sasl_plain_password

    common_configs = {
        "bootstrap_servers": effective_bootstrap_servers,
        "client_id": effective_client_id,
        "value_serializer": lambda v: json.dumps(v).encode('utf-8'),
        "key_serializer": lambda k: str(k).encode('utf-8') if k is not None else None,
        "retries": retries, 
        **kwargs
    }

    if effective_security_protocol and effective_sasl_mechanism:
        logger.info(f"Attempting to create Kafka producer with SASL: {effective_security_protocol}, {effective_sasl_mechanism}")
        common_configs["security_protocol"] = effective_security_protocol
        common_configs["sasl_mechanism"] = effective_sasl_mechanism
        if effective_sasl_mechanism in ["PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"]:
            if not effective_sasl_plain_username or not effective_sasl_plain_password:
                logger.error("SASL mechanism requires username and password, but they were not provided.")
                return None
            common_configs["sasl_plain_username"] = effective_sasl_plain_username
            common_configs["sasl_plain_password"] = effective_sasl_plain_password
    else:
        logger.info(f"Attempting to create Kafka producer without SASL for servers: {effective_bootstrap_servers}")

    try:
        producer = KafkaProducer(**common_configs)
        logger.info(f"KafkaProducer initialized successfully for {effective_bootstrap_servers}.")
        # Test connection (optional, KafkaProducer tries to connect lazily)
        # if hasattr(producer, 'bootstrap_connected') and not producer.bootstrap_connected():
        #     logger.warning("Kafka producer initialized but not yet connected to bootstrap servers.")
        # elif hasattr(producer, 'partitions_for'): # A way to force metadata fetch
        #     try:
        #         producer.partitions_for(settings.kafka_causality_topic) # Check a known topic
        #         logger.info("Kafka producer seems connected (checked partitions for a topic).")
        #     except Exception as e:
        #         logger.warning(f"Kafka producer initialized but could not fetch partitions: {e}")
        return producer
    except KafkaError as e:
        logger.error(f"Failed to create KafkaProducer for {effective_bootstrap_servers}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while creating KafkaProducer for {effective_bootstrap_servers}: {e}", exc_info=True)
    return None

def send_to_kafka(producer: KafkaProducer, topic: str, value: Dict, key: Optional[str] = None) -> bool:
    """
    Sends a message to the specified Kafka topic.
    Returns True on success, False on failure.
    """
    if not producer:
        logger.error(f"Kafka producer is None. Cannot send message to topic '{topic}'.")
        return False
    try:
        future = producer.send(topic, value=value, key=key)
        # Block for synchronous send for up to e.g., 1 second, or use callbacks for async
        record_metadata = future.get(timeout=10) 
        logger.debug(f"Message sent to Kafka topic '{record_metadata.topic}', partition '{record_metadata.partition}', offset '{record_metadata.offset}'")
        return True
    except KafkaError as e:
        logger.error(f"Error sending message to Kafka topic '{topic}': {e}", exc_info=True)
    except Exception as e:
        logger.error(f"An unexpected error occurred while sending to Kafka topic '{topic}': {e}", exc_info=True)
    return False

if __name__ == '__main__':
    # Configure basic logging for this test script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running kafka_utils.py direct test...")

    # Attempt to create a producer using settings (ensure Kafka is running for this to succeed)
    # For this test, it will use localhost:9092 by default from app.config.settings
    test_producer = create_kafka_producer()

    if test_producer:
        logger.info("Test Kafka producer CREATED successfully.")
        test_topic_name = settings.kafka_causality_topic + "_util_test"
        logger.info(f"Attempting to send test message to topic: {test_topic_name}")
        
        sample_data = {"test_key": "test_value", "timestamp": time.time()}
        
        # The send_to_kafka now blocks and gets result, so we can check its return.
        success = send_to_kafka(test_producer, test_topic_name, sample_data, key="test_message_key")
        
        if success:
            logger.info(f"Test message successfully sent to {test_topic_name}.")
        else:
            logger.error(f"Failed to send test message to {test_topic_name}.")
        
        logger.info("Flushing and closing test producer...")
        try:
            test_producer.flush(timeout=10)
            test_producer.close(timeout=10)
            logger.info("Test producer closed.")
        except Exception as e:
            logger.error(f"Error during test producer flush/close: {e}")
    else:
        logger.error("Test Kafka producer FAILED to create. Check Kafka server and settings.")

    logger.info("kafka_utils.py direct test finished.") 