import json
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
import os
import time # For timestamp in example
import logging

# Assuming app.config is accessible
import sys
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_APP_DIR)) # Adjust if graph is not directly under app/graph
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)
if _APP_DIR not in sys.path: # Ensure app dir itself is available
    sys.path.append(os.path.dirname(_APP_DIR))

from app.config import settings
from app.kafka_utils import send_to_kafka, KafkaProducer # KafkaProducer for type hint

logger = logging.getLogger(__name__)

# Define path to the relationship map. Assumes this script is in app/graph/
# and the map is in the same directory.
_DEFAULT_RELATION_MAP_PATH = os.path.join(os.path.dirname(__file__), "entity_relation_map.json")

RELATION_TYPES = ["peers", "suppliers", "customers", "etf_inclusion"]

# Predefined decay factors for different relationship types
# These are examples; they could be more dynamic or configurable.
RELATION_DECAY_FACTORS = {
    "peers": 0.7,
    "suppliers": 0.8,
    "customers": 0.75,
    "etf_inclusion": 0.5, # ETF impact might be more diffuse
    "same_sector": 0.4,   # For broader sector propagation not explicitly in map peers
    "same_industry": 0.5  # For broader industry propagation
}

class EventPropagator:
    def __init__(self, entity_map_path: Optional[str] = None, decay_factor: float = 0.75):
        """
        Initializes the EventPropagator.

        Args:
            entity_map_path: Path to the JSON file containing entity relationships.
                             Uses settings.entity_relation_map_path if None.
            decay_factor: Factor by which sentiment/signal strength decays with each propagation step.
        """
        self.graph = nx.Graph()
        self.decay_factor = decay_factor
        
        actual_map_path = entity_map_path
        if actual_map_path is None:
            actual_map_path = os.path.join(_PROJECT_ROOT, settings.entity_relation_map_path)
        
        self.entity_relations = self._load_entity_map(actual_map_path)
        if self.entity_relations:
            self._build_graph()
            logger.info(f"EventPropagator initialized with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges from '{actual_map_path}'.")
        else:
            logger.warning(f"EventPropagator initialized with an empty graph due to issues loading map from '{actual_map_path}'.")

    def _load_entity_map(self, map_path: str) -> Dict[str, Dict[str, Any]]:
        """Loads the entity relationship map from a JSON file."""
        if not os.path.exists(map_path):
            logger.error(f"Entity relationship map file not found at: {map_path}")
            return {}
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded entity map from {map_path} with {len(data)} root entities.")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from entity map file {map_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load entity map from {map_path}: {e}", exc_info=True)
        return {}

    def _build_graph(self):
        """Builds the graph from the loaded entity relationships."""
        if not self.entity_relations:
            logger.warning("Cannot build graph: entity relations not loaded.")
            return

        for entity, relations in self.entity_relations.items():
            self.graph.add_node(entity) # Ensure entity itself is a node
            for relation_type, related_entities in relations.items():
                if isinstance(related_entities, list):
                    for related_entity in related_entities:
                        self.graph.add_node(related_entity) # Ensure related entity is a node
                        self.graph.add_edge(entity, related_entity, type=relation_type)
                        logger.debug(f"Graph: Added edge {entity} <-> {related_entity} (type: {relation_type})")
                else:
                     logger.warning(f"Expected list for entities under '{relation_type}' for entity '{entity}', got {type(related_entities)}")

    def propagate_event(
        self, 
        subject_id: str, 
        original_sentiment: float, 
        max_depth: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Propagates an event (sentiment) starting from subject_id through the graph.

        Args:
            subject_id: The ID of the entity where the event originated.
            original_sentiment: The sentiment score of the original event (-1.0 to 1.0).
            max_depth: Maximum propagation depth (number of hops).

        Returns:
            A dictionary where keys are related entity IDs and values are dicts 
            containing 'propagated_sentiment' and 'relation_type' (from the first hop).
        """
        propagated_signals = {}
        if not self.graph.has_node(subject_id):
            logger.warning(f"Cannot propagate event: Subject ID '{subject_id}' not found in the graph.")
            return propagated_signals

        # Use BFS to find neighbors up to max_depth
        # queue stores (node, depth, path_sentiment, first_hop_relation_type)
        queue = [(subject_id, 0, original_sentiment, None)]
        visited_for_propagation = {subject_id} # To avoid cycles in propagation paths for this event

        head = 0
        while head < len(queue):
            current_node, depth, current_sentiment, _ = queue[head]
            head += 1

            if depth >= max_depth:
                continue

            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in visited_for_propagation:
                    visited_for_propagation.add(neighbor)
                    edge_data = self.graph.get_edge_data(current_node, neighbor)
                    relation_type = edge_data.get('type', 'related') if edge_data else 'related'
                    
                    propagated_sentiment = current_sentiment * self.decay_factor
                    
                    # Store the signal for this neighbor
                    # If neighbor is reached by multiple paths, we could average, take max, or overwrite.
                    # Current logic overwrites with the first path found via BFS (shortest path).
                    # For simplicity, let's store based on the first encounter.
                    if neighbor not in propagated_signals:
                        propagated_signals[neighbor] = {
                            "propagated_sentiment": round(propagated_sentiment, 3),
                            "relation_type": relation_type,
                            "source_event_entity": subject_id,
                            "depth": depth + 1
                        }
                        logger.debug(f"Propagated event from {subject_id} to {neighbor} (depth {depth+1}): sentiment {propagated_sentiment:.3f}, relation {relation_type}")
                    
                    queue.append((neighbor, depth + 1, propagated_sentiment, relation_type))
        
        return propagated_signals

    def propagate_and_send_relations(
        self,
        producer: Optional[KafkaProducer],
        topic: str,
        subject_id: str,
        original_sentiment: float,
        original_ts: int, # Unix timestamp in milliseconds
        max_depth: int = 2,
        message_key_prefix: Optional[str] = None
    ) -> int:
        """
        Propagates events and sends the resulting relations to a Kafka topic.
        """
        if not producer:
            logger.warning(f"Kafka producer not provided. Skipping sending propagated relations for {subject_id}.")
            return 0
        if not self.graph.has_node(subject_id):
            logger.warning(f"Cannot propagate and send: Subject ID '{subject_id}' not found in graph.")
            return 0

        propagated_signals = self.propagate_event(subject_id, original_sentiment, max_depth)
        sent_count = 0

        if not propagated_signals:
            logger.info(f"No propagated signals found for subject '{subject_id}'.")
            return 0

        for related_entity_id, signal_data in propagated_signals.items():
            message = {
                "subject_id": subject_id,
                "relation": signal_data["relation_type"],
                "object_id": related_entity_id,
                "propagated_sentiment": signal_data["propagated_sentiment"],
                "original_event_ts": original_ts,
                "propagation_depth": signal_data["depth"]
            }
            
            kafka_key = None
            if message_key_prefix:
                kafka_key = f"{message_key_prefix}_relation_{subject_id}_{related_entity_id}"
            else:
                kafka_key = f"{subject_id}_{related_entity_id}"
            
            if send_to_kafka(producer, topic, message, key=kafka_key):
                sent_count += 1
            else:
                logger.error(f"Failed to send propagated relation to Kafka: {message}")
        
        if sent_count > 0:
            logger.info(f"Successfully sent {sent_count} propagated relation(s) for '{subject_id}' to Kafka topic '{topic}'.")
        return sent_count

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running EventPropagator direct test...")

    # Create a dummy entity_relation_map.json if it doesn't exist for the test
    # Uses path from settings
    dummy_map_path = os.path.join(_PROJECT_ROOT, settings.entity_relation_map_path)
    if not os.path.exists(dummy_map_path):
        logger.warning(f"Test: Entity relation map not found at '{dummy_map_path}'. Creating dummy map for test.")
        os.makedirs(os.path.dirname(dummy_map_path), exist_ok=True)
        dummy_data = {
            "AAPL": {"sector_peer": ["MSFT", "GOOGL"], "supplier": ["TSMC"]},
            "MSFT": {"sector_peer": ["AAPL", "GOOGL"], "competitor": ["AMZN"]},
            "TSMC": {"customer": ["AAPL"]},
            "GOOGL": {"sector_peer": ["AAPL", "MSFT"]},
            "AMZN": {"competitor": ["MSFT"]}
        }
        with open(dummy_map_path, 'w') as f:
            json.dump(dummy_data, f, indent=2)
        logger.info(f"Dummy map created at {dummy_map_path}")
    
    # Initialize EventPropagator (will use path from settings by default if None passed)
    propagator = EventPropagator()

    if not propagator.entity_relations:
        logger.error("Propagator could not load entity relations. Aborting test.")
    else:
        logger.info(f"Propagator graph nodes: {list(propagator.graph.nodes)}")
        logger.info(f"Propagator graph edges: {list(propagator.graph.edges(data=True))}")

        # Test propagation
        subject = "AAPL"
        sentiment = 0.8
        logger.info(f"\n--- Testing propagation for {subject} with sentiment {sentiment} ---")
        signals = propagator.propagate_event(subject, sentiment, max_depth=2)
        if signals:
            logger.info(f"Propagated signals for {subject}:")
            for entity, data in signals.items():
                logger.info(f"  To: {entity}, Sentiment: {data['propagated_sentiment']}, Type: {data['relation_type']}, Depth: {data['depth']}")
        else:
            logger.info(f"No signals propagated for {subject}.")

        # Test Kafka sending (mocked for this __main__)
        class MockKafkaProducer:
            def send(self, topic, value, key=None):
                logger.info(f"MockKafka SEND: Topic={topic}, Key={key}, Value={value}")
                return True # Simulate success
            def flush(self, timeout=None): logger.info("MockKafka FLUSH called")
            def close(self, timeout=None): logger.info("MockKafka CLOSE called")
        
        mock_producer = MockKafkaProducer()
        test_relations_topic = settings.kafka_relations_topic + "_propagator_test"

        logger.info(f"\n--- Testing propagate_and_send_relations for {subject} to topic '{test_relations_topic}' ---")
        sent_count = propagator.propagate_and_send_relations(
            producer=mock_producer,
            topic=test_relations_topic,
            subject_id=subject,
            original_sentiment=sentiment,
            original_ts=int(time.time() * 1000), # Python equivalent for current time in milliseconds
            message_key_prefix="test_article_1"
        )
        logger.info(f"Propagate and send completed. Messages sent: {sent_count}")

    logger.info("\nEventPropagator direct test finished.") 