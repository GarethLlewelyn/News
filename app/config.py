'''
Centralized configuration management for the application.

This module uses Pydantic's BaseSettings to load configuration from environment
variables and/or a .env file.
'''
import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

# Determine the base directory of the application (e.g., the parent of 'app')
# This allows .env file to be located at the project root.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # Application settings
    app_name: str = "Financial News Ingestion Service"
    app_version: str = "0.2.0"
    log_level: str = "INFO"

    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_syndication_key: str = "SEEN_ARTICLE_HASHES"

    # Kafka configuration
    kafka_bootstrap_servers: List[str] = ["localhost:9092"]
    kafka_client_id: str = "financial-news-processor"
    kafka_causality_topic: str = "news_causality"
    kafka_relations_topic: str = "news_relations"
    # Optional: Kafka security protocol, SASL mechanism, username, password
    kafka_security_protocol: Optional[str] = None
    kafka_sasl_mechanism: Optional[str] = None
    kafka_sasl_plain_username: Optional[str] = None
    kafka_sasl_plain_password: Optional[str] = None

    # spaCy model
    spacy_model_name: str = "en_core_web_sm"

    # Event Propagator
    entity_relation_map_path: str = "app/graph/entity_relation_map.json" # Relative to project root

    # Filtering thresholds (NewsProcessorService defaults)
    default_min_source_trust: float = 0.4
    default_min_tokens_density: int = 30 # As in advanced.py apply_advanced_filters default
    default_min_org_entities_density: int = 0 # As in advanced.py apply_advanced_filters default

    # Clickbait and Source Profile paths (relative to project root)
    clickbait_patterns_path: str = "app/config/clickbait_patterns.json"
    source_profiles_path: str = "app/config/source_profiles.json"

    # For Pydantic settings, specify the .env file location if used
    model_config = SettingsConfigDict(
        env_file=os.path.join(_PROJECT_DIR, '.env'), 
        env_file_encoding='utf-8',
        extra='ignore' # Ignore extra fields from .env that are not in the model
    )

# Instantiate the settings so they can be imported and used elsewhere
settings = Settings()

logger.info(f"Settings loaded. Kafka Bootstrap Servers: {settings.kafka_bootstrap_servers}")

if __name__ == "__main__":
    # Print out the loaded settings for verification
    print("Loaded application settings:")
    for field_name, value in settings.model_dump().items():
        print(f"  {field_name}: {value}")
    
    print(f"\nProject directory considered for .env: {_PROJECT_DIR}")
    env_file_path = os.path.join(_PROJECT_DIR, '.env')
    if os.path.exists(env_file_path):
        print(f".env file found at: {env_file_path}")
    else:
        print(f".env file NOT found at: {env_file_path} - using defaults or other env vars.") 