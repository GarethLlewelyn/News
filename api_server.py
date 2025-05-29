from fastapi import FastAPI
from contextlib import asynccontextmanager
import datetime
import os
import json

# Import pipeline functions - ensure main_pipeline.py is in the same directory or PYTHONPATH
# We need the initialize_components function for startup
from main_pipeline import initialize_components, run_full_pipeline_cycle
from storage.writer import PROCESSED_DATA_DIR # To find sample data

# --- FastAPI Application Lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    print("FastAPI server starting up...")
    print("Initializing pipeline components for the API server...")
    initialize_components() # This will load all ML models and other components
    print("Pipeline components initialized.")
    yield
    # Code to run on shutdown (if any)
    print("FastAPI server shutting down...")

# Create FastAPI app instance with lifespan management
app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---

@app.get("/status")
async def get_status():
    """Returns the current status of the API and pipeline components."""
    # Basic status, can be expanded (e.g., check model loading status from global vars in main_pipeline)
    # For now, if initialize_components() didn't raise an error, we assume basic readiness.
    from main_pipeline import NLP_PROCESSOR_INSTANCE, ARTICLE_FILTER_INSTANCE, RELEVANCE_CLASSIFIER_INSTANCE
    
    nlp_status = "NLP Processor: Ready" if NLP_PROCESSOR_INSTANCE and NLP_PROCESSOR_INSTANCE.nlp_spacy and NLP_PROCESSOR_INSTANCE.sentiment_analyzer else "NLP Processor: Error/Not Loaded"
    filter_status = "Article Filter: Ready" if ARTICLE_FILTER_INSTANCE else "Article Filter: Not Initialized"
    relevance_status = "Relevance Classifier: Ready" if RELEVANCE_CLASSIFIER_INSTANCE else "Relevance Classifier: Not Initialized"
    
    return {
        "status": "API is running",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "pipeline_components": {
            "nlp_processor": nlp_status,
            "article_filter": filter_status,
            "relevance_classifier": relevance_status
        },
        "message": "Financial News AI Pipeline API"
    }

@app.get("/sample")
async def get_sample_processed_articles(count: int = 5):
    """Returns a sample of recently processed articles.
       Reads from the latest .jsonl file in the processed data directory.
    """
    if count <= 0:
        return {"error": "Count must be a positive integer."}
    
    samples = []
    try:
        # Find the most recent JSONL file in the processed directory
        # This is a simple way; a more robust system might query a database or dedicated service
        processed_files = sorted(
            [
                os.path.join(PROCESSED_DATA_DIR, f)
                for f in os.listdir(PROCESSED_DATA_DIR)
                if f.endswith(".jsonl") and os.path.isfile(os.path.join(PROCESSED_DATA_DIR, f))
            ],
            key=os.path.getmtime,
            reverse=True,
        )

        if not processed_files:
            return {"message": "No processed articles found yet.", "samples": []}

        # Read last 'count' lines from the most recent file(s)
        # For simplicity, reading from the newest file first.
        # A more robust implementation might need to read across files if one file has too few lines.
        lines_to_fetch = count
        for file_path in processed_files:
            if lines_to_fetch <= 0:
                break
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Read all lines and then take the last `lines_to_fetch`
                    # This is not memory efficient for very large files, but simple for typical JSONL.
                    all_lines_in_file = f.readlines()
                    start_index = max(0, len(all_lines_in_file) - lines_to_fetch)
                    fetched_lines = all_lines_in_file[start_index:]
                    
                    for line in reversed(fetched_lines): # Get newest first from the selection
                        if lines_to_fetch <= 0:
                            break
                        try:
                            samples.append(json.loads(line.strip()))
                            lines_to_fetch -= 1
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line from {file_path}: {line.strip()}")
                            continue # Skip malformed lines
            except Exception as e:
                print(f"Error reading sample file {file_path}: {e}")
                continue # Try next file if one fails

        return {"message": f"Returning up to {count} sample(s).", "count": len(samples), "samples": samples}

    except Exception as e:
        print(f"Error fetching samples: {e}")
        return {"error": "Could not fetch samples.", "details": str(e)}


@app.post("/run-pipeline-cycle")
async def trigger_pipeline_run():
    """Manually triggers one full cycle of the news processing pipeline.
       Note: This is a long-running task and will block until complete.
       In a production system, this should be handled by a background worker (e.g., Celery).
    """
    print("Received request to run pipeline cycle...")
    try:
        # In a real async setup, this should be: await asyncio.to_thread(run_full_pipeline_cycle)
        # Or better, offload to a task queue.
        # For simplicity now, running it synchronously but acknowledging it's not ideal for FastAPI.
        run_full_pipeline_cycle() 
        return {"message": "Pipeline cycle triggered and completed successfully."}
    except Exception as e:
        print(f"Error during triggered pipeline run: {e}")
        return {"error": "Pipeline cycle run failed.", "details": str(e)}

# To run this server (from the project root directory):
# uvicorn api_server:app --reload
# The Dockerfile will use: uvicorn api_server:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    # This is for local development testing if you run `python api_server.py`
    # However, uvicorn is the recommended way to run FastAPI apps.
    import uvicorn
    print("Starting API server with Uvicorn for local development...")
    # Initialize components here if not using lifespan for this direct run, 
    # or rely on lifespan to do it.
    # initialize_components() # Lifespan will handle this if run via uvicorn properly.
    uvicorn.run(app, host="127.0.0.1", port=8000) 