# config_loader.py

# Config Loading — Concept First
# The config loader's job is simple: read the JSON file from disk and give the rest of the pipeline access to it. 
# The simplest approach is just returning a plain Python dictionary — json.load() gives you that directly. 
# The rest of your modules then access values like config["data"]["target_column"]. 
# We'll keep it in its own module config_loader.py rather than burying it in pipeline.py, because every module in your pipeline will need access to config, and you don't want circular imports later.
# config_path is the path to the JSON config file. The function opens the file, reads it, and returns the parsed JSON as a Python dictionary.

# from ml_pipeline.config_loader import load_config
# config = load_config("configs/example_run.json")
# print(config["run_name"])
# print(config["data"]["target_column"])

import json

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config