# Configuration file
from langchain.prompts import PromptTemplate
import argparse
import configparser
from configparser import ConfigParser

config = {
    "BASE_FOLDER_PATH": "",
    "template": "",
    "query": ""
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
ANSWERING_MODEL_NAME = "incept5/llama3.1-claude"
ANSWERING_MODEL_URL = "http://127.0.0.1:11434"
CHUNK_SIZE = 1500 #used to be 1024 for Reddit
CHUNK_OVERLAP = 400 #used to be 64 for Reddit
template=""
query=""
BASE_FOLDER_PATH="" # Base folder containing CSV and DOCX files

retrieval_qa_chat_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template=config['template'] + """
Context: {context}

Question: {input}
Response:
"""
)

def import_config():
    global config  # Ensure updates affect module-level variables

    parser = argparse.ArgumentParser(description="Run the RAG system with a specified configuration file.")
    parser.add_argument(
        "-c", "--config", required=True, help="Path to the .ini configuration file."
    )
    args = parser.parse_args()

    # Read the .ini file
    cfg_parser = ConfigParser()
    cfg_parser.read(args.config)
    print(f"Reading configuration file: {args.config}")
    print(f"Sections in config: {cfg_parser.sections()}")
    if "Settings" in cfg_parser:
        print(f"Options in [Settings]: {cfg_parser.options('Settings')}")
        print(f"path value: {cfg_parser.get('Settings', 'path', fallback='Not Found')}")
    else:
        print("Section [Settings] not found.")
    # Combine multiline template manually
    config["BASE_FOLDER_PATH"] = cfg_parser.get("Settings", "path", fallback="").strip('"')
    config["template"] = " ".join(line.strip() for line in cfg_parser["Settings"]["template"].splitlines())
    config["query"] = cfg_parser.get("Settings", "query", fallback="").strip('"')

    # Validate configuration
    if not config["BASE_FOLDER_PATH"]:
        raise ValueError("Missing required 'path' in the configuration file.")
    if not config["query"]:
        raise ValueError("Missing required 'query' in the configuration file.")
    if not config["template"]:
        raise ValueError("Missing required 'template' in the configuration file.")

    # Print for debugging
    print(f"Using path: {config['BASE_FOLDER_PATH']}")
    print(f"Using query: {config['query']}")
    print(f"Using template: {config['template']}")


