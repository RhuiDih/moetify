#python3 scripts/serve.py --model ~/llama-3-3x8B --trust_remote_code  --chat_template ./scripts/alpaca.jinja2
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"

import argparse, runpy
import moetify
runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')