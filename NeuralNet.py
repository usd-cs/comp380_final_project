import modal
from modal import Volume, Stub, Image
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

stub = Stub(name="comp380FP")

final_proj_image = Image.from_registry("nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.10").run_commands(
    "apt-get update",
    "apt-get install -y software-properties-common git wget",
    "apt-get install -y poppler-utils",
    "pip install requests",
    "mkdir -p /root/weights",
    "mkdir -p /root/analysis",
    "mkdir -p /root/weights",
    "cd /root && wget -q -0- kaggle competitions download -c planttraits2024", # need to fix this one
    "pip install tensorflow==2.10.1",
    "pip install scikit-learn",
    "pip install numpy",
    "pip install pandas",
    "pip install itertools"
)