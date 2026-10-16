"""Tests never depend on the machine's public-demo settings: the server section is switched off unless a test sets it itself."""
import os
from pathlib import Path

os.environ.setdefault("CGRAG_OVERLAY", str(Path(__file__).with_name("server_off.toml")))
