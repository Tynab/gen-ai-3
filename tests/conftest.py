"""Cấu hình chung cho pytest: đảm bảo import được streamlit_app từ gốc repo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
