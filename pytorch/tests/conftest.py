"""把 pytorch/ 加进 sys.path，让 `from nndl.runner import ...` 在 tests 里能解析。"""
import sys
from pathlib import Path

PYTORCH_ROOT = Path(__file__).resolve().parent.parent
if str(PYTORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTORCH_ROOT))
