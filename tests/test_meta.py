import json
import re

from hybridquant import run_metadata
from hybridquant.meta import REQUIRED_KEYS


def test_run_metadata_has_required_keys_and_is_json_serialisable():
    m = run_metadata()
    assert set(REQUIRED_KEYS) <= set(m), set(REQUIRED_KEYS) - set(m)
    json.dumps(m)   # must be writable into the results jsonl
    assert isinstance(m["git_dirty"], bool)
    assert m["git_commit"] == "unknown" or re.fullmatch(r"[0-9a-f]{7,40}", m["git_commit"])
    assert m["device"] and m["torch"] and m["transformers"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", m["python"])
    assert m["timestamp_utc"].endswith("+00:00")
