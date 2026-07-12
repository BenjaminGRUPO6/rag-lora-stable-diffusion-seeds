import json
from pathlib import Path


def test_label_map_has_five_classes() -> None:
    data = json.loads(Path("data/metadata/label_map.json").read_text(encoding="utf-8"))
    assert set(data["classes"]) == {"intact", "spotted", "immature", "broken", "skin_damaged"}
    assert "no confirma" in data["classes"]["spotted"]["description"]
