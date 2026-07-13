from __future__ import annotations

import pandas as pd

from src.data.cleaning import (
    AuditReports,
    build_cleaning_manifest,
    create_exclusion_list,
    create_near_duplicates_review,
)


def test_create_exclusion_list_keeps_one_exact_duplicate() -> None:
    reports = _reports()

    exclusions = create_exclusion_list(reports)

    assert len(exclusions) == 2
    assert set(exclusions["exclusion_reason"]) == {
        "corrupted_file",
        "exact_duplicate",
    }

    duplicate = exclusions[exclusions["exclusion_reason"] == "exact_duplicate"].iloc[0]
    assert duplicate["relative_path"] == "intact/b.jpg"
    assert duplicate["kept_relative_path"] == "intact/a.jpg"


def test_cleaning_manifest_does_not_exclude_near_duplicates() -> None:
    reports = _reports()
    exclusions = create_exclusion_list(reports)

    manifest = build_cleaning_manifest(reports, exclusions)
    review = create_near_duplicates_review(reports)

    assert set(review["review_status"]) == {"pending"}
    assert manifest.loc[
        manifest["relative_path"] == "spotted/c.jpg",
        "exclusion_status",
    ].item() == "included"
    assert manifest.loc[
        manifest["relative_path"] == "spotted/d.jpg",
        "exclusion_status",
    ].item() == "included"


def _reports() -> AuditReports:
    images = pd.DataFrame(
        [
            _image("intact/a.jpg", "intact", "hash-a"),
            _image("intact/b.jpg", "intact", "hash-a"),
            _image("spotted/c.jpg", "spotted", "hash-c"),
            _image("spotted/d.jpg", "spotted", "hash-d"),
        ]
    )
    corrupted = pd.DataFrame(
        [
            {
                "relative_path": "broken/bad.jpg",
                "category": "broken",
                "extension": ".jpg",
                "size_bytes": 10,
                "error": "cannot identify image file",
            }
        ]
    )
    exact_duplicates = pd.DataFrame(
        [
            {
                "group_id": "exact_0001",
                "sha256": "hash-a",
                "relative_path": "intact/b.jpg",
                "category": "intact",
                "size_bytes": 100,
            },
            {
                "group_id": "exact_0001",
                "sha256": "hash-a",
                "relative_path": "intact/a.jpg",
                "category": "intact",
                "size_bytes": 100,
            },
        ]
    )
    near_duplicates = pd.DataFrame(
        [
            {
                "category": "spotted",
                "path_a": "spotted/c.jpg",
                "path_b": "spotted/d.jpg",
                "phash_a": "aaa",
                "phash_b": "aab",
                "phash_distance": 1,
            }
        ]
    )

    return AuditReports(
        summary={"total_files": 5},
        images=images,
        category_distribution=pd.DataFrame(),
        corrupted=corrupted,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
    )


def _image(relative_path: str, category: str, sha256: str) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "category": category,
        "extension": ".jpg",
        "width": 227,
        "height": 227,
        "mode": "RGB",
        "size_bytes": 100,
        "sha256": sha256,
        "perceptual_hash": "abc",
        "valid": True,
        "is_small": False,
        "error": "",
    }
