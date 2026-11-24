"""BG-REL R4 candidate: explicit paths and audit configuration.

V3.2 cutover paths. Immutable manifest values remain unchanged.
No legacy-data fallback, directory discovery, or data mutation.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DICOM_ROOT = RAW_DIR / "RSNA Pneumonia" / "raw"
MAPPINGS_JSON = RAW_DIR / "pneumonia-challenge-dataset-mappings_2018.json"
ANNOTATIONS_JSON = RAW_DIR / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_CSV = PROCESSED_DIR / "split.csv"
MASTER_CSV = PROCESSED_DIR / "master_manifest.csv"
BOXES_CSV = PROCESSED_DIR / "boxes.csv"
LUNG_MASK_DIR = PROJECT_ROOT / "data" / "lung_masks"
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
EVALUATION_DIR = PROJECT_ROOT / "results" / "evaluation"
QA_DIR = PROJECT_ROOT / "results" / "qa"

FROZEN_DICOM_PREFIX = ("RSNA Pneumonia", "raw")


def migrated_dicom_relpath(frozen_manifest_path: str) -> Path:
    """Convert immutable historical manifest path to explicit new layout.

    The value inside master_manifest.csv MUST remain unchanged. Only the
    physical lookup path is mapped. This is not a filesystem search.
    """
    if not isinstance(frozen_manifest_path, str) or not frozen_manifest_path:
        raise ValueError("Empty DICOM manifest path")
    normalized = frozen_manifest_path.replace("\\", "/")
    if normalized.startswith("/") or "//" in normalized or ":" in normalized:
        raise ValueError("Unexpected/unsafe absolute or malformed DICOM path")
    rel = PurePosixPath(normalized)
    parts = rel.parts
    if len(parts) != 5 or parts[:2] != FROZEN_DICOM_PREFIX:
        raise ValueError("DICOM path does not use frozen RSNA Study/Series/SOP layout")
    if any(p in ("", ".", "..") for p in parts):
        raise ValueError("Unsafe DICOM manifest path")
    if rel.suffix.lower() != ".dcm":
        raise ValueError("Expected DICOM file")
    return Path("data", "raw", "RSNA Pneumonia", "raw", *parts[2:])


def dicom_from_manifest(frozen_manifest_path: str, *, root: Path = PROJECT_ROOT) -> Path:
    """Physical path at data/raw; no fallback to old root."""
    return Path(root) / migrated_dicom_relpath(frozen_manifest_path)


from dataclasses import dataclass

@dataclass(frozen=True)
class AuditPaths:
    root: Path
    dicom_root: Path
    mappings: Path
    annotations: Path
    processed: Path
    qa: Path
    logs: Path


def get_audit_paths(
    dicom_root: Path | None = None,
    mappings: Path | None = None,
    annotations: Path | None = None,
) -> AuditPaths:
    """Choose the single verified data source; never silently prefer a duplicate."""
    root = PROJECT_ROOT
    return AuditPaths(
        root=root,
        dicom_root=Path(dicom_root).resolve() if dicom_root else DICOM_ROOT,
        mappings=Path(mappings).resolve() if mappings else MAPPINGS_JSON,
        annotations=Path(annotations).resolve() if annotations else ANNOTATIONS_JSON,
        processed=root / "data/processed",
        qa=root / "results/qa",
        logs=root / "results/logs",
    )
