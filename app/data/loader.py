"""Data loading utilities for the Streamlit application."""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Any, MutableMapping, Optional, Sequence

import pandas as pd
import requests


class DataLoaderError(RuntimeError):
    """Raised when a data source could not be loaded."""


@dataclass
class LoadedData:
    """Container for the frames stored in Streamlit's session state."""

    frames: list[pd.DataFrame]
    file_name_map: dict[str, int]
    source: str = "none"

    @property
    def has_data(self) -> bool:
        return bool(self.frames)


SessionState = MutableMapping[str, Any]


def ensure_state_defaults(state: SessionState) -> None:
    """Guarantee that commonly used keys exist in ``st.session_state``."""
    state.setdefault("curr_filtered_df", pd.DataFrame())
    state.setdefault("file_uploaded", False)
    state.setdefault("files", [])
    state.setdefault("file_name", {})
    state.setdefault("select_df", None)
    state.setdefault("filtered_df", pd.DataFrame())


def fetch_github_csv(url: str, token: Optional[str] = None, *, timeout: int = 15) -> pd.DataFrame:
    """Load a CSV file from GitHub using an optional personal access token."""
    headers = {"Accept": "text/csv"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        raise DataLoaderError(f"Could not load CSV from GitHub: {exc}") from exc

    return pd.read_csv(StringIO(response.text))


def prepare_uploaded_data(uploaded_files: Optional[Sequence[Any]]) -> LoadedData:
    """Convert uploaded files into ``pandas`` DataFrames with row numbers."""
    frames: list[pd.DataFrame] = []
    name_map: dict[str, int] = {}

    if not uploaded_files:
        return LoadedData(frames=frames, file_name_map=name_map, source="none")

    for index, uploaded in enumerate(uploaded_files):
        frame = pd.read_csv(uploaded)
        frame["Row_Number_"] = range(len(frame))
        frames.append(frame)
        name = getattr(uploaded, "name", f"Dataset {index + 1}")
        name_map[name] = index

    return LoadedData(frames=frames, file_name_map=name_map, source="upload")


def prepare_default_data(default_df: Optional[pd.DataFrame], default_name: str) -> LoadedData:
    """Wrap a default DataFrame (if available) into a ``LoadedData`` object."""
    if default_df is None or default_df.empty:
        return LoadedData(frames=[], file_name_map={}, source="none")

    frame = default_df.copy()
    frame["Row_Number_"] = range(len(frame))
    return LoadedData(frames=[frame], file_name_map={default_name: 0}, source="default")


def update_state_with_data(state: SessionState, loaded: LoadedData) -> None:
    """Store loaded data details back into the Streamlit session state."""
    state["files"] = loaded.frames
    state["file_name"] = loaded.file_name_map
    state["file_uploaded"] = loaded.has_data

    if state.get("select_df") not in loaded.file_name_map:
        state["select_df"] = next(iter(loaded.file_name_map), None)

    if not loaded.has_data:
        state["filtered_df"] = pd.DataFrame()


__all__ = [
    "DataLoaderError",
    "LoadedData",
    "ensure_state_defaults",
    "fetch_github_csv",
    "prepare_default_data",
    "prepare_uploaded_data",
    "update_state_with_data",
]
