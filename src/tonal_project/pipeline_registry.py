"""Project pipelines."""
from __future__ import annotations

from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to Pipeline objects.
    """
    pipelines = find_pipelines()
    # Concatenate all pipelines
    all_pipelines = sum(pipelines.values(), Pipeline([]))

    # Define "__default__" pipeline
    pipelines["__default__"] = all_pipelines
    return pipelines
