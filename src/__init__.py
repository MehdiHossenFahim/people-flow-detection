"""
People Flow Detection Package
Real-time pedestrian tracking, bidirectional counting, and spatial density heatmaps.
"""

from .pipeline import (
    LineCrossingCounter,
    SpatialHeatmapAccumulator,
    PeopleFlowPipeline,
    main,
)

__all__ = [
    "LineCrossingCounter",
    "SpatialHeatmapAccumulator",
    "PeopleFlowPipeline",
    "main",
]
__version__ = "1.0.0"
