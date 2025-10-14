"""
Algorithms module for text segmentation.

This module provides a factory interface for creating various segmentation algorithm instances.
"""

from .nsp import create_segmenter as create_nsp_segmenter
from .cnn_bilstm import create_segmenter as create_cnn_bilstm_segmenter
from .topseg import create_segmenter as create_topseg_segmenter
from .texttiling import create_segmenter as create_texttiling_segmenter
from .lumberchunker import LumberChunkerSegmenter

# Dictionary mapping algorithm names to their factory functions
ALGORITHM_FACTORIES = {
    "nsp": create_nsp_segmenter,
    "cnn_bilstm": create_cnn_bilstm_segmenter,
    "topseg": create_topseg_segmenter,
    "texttiling": create_texttiling_segmenter,
    "lumberchunker": lambda **kwargs: LumberChunkerSegmenter(**kwargs)
}

def get_available_algorithms():
    """
    Get the list of available segmentation algorithm names
    
    Returns:
        list: Names of available algorithms
    """
    return list(ALGORITHM_FACTORIES.keys())

def create_segmenter(algorithm_name, **kwargs):
    """
    Factory function to create a segmenter by name
    
    Args:
        algorithm_name (str): Name of the algorithm to create ("nsp", "cvs")
        **kwargs: Additional parameters to pass to the specific algorithm factory
        
    Returns:
        SegmentationAlgorithm: An instance of the requested segmentation algorithm
        
    Raises:
        ValueError: If the algorithm name is not recognized
    """
    if algorithm_name not in ALGORITHM_FACTORIES:
        available = ", ".join(get_available_algorithms())
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available algorithms: {available}")
    
    return ALGORITHM_FACTORIES[algorithm_name](**kwargs)

# Alias for compatibility
create_algorithm = create_segmenter