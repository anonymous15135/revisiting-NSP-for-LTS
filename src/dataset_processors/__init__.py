"""
Datasets module for topic segmentation.

This module provides a unified interface for loading and processing different datasets
for topic segmentation tasks.
"""

from .base import DatasetProcessor
from .wikisection import WikiSectionProcessor
from .councilseg import CouncilSegProcessor

# Registry of available dataset processors
DATASET_PROCESSORS = {
    "wikisection": WikiSectionProcessor,
    "councilseg": CouncilSegProcessor,
}

def create_dataset_processor(dataset_name, **kwargs):
    """
    Factory function to create a dataset processor instance.
    
    Args:
        dataset_name (str): Name of the dataset processor to create
        **kwargs: Additional arguments to pass to the processor constructor
        
    Returns:
        DatasetProcessor: An instance of the requested dataset processor
        
    Raises:
        ValueError: If the dataset name is not recognized
    """
    if dataset_name not in DATASET_PROCESSORS:
        available = ", ".join(DATASET_PROCESSORS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {available}")
    
    processor_class = DATASET_PROCESSORS[dataset_name]
    return processor_class(**kwargs)

def get_available_datasets():
    """
    Get a list of available dataset names.
    
    Returns:
        list: List of available dataset names
    """
    return list(DATASET_PROCESSORS.keys())
