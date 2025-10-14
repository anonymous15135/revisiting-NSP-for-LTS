"""
Base class for all topic segmentation algorithms.

This module defines the interface that all segmentation algorithms should implement.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class SegmentationAlgorithm(ABC):
    """
    Abstract base class for text segmentation algorithms.
    """
    
    @abstractmethod
    def segment_text(self, text: str, sentences: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """
        Segment text into topically coherent segments.
        
        Args:
            text (str): The full text to segment
            sentences (list): List of sentence dictionaries with text and spans
            
        Returns:
            tuple: (annotated_text, segment_boundaries)
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of the segmentation algorithm.
        
        Returns:
            str: Algorithm name
        """
        return self.__class__.__name__.replace('Segmenter', '')
    
    def load_model(self):
        """
        Load the model/resources required for segmentation.
        Subclasses should override this if they need to load models.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return True
    
    def train_model(self, train_documents=None, val_documents=None):
        """
        Train or fine-tune the model if supported.
        Subclasses should override this if they support training.
        
        Args:
            train_documents: List of training documents
            val_documents: List of validation documents
        """
        pass