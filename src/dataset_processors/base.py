"""
Base class for dataset processors.

This module defines the interface that all dataset processors must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class DatasetProcessor(ABC):
    """
    Abstract base class for dataset processors.
    
    All dataset processors must inherit from this class and implement the required methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the dataset processor.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        pass
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """
        Load the raw dataset.
        
        Returns:
            The loaded dataset object
        """
        pass
    
    @abstractmethod
    def process_dataset(self, dataset: Any, split: str = 'test') -> List[Dict[str, Any]]:
        """
        Process the dataset to create meeting documents with ground truth segments.
        
        Args:
            dataset: The raw dataset object
            split: Dataset split to use ('train', 'test', or 'validation')
            
        Returns:
            List[Dict]: List of processed meetings with the following structure:
                {
                    'meeting_id': str,
                    'text': str,
                    'segments': List[Dict],
                    'num_segments': int
                }
        """
        pass
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dict containing dataset information like name, description, splits, etc.
        """
        pass
    
    def save_processed_meetings(self, processed_meetings: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save processed meetings to files (optional implementation).
        
        Args:
            processed_meetings: List of processed meeting dictionaries
            output_dir: Directory to save the processed meetings
        """
        pass
    
    def create_annotated_text(self, meeting_text: str, segments: List[Dict[str, Any]]) -> str:
        """
        Create an annotated text with segment boundaries marked.
        
        Args:
            meeting_text: The complete meeting text
            segments: List of segment dictionaries with start and end positions
            
        Returns:
            str: Annotated text with ##SEQ# and #SEQ## markers
        """
        annotated_text = meeting_text
        offset = 0  # Keep track of how much we've shifted the text
        
        # Sort segments by start position
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        for i, segment in enumerate(sorted_segments, 1):
            # Calculate adjusted start position
            adjusted_start = segment['start'] + offset
            
            # Create markers
            start_marker = f"##{i}#"
            end_marker = f"#{i}##"
            
            # Insert the start marker
            annotated_text = (
                annotated_text[:adjusted_start] + 
                start_marker + 
                annotated_text[adjusted_start:]
            )
            
            # Update offset
            offset += len(start_marker)
            
            # Calculate adjusted end position
            adjusted_end = segment['end'] + offset
            
            # Insert the end marker
            annotated_text = (
                annotated_text[:adjusted_end] + 
                end_marker + 
                annotated_text[adjusted_end:]
            )
            
            # Update offset
            offset += len(end_marker)
        
        return annotated_text
