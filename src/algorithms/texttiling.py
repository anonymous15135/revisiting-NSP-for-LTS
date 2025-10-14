"""
TextTiling Topic Segmentation Algorithm

This module provides a class for segmenting text into topically coherent segments
using the TextTiling algorithm from NLTK.
"""

import logging
import re
from nltk.tokenize.texttiling import TextTilingTokenizer
from .base import SegmentationAlgorithm

class TextTilingSegmenter(SegmentationAlgorithm):
    """
    Topic segmentation algorithm based on TextTiling.
    
    TextTiling is a domain-independent algorithm that uses lexical cohesion
    to identify topic boundaries in text.
    """
    
    def __init__(self, w=20, k=15, similarity_method=0, stopwords=None, smoothing_method=[0], smoothing_width=2, smoothing_rounds=1, cutoff_policy=1):
        """
        Initialize the TextTiling segmenter.
        
        Args:
            w (int): Pseudosentence size. Default is 20.
            k (int): Size (in sentences) of the block used in the block comparison method. Default is 10.
            similarity_method (int): The method used for determining similarity scores:
                                   0 (default) uses the cosine similarity.
                                   1 uses the Dice coefficient.
            stopwords (list): List of stopwords to filter out. If None, uses default English stopwords.
            smoothing_method (list): List of smoothing methods to apply:
                                   [0] (default) uses no smoothing.
                                   [1] uses the average.
            smoothing_width (int): The width of the window used for smoothing. Default is 2.
            smoothing_rounds (int): Number of smoothing rounds. Default is 1.
            cutoff_policy (int): The cutoff policy. Default is 1.
        """
        self.w = w
        self.k = k
        self.similarity_method = similarity_method
        self.stopwords = stopwords
        self.smoothing_method = smoothing_method
        self.smoothing_width = smoothing_width
        self.smoothing_rounds = smoothing_rounds
        self.cutoff_policy = cutoff_policy
        
        # Initialize the TextTiling tokenizer
        self.tokenizer = TextTilingTokenizer(
            w=self.w,
            k=self.k,
            similarity_method=self.similarity_method,
            stopwords=self.stopwords,
            smoothing_method=self.smoothing_method,
            smoothing_width=self.smoothing_width,
            smoothing_rounds=self.smoothing_rounds,
            cutoff_policy=self.cutoff_policy
        )
        
        logging.info(f"Initialized TextTiling segmenter with w={w}, k={k}, similarity_method={similarity_method}")
    
    def segment_text(self, text, sentences):
        """
        Segment text using the TextTiling algorithm.
        
        Args:
            text (str): The full text to segment
            sentences (list): List of sentence dictionaries with text and spans
            
        Returns:
            tuple: (annotated_text, segment_boundaries)
        """
        logging.info("Starting TextTiling segmentation...")
        
        if not sentences:
            logging.warning("No sentences provided for segmentation")
            return text, []
        
        try:
            # Prepare the text for TextTiling by joining sentences with newlines
            # TextTiling expects paragraph-separated text
            sentence_texts = [sent['text'].strip() for sent in sentences if sent['text'].strip()]
            tiling_input = '\n\n'.join(sentence_texts)
            
            # Perform TextTiling segmentation
            segments = self.tokenizer.tokenize(tiling_input)
            
            logging.info(f"TextTiling found {len(segments)} segments")
            
            # Convert TextTiling segments back to character-based boundaries
            segment_boundaries = self._map_segments_to_boundaries(segments, sentences, text)
            
            # Create annotated text with segment markers
            annotated_text = self._create_annotated_text(text, segment_boundaries)
            
            logging.info(f"Created {len(segment_boundaries)} segment boundaries")
            
            return annotated_text, segment_boundaries
            
        except Exception as e:
            logging.error(f"Error during TextTiling segmentation: {e}")
            # Return original text with no segmentation if there's an error
            return text, []
    
    def _map_segments_to_boundaries(self, tiling_segments, sentences, original_text):
        """
        Map TextTiling segments back to character positions in the original text.
        
        Args:
            tiling_segments (list): List of text segments from TextTiling
            sentences (list): List of sentence dictionaries with text and spans
            original_text (str): The original text
            
        Returns:
            list: List of segment boundary dictionaries
        """
        boundaries = []
        current_char_pos = 0
        segment_id = 1
        
        # Create a mapping from sentence text to sentence spans
        sentence_map = {}
        for sent in sentences:
            clean_text = sent['text'].strip()
            if clean_text:
                sentence_map[clean_text] = sent
        
        for segment in tiling_segments:
            segment_start = current_char_pos
            
            # Find all sentences in this segment
            segment_sentences = segment.strip().split('\n\n')
            segment_end = segment_start
            
            for sent_text in segment_sentences:
                sent_text = sent_text.strip()
                if sent_text and sent_text in sentence_map:
                    sent_span = sentence_map[sent_text]
                    # Update the segment end to include this sentence
                    segment_end = max(segment_end, sent_span['end'])
                    current_char_pos = max(current_char_pos, sent_span['end'])
            
            # Create boundary object
            if segment_end > segment_start:
                boundaries.append({
                    'seq': segment_id,
                    'start': segment_start,
                    'end': segment_end,
                    'text': original_text[segment_start:segment_end]
                })
                segment_id += 1
        
        return boundaries
    
    def _create_annotated_text(self, text, boundaries):
        """
        Create annotated text with segment markers.
        
        Args:
            text (str): Original text
            boundaries (list): List of segment boundaries
            
        Returns:
            str: Text with ##SEQ# and #SEQ## markers
        """
        if not boundaries:
            return text
        
        annotated_text = text
        offset = 0
        
        # Sort boundaries by start position
        sorted_boundaries = sorted(boundaries, key=lambda x: x['start'])
        
        for boundary in sorted_boundaries:
            # Calculate positions in the modified text
            adjusted_start = boundary['start'] + offset
            
            # Create the markers
            start_marker = f"##{boundary['seq']}#"
            end_marker = f"#{boundary['seq']}##"
            
            # Insert the start marker
            annotated_text = (
                annotated_text[:adjusted_start] + 
                start_marker + 
                annotated_text[adjusted_start:]
            )
            
            # Update offset
            offset += len(start_marker)
            
            # Calculate adjusted end position
            adjusted_end = boundary['end'] + offset
            
            # Insert the end marker
            annotated_text = (
                annotated_text[:adjusted_end] + 
                end_marker + 
                annotated_text[adjusted_end:]
            )
            
            # Update offset
            offset += len(end_marker)
        
        return annotated_text

def create_segmenter(**kwargs):
    """
    Factory function to create a TextTiling segmenter.
    
    Args:
        **kwargs: Parameters for the TextTiling algorithm
        
    Returns:
        TextTilingSegmenter: Configured TextTiling segmenter instance
    """
    # Convert smoothing_method to list if it's an integer
    if 'smoothing_method' in kwargs and isinstance(kwargs['smoothing_method'], int):
        kwargs['smoothing_method'] = [kwargs['smoothing_method']]
    
    return TextTilingSegmenter(**kwargs)
