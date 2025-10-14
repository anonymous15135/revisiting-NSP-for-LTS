"""
Utility functions for text segmentation algorithms.

This module provides common utility functions used across different segmentation algorithms,
including sentence splitting, validation, and text preprocessing.
"""

import re
import logging
from typing import List

# Try to import NLTK for proper sentence tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    NLTK_AVAILABLE = True
except ImportError:
    logging.warning("NLTK not available, falling back to simple sentence splitting")
    NLTK_AVAILABLE = False


def split_text_into_sentences(text: str, language: str = 'english') -> List[str]:
    """
    Split text into sentences using NLTK sent_tokenize if available, 
    otherwise fall back to simple sentence splitting.
    
    Args:
        text (str): Text to split into sentences
        language (str): Language for sentence tokenization ('english' or 'portuguese')
        
    Returns:
        list: List of sentence strings
    """
    if NLTK_AVAILABLE:
        try:
            # Use NLTK sentence tokenizer with appropriate language
            sentences = sent_tokenize(text, language=language)
            # Filter out very short sentences, but keep numbered items
            sentences = [s.strip() for s in sentences if s.strip() and is_valid_sentence(s.strip())]
        except Exception as e:
            logging.warning(f"NLTK sentence tokenization failed: {e}. Using fallback method.")
            sentences = _simple_sentence_split(text)
    else:
        sentences = _simple_sentence_split(text)
    
    return sentences


def _simple_sentence_split(text: str) -> List[str]:
    """
    Simple sentence splitting using regex patterns.
    
    Args:
        text (str): Text to split
        
    Returns:
        list: List of sentence strings
    """
    # Split on sentence-ending punctuation followed by whitespace and capital letter or number
    sentences = re.split(r'[.!?]+\s+(?=[A-Z0-9])', text)
    
    # Clean and filter sentences
    result = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and is_valid_sentence(sentence):
            result.append(sentence)
    
    return result


def is_valid_sentence(sentence: str) -> bool:
    """
    Check if a sentence is valid and should be kept.
    
    Args:
        sentence (str): Sentence to validate
        
    Returns:
        bool: True if sentence should be kept
    """
    # Empty sentences are invalid
    if not sentence:
        return False
    
    # Check if it's a numbered item pattern (these should be kept even if short)
    # Enhanced pattern to capture various numbering formats including prefixed ones
    # Includes decimal numbering like "12.2." common in some documents
    numbered_pattern = r'''
        ^\s*                          # Optional whitespace at start
        (?:                           # Non-capturing group for prefixes
            [-–—]+\s*                 # Dashes (regular, en-dash, em-dash)
            |[•·*+]\s*                # Bullet points
            |§\s*                     # Section symbol
        )?                            # Prefix is optional
        (?:                           # Main numbering patterns
            \d+(?:\.\d+)*[\.\)]\s*    # Numbers with optional decimal parts: "1.", "2.1.", "12.2."
            |[a-z]\)\s*               # Letters with parenthesis: "a)", "b)"
            |[ivxlcdm]+[\.\)]\s*      # Roman numerals: "i.", "ii)", "iv."
            |[-–—]+\s*                # Just dashes: "--", "---"
        )
        $                             # End of string
    '''
    if re.match(numbered_pattern, sentence, re.IGNORECASE | re.VERBOSE):
        return True
    
    # For regular sentences, require at least 3 characters
    return len(sentence) > 3


def merge_numbered_sentences(sentences: List[str]) -> List[str]:
    """
    Merge sentences that are just numbers/bullets (e.g., "2.", "3.") 
    with the following sentence to help segmentation predictions.
    
    Args:
        sentences (list): List of sentence strings
        
    Returns:
        list: List of merged sentence strings
    """
    if not sentences:
        return sentences
    
    merged_sentences = []
    i = 0
    
    # Pattern for numbered items
    numbered_pattern = r'''
        ^\s*                          # Optional whitespace at start
        (?:                           # Non-capturing group for prefixes
            [-–—]+\s*                 # Dashes (regular, en-dash, em-dash)
            |[•·*+]\s*                # Bullet points
            |§\s*                     # Section symbol
        )?                            # Prefix is optional
        (?:                           # Main numbering patterns
            \d+(?:\.\d+)*[\.\)]\s*    # Numbers with optional decimal parts: "1.", "2.1.", "12.2."
            |[a-z]\)\s*               # Letters with parenthesis: "a)", "b)"
            |[ivxlcdm]+[\.\)]\s*      # Roman numerals: "i.", "ii)", "iv."
            |[-–—]+\s*                # Just dashes: "--", "---"
        )
        $                             # End of string
    '''
    
    while i < len(sentences):
        current_sentence = sentences[i].strip()
        
        if re.match(numbered_pattern, current_sentence, re.IGNORECASE | re.VERBOSE):
            # This is a numbered item, find the next non-numbered sentence to merge with
            merge_target_idx = i + 1
            
            # Skip over consecutive numbered items to find actual content
            while (merge_target_idx < len(sentences) and 
                   re.match(numbered_pattern, sentences[merge_target_idx].strip(), re.IGNORECASE | re.VERBOSE)):
                merge_target_idx += 1
            
            if merge_target_idx < len(sentences):
                # Found a non-numbered sentence to merge with
                target_sentence = sentences[merge_target_idx].strip()
                merged_sentence = f"{current_sentence} {target_sentence}"
                merged_sentences.append(merged_sentence)
                i = merge_target_idx + 1  # Skip to after the merged target
            else:
                # No non-numbered sentence found, keep as is
                merged_sentences.append(current_sentence)
                i += 1
        else:
            # Regular sentence, keep as is
            merged_sentences.append(current_sentence)
            i += 1
    
    return merged_sentences


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text (str): Input text
        
    Returns:
        int: Number of words
    """
    return len(text.split())


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility across different libraries.
    
    Args:
        seed (int): Seed value to set
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass  # torch not available
