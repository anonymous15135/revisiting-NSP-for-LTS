"""
CouncilSeg dataset processor.

This module handles loading and processing of the CouncilSeg dataset for topic segmentation.
The dataset consists of Portuguese and English municipal meeting minutes with topic boundaries.
CouncilSeg uses the consolidated format with corrected character offsets.
"""

import json
import logging
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path

from tqdm import tqdm

from .base import DatasetProcessor


class CouncilSegProcessor(DatasetProcessor):
    """
    Dataset processor for the CouncilSeg dataset.
    
    CouncilSeg is available in both Portuguese (PT) and English (EN) versions,
    with parallel aligned documents and segments.
    """
    
    def __init__(self, 
                 dataset_path="data/councilseg",
                 language="pt",  # 'pt' or 'en'
                 min_segment_length=50, 
                 min_segments_per_document=2,
                 max_documents=None,
                 random_seed=42, 
                 train_ratio=0.6,
                 val_ratio=0.2,
                 test_ratio=0.2,
                 **kwargs):
        """
        Initialize the CouncilSeg processor.
        
        Args:
            dataset_path (str): Path to the councilseg dataset directory
            language (str): Language version to use ('pt' for Portuguese, 'en' for English)
            min_segment_length (int): Minimum character length for valid segments
            min_segments_per_document (int): Minimum number of segments required per document
            max_documents (int): Maximum number of documents to load (None = all)
            random_seed (int): Random seed for reproducibility
            train_ratio (float): Proportion of data for training (default: 0.7)
            val_ratio (float): Proportion of data for validation (default: 0.15)
            test_ratio (float): Proportion of data for testing (default: 0.15)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        # Validate language
        if language not in ['pt', 'en']:
            raise ValueError(f"Invalid language: {language}. Must be 'pt' or 'en'")
        
        self.dataset_path = Path(dataset_path)
        self.language = language
        self.min_segment_length = min_segment_length
        self.min_segments_per_document = min_segments_per_document
        self.max_documents = max_documents
        self.random_seed = random_seed
        
        # Use ratio-based splits
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate split ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Determine dataset file based on language
        if self.language == 'pt':
            self.dataset_file = self.dataset_path / "councilseg.json"
        else:  # 'en'
            self.dataset_file = self.dataset_path / "councilseg_en.json"
        
        # Check if file exists
        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_file}")
        
        # Cache for loaded data
        self._all_documents = None
        self._splits = None
        self._split_info = None
        
        # Path to split info file (contains temporal split definition)
        self.split_info_file = self.dataset_path / "split_info.json"
        
        # Check if split_info.json exists
        if not self.split_info_file.exists():
            logging.warning(f"split_info.json not found at {self.split_info_file}")
            logging.warning("Will use ratio-based random splits instead of temporal split")
            self._use_predefined_split = False
        else:
            logging.info(f"Using predefined temporal split from {self.split_info_file}")
            self._use_predefined_split = True
        
        logging.info(f"Initialized CouncilSeg processor with language={language}, "
                    f"dataset_path={dataset_path}, min_segment_length={min_segment_length}, "
                    f"min_segments_per_document={min_segments_per_document}, "
                    f"random_seed={random_seed}")
    
    def load_dataset(self) -> Dict[str, Any]:
        """
        Load the CouncilSeg dataset from JSON file.
        Cache the results to avoid reloading for splits.
        
        Returns:
            Dict: Nested dictionary with structure {municipality: {documents: [...]}}
        """
        if self._all_documents is not None:
            logging.info("Using cached dataset")
            return self._all_documents
        
        logging.info(f"Loading CouncilSeg dataset from {self.dataset_file}")
        
        try:
            with open(self.dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flatten the nested structure and filter documents
            all_documents = []
            total_docs = 0
            filtered_docs = 0
            
            for municipality, municipality_data in data.items():
                for doc in municipality_data['documents']:
                    total_docs += 1
                    
                    # Add municipality info to document
                    doc['municipality'] = municipality
                    
                    # Filter based on segment criteria
                    if self._is_valid_document(doc):
                        all_documents.append(doc)
                    else:
                        filtered_docs += 1
            
            logging.info(f"Loaded {len(all_documents)} valid documents from {len(data)} municipalities")
            logging.info(f"Total documents: {total_docs}, Filtered: {filtered_docs}")
            
            # Apply max_documents limit if specified
            if self.max_documents is not None and len(all_documents) > self.max_documents:
                logging.info(f"Limiting to {self.max_documents} documents (from {len(all_documents)})")
                random.shuffle(all_documents)
                all_documents = all_documents[:self.max_documents]
            
            self._all_documents = all_documents
            return all_documents
            
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise
    
    def _load_split_info(self) -> Dict[str, Any]:
        """Load split information from split_info.json file."""
        if self._split_info is not None:
            return self._split_info
        
        if not self._use_predefined_split:
            return {}
        
        try:
            with open(self.split_info_file, 'r', encoding='utf-8') as f:
                self._split_info = json.load(f)
            logging.info(f"Loaded split info: strategy={self._split_info.get('strategy')}, "
                        f"train={self._split_info.get('train_count')}, "
                        f"val={self._split_info.get('val_count')}, "
                        f"test={self._split_info.get('test_count')}")
            return self._split_info
        except Exception as e:
            logging.error(f"Error loading split info from {self.split_info_file}: {e}")
            self._use_predefined_split = False
            return {}
    
    def _is_valid_document(self, doc: Dict[str, Any]) -> bool:
        """
        Check if a document meets the minimum requirements.
        
        Args:
            doc: Document dictionary
            
        Returns:
            bool: True if document is valid
        """
        segments = doc.get('segments', [])
        
        # Check minimum number of segments
        if len(segments) < self.min_segments_per_document:
            return False
        
        # Check minimum segment length
        valid_segments = 0
        for seg in segments:
            seg_length = seg.get('end', 0) - seg.get('start', 0)
            if seg_length >= self.min_segment_length:
                valid_segments += 1
        
        return valid_segments >= self.min_segments_per_document
    
    def _create_splits(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Create train/val/test splits from the dataset.
        Uses predefined temporal split from split_info.json if available,
        otherwise falls back to ratio-based random splits.
        
        Returns:
            Dict with 'train', 'val', 'test' keys containing document lists
        """
        if self._splits is not None:
            return self._splits
        
        documents = self.load_dataset()
        
        # Try to use predefined split from split_info.json
        if self._use_predefined_split:
            split_info = self._load_split_info()
            
            if split_info:
                logging.info("Using predefined temporal split from split_info.json")
                
                # Create sets of filenames (without .json extension) for each split
                train_files = set(f.replace('.json', '') for f in split_info.get('train_files', []))
                val_files = set(f.replace('.json', '') for f in split_info.get('val_files', []))
                test_files = set(f.replace('.json', '') for f in split_info.get('test_files', []))
                
                # Assign documents to splits based on document_id
                train_docs = []
                val_docs = []
                test_docs = []
                unassigned_docs = []
                
                for doc in documents:
                    doc_id = doc.get('document_id', '')
                    
                    if doc_id in train_files:
                        train_docs.append(doc)
                    elif doc_id in val_files:
                        val_docs.append(doc)
                    elif doc_id in test_files:
                        test_docs.append(doc)
                    else:
                        unassigned_docs.append(doc)
                        logging.warning(f"Document {doc_id} not found in split_info.json")
                
                # Verify split sizes
                expected_train = split_info.get('train_count', 0)
                expected_val = split_info.get('val_count', 0)
                expected_test = split_info.get('test_count', 0)
                
                if len(train_docs) != expected_train:
                    logging.warning(f"Train split size mismatch: got {len(train_docs)}, expected {expected_train}")
                if len(val_docs) != expected_val:
                    logging.warning(f"Val split size mismatch: got {len(val_docs)}, expected {expected_val}")
                if len(test_docs) != expected_test:
                    logging.warning(f"Test split size mismatch: got {len(test_docs)}, expected {expected_test}")
                
                if unassigned_docs:
                    logging.warning(f"{len(unassigned_docs)} documents not assigned to any split")
                
                splits = {
                    'train': train_docs,
                    'val': val_docs,
                    'test': test_docs
                }
                
                logging.info(f"Created predefined temporal splits: "
                            f"train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}")
                
                self._splits = splits
                return splits
        
        # Fallback to ratio-based random splits
        logging.warning("Using ratio-based random splits (not temporal!)")
        logging.warning("This may affect reproducibility and temporal evaluation!")
        
        shuffled_docs = documents.copy()
        random.shuffle(shuffled_docs)
        
        total = len(shuffled_docs)
        
        # Use ratio-based splits
        train_size = int(total * self.train_ratio)
        val_size = int(total * self.val_ratio)
        test_size = total - train_size - val_size  # Remainder
        
        # Create splits
        train_docs = shuffled_docs[:train_size]
        val_docs = shuffled_docs[train_size:train_size + val_size]
        test_docs = shuffled_docs[train_size + val_size:]
        
        splits = {
            'train': train_docs,
            'val': val_docs,
            'test': test_docs
        }
        
        logging.info(f"Created random splits from {total} total documents: "
                    f"train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}")
        
        self._splits = splits
        return splits
    
    def get_documents(self, split="test", max_documents=None) -> List[Dict[str, Any]]:
        """
        Get processed documents for a specific split.
        
        Args:
            split (str): Split to use ('train', 'val', or 'test')
            max_documents (int): Maximum number of documents to return (None = all)
            
        Returns:
            List[Dict]: List of processed document dictionaries
        """
        splits = self._create_splits()
        
        if split not in splits:
            raise ValueError(f"Invalid split: {split}. Must be one of {list(splits.keys())}")
        
        raw_documents = splits[split]
        
        # Apply max_documents limit to raw documents first for efficiency
        if max_documents is not None and len(raw_documents) > max_documents:
            raw_documents = raw_documents[:max_documents]
        
        # Process the documents to add 'text' field and boundaries
        processed_documents = []
        for doc in tqdm(raw_documents, desc=f"Processing {split} documents"):
            text, boundaries = self.create_ground_truth_segments(doc)
            
            processed_doc = {
                'meeting_id': doc.get('document_id', ''),
                'document_id': doc.get('document_id', ''),
                'municipality': doc.get('municipality', ''),
                'text': text,
                'segments': doc.get('segments', []),
                'num_segments': len(doc.get('segments', [])),
                'boundaries': boundaries
            }
            processed_documents.append(processed_doc)
        
        logging.info(f"Retrieved and processed {len(processed_documents)} documents for split '{split}'")
        return processed_documents
    
    def create_ground_truth_segments(self, document: Dict[str, Any]) -> Tuple[str, List[int]]:
        """
        Create ground truth segment boundaries from a document.
        
        Args:
            document: Document dictionary with 'full_text' and 'segments'
            
        Returns:
            Tuple[str, List[int]]: (full_text, list of segment boundary positions)
        """
        full_text = document.get('full_text', '')
        segments = document.get('segments', [])
        
        # Sort segments by start position
        sorted_segments = sorted(segments, key=lambda x: x.get('start', 0))
        
        # Extract boundary positions (start of each segment after the first)
        boundaries = []
        for i, seg in enumerate(sorted_segments):
            if i > 0:  # Skip first segment (no boundary before it)
                boundaries.append(seg['start'])
        
        return full_text, boundaries
    
    def get_evaluation_data(self, documents: List[Dict[str, Any]]) -> List[Tuple[str, List[int]]]:
        """
        Convert documents to evaluation format (text, boundaries).
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List[Tuple[str, List[int]]]: List of (text, boundaries) tuples
        """
        evaluation_data = []
        
        for doc in tqdm(documents, desc="Preparing evaluation data"):
            text, boundaries = self.create_ground_truth_segments(doc)
            evaluation_data.append((text, boundaries))
        
        return evaluation_data
    
    def process_dataset(self, dataset: List[Dict[str, Any]], split: str = 'test') -> List[Dict[str, Any]]:
        """
        Process the dataset to create meeting documents with ground truth segments.
        
        Args:
            dataset: List of document dictionaries (ignored, uses internal cache)
            split: Dataset split to use ('train', 'val', or 'test')
            
        Returns:
            List[Dict]: List of processed meetings with the following structure:
                {
                    'meeting_id': str,
                    'document_id': str,
                    'municipality': str,
                    'text': str,
                    'segments': List[Dict],
                    'num_segments': int,
                    'boundaries': List[int]
                }
        """
        documents = self.get_documents(split=split)
        
        processed_meetings = []
        
        for doc in tqdm(documents, desc=f"Processing {split} documents"):
            text, boundaries = self.create_ground_truth_segments(doc)
            
            processed_meeting = {
                'meeting_id': doc.get('document_id', ''),
                'document_id': doc.get('document_id', ''),
                'municipality': doc.get('municipality', ''),
                'text': text,
                'segments': doc.get('segments', []),
                'num_segments': len(doc.get('segments', [])),
                'boundaries': boundaries
            }
            
            processed_meetings.append(processed_meeting)
        
        logging.info(f"Processed {len(processed_meetings)} meetings for split '{split}'")
        return processed_meetings
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the CouncilSeg dataset.
        
        Returns:
            Dict containing dataset information
        """
        documents = self.load_dataset()
        splits = self._create_splits()
        
        # Calculate statistics
        total_segments = sum(len(doc.get('segments', [])) for doc in documents)
        avg_segments = total_segments / len(documents) if documents else 0
        
        total_chars = sum(len(doc.get('full_text', '')) for doc in documents)
        avg_chars = total_chars / len(documents) if documents else 0
        
        # Get unique municipalities
        municipalities = set(doc.get('municipality', '') for doc in documents)
        
        info = {
            'name': 'CouncilSeg',
            'language': 'Portuguese' if self.language == 'pt' else 'English',
            'language_code': self.language,
            'description': f'Municipal meeting minutes with topic segmentation ({self.language.upper()})',
            'total_documents': len(documents),
            'total_segments': total_segments,
            'avg_segments_per_doc': round(avg_segments, 2),
            'avg_chars_per_doc': round(avg_chars, 2),
            'municipalities': len(municipalities),
            'municipality_list': sorted(municipalities),
            'splits': {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            },
            'split_ratios': {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            },
            'min_segment_length': self.min_segment_length,
            'min_segments_per_document': self.min_segments_per_document,
            'random_seed': self.random_seed,
            'dataset_file': str(self.dataset_file)
        }
        
        return info
    
    def save_processed_meetings(self, processed_meetings: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save processed meetings as individual text files and ground truth annotations.
        
        Args:
            processed_meetings: List of processed meeting dictionaries
            output_dir: Directory to save the processed meetings
        """
        output_path = Path(output_dir)
        meetings_dir = output_path / "meetings"
        ground_truth_dir = output_path / "ground_truth"
        
        meetings_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in tqdm(processed_meetings, desc="Saving processed documents"):
            # Get document ID - check both meeting_id and document_id
            doc_id = doc.get('meeting_id') or doc.get('document_id')
            if not doc_id:
                logging.warning(f"Document missing ID, skipping: {doc.keys()}")
                continue
            
            doc_text = doc.get('text', '')
            segments = doc.get('segments', [])
            
            # Save raw document text
            doc_file_path = meetings_dir / f"{doc_id}.txt"
            with open(doc_file_path, 'w', encoding='utf-8') as f:
                f.write(doc_text)
            
            # Create and save ground truth annotation
            annotated_text = self.create_annotated_text(doc_text, segments)
            gt_file_path = ground_truth_dir / f"{doc_id}_annotated.txt"
            with open(gt_file_path, 'w', encoding='utf-8') as f:
                f.write(annotated_text)
        
        logging.info(f"Saved {len(processed_meetings)} documents and their ground truth files to {output_dir}")


def create_processor(**kwargs) -> CouncilSegProcessor:
    """
    Factory function to create a CouncilSeg processor.
    
    Args:
        **kwargs: Configuration parameters for the processor
    
    Returns:
        CouncilSegProcessor instance
    """
    return CouncilSegProcessor(**kwargs)
