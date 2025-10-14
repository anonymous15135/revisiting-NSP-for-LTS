"""
WikiSection dataset processor.

This module handles loading and processing of the WikiSection dataset for topic segmentation.
"""

import os
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm

from .base import DatasetProcessor


class WikiSectionProcessor(DatasetProcessor):
    """
    Dataset processor for the WikiSection dataset.
    """
    
    def __init__(self, dataset_path=None, subset="en_city", 
                 min_segment_length=50, min_segments_per_document=2, 
                 train_size=None, val_size=None, test_size=None, random_seed=42, **kwargs):
        """
        Initialize the WikiSection processor.
        
        Args:
            dataset_path (str): Path to the WikiSection dataset directory (default: auto-detect)
            subset (str): Which subset to use ('en_city' or 'en_disease')
            min_segment_length (int): Minimum character length for valid segments
            min_segments_per_document (int): Minimum number of segments required per document
            train_size (int): Number of documents for training split (None = use all available)
            val_size (int): Number of documents for validation split (None = use all available)
            test_size (int): Number of documents for test split (None = use all available)
            random_seed (int): Random seed for reproducibility
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        # Auto-detect dataset path if not provided
        if dataset_path is None:
            # Try to find the dataset relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up to src/topic_segmentation, then to project root, then to data
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            dataset_path = os.path.join(project_root, "data", "wikisection_dataset")
            logging.info(f"Auto-detected dataset path: {dataset_path}")
        
        # Convert to absolute path
        self.dataset_path = os.path.abspath(dataset_path)
        self.subset = subset
        self.min_segment_length = min_segment_length
        self.min_segments_per_document = min_segments_per_document
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
        
        logging.info(f"Initialized WikiSection processor with subset={subset}, "
                    f"min_segment_length={min_segment_length}, "
                    f"min_segments_per_document={min_segments_per_document}, "
                    f"train_size={train_size}, val_size={val_size}, test_size={test_size}, "
                    f"random_seed={random_seed}")
    
    def load_dataset(self) -> Dict[str, List[Dict]]:
        """
        Load the WikiSection dataset from local JSON files.
        
        Returns:
            Dict[str, List[Dict]]: Dictionary with train/test/validation splits
        """
        logging.info(f"Loading WikiSection dataset from {self.dataset_path}...")
        
        dataset = {}
        splits = ['train', 'test', 'validation']
        
        for split in splits:
            filename = f"wikisection_{self.subset}_{split}.json"
            filepath = os.path.join(self.dataset_path, filename)
            
            if not os.path.exists(filepath):
                logging.warning(f"Split file not found: {filepath}")
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                dataset[split] = data
                logging.info(f"Loaded {split} split: {len(data)} documents")
            except Exception as e:
                logging.error(f"Error loading {filepath}: {e}")
                continue
        
        if not dataset:
            logging.error("No dataset splits could be loaded")
            return None
            
        # Log sample structure for debugging
        if 'test' in dataset and dataset['test']:
            sample = dataset['test'][0]
            logging.info(f"Sample document structure: {list(sample.keys())}")
            if 'annotations' in sample:
                logging.info(f"Sample has {len(sample['annotations'])} annotations")
                if sample['annotations']:
                    logging.info(f"First annotation: {sample['annotations'][0]}")
        
        logging.info(f"Successfully loaded WikiSection dataset with splits: {list(dataset.keys())}")
        return dataset
    
    def process_dataset(self, dataset: Dict[str, List[Dict]], split: str = 'test') -> List[Dict[str, Any]]:
        """
        Process the WikiSection dataset to create documents with ground truth segments.
        
        Args:
            dataset: The WikiSection dataset dictionary
            split: Dataset split to use ('train', 'test', or 'validation')
            
        Returns:
            List[Dict]: List of processed documents with ground truth segments
        """
        if dataset is None or split not in dataset:
            logging.warning(f"Dataset is None or split '{split}' not found, returning empty list")
            return []
        
        data_split = dataset[split]
        logging.info(f"Processing {split} split with {len(data_split)} documents")
        
        processed_documents = []
        
        for doc in tqdm(data_split, desc="Processing documents"):
            # Extract basic information
            doc_id = doc.get('id', '').split('/')[-1]  # Get the last part of the URL as ID
            title = doc.get('title', '')
            text = doc.get('text', '')
            
            # Skip documents that are too short
            if len(text) < self.min_segment_length:
                continue
            
            # Process annotations to create segments
            annotations = doc.get('annotations', [])
            
            # Skip documents with too few segments
            if len(annotations) < self.min_segments_per_document:
                continue
            
            # Create segments from annotations
            segments = []
            for i, annotation in enumerate(annotations):
                segment_start = annotation.get('begin', 0)
                segment_length = annotation.get('length', 0)
                segment_end = segment_start + segment_length
                
                # Extract segment text
                segment_text = text[segment_start:segment_end]
                
                # Skip segments that are too short
                if len(segment_text) < self.min_segment_length:
                    continue
                
                segments.append({
                    'segment_id': str(i + 1),  # Use simple numeric ID as string
                    'start': segment_start,
                    'end': segment_end,
                    'text': segment_text,
                    'section_heading': annotation.get('sectionHeading', ''),
                    'section_label': annotation.get('sectionLabel', ''),
                    'class': annotation.get('class', '')
                })
            
            # Skip documents with too few valid segments after filtering
            if len(segments) < self.min_segments_per_document:
                continue
            
            # Create document entry
            processed_documents.append({
                'meeting_id': doc_id,  # Using 'meeting_id' for compatibility
                'document_id': doc_id,  # Add document_id for consistency with CouncilSeg
                'title': title,
                'text': text,
                'segments': segments,
                'num_segments': len(segments),
                'original_url': doc.get('id', ''),
                'document_type': doc.get('type', ''),
                'subset': self.subset
            })
        
        logging.info(f"Created {len(processed_documents)} processed documents from {split} split")
        if processed_documents:
            avg_segments = sum(d['num_segments'] for d in processed_documents) / len(processed_documents)
            logging.info(f"Average segments per document: {avg_segments:.1f}")
            
            # Log segment length statistics
            all_segment_lengths = []
            for doc in processed_documents:
                for segment in doc['segments']:
                    all_segment_lengths.append(len(segment['text']))
            
            if all_segment_lengths:
                avg_length = sum(all_segment_lengths) / len(all_segment_lengths)
                min_length = min(all_segment_lengths)
                max_length = max(all_segment_lengths)
                logging.info(f"Segment length stats - Avg: {avg_length:.1f}, Min: {min_length}, Max: {max_length}")
        
        return processed_documents
    
    def get_documents(self, split="test", max_documents=None) -> List[Dict[str, Any]]:
        """
        Get documents for processing with specified split.
        
        Args:
            split (str): Dataset split ('train', 'test', 'validation', or 'all')
            max_documents (int): Maximum number of documents to return
            
        Returns:
            List of document dictionaries
        """
        # Load the dataset
        dataset = self.load_dataset()
        if not dataset:
            logging.warning(f"Dataset is None or split '{split}' not found, returning empty list")
            return []
        
        # Handle 'all' split by combining all splits
        if split == "all":
            documents = []
            for split_name in ['train', 'test', 'validation']:
                if split_name in dataset:
                    processed_docs = self.process_dataset(dataset, split_name)
                    documents.extend(processed_docs)
        else:
            # Handle specific split
            if split == 'val':
                split = 'validation'  # Handle 'val' as alias for 'validation'
            
            if split not in dataset:
                logging.warning(f"Split '{split}' not found in dataset, returning empty list")
                return []
            
            documents = self.process_dataset(dataset, split)
        
        # Apply split-specific limits first (from config)
        # Use early sampling for efficiency on large datasets
        if split == 'train' and self.train_size is not None and len(documents) > self.train_size:
            import random
            random.seed(self.random_seed)
            logging.info(f"Limiting training set to {self.train_size} documents (out of {len(documents)} available)")
            documents = random.sample(documents, self.train_size)
        elif split in ['val', 'validation'] and self.val_size is not None and len(documents) > self.val_size:
            import random
            random.seed(self.random_seed)
            logging.info(f"Limiting validation set to {self.val_size} documents (out of {len(documents)} available)")
            documents = random.sample(documents, self.val_size)
        elif split == 'test' and hasattr(self, 'test_size') and self.test_size is not None and len(documents) > self.test_size:
            import random
            random.seed(self.random_seed)
            logging.info(f"Limiting test set to {self.test_size} documents (out of {len(documents)} available)")
            documents = random.sample(documents, self.test_size)
        
        # Apply max_documents limit if specified (for test/evaluation)
        if max_documents and max_documents < len(documents):
            import random
            random.seed(self.random_seed)
            logging.info(f"Limiting to {max_documents} documents (out of {len(documents)} available for split '{split}')")
            documents = random.sample(documents, max_documents)
        
        logging.info(f"Returning {len(documents)} documents for split '{split}'")
        return documents
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the WikiSection dataset.
        
        Returns:
            Dict containing dataset information
        """
        return {
            'name': 'WikiSection',
            'description': f'Wikipedia articles with section-level segmentation ({self.subset} subset)',
            'source': 'sebastianarnold/WikiSection',
            'splits': ['train', 'test', 'validation'],
            'task': 'topic_segmentation',
            'language': 'english',
            'subset': self.subset,
            'synthetic_meetings': False,
            'min_segment_length': self.min_segment_length,
            'min_segments_per_document': self.min_segments_per_document,
            'dataset_path': self.dataset_path
        }
    
    def save_processed_meetings(self, processed_documents: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Save processed documents and ground truth files.
        
        Args:
            processed_documents: List of processed document dictionaries
            output_dir: Directory to save the processed documents
        """
        # Create subdirectories
        meetings_dir = os.path.join(output_dir, "meetings")
        ground_truth_dir = os.path.join(output_dir, "ground_truth")
        os.makedirs(meetings_dir, exist_ok=True)
        os.makedirs(ground_truth_dir, exist_ok=True)
        
        for doc in tqdm(processed_documents, desc="Saving processed documents"):
            doc_id = doc['meeting_id']
            doc_text = doc['text']
            segments = doc['segments']
            
            # Save raw document text
            doc_file_path = os.path.join(meetings_dir, f"{doc_id}.txt")
            with open(doc_file_path, 'w', encoding='utf-8') as f:
                f.write(doc_text)
            
            # Create and save ground truth annotation
            annotated_text = self.create_annotated_text(doc_text, segments)
            gt_file_path = os.path.join(ground_truth_dir, f"{doc_id}_annotated.txt")
            with open(gt_file_path, 'w', encoding='utf-8') as f:
                f.write(annotated_text)
        
        logging.info(f"Saved {len(processed_documents)} documents and their ground truth files to {output_dir}")

def create_wikisection_processor(**kwargs):
    """
    Factory function to create a WikiSection processor instance.
    
    Args:
        **kwargs: Configuration parameters for the processor
        
    Returns:
        WikiSectionProcessor: A configured WikiSection processor instance
    """
    return WikiSectionProcessor(**kwargs)
