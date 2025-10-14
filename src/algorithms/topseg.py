"""
TopSeg (RoBERTa Topic Segmentation) Algorithm

This module implements the TopSeg model from the NSE-TopicSegmentation paper,
which fine-tunes RoBERTa with a coherence loss function to make embeddings from
the same topic segment closer in space and embeddings from different segments
further apart.
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from .base import SegmentationAlgorithm

# Import dataset processor for internal data loading
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset_processors import create_dataset_processor


class TopSegCoherenceDataset(Dataset):
    """
    Dataset class for TopSeg coherence pre-training.
    Creates sentence pairs with labels indicating whether they come from the same segment.
    """
    
    def __init__(self, documents, tokenizer, max_length=512):
        """
        Initialize the TopSeg coherence dataset.
        
        Args:
            documents: List of document dictionaries with segments
            tokenizer: Tokenizer for encoding sentences
            max_length: Maximum sequence length
        """
        self.sentence_pairs = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self._create_training_pairs(documents)
    
    def _create_training_pairs(self, documents):
        """Create sentence pairs for coherence training following the TopSeg approach."""
        logging.info(f"Creating training pairs from {len(documents)} documents")
        
        for doc_idx, doc in enumerate(documents):
            logging.info(f"Processing document {doc_idx}: type={type(doc)}")
            
            # Handle both dict and object-like structures - similar to NSP
            if hasattr(doc, 'segments'):
                segments = doc.segments
                logging.info(f"Found {len(segments)} segments via .segments")
            elif isinstance(doc, dict):
                segments = doc.get('segments', [])
                logging.info(f"Found {len(segments)} segments via dict['segments']")
            else:
                logging.warning(f"Unknown document format: {type(doc)}, skipping")
                continue
            
            if not segments or len(segments) < 2:
                logging.warning(f"Document {doc_idx} has insufficient segments ({len(segments)}), skipping")
                continue
            
            # Extract sentences from segments (like NSP does)
            all_sentences = []
            segment_boundaries = []
            
            for segment in segments:
                if hasattr(segment, 'text'):
                    segment_text = segment.text
                elif isinstance(segment, dict):
                    segment_text = segment.get('text', '')
                else:
                    segment_text = str(segment)
                
                # Split segment into sentences (simple approach)
                sentences = [s.strip() for s in segment_text.split('.') if s.strip() and len(s.strip()) > 10]
                
                start_idx = len(all_sentences)
                all_sentences.extend(sentences)
                end_idx = len(all_sentences)
                segment_boundaries.append((start_idx, end_idx))
            
            if len(all_sentences) < 2:
                logging.warning(f"Document {doc_idx} has too few sentences ({len(all_sentences)}), skipping")
                continue
            
            logging.info(f"Document {doc_idx}: extracted {len(all_sentences)} sentences from {len(segments)} segments")
            
            doc_pairs_before = len(self.sentence_pairs)
            
            # Create positive pairs (same segment) and negative pairs (different segments)
            for i in range(len(all_sentences) - 1):
                sentence_a = all_sentences[i]
                sentence_b = all_sentences[i + 1]
                
                # Check if consecutive sentences are in the same segment
                same_segment = self._are_sentences_in_same_segment(i, i + 1, segment_boundaries)
                
                # Convert to CosineSimilarityLoss format: 1.0 for same segment, 0.0 for different segments
                if same_segment:
                    label = 1.0  # Similar sentences (same segment)
                else:
                    label = 0.0  # Dissimilar sentences (different segments)
                
                self.sentence_pairs.append((sentence_a, sentence_b))
                self.labels.append(label)
            
            # Add some random negative pairs from different segments for better training
            for _ in range(min(5, len(all_sentences) // 4)):
                if len(all_sentences) >= 4:
                    i = np.random.randint(0, len(all_sentences) - 2)
                    j = np.random.randint(i + 2, len(all_sentences))
                    
                    if not self._are_sentences_in_same_segment(i, j, segment_boundaries):
                        sentence_a = all_sentences[i]
                        sentence_b = all_sentences[j]
                        
                        self.sentence_pairs.append((sentence_a, sentence_b))
                        self.labels.append(0.0)  # Different segments (dissimilar)
            
            doc_pairs_after = len(self.sentence_pairs)
            logging.info(f"Document {doc_idx}: created {doc_pairs_after - doc_pairs_before} pairs")
        
        logging.info(f"Created {len(self.sentence_pairs)} sentence pairs for TopSeg training")
        positive_pairs = sum(1 for label in self.labels if label > 0)
        negative_pairs = len(self.labels) - positive_pairs
        logging.info(f"Label distribution: {positive_pairs} same segment, {negative_pairs} different segments")
    
    def _are_sentences_in_same_segment(self, sent_idx_a, sent_idx_b, segment_boundaries):
        """Check if two sentence indices are in the same segment."""
        for start, end in segment_boundaries:
            if start <= sent_idx_a < end and start <= sent_idx_b < end:
                return True
        return False
    
    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, idx):
        sentence_a, sentence_b = self.sentence_pairs[idx]
        label = self.labels[idx]
        
        return InputExample(texts=[sentence_a, sentence_b], label=label)


class TopSegSegmenter(SegmentationAlgorithm):
    """
    TopSeg topic segmentation algorithm that fine-tunes RoBERTa with coherence loss
    and then uses it for segmentation.
    """
    
    def __init__(
        self,
        model_name="neuralmind/bert-base-portuguese-cased",
        pooling_mode="mean",
        fine_tuning=True,
        learning_rate=2e-5,
        batch_size=16,
        epochs=10,
        warmup_steps=100,
        max_length=512,
        threshold=0.5,
        save_model_path=None,
        config=None,
        **kwargs
    ):
        """
        Initialize the TopSeg segmenter.
        
        Args:
            model_name (str): Name of the pretrained model to use
            pooling_mode (str): Pooling strategy ("mean", "cls", "max")
            fine_tuning (bool): Whether to perform fine-tuning
            learning_rate (float): Learning rate for fine-tuning
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            warmup_steps (int): Warmup steps for scheduler
            max_length (int): Maximum sequence length
            threshold (float): Threshold for boundary prediction
            save_model_path (str): Path to save fine-tuned model
            config (dict): Configuration dictionary for dataset loading
            **kwargs: Additional keyword arguments
        """
        self.model_name = model_name
        self.pooling_mode = pooling_mode
        self.fine_tuning = fine_tuning
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.threshold = threshold
        self.save_model_path = save_model_path
        
        # Store config for dataset loading
        self.config = config or {}
        self.config.update(kwargs)
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.training_mode = fine_tuning
        
    def load_model(self):
        """Load the SentenceTransformer model for TopSeg."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading TopSeg model '{self.model_name}' on {self.device}")
        
        try:
            if self.fine_tuning:
                # Create SentenceTransformer model for fine-tuning
                word_embedding_model = models.Transformer(
                    self.model_name, 
                    max_seq_length=self.max_length
                )
                pooling_model = models.Pooling(
                    word_embedding_model.get_word_embedding_dimension(),
                    pooling_mode=self.pooling_mode
                )
                self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            else:
                # Load pre-trained SentenceTransformer if available
                if self.save_model_path and os.path.exists(self.save_model_path):
                    self.model = SentenceTransformer(self.save_model_path)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                else:
                    # Fallback to basic model
                    word_embedding_model = models.Transformer(
                        self.model_name, 
                        max_seq_length=self.max_length
                    )
                    pooling_model = models.Pooling(
                        word_embedding_model.get_word_embedding_dimension(),
                        pooling_mode=self.pooling_mode
                    )
                    self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            self.model.to(self.device)
            logging.info("TopSeg model loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading TopSeg model: {e}")
            return False
    
    def train_model(self, train_documents=None, val_documents=None):
        """
        Fine-tune the TopSeg model on the provided documents using coherence loss.
        
        Args:
            train_documents: List of training documents
            val_documents: List of validation documents
        """
        if not self.fine_tuning:
            logging.info("Fine-tuning disabled, skipping training")
            return
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model for training")
        
        logging.info("Starting TopSeg coherence fine-tuning...")
        
        # Load training data if not provided
        if train_documents is None:
            train_documents, val_documents = self._load_training_data()
        
        if not train_documents:
            logging.warning("No training documents provided, skipping training")
            return
        
        # Create training dataset for coherence learning
        train_dataset = TopSegCoherenceDataset(
            train_documents, 
            self.tokenizer, 
            self.max_length
        )
        
        if len(train_dataset) == 0:
            logging.warning("No training pairs created, skipping training")
            return
        
        # Create data loader
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Use the coherence loss from the TopSeg paper
        # This is a modified cosine similarity loss that treats same-segment pairs differently
        from sentence_transformers.losses import CosineSimilarityLoss
        
        # Use the standard CosineSimilarityLoss for coherence training
        # It expects labels: 1.0 for similar pairs, 0.0 for dissimilar pairs
        train_loss = losses.CosineSimilarityLoss(self.model)
        
        # Create validation evaluator if validation data is available
        evaluator = None
        if val_documents:
            val_dataset = TopSegCoherenceDataset(val_documents, self.tokenizer, self.max_length)
            if len(val_dataset) > 0:
                # Create evaluation pairs
                first_sentences = []
                second_sentences = []
                labels = []
                
                for example in val_dataset:
                    first_sentences.append(example.texts[0])
                    second_sentences.append(example.texts[1])
                    # Convert coherence labels to binary classification
                    labels.append(1 if example.label > 0 else 0)
                
                from sentence_transformers.evaluation import BinaryClassificationEvaluator
                evaluator = BinaryClassificationEvaluator(
                    first_sentences, 
                    second_sentences, 
                    labels,
                    name="TopSeg_Coherence_Validation"
                )
        
        # Training objectives
        train_objectives = [(train_dataloader, train_loss)]
        
        # Fine-tune the model
        self.model.fit(
            train_objectives=train_objectives,
            evaluator=evaluator,
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            show_progress_bar=True,
            output_path=self.save_model_path if self.save_model_path else None
        )
        
        logging.info("TopSeg fine-tuning completed")
        
        # Save the model if path is provided
        if self.save_model_path:
            self.model.save(self.save_model_path)
            logging.info(f"TopSeg model saved to {self.save_model_path}")
    
    def segment_text(self, meeting_text, sentences):
        """
        Segment text using the fine-tuned TopSeg model.
        
        Args:
            meeting_text (str): The full text to segment
            sentences (list): List of sentence dictionaries with text and spans
            
        Returns:
            tuple: (annotated_text, segment_boundaries)
        """
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model for segmentation")
        
        if len(sentences) <= 1:
            return meeting_text, []
        
        # Extract sentence texts - handle different sentence formats
        sentence_texts = []
        for sent in sentences:
            if hasattr(sent, 'text'):
                text = sent.text
            elif isinstance(sent, dict):
                text = sent.get('text', '')
            else:
                text = str(sent)
            sentence_texts.append(text)
        
        # Filter out empty sentences
        sentence_texts = [text for text in sentence_texts if text.strip()]
        
        if len(sentence_texts) <= 1:
            return meeting_text, []
        
        # Compute embeddings for all sentences
        try:
            embeddings = self.model.encode(
                sentence_texts, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
        except Exception as e:
            logging.error(f"Error computing embeddings: {e}")
            return meeting_text, []
        
        # Calculate coherence scores between adjacent sentences
        coherence_scores = []
        
        for i in range(len(embeddings) - 1):
            # Compute cosine similarity between adjacent sentence embeddings
            similarity = F.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[i + 1].unsqueeze(0)
            ).item()
            
            # Following the paper's approach: lower similarity indicates boundary
            # Convert similarity to boundary score (inverse relationship)
            boundary_score = 1.0 - similarity
            coherence_scores.append(boundary_score)
        
        # Apply threshold to determine boundaries
        boundaries = []
        for i, score in enumerate(coherence_scores):
            if score > self.threshold:
                boundaries.append(i + 1)  # Boundary after sentence i
        
        # Create segments based on boundaries (like NSP does)
        segments = []
        current_segment_sentences = [sentences[0]]  # Start with the first sentence
        
        for i in range(1, len(sentences)):
            if i in boundaries:
                # Boundary detected, finalize current segment
                segments.append(current_segment_sentences)
                # Start new segment
                current_segment_sentences = [sentences[i]]
            else:
                # No boundary, add to current segment
                current_segment_sentences.append(sentences[i])
        
        # Add the last segment
        if current_segment_sentences:
            segments.append(current_segment_sentences)
        
        logging.info(f"TopSeg segmentation found {len(segments)} segments with {len(boundaries)} boundaries")
        
        # Format output like NSP - create annotated text with segment markers
        segment_boundaries = []
        annotated_text = ""
        current_char_pos = 0
        seq_num = 1
        
        for segment_sentence_list in segments:
            start_char = segment_sentence_list[0]["start"]
            end_char = segment_sentence_list[-1]["end"]
            
            # Add text before this segment (if any)
            annotated_text += meeting_text[current_char_pos:start_char]
            
            # Store boundary position
            boundary_pos = len(annotated_text)
            
            # Add annotation start tag
            start_tag = f"##{seq_num}#"
            annotated_text += start_tag
            
            # Add segment content
            annotated_text += meeting_text[start_char:end_char]
            
            # Add annotation end tag
            end_tag = f"#{seq_num}##"
            annotated_text += end_tag
            
            # Store boundary information (format expected by evaluation)
            segment_boundaries.append({
                'segment_id': str(seq_num),
                'start': boundary_pos,
                'end': boundary_pos + len(start_tag) + (end_char - start_char) + len(end_tag),
                'text': meeting_text[start_char:end_char]
            })
            
            current_char_pos = end_char
            seq_num += 1
        
        # Add any remaining text after the last segment
        annotated_text += meeting_text[current_char_pos:]
        
        return annotated_text, segment_boundaries
    
    def _load_training_data(self):
        """Load training data using the internal dataset processor."""
        try:
            dataset_type = self.config.get('dataset_type', 'wikisection')
            subset = self.config.get('dataset_subset', None)
            split_type = self.config.get('split_type', None)
            min_segment_length = self.config.get('min_segment_length', 100)
            min_segments_per_document = self.config.get('min_segments_per_document', 2)
            
            # Create dataset processor
            processor_kwargs = {
                'min_segment_length': min_segment_length,
                'min_segments_per_document': min_segments_per_document
            }
            
            if split_type:
                processor_kwargs['split_type'] = split_type
            if subset:
                processor_kwargs['subset'] = subset
                
            processor = create_dataset_processor(
                dataset_type=dataset_type,
                **processor_kwargs
            )
            
            # Load and process data
            train_docs = processor.load_and_process_documents('train', max_docs=50)
            val_docs = processor.load_and_process_documents('val', max_docs=20) if hasattr(processor, 'load_and_process_documents') else []
            
            # If the processor doesn't support separate train/val splits, split manually
            if not val_docs and train_docs:
                split_idx = int(0.8 * len(train_docs))
                val_docs = train_docs[split_idx:]
                train_docs = train_docs[:split_idx]
            
            logging.info(f"Loaded {len(train_docs)} training documents and {len(val_docs)} validation documents")
            
            # Debug: print structure of first document
            if train_docs:
                first_doc = train_docs[0]
                logging.info(f"First document type: {type(first_doc)}")
                if hasattr(first_doc, '__dict__'):
                    logging.info(f"First document attributes: {list(first_doc.__dict__.keys())}")
                elif isinstance(first_doc, dict):
                    logging.info(f"First document keys: {list(first_doc.keys())}")
            
            return train_docs, val_docs
            
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return [], []


# Factory function to create TopSeg segmenter
def create_segmenter(**kwargs):
    """
    Factory function to create a TopSeg segmenter.
    
    Args:
        model_name (str): Name of the model to use
        fine_tuning (bool): Whether to enable fine-tuning
        train_size (int): Number of documents to use for training (optional)
        val_size (int): Number of documents to use for validation (optional)
        config (dict): Configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        TopSegSegmenter: An instance of the TopSeg segmentation algorithm
    """
    # Extract main parameters and remove them from kwargs to avoid duplicates
    model_name = kwargs.pop("model_name", "neuralmind/bert-base-portuguese-cased")
    fine_tuning = kwargs.pop("fine_tuning", True)
    
    # Remove ONLY random_seed and test_size (dataset processor specific)
    # Keep train_size and val_size as they're used by the algorithm's training logic
    kwargs.pop("random_seed", None)
    kwargs.pop("test_size", None)
    
    # Pass all remaining kwargs directly to the segmenter (includes train_size, val_size)
    return TopSegSegmenter(
        model_name=model_name,
        fine_tuning=fine_tuning,
        **kwargs
    )
