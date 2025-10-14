"""
Next Sentence Prediction (NSP) Topic Segmentation Algorithm

This module provides a class for segmenting text into topically coherent segments
using BERT's Next Sentence Prediction capability with optional fine-tuning support.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForNextSentencePrediction, get_linear_schedule_with_warmup
from torch.optim import AdamW 
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys

from .base import SegmentationAlgorithm
from .utils import split_text_into_sentences, merge_numbered_sentences, set_random_seeds

# Import dataset processor for internal data loading
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset_processors import create_dataset_processor

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
set_random_seeds()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in NSP training.
    Focuses learning on hard examples and reduces the relative loss for well-classified examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SegmentationAwareLoss(nn.Module):
    """
    Custom loss that combines multiple objectives for better segmentation performance:
    1. Focal loss for handling class imbalance
    2. Confidence penalty for overconfident wrong predictions
    3. Boundary coherence loss
    """
    def __init__(self, focal_alpha=1.0, focal_gamma=2.0, confidence_penalty=0.1, boundary_weight=0.2):
        super(SegmentationAwareLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.confidence_penalty = confidence_penalty
        self.boundary_weight = boundary_weight
    
    def forward(self, logits, labels, boundary_weights=None):
        # Basic focal loss
        focal_loss = self.focal_loss(logits, labels)
        
        # Confidence penalty: penalize overconfident wrong predictions
        probs = F.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        predictions = torch.argmax(logits, dim=1)
        wrong_predictions = (predictions != labels).float()
        
        # Penalty for being confident and wrong
        confidence_loss = (max_probs * wrong_predictions).mean()
        
        # Boundary-aware weighting
        if boundary_weights is not None:
            # Weight the loss based on proximity to actual boundaries
            weighted_focal = focal_loss * boundary_weights.mean()
        else:
            weighted_focal = focal_loss
        
        # Combine losses
        total_loss = weighted_focal + self.confidence_penalty * confidence_loss
        
        return total_loss, {
            'focal_loss': focal_loss.item(),
            'confidence_loss': confidence_loss.item(),
            'total_loss': total_loss.item()
        }


class NSPDataset(Dataset):
    """
    Dataset class for NSP fine-tuning with enhanced features for segmentation.
    """
    
    def __init__(self, sentence_pairs, labels, tokenizer, max_length=512, boundary_weights=None):
        """
        Initialize the NSP dataset.
        
        Args:
            sentence_pairs: List of (sentence_a, sentence_b) tuples
            labels: List of labels (0 = is_next, 1 = not_next)
            tokenizer: Tokenizer for encoding sentences
            max_length: Maximum sequence length
            boundary_weights: Optional weights for each pair based on boundary proximity
        """
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.boundary_weights = boundary_weights or [1.0] * len(sentence_pairs)
    
    def __len__(self):
        return len(self.sentence_pairs)
    
    def __getitem__(self, idx):
        sentence_a, sentence_b = self.sentence_pairs[idx]
        label = self.labels[idx]
        boundary_weight = self.boundary_weights[idx]
        
        # Tokenize the sentence pair
        encoded = self.tokenizer(
            sentence_a, 
            sentence_b,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'token_type_ids': encoded['token_type_ids'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long),
            'boundary_weight': torch.tensor(boundary_weight, dtype=torch.float)
        }


class NSPSegmenter(SegmentationAlgorithm):
    """
    Topic segmentation algorithm based on BERT's Next Sentence Prediction with fine-tuning support.
    """
    
    def __init__(
        self, 
        model_name="bert-base-uncased",
        fine_tuning=False,
        learning_rate=2e-5,
        batch_size=16,
        epochs=3,
        warmup_steps=100,
        max_length=512,
        threshold=0.5,
        save_model_path=None,
        config=None,
        merge_numbered_sentences=True,
        # Dataset configuration parameters
        dataset_type=None,
        dataset_subset=None,
        # New loss function parameters
        use_focal_loss=True,
        focal_alpha=1.0,
        focal_gamma=2.0,
        confidence_penalty=0.1,
        boundary_weight=0.2,
        # Early stopping parameters
        early_stopping_patience=3,
        min_improvement=0.001,
        **kwargs
    ):
        """
        Initialize the NSP segmenter with a specific BERT model.
        
        Args:
            model_name (str): Name of the pretrained BERT model to use
            fine_tuning (bool): Whether to enable fine-tuning mode
            learning_rate (float): Learning rate for fine-tuning
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            warmup_steps (int): Warmup steps for scheduler
            max_length (int): Maximum sequence length
            threshold (float): Threshold for boundary prediction
            save_model_path (str): Path to save fine-tuned model
            config (dict): Configuration dictionary for dataset loading
            merge_numbered_sentences (bool): Whether to merge numbered items (e.g., "2.") with the following sentence
            use_focal_loss (bool): Whether to use focal loss instead of standard cross-entropy
            focal_alpha (float): Alpha parameter for focal loss
            focal_gamma (float): Gamma parameter for focal loss
            confidence_penalty (float): Weight for confidence penalty in loss
            boundary_weight (float): Weight for boundary-aware loss component
            early_stopping_patience (int): Number of epochs to wait before early stopping
            min_improvement (float): Minimum improvement required to reset patience
            **kwargs: Additional keyword arguments
        """
        self.model_name = model_name
        self.fine_tuning = fine_tuning
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.max_length = max_length
        self.threshold = threshold
        self.save_model_path = save_model_path
        self.merge_numbered_sentences = merge_numbered_sentences
        
        # Loss function parameters
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.confidence_penalty = confidence_penalty
        self.boundary_weight = boundary_weight
        
        # Early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.min_improvement = min_improvement
        
        # Store dataset configuration
        self.dataset_type = dataset_type
        self.dataset_subset = dataset_subset
        
        # Store config for dataset loading, merge with kwargs
        self.config = config or {}
        self.config.update(kwargs)
        # Ensure dataset_type and dataset_subset are in config
        if dataset_type:
            self.config['dataset_type'] = dataset_type
        if dataset_subset:
            self.config['dataset_subset'] = dataset_subset
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.training_mode = fine_tuning  # Flag for the pipeline to know if training is needed
        
    def load_model(self):
        """
        Load the BERT model and tokenizer for Next Sentence Prediction.
        If fine-tuning is disabled, try to load a pre-trained model from models/ directory.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # If not fine-tuning, try to load a pre-trained model first
        if not self.fine_tuning:
            fine_tuned_model_path = self._get_fine_tuned_model_path()
            if fine_tuned_model_path and self.load_fine_tuned_model(fine_tuned_model_path):
                return True
        
        # Load base model if fine-tuning or if no fine-tuned model available
        logging.info(f"Loading BERT model '{self.model_name}' for NSP on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = BertForNextSentencePrediction.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            # Set model mode based on whether we're fine-tuning
            if self.fine_tuning:
                self.model.train()
                logging.info("Model set to training mode for fine-tuning")
            else:
                self.model.eval()
                logging.info("Model set to evaluation mode")
            
            logging.info(f"Successfully loaded NSP model")
            return True
        except Exception as e:
            logging.error(f"Error loading BERT model: {e}")
            return False
    
    def _create_training_data(self, documents):
        """
        Create training data from documents for NSP fine-tuning with boundary weights.
        
        Args:
            documents: List of document dictionaries with segments
            
        Returns:
            Tuple of (sentence_pairs, labels, boundary_weights)
        """
        sentence_pairs = []
        labels = []
        boundary_weights = []
        
        logging.info(f"Creating training data from {len(documents)} documents...")
        
        # Determine language from dataset type
        dataset_type = self.config.get('dataset_type', 'wikisection')
        language = 'portuguese' if dataset_type == 'councilseg' else 'english'
        
        for doc_idx, doc in enumerate(documents):
            if doc_idx % 100 == 0 and doc_idx > 0:
                logging.info(f"Processed {doc_idx}/{len(documents)} documents, created {len(sentence_pairs)} pairs so far")
            
            if 'segments' not in doc:
                continue
                
            segments = doc['segments']
            if len(segments) < 2:
                continue
            
            # Extract sentences from segments
            all_sentences = []
            segment_boundaries = []
            
            for segment in segments:
                segment_text = segment.get('text', '')
                # Use utility functions for sentence splitting
                sentences = split_text_into_sentences(segment_text, language=language)
                
                # Apply numbered sentence merging if enabled
                if self.merge_numbered_sentences:
                    sentences = merge_numbered_sentences(sentences)
                
                start_idx = len(all_sentences)
                all_sentences.extend(sentences)
                end_idx = len(all_sentences)
                segment_boundaries.append((start_idx, end_idx))
            
            # Create positive and negative pairs with boundary weights
            for i in range(len(all_sentences) - 1):
                sentence_a = all_sentences[i]
                sentence_b = all_sentences[i + 1]
                
                # Determine if sentence_b follows sentence_a logically
                is_consecutive = self._are_sentences_in_same_segment(i, i + 1, segment_boundaries)
                
                # Calculate boundary weight based on proximity to actual boundaries
                boundary_weight = self._calculate_boundary_weight(i, i + 1, segment_boundaries)
                
                if len(sentence_a.strip()) > 10 and len(sentence_b.strip()) > 10:
                    sentence_pairs.append((sentence_a, sentence_b))
                    labels.append(0 if is_consecutive else 1)  # 0 = is_next, 1 = not_next
                    boundary_weights.append(boundary_weight)
            
            # Add some random negative pairs for better training
            num_random_pairs = min(10, len(all_sentences) // 2)
            for _ in range(num_random_pairs):
                if len(all_sentences) >= 4:
                    i = torch.randint(0, len(all_sentences) - 2, (1,)).item()
                    j = torch.randint(i + 2, len(all_sentences), (1,)).item()
                    
                    sentence_a = all_sentences[i]
                    sentence_b = all_sentences[j]
                    
                    if len(sentence_a.strip()) > 10 and len(sentence_b.strip()) > 10:
                        sentence_pairs.append((sentence_a, sentence_b))
                        labels.append(1)  # not_next
                        # Random pairs get standard weight
                        boundary_weights.append(1.0)
        
        logging.info(f"Created {len(sentence_pairs)} training pairs from {len(documents)} documents")
        if len(labels) > 0:
            logging.info(f"Label distribution: {sum(labels)} not_next, {len(labels) - sum(labels)} is_next")
            logging.info(f"Average boundary weight: {sum(boundary_weights) / len(boundary_weights):.3f}")
        else:
            logging.warning("No training pairs created - documents may be too short or have insufficient segments")
        
        return sentence_pairs, labels, boundary_weights
    
    def _calculate_boundary_weight(self, sent_idx_a, sent_idx_b, segment_boundaries):
        """
        Calculate weight based on proximity to actual segment boundaries.
        Higher weights for pairs near actual boundaries to focus learning.
        
        Args:
            sent_idx_a: Index of first sentence
            sent_idx_b: Index of second sentence  
            segment_boundaries: List of (start, end) tuples for each segment
            
        Returns:
            float: Weight for this sentence pair
        """
        # Base weight
        weight = 1.0
        
        # Check if this is a true boundary (between segments)
        is_boundary = not self._are_sentences_in_same_segment(sent_idx_a, sent_idx_b, segment_boundaries)
        
        if is_boundary:
            # True boundaries get higher weight
            weight = 2.0
        else:
            # Check proximity to boundaries - closer pairs get slightly higher weight
            min_distance_to_boundary = float('inf')
            for start, end in segment_boundaries:
                # Distance to start of segment
                if start > 0:  # Not the first segment
                    distance_to_start = min(abs(sent_idx_a - start), abs(sent_idx_b - start))
                    min_distance_to_boundary = min(min_distance_to_boundary, distance_to_start)
                # Distance to end of segment  
                if end < max(max(boundary) for boundary in segment_boundaries):  # Not the last segment
                    distance_to_end = min(abs(sent_idx_a - (end - 1)), abs(sent_idx_b - (end - 1)))
                    min_distance_to_boundary = min(min_distance_to_boundary, distance_to_end)
            
            # Closer to boundary = slightly higher weight
            if min_distance_to_boundary < 3:
                weight = 1.5
            elif min_distance_to_boundary < 5:
                weight = 1.2
        
        return weight
    
    def _are_sentences_in_same_segment(self, sent_idx_a, sent_idx_b, segment_boundaries):
        """
        Check if two sentence indices are in the same segment.
        
        Args:
            sent_idx_a: Index of first sentence
            sent_idx_b: Index of second sentence
            segment_boundaries: List of (start, end) tuples for each segment
            
        Returns:
            bool: True if sentences are in the same segment
        """
        for start, end in segment_boundaries:
            if start <= sent_idx_a < end and start <= sent_idx_b < end:
                return True
        return False
    
    def train_model(self, train_documents=None, val_documents=None):
        """
        Fine-tune the NSP model on the provided documents.
        If no documents are provided, will load data using internal dataset processor.
        
        Args:
            train_documents: List of training documents (optional)
            val_documents: List of validation documents (optional)
        """
        if not self.fine_tuning:
            logging.info("Fine-tuning is disabled. Skipping training.")
            return
        
        # Load model if not already loaded
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("Failed to load model for training")
        
        logging.info("Starting NSP model fine-tuning...")
        
        # Load training data if not provided
        if train_documents is None or val_documents is None:
            logging.info("Loading training data using internal dataset processor...")
            loaded_train, loaded_val = self._load_training_data()
            train_documents = train_documents or loaded_train
            val_documents = val_documents or loaded_val
        
        # Create training data
        train_pairs, train_labels, train_weights = self._create_training_data(train_documents)
        if not train_pairs:
            logging.warning("No training data created. Skipping fine-tuning.")
            return
        
        # Create validation data if available
        val_pairs, val_labels, val_weights = [], [], []
        if val_documents:
            val_pairs, val_labels, val_weights = self._create_training_data(val_documents)
        
        # Create datasets
        train_dataset = NSPDataset(train_pairs, train_labels, self.tokenizer, self.max_length, train_weights)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_loader = None
        if val_pairs:
            val_dataset = NSPDataset(val_pairs, val_labels, self.tokenizer, self.max_length, val_weights)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup loss function
        if self.use_focal_loss:
            criterion = SegmentationAwareLoss(
                focal_alpha=self.focal_alpha,
                focal_gamma=self.focal_gamma,
                confidence_penalty=self.confidence_penalty,
                boundary_weight=self.boundary_weight
            )
            logging.info(f"Using SegmentationAwareLoss with focal_gamma={self.focal_gamma}, confidence_penalty={self.confidence_penalty}")
        else:
            criterion = nn.CrossEntropyLoss()
            logging.info("Using standard CrossEntropyLoss")
        
        # Training loop with early stopping
        self.model.train()
        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None
        
        # Track loss components
        loss_history = {'focal': [], 'confidence': [], 'total': []}
        
        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Training
            total_loss = 0
            epoch_loss_components = {'focal': 0, 'confidence': 0, 'total': 0}
            train_predictions, train_true_labels = [], []
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                boundary_weights = batch['boundary_weight'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                # Calculate loss using our custom loss function
                if self.use_focal_loss:
                    loss, loss_components = criterion(outputs.logits, labels, boundary_weights)
                    epoch_loss_components['focal'] += loss_components['focal_loss']
                    epoch_loss_components['confidence'] += loss_components['confidence_loss']
                    epoch_loss_components['total'] += loss_components['total_loss']
                else:
                    loss = criterion(outputs.logits, labels)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
                # Collect predictions for accuracy calculation
                predictions = torch.argmax(outputs.logits, dim=1)
                train_predictions.extend(predictions.cpu().numpy())
                train_true_labels.extend(labels.cpu().numpy())
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(train_true_labels, train_predictions)
            
            # Log detailed loss information
            if self.use_focal_loss:
                num_batches = len(train_loader)
                logging.info(f"Training Loss Components:")
                logging.info(f"  Focal Loss: {epoch_loss_components['focal'] / num_batches:.4f}")
                logging.info(f"  Confidence Loss: {epoch_loss_components['confidence'] / num_batches:.4f}")
                logging.info(f"  Total Loss: {epoch_loss_components['total'] / num_batches:.4f}")
                
                # Store loss history
                loss_history['focal'].append(epoch_loss_components['focal'] / num_batches)
                loss_history['confidence'].append(epoch_loss_components['confidence'] / num_batches)
                loss_history['total'].append(epoch_loss_components['total'] / num_batches)
            
            logging.info(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
            
            # Validation
            if val_loader:
                val_acc = self._evaluate_model(val_loader)
                logging.info(f"Validation Accuracy: {val_acc:.4f}")
                
                # Early stopping logic
                if val_acc > best_val_acc + self.min_improvement:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model state
                    best_model_state = self.model.state_dict().copy()
                    if self.save_model_path:
                        self._save_model()
                        logging.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
                else:
                    patience_counter += 1
                    logging.info(f"No improvement for {patience_counter} epochs (best: {best_val_acc:.4f})")
                
                # Early stopping
                if patience_counter >= self.early_stopping_patience:
                    logging.info(f"Early stopping after {epoch + 1} epochs")
                    # Restore best model
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                        logging.info(f"Restored best model with validation accuracy: {best_val_acc:.4f}")
                    break
        
        # Save final model if no validation or path specified
        if self.save_model_path and not val_loader:
            self._save_model()
            logging.info("Saved final fine-tuned model")
        
        # Log final loss summary
        if self.use_focal_loss and loss_history['total']:
            logging.info("Loss progression summary:")
            logging.info(f"  Initial total loss: {loss_history['total'][0]:.4f}")
            logging.info(f"  Final total loss: {loss_history['total'][-1]:.4f}")
            logging.info(f"  Improvement: {loss_history['total'][0] - loss_history['total'][-1]:.4f}")
        
        # Set back to evaluation mode
        self.model.eval()
        logging.info("Fine-tuning completed")
    
    def _evaluate_model(self, val_loader):
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            float: Validation accuracy
        """
        self.model.eval()
        val_predictions, val_true_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                predictions = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())
        
        self.model.train()
        return accuracy_score(val_true_labels, val_predictions)
    
    def _save_model(self):
        """Save the fine-tuned model."""
        if self.save_model_path:
            os.makedirs(os.path.dirname(self.save_model_path), exist_ok=True)
            self.model.save_pretrained(self.save_model_path)
            self.tokenizer.save_pretrained(self.save_model_path)
    
    def _get_fine_tuned_model_path(self):
        """
        Determine the path to a fine-tuned model based on configuration.
        
        Returns:
            str or None: Path to fine-tuned model if available, None otherwise
        """
        # Get dataset information from config and instance attributes
        dataset_type = self.config.get('dataset_type', '') or getattr(self, 'dataset_type', '') or ''
        dataset_subset = self.config.get('dataset_subset', '') or getattr(self, 'dataset_subset', '') or ''
        
        # Determine base model directory
        models_dir = os.path.join(os.path.dirname(current_dir), 'models')
        
        # Build model directory name based on dataset
        if dataset_type and dataset_subset:
            model_dir_name = f"nsp_{dataset_type}_{dataset_subset}"
        elif dataset_type:
            model_dir_name = f"nsp_{dataset_type}"
        else:
            logging.info("No dataset type specified for fine-tuned model lookup")
            return None
        
        model_path = os.path.join(models_dir, model_dir_name)
        
        # Check if model directory exists and contains required files
        if os.path.exists(model_path):
            required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
            if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                logging.info(f"Found fine-tuned model at: {model_path}")
                return model_path
            else:
                logging.warning(f"Model directory exists but missing required files: {model_path}")
        else:
            logging.info(f"Fine-tuned model directory not found: {model_path}")
        
        return None

    def load_fine_tuned_model(self, model_path):
        """
        Load a previously fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = BertForNextSentencePrediction.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            logging.info(f"Successfully loaded fine-tuned model from {model_path}")
            return True
        except Exception as e:
            logging.error(f"Error loading fine-tuned model: {e}")
            return False
    
    def predict_next_sentence(self, sentence_a, sentence_b):
        """
        Predict if sentence_b is the logical next sentence after sentence_a using BERT NSP.
        
        Args:
            sentence_a (str): The first sentence
            sentence_b (str): The second sentence
            
        Returns:
            bool: True if sentence_b is predicted to follow sentence_a, False otherwise
        """
        inputs = self.tokenizer(sentence_a, sentence_b, return_tensors='pt', truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [1, 2] -> [is_next_logit, not_next_logit]

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        is_next_prob = probs[0, 0].item()  # Probability of "is_next"
        not_next_prob = probs[0, 1].item()  # Probability of "not_next"
        
        # Use threshold to make decision
        # If not_next probability is above threshold, predict a segment boundary
        is_next = not_next_prob < self.threshold
        
        return is_next
    
    def segment_text(self, meeting_text, sentences):
        """
        Segment text into topically coherent segments using NSP.
        
        Args:
            meeting_text (str): The full text to segment
            sentences (list): List of sentence dictionaries with text and spans
            
        Returns:
            tuple: (annotated_text, segment_boundaries)
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                logging.error("Failed to load NSP model. Cannot perform segmentation.")
                return meeting_text, []
        
        if not sentences:
            return "", []

        segments = []
        current_segment_sentences = [sentences[0]]  # Start with the first sentence

        logging.info(f"Processing {len(sentences)} sentences...")
        for i in range(1, len(sentences)):
            prev_sentence = sentences[i-1]["text"]
            curr_sentence = sentences[i]["text"]

            # Predict if current sentence follows the previous one logically
            is_next = self.predict_next_sentence(prev_sentence, curr_sentence)

            if not is_next:
                # NSP predicts a break, so finalize the current segment
                segments.append(current_segment_sentences)
                # Start a new segment
                current_segment_sentences = [sentences[i]]
            else:
                # No break, add sentence to the current segment
                current_segment_sentences.append(sentences[i])

        # Add the last segment
        if current_segment_sentences:
            segments.append(current_segment_sentences)

        logging.info(f"Generated {len(segments)} segments.")

        # Reconstruct annotated text
        segment_boundaries = []  # Store boundary objects for evaluation
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

            # Store boundary information
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
        """
        Load training and validation data using dataset processor with train/val splits.
        
        Returns:
            Tuple of (training_documents, validation_documents)
        """
        try:
            # Get dataset configuration from the config
            dataset_type = self.config.get('dataset_type', 'wikisection')
            dataset_subset = self.config.get('dataset_subset', None)
            
            # Set up dataset path based on dataset type
            dataset_path = os.path.join(os.path.dirname(current_dir), '..', '..', 'data', f'{dataset_type}_dataset')
            
            # Get train/val sizes from config
            train_size = self.config.get('train_size', None)
            val_size = self.config.get('val_size', None)
            
            logging.info(f"Loading NSP training data with train_size={train_size}, val_size={val_size}")
            
            # Initialize the processor with train/val sizes
            processor_kwargs = {
                'dataset_path': dataset_path,
                'min_segment_length': self.config.get('min_segment_length', 50),
                'min_segments_per_document': self.config.get('min_segments_per_document', 2),
                'train_size': train_size,
                'val_size': val_size,
                'random_seed': self.config.get('random_seed', 42)
            }
            
            # Add subset if provided
            if dataset_subset:
                processor_kwargs['subset'] = dataset_subset
            
            processor = create_dataset_processor(dataset_type, **processor_kwargs)
            
            # Load the dataset
            dataset = processor.load_dataset()
            if not dataset:
                logging.warning(f"Failed to load {dataset_type} dataset")
                return [], []

            # Check if processor supports get_documents method (newer approach)
            if hasattr(processor, 'get_documents'):
                # Get documents with proper splits - these are already processed
                training_documents = processor.get_documents(split='train')
                validation_documents = processor.get_documents(split='val')
                
                # Limit documents if sizes are specified (as an additional safety measure)
                if train_size is not None and len(training_documents) > train_size:
                    training_documents = training_documents[:train_size]
                if val_size is not None and len(validation_documents) > val_size:
                    validation_documents = validation_documents[:val_size]
                
            else:
                # Fallback approach for processors without get_documents method
                logging.info(f"Dataset processor doesn't support get_documents, using fallback approach")
                
                # Process all documents first
                all_documents = processor.process_dataset(dataset)
                
                # Manually split the documents
                from sklearn.model_selection import train_test_split
                
                if len(all_documents) < 3:
                    # Too few documents, use all for training
                    training_documents = all_documents
                    validation_documents = []
                    logging.warning(f"Only {len(all_documents)} documents available, using all for training")
                else:
                    # Calculate split sizes
                    total_docs = len(all_documents)
                    if train_size is not None and val_size is not None:
                        # Use specified sizes
                        actual_train_size = min(train_size, total_docs - 1)
                        actual_val_size = min(val_size, total_docs - actual_train_size)
                        
                        # Take first train_size documents for training, next val_size for validation
                        training_documents = all_documents[:actual_train_size]
                        validation_documents = all_documents[actual_train_size:actual_train_size + actual_val_size]
                    else:
                        # Use proportional split (80/20)
                        train_ratio = 0.8 if train_size is None else min(train_size / total_docs, 0.9)
                        training_documents, validation_documents = train_test_split(
                            all_documents, 
                            train_size=train_ratio,
                            random_state=self.config.get('random_seed', 42)
                        )
            
            logging.info(f"Loaded {len(training_documents)} training documents and {len(validation_documents)} validation documents for NSP")
            
            return training_documents, validation_documents
            
        except Exception as e:
            logging.warning(f"Error loading training data: {e}")
            return [], []


# Factory function to easily create an instance
def create_segmenter(**kwargs):
    """
    Factory function to create an NSP segmenter.
    
    Args:
        model_name (str): Name of the BERT model to use
        fine_tuning (bool): Whether to enable fine-tuning
        merge_numbered_sentences (bool): Whether to merge numbered items with following sentences
        train_size (int): Number of documents to use for training (optional)
        val_size (int): Number of documents to use for validation (optional)
        config (dict): Configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        NSPSegmenter: An instance of the NSP segmentation algorithm
    """
    # Extract main parameters and remove them from kwargs to avoid duplicates
    model_name = kwargs.pop("model_name", "bert-base-uncased")
    fine_tuning = kwargs.pop("fine_tuning", False)
    merge_numbered_sentences = kwargs.pop("merge_numbered_sentences", True)
    
    # Remove ONLY random_seed and test_size (dataset processor specific)
    # Keep train_size and val_size as they're used by the algorithm's training logic
    kwargs.pop("random_seed", None)
    kwargs.pop("test_size", None)
    
    # Pass all remaining kwargs directly to the segmenter (includes train_size, val_size)
    return NSPSegmenter(
        model_name=model_name,
        fine_tuning=fine_tuning,
        merge_numbered_sentences=merge_numbered_sentences,
        **kwargs
    )
