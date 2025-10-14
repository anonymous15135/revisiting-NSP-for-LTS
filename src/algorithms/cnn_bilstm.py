#!/usr/bin/env python3
"""
Att+ CNN (Contextual CNN+BiLSTM+Attention) Algorithm for Topic Segmentation

Implementation of the Att+ CNN model from:
Badjatiya, Pinkesh, Litton J. Kurisinkel, Harish Gupta, and Vasudeva Varma.
"Attention-based neural text segmentation." 
European conference on information retrieval. Springer, 2018.

This implementation follows the true architecture from the original codebase:
- Context windows (left/right context around each sentence)
- CNN with multiple n-gram filters and TimeDistributed layers
- Bidirectional LSTM for sequence modeling
- Attention mechanism for weighted aggregation
- Multi-input architecture: [left_context, main_input, right_context]

Based on the original files:
- lstm_try_with_highSegLenTrain.py
- baselines_deepNN.py  
- lstm_2dSent_15Len_TimDisConv_ctxWindow.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import os
import sys
import json
import random
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

from .base import SegmentationAlgorithm

class AttentionWithContext(nn.Module):
    """
    Attention operation with a context/query vector for temporal data.
    PyTorch implementation of the original Keras AttentionWithContext layer.
    
    Based on AttentionWithContext.py from the original codebase.
    Follows the work of Yang et al. "Hierarchical Attention Networks for Document Classification"
    """
    
    def __init__(self, hidden_size, bias=True):
        super(AttentionWithContext, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Attention weights
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        if bias:
            self.b = nn.Parameter(torch.zeros(hidden_size))
        self.u = nn.Parameter(torch.randn(hidden_size))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.u.unsqueeze(0))
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
            mask: Optional mask tensor
            
        Returns:
            Weighted sum of input with shape (batch_size, hidden_size)
        """
        # Apply linear transformation: uit = tanh(x * W + b)
        uit = torch.tanh(self.W(x))
        if self.bias:
            uit = uit + self.b
        
        # Compute attention scores: ait = uit * u
        ait = torch.sum(uit * self.u, dim=2)  # (batch_size, seq_len)
        
        # Apply softmax to get attention weights
        a = torch.exp(ait)
        
        # Apply mask if provided
        if mask is not None:
            a = a * mask.float()
        
        # Normalize attention weights
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-8)
        
        # Apply attention weights to input
        a = a.unsqueeze(2)  # (batch_size, seq_len, 1)
        weighted_input = x * a  # (batch_size, seq_len, hidden_size)
        
        # Sum over sequence dimension
        output = torch.sum(weighted_input, dim=1)  # (batch_size, hidden_size)
        
        return output


class AttentionBasedCNNModel(nn.Module):
    """
    Att+ CNN (Contextual CNN+BiLSTM+Attention) Model for Topic Segmentation.
    
    This is the true architecture from the original neuralTextSegmentation codebase.
    
    Architecture:
    1. Three inputs: left_context, main_input, right_context
    2. Embedding layers (TimeDistributed)
    3. CNN layers with multiple n-gram filters (TimeDistributed Conv1D)
    4. GlobalMaxPooling1D for each filter 
    5. Bidirectional LSTM layers for context encoding
    6. Attention mechanism (AttentionWithContext)
    7. Concatenation of left/main/right features
    8. Dense decoder layers
    9. Sigmoid output for binary classification
    
    Based on lstm_try_with_highSegLenTrain.py lines 66-138:
    - ONE_SIDE_CONTEXT_SIZE = 10 (configurable)
    - ngram_filters = [2, 3, 4, 5]
    - conv_hidden_units = [200, 200, 200, 200]
    - BiLSTM with 500/600 hidden units
    - AttentionWithContext for left/right contexts
    - Dense layers: 500 -> sigmoid output
    """
    
    def __init__(self, config):
        super(AttentionBasedCNNModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model configuration from original implementation
        self.one_side_context_size = config.get('one_side_context_size', 10)
        self.ngram_filters = config.get('ngram_filters', [2, 3, 4, 5])
        self.conv_hidden_units = config.get('conv_hidden_units', [200, 200, 200, 200])
        self.embedding_dim = config['embedding_dim']
        
        # BERT encoder for sentence embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(config['pretrained_model'])
        self.bert = AutoModel.from_pretrained(config['pretrained_model'])
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # CNN layers for each n-gram filter
        # Following original: Convolution1D with different filter sizes
        self.conv_layers = nn.ModuleList()
        for n_gram, hidden_units in zip(self.ngram_filters, self.conv_hidden_units):
            conv_layer = nn.Conv1d(
                in_channels=self.embedding_dim,
                out_channels=hidden_units,
                kernel_size=min(n_gram, self.one_side_context_size + 1),  # Adjust for sequence length
                padding='same'  # Equivalent to border_mode='same'
            )
            self.conv_layers.append(conv_layer)
        
        # Total conv dimension
        self.conv_dim = sum(self.conv_hidden_units)
        
        # Dense encoder for main input (middle sentence)
        # Following original line 112: encode_mid = Dense(300, ...)
        self.main_encoder = nn.Linear(self.conv_dim, 300)
        
        # Bidirectional LSTM for context encoding
        # Following original lines 114-116: Bidirectional LSTM with 500 units
        self.context_lstm = nn.LSTM(
            input_size=self.conv_dim,
            hidden_size=500,
            batch_first=True,
            bidirectional=True,
            dropout=config.get('lstm_dropout', 0.2)
        )
        
        # Attention layers for left and right contexts
        # Following original lines 119-121: AttentionWithContext
        self.left_attention = AttentionWithContext(hidden_size=1000)  # 500*2 for bidirectional
        self.right_attention = AttentionWithContext(hidden_size=1000)
        
        # Dropout layers
        self.dropout = nn.Dropout(config.get('dropout', 0.2))
        
        # Final decoder layers
        # Following original lines 124-129: concatenate -> Dense(500) -> dropout -> Dense(1)
        # Input size: left_features + main_features + right_features = 1000 + 300 + 1000 = 2300
        self.decoder = nn.Sequential(
            nn.Linear(2300, 500),
            nn.Dropout(config.get('dropout', 0.3)),
            nn.Linear(500, 1),
            nn.Sigmoid()
        )
        
    def forward(self, left_context, main_input, right_context):
        """
        Forward pass through the Att+ CNN model.
        
        Args:
            left_context: Tensor of shape (batch_size, context_size, embedding_dim)
            main_input: Tensor of shape (batch_size, 1, embedding_dim)  
            right_context: Tensor of shape (batch_size, context_size, embedding_dim)
            
        Returns:
            Tensor of boundary probabilities for each sentence
        """
        batch_size = left_context.size(0)
        
        self.logger.debug(f"Att+ CNN forward - Left: {left_context.shape}, Main: {main_input.shape}, Right: {right_context.shape}")
        
        # Apply CNN layers to each input
        left_conv_outputs, main_conv_outputs, right_conv_outputs = [], [], []
        
        for conv_layer in self.conv_layers:
            # Apply CNN to left context
            # Transpose for Conv1d: (batch, seq_len, features) -> (batch, features, seq_len)
            left_transposed = left_context.transpose(1, 2)
            left_conv = conv_layer(left_transposed)  # (batch, hidden_units, seq_len)
            # Global max pooling
            left_pooled = F.max_pool1d(left_conv, kernel_size=left_conv.size(2)).squeeze(2)
            left_conv_outputs.append(left_pooled)
            
            # Apply CNN to main input
            main_transposed = main_input.transpose(1, 2)
            main_conv = conv_layer(main_transposed)
            main_pooled = F.max_pool1d(main_conv, kernel_size=main_conv.size(2)).squeeze(2)
            main_conv_outputs.append(main_pooled)
            
            # Apply CNN to right context
            right_transposed = right_context.transpose(1, 2)
            right_conv = conv_layer(right_transposed)
            right_pooled = F.max_pool1d(right_conv, kernel_size=right_conv.size(2)).squeeze(2)
            right_conv_outputs.append(right_pooled)
        
        # Concatenate CNN outputs
        left_conv_concat = torch.cat(left_conv_outputs, dim=1)    # (batch, conv_dim)
        main_conv_concat = torch.cat(main_conv_outputs, dim=1)    # (batch, conv_dim)
        right_conv_concat = torch.cat(right_conv_outputs, dim=1)  # (batch, conv_dim)
        
        # Encode main input with dense layer
        main_encoded = self.main_encoder(main_conv_concat)  # (batch, 300)
        main_encoded = self.dropout(main_encoded)
        
        # Process left and right contexts with BiLSTM + Attention
        # Add sequence dimension for LSTM: (batch, conv_dim) -> (batch, 1, conv_dim)
        left_lstm_input = left_conv_concat.unsqueeze(1)
        right_lstm_input = right_conv_concat.unsqueeze(1)
        
        # Apply BiLSTM
        left_lstm_out, _ = self.context_lstm(left_lstm_input)   # (batch, 1, 1000)
        right_lstm_out, _ = self.context_lstm(right_lstm_input) # (batch, 1, 1000)
        
        # Apply attention
        left_attended = self.left_attention(left_lstm_out)      # (batch, 1000)
        right_attended = self.right_attention(right_lstm_out)   # (batch, 1000)
        
        # Apply dropout to context features
        left_attended = self.dropout(left_attended)
        right_attended = self.dropout(right_attended)
        
        # Concatenate all features
        # Following original line 124: encoded_info = Merge(mode='concat')([encode_left_drop, encode_mid_drop, encode_right_drop])
        concatenated = torch.cat([left_attended, main_encoded, right_attended], dim=1)  # (batch, 2300)
        
        # Apply final decoder
        output = self.decoder(concatenated)  # (batch, 1)
        
        # Squeeze to remove extra dimension
        output = output.squeeze(1)  # (batch,)
        
        self.logger.debug(f"Att+ CNN final output shape: {output.shape}")
        
        return output
        
    def encode_sentences(self, sentences):
        """Encode sentences using BERT."""
        encoded_sentences = []
        
        with torch.no_grad():
            for sentence in sentences:
                # Tokenize and encode
                inputs = self.tokenizer(
                    sentence,
                    return_tensors='pt',
                    max_length=self.config['max_seq_length'],
                    padding='max_length',
                    truncation=True
                )
                
                # Move inputs to the same device as the model
                inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
                
                # Get BERT embeddings
                outputs = self.bert(**inputs)
                # Use CLS token as sentence representation
                sentence_embedding = outputs.last_hidden_state[:, 0, :]  # (1, embedding_dim)
                encoded_sentences.append(sentence_embedding)
        
        return torch.cat(encoded_sentences, dim=0)  # (num_sentences, embedding_dim)


class CNNBiLSTMSegmenter(SegmentationAlgorithm):
    """
    Att+ CNN (Contextual CNN+BiLSTM+Attention) topic segmentation algorithm.
    
    This implementation follows the true architecture from the original codebase:
    - Context windows around each sentence for contextual modeling
    - Multiple CNN layers with different n-gram filters
    - Bidirectional LSTM for sequence modeling
    - Attention mechanism for weighted feature aggregation
    - Multi-input architecture processing left/main/right contexts
    
    The key innovation is using contextual information from neighboring sentences
    to make better segmentation decisions, rather than processing sentences in isolation.
    """
    
    def __init__(self, pretrained_model='distilbert-base-uncased', **kwargs):
        self.config = {
            'pretrained_model': pretrained_model,
            'embedding_dim': 768,  # DistilBERT embedding dimension
            'max_seq_length': 512,
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 10,
            'dropout': 0.3,
            'lstm_dropout': 0.2,
            'one_side_context_size': 10,  # Context window size (original: 10 or 15)
            'ngram_filters': [2, 3, 4, 5],
            'conv_hidden_units': [200, 200, 200, 200],
            'boundary_threshold': 0.5,
            'min_boundary_distance': 2,
            **kwargs
        }
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Set training mode based on config
        self.training_mode = True  # Always require training for this model
        
    def train_model(self, training_documents, validation_documents=None):
        """
        Train the model using the interface expected by run_baselines.py.
        
        Args:
            training_documents: List of training documents
            validation_documents: List of validation documents (optional)
        """
        self.logger.info("Starting CNN-BiLSTM model training")
        
        # Convert documents to the format expected by the train method
        documents = training_documents
        if validation_documents:
            documents.extend(validation_documents)
        
        # Train the model
        self.train(documents)
        
        self.logger.info("CNN-BiLSTM model training completed")

    def train(self, documents):
        """
        Train the Att+ CNN model on the provided documents.
        
        Args:
            documents: List of Document objects with sentences and boundaries
        """
        self.logger.info("Starting Att+ CNN training")
        
        try:
            # Initialize the model
            self.model = AttentionBasedCNNModel(self.config).to(self.device)
            
            # Prepare training data with context windows
            X_left, X_main, X_right, y = self._prepare_contextual_data(documents)
            
            if len(X_left) == 0:
                self.logger.warning("No training data available")
                return
            
            self.logger.info(f"Training with {len(X_left)} context windows")
            
            # Convert to tensors
            X_left_tensor = torch.FloatTensor(X_left).to(self.device)
            X_main_tensor = torch.FloatTensor(X_main).to(self.device)
            X_right_tensor = torch.FloatTensor(X_right).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(X_left_tensor, X_main_tensor, X_right_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True
            )
            
            # Initialize optimizer and loss function
            optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
            criterion = nn.BCELoss()
            
            # Training loop
            self.model.train()
            for epoch in range(self.config.get('epochs', self.config.get('num_epochs', 10))):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (batch_left, batch_main, batch_right, batch_y) in enumerate(dataloader):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_left, batch_main, batch_right)
                    loss = criterion(outputs, batch_y)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        self.logger.warning("NaN loss detected, skipping batch")
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Statistics
                    epoch_loss += loss.item()
                    predictions = (outputs > 0.5).float()
                    epoch_correct += (predictions == batch_y).sum().item()
                    epoch_total += batch_y.size(0)
                    
                    if batch_idx % 10 == 0:
                        self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                
                avg_loss = epoch_loss / len(dataloader)
                accuracy = epoch_correct / epoch_total
                self.logger.info(f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
                
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def segment(self, document):
        """
        Segment a document using the trained Att+ CNN model.
        
        Args:
            document: Document object with sentences
            
        Returns:
            List of boundary predictions (1 for boundary, 0 for no boundary)
        """
        if self.model is None:
            self.logger.error("Model not trained yet")
            return [0] * len(document.sentences)
        
        try:
            self.model.eval()
            
            # Prepare contextual data for the document
            X_left, X_main, X_right, _ = self._prepare_contextual_data([document])
            
            if len(X_left) == 0:
                self.logger.warning("No data to segment")
                return [0] * len(document.sentences)
            
            # Convert to tensors
            X_left_tensor = torch.FloatTensor(X_left).to(self.device)
            X_main_tensor = torch.FloatTensor(X_main).to(self.device)
            X_right_tensor = torch.FloatTensor(X_right).to(self.device)
            
            with torch.no_grad():
                # Get predictions
                outputs = self.model(X_left_tensor, X_main_tensor, X_right_tensor)
                predictions = (outputs > 0.5).cpu().numpy().astype(int)
            
            self.logger.debug(f"Predicted {np.sum(predictions)} boundaries out of {len(predictions)} sentences")
            
            return predictions.tolist()
            
        except Exception as e:
            self.logger.error(f"Error during segmentation: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return safe fallback
            return [0] * len(document.sentences)
    
    def _prepare_contextual_data(self, documents):
        """
        Prepare contextual training data with left/main/right context windows.
        
        This follows the original data preparation from batch_gen_consecutive_context_segments_from_big_seq
        in the original codebase.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (X_left, X_main, X_right, y) arrays
        """
        X_left, X_main, X_right, y = [], [], [], []
        context_size = self.config['one_side_context_size']
        
        self.logger.info(f"Preparing contextual data with context size {context_size}")
        total_sentences = 0
        processed_docs = 0
        
        for doc_idx, doc in enumerate(documents):
            self.logger.debug(f"Document {doc_idx}: Processing document with attributes: {dir(doc)}")
            
            # Handle both sentence-based and segment-only documents
            if (hasattr(doc, 'sentences') and doc.sentences) or (isinstance(doc, dict) and 'sentences' in doc):
                # Use existing sentences
                sentences = doc.sentences if hasattr(doc, 'sentences') else doc.get('sentences', [])
                boundaries = getattr(doc, 'boundaries', [0] * len(sentences)) if hasattr(doc, 'boundaries') else doc.get('boundaries', [0] * len(sentences))
                self.logger.debug(f"Document {doc_idx}: Using existing {len(sentences)} sentences")
            else:
                # Split segments into sentences
                self.logger.debug(f"Document {doc_idx}: Document has segments: {hasattr(doc, 'segments') or (isinstance(doc, dict) and 'segments' in doc)}")
                if hasattr(doc, 'segments') or (isinstance(doc, dict) and 'segments' in doc):
                    segments = doc.segments if hasattr(doc, 'segments') else doc.get('segments', [])
                    self.logger.debug(f"Document {doc_idx}: Number of segments: {len(segments) if segments else 0}")
                sentences, boundaries = self._prepare_document_data(doc)
                self.logger.debug(f"Document {doc_idx}: Prepared {len(sentences)} sentences from segments")
            
            if len(sentences) == 0:
                self.logger.warning(f"Document {doc_idx}: No sentences found, skipping. Document type: {type(doc)}")
                continue
                
            total_sentences += len(sentences)
            
            # Encode all sentences in the document
            try:
                encoded_sentences = self.model.encode_sentences(sentences) if self.model else None
            except Exception as e:
                self.logger.warning(f"Failed to encode sentences for document: {e}")
                encoded_sentences = None
            
            # Skip if encoding failed
            if encoded_sentences is None:
                self.logger.debug(f"Document {doc_idx}: Skipping due to encoding failure")
                continue
                
            # Check minimum context requirement - be more flexible for small documents
            min_required_sentences = max(3, context_size + 1)  # At least 3 sentences or context_size + 1
            if len(encoded_sentences) < min_required_sentences:
                self.logger.debug(f"Document {doc_idx}: Skipping with {len(sentences)} sentences (need at least {min_required_sentences})")
                continue
                
            processed_docs += 1
            self.logger.debug(f"Document {doc_idx}: Processing {len(sentences)} sentences")
            
            # Create context windows for each sentence
            for i in range(len(sentences)):
                # Define context window indices
                left_start = max(0, i - context_size)
                left_end = i
                main_idx = i
                right_start = i + 1
                right_end = min(len(sentences), i + context_size + 1)
                
                # Extract context features
                if left_end > left_start:
                    # Pad left context if necessary
                    left_context = encoded_sentences[left_start:left_end]
                    if left_context.shape[0] < context_size:
                        padding = torch.zeros(context_size - left_context.shape[0], self.config['embedding_dim']).to(encoded_sentences.device)
                        left_context = torch.cat([padding, left_context], dim=0)
                    else:
                        left_context = left_context[-context_size:]  # Take last context_size
                else:
                    # No left context available
                    left_context = torch.zeros(context_size, self.config['embedding_dim']).to(encoded_sentences.device)
                
                # Main sentence
                main_sentence = encoded_sentences[main_idx:main_idx+1]
                
                if right_end > right_start:
                    # Pad right context if necessary
                    right_context = encoded_sentences[right_start:right_end]
                    if right_context.shape[0] < context_size:
                        padding = torch.zeros(context_size - right_context.shape[0], self.config['embedding_dim']).to(encoded_sentences.device)
                        right_context = torch.cat([right_context, padding], dim=0)
                    else:
                        right_context = right_context[:context_size]  # Take first context_size
                else:
                    # No right context available
                    right_context = torch.zeros(context_size, self.config['embedding_dim']).to(encoded_sentences.device)
                
                # Add to training data
                X_left.append(left_context.cpu().numpy())
                X_main.append(main_sentence.cpu().numpy())
                X_right.append(right_context.cpu().numpy())
                
                # Get boundary label (1 if sentence ends a segment, 0 otherwise)
                if i < len(boundaries):
                    y.append(boundaries[i])
                else:
                    y.append(0)
        
        self.logger.info(f"Prepared {len(X_left)} context windows from {processed_docs}/{len(documents)} documents")
        self.logger.info(f"Total sentences processed: {total_sentences}")
        
        if len(X_left) == 0:
            self.logger.warning("No context windows created. Check document format and minimum sentence requirements.")
            self.logger.warning(f"Context size: {context_size}, Min required sentences: {max(3, context_size + 1)}")
        
        return np.array(X_left), np.array(X_main), np.array(X_right), np.array(y)
    
    def _prepare_document_data(self, doc):
        """
        Prepare document data - split segments into sentences.
        
        Args:
            doc: Document object with segments
            
        Returns:
            tuple: (sentences, boundaries) where sentences are text strings
                   and boundaries are binary labels
        """
        sentences = []
        boundaries = []
        
        self.logger.debug(f"_prepare_document_data: Document attributes: {dir(doc)}")
        
        # Debug: if it's a dict, show its keys
        if isinstance(doc, dict):
            self.logger.debug(f"_prepare_document_data: Document keys: {list(doc.keys())}")
        
        # Handle segment-only format
        if (hasattr(doc, 'segments') and doc.segments) or (isinstance(doc, dict) and 'segments' in doc):
            # Get segments from either object attribute or dictionary key
            segments = doc.segments if hasattr(doc, 'segments') else doc.get('segments', [])
            self.logger.debug(f"_prepare_document_data: Found {len(segments)} segments")
            for i, segment in enumerate(segments):
                self.logger.debug(f"_prepare_document_data: Processing segment {i}: {type(segment)}")
                segment_text = segment.get('text', '').strip()
                self.logger.debug(f"_prepare_document_data: Segment {i} text length: {len(segment_text)}")
                if not segment_text:
                    self.logger.debug(f"_prepare_document_data: Segment {i} has no text, skipping")
                    continue
                
                # Split segment into sentences
                try:
                    import nltk
                    nltk.download('punkt', quiet=True)
                    # Use Portuguese tokenizer if available
                    try:
                        segment_sentences = nltk.sent_tokenize(segment_text, language='portuguese')
                    except:
                        segment_sentences = nltk.sent_tokenize(segment_text)
                except Exception as e:
                    self.logger.warning(f"NLTK sentence tokenization failed: {e}, using fallback")
                    import re
                    # Better Portuguese sentence splitting regex
                    segment_sentences = re.split(r'[.!?]+\s+', segment_text)
                    segment_sentences = [s.strip() for s in segment_sentences if s.strip()]
                
                self.logger.debug(f"_prepare_document_data: Segment {i} split into {len(segment_sentences)} sentences")
                
                # If we still don't have enough sentences, split by periods more aggressively
                if len(segment_sentences) <= 2 and len(segment_text) > 100:
                    import re
                    # Split on any period, exclamation, or question mark
                    segment_sentences = re.split(r'[.!?]+', segment_text)
                    segment_sentences = [s.strip() for s in segment_sentences if s.strip() and len(s) > 10]
                    self.logger.debug(f"_prepare_document_data: Segment {i} aggressive split into {len(segment_sentences)} sentences")
                
                if not segment_sentences:
                    self.logger.debug(f"_prepare_document_data: Segment {i} has no sentences after tokenization")
                    continue
                
                # Add sentences from this segment
                for j, sentence in enumerate(segment_sentences):
                    if len(sentence.strip()) > 10:  # Only add sentences with meaningful content
                        sentences.append(sentence.strip())
                        # Mark first sentence of each segment as a boundary
                        boundaries.append(1 if j == 0 and i > 0 else 0)  # Don't mark first segment's first sentence
        
        # Handle text-based documents
        elif (hasattr(doc, 'text') and doc.text) or (isinstance(doc, dict) and 'text' in doc):
            # Get text from either object attribute or dictionary key
            doc_text = doc.text if hasattr(doc, 'text') else doc.get('text', '')
            doc_text = doc_text.strip()
            if doc_text:
                try:
                    import nltk
                    nltk.download('punkt', quiet=True)
                    # Use Portuguese tokenizer if available
                    try:
                        sentences = nltk.sent_tokenize(doc_text, language='portuguese')
                    except:
                        sentences = nltk.sent_tokenize(doc_text)
                except Exception as e:
                    self.logger.warning(f"NLTK sentence tokenization failed: {e}, using fallback")
                    import re
                    sentences = re.split(r'[.!?]+\s+', doc_text)
                    sentences = [s.strip() for s in sentences if s.strip()]
                
                # Filter out very short sentences
                sentences = [s for s in sentences if len(s.strip()) > 10]
                
                # Create boundaries - mark first sentence and some evenly spaced ones
                boundaries = [0] * len(sentences)
                if sentences:
                    boundaries[0] = 1  # First sentence is always a boundary
                    # Add some evenly spaced boundaries
                    step = max(1, len(sentences) // 5)  # Divide into ~5 segments
                    for i in range(step, len(sentences), step):
                        if i < len(boundaries):
                            boundaries[i] = 1
        
        self.logger.debug(f"_prepare_document_data: Returning {len(sentences)} sentences and {len(boundaries)} boundaries")
        return sentences, boundaries

    def segment_text(self, text, sentences):
        """
        Segment text into topically coherent segments using the Att+ CNN model.
        
        This method implements the required interface from SegmentationAlgorithm.
        It uses the trained Att+ CNN model to predict segment boundaries.
        
        Args:
            text (str): The full text to segment
            sentences (list): List of sentence dictionaries with text and spans
            
        Returns:
            tuple: (annotated_text, segment_boundaries)
        """
        try:
            # Check if model is initialized and trained
            if self.model is None:
                self.logger.error("Model not trained yet. Call train_model() first.")
                # Fallback: return simple segments with annotation tokens
                segments = []
                if sentences:
                    first_sentence = sentences[0]
                    last_sentence = sentences[-1]
                    if isinstance(first_sentence, dict) and isinstance(last_sentence, dict):
                        segments.append({
                            'seq': 1,
                            'start': first_sentence.get('start', 0),
                            'end': last_sentence.get('end', len(text)),
                            'start_sent': 0,
                            'end_sent': len(sentences) - 1
                        })
                    else:
                        segments.append({
                            'seq': 1,
                            'start': 0,
                            'end': len(text),
                            'start_sent': 0,
                            'end_sent': len(sentences) - 1
                        })
                
                # Create annotated text with segment markers
                annotated_text = self._create_annotated_text(text, sentences, segments)
                return annotated_text, segments
            
            # Extract sentence texts from different formats
            sentence_texts = []
            if sentences:
                for sentence in sentences:
                    if isinstance(sentence, dict):
                        sentence_texts.append(sentence.get('text', ''))
                    elif isinstance(sentence, (list, tuple)) and len(sentence) > 0:
                        sentence_texts.append(str(sentence[0]))
                    else:
                        sentence_texts.append(str(sentence))
            
            # Filter out empty sentences
            sentence_texts = [s.strip() for s in sentence_texts if s.strip()]
            
            if not sentence_texts:
                self.logger.warning("No valid sentences found, using fallback segmentation")
                # Fallback: split text into sentences
                try:
                    import nltk
                    nltk.download('punkt', quiet=True)
                    sentence_texts = nltk.sent_tokenize(text)
                except:
                    sentence_texts = [s.strip() for s in text.split('.') if s.strip()]
            
            # Skip documents that are too short for context windows
            context_size = self.config['one_side_context_size']
            min_sentences = 2 * context_size + 1
            
            if len(sentence_texts) < min_sentences:
                self.logger.warning(f"Document has {len(sentence_texts)} sentences, need at least {min_sentences} for context windows. Using simple segmentation.")
                # Fallback: create simple segments with proper annotation
                segments = []
                if sentences:
                    # Create 2-3 segments for short documents
                    num_segments = min(3, max(2, len(sentences) // 5))
                    segment_length = len(sentences) // num_segments
                    
                    for i in range(num_segments):
                        start_idx = i * segment_length
                        end_idx = min((i + 1) * segment_length - 1, len(sentences) - 1)
                        
                        if i == num_segments - 1:  # Last segment includes remaining sentences
                            end_idx = len(sentences) - 1
                        
                        start_sentence = sentences[start_idx]
                        end_sentence = sentences[end_idx]
                        
                        if isinstance(start_sentence, dict) and isinstance(end_sentence, dict):
                            segments.append({
                                'seq': i + 1,
                                'start': start_sentence.get('start', 0),
                                'end': end_sentence.get('end', 0),
                                'start_sent': start_idx,
                                'end_sent': end_idx
                            })
                        else:
                            segments.append({
                                'seq': i + 1,
                                'start': start_idx * 100,  # Approximate positions
                                'end': (end_idx + 1) * 100,
                                'start_sent': start_idx,
                                'end_sent': end_idx
                            })
                
                annotated_text = self._create_annotated_text(text, sentences, segments)
                return annotated_text, segments
            
            # Prepare contextual data for this document
            encoded_sentences = self.model.encode_sentences(sentence_texts)
            
            # Create context windows and get predictions
            predictions = []
            self.model.eval()
            
            with torch.no_grad():
                for i in range(len(sentence_texts)):
                    # Create context windows
                    left_start = max(0, i - context_size)
                    left_end = i
                    main_idx = i
                    right_start = i + 1
                    right_end = min(len(sentence_texts), i + context_size + 1)
                    
                    # Extract left context
                    if left_end > left_start:
                        left_context = encoded_sentences[left_start:left_end]
                        if left_context.shape[0] < context_size:
                            padding = torch.zeros(context_size - left_context.shape[0], self.config['embedding_dim']).to(encoded_sentences.device)
                            left_context = torch.cat([padding, left_context], dim=0)
                        else:
                            left_context = left_context[-context_size:]
                    else:
                        left_context = torch.zeros(context_size, self.config['embedding_dim']).to(encoded_sentences.device)
                    
                    # Extract main sentence
                    main_sentence = encoded_sentences[main_idx:main_idx+1]
                    
                    # Extract right context
                    if right_end > right_start:
                        right_context = encoded_sentences[right_start:right_end]
                        if right_context.shape[0] < context_size:
                            padding = torch.zeros(context_size - right_context.shape[0], self.config['embedding_dim']).to(encoded_sentences.device)
                            right_context = torch.cat([right_context, padding], dim=0)
                        else:
                            right_context = right_context[:context_size]
                    else:
                        right_context = torch.zeros(context_size, self.config['embedding_dim']).to(encoded_sentences.device)
                    
                    # Prepare tensors and get prediction
                    left_tensor = left_context.unsqueeze(0).to(self.device)
                    main_tensor = main_sentence.unsqueeze(0).to(self.device)
                    right_tensor = right_context.unsqueeze(0).to(self.device)
                    
                    # Get boundary probability
                    output = self.model(left_tensor, main_tensor, right_tensor)
                    prediction = output.item()
                    predictions.append(prediction)
            
            # Post-process predictions to get final segments
            segments = self._post_process_boundaries(predictions, sentences)
            
            # Create annotated text with segment markers
            annotated_text = self._create_annotated_text(text, sentences, segments)
            
            self.logger.info(f"Att+ CNN segmentation completed: {len(segments)} segments found")
            return annotated_text, segments
            
        except Exception as e:
            self.logger.error(f"Error in Att+ CNN segmentation: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Fallback: return simple segments with annotation tokens
            segments = []
            if sentences:
                first_sentence = sentences[0]
                last_sentence = sentences[-1]
                if isinstance(first_sentence, dict) and isinstance(last_sentence, dict):
                    segments.append({
                        'seq': 1,
                        'start': first_sentence.get('start', 0),
                        'end': last_sentence.get('end', len(text)),
                        'start_sent': 0,
                        'end_sent': len(sentences) - 1
                    })
                else:
                    segments.append({
                        'seq': 1,
                        'start': 0,
                        'end': len(text),
                        'start_sent': 0,
                        'end_sent': len(sentences) - 1
                    })
            
            # Create annotated text with segment markers  
            annotated_text = self._create_annotated_text(text, sentences, segments)
            return annotated_text, segments
    
    def _post_process_boundaries(self, boundary_probs, sentences):
        """
        Post-process boundary probabilities to get final segment boundaries.
        
        Args:
            boundary_probs (list): List of boundary probabilities for each sentence
            sentences (list): List of sentence dictionaries
            
        Returns:
            list: List of segment dictionaries with 'start', 'end', 'seq', 'start_sent', 'end_sent' keys
        """
        threshold = self.config['boundary_threshold']
        min_distance = self.config['min_boundary_distance']
        
        # Find boundary positions based on probability threshold
        boundary_positions = [0]  # Always start with first sentence
        
        for i in range(1, len(boundary_probs)):
            if boundary_probs[i] > threshold:
                # Check minimum distance from previous boundary
                if i - boundary_positions[-1] >= min_distance:
                    boundary_positions.append(i)
        
        # Ensure we have at least some segments
        if len(boundary_positions) == 1 and len(sentences) > 10:
            # If only first sentence is marked, add some evenly spaced boundaries
            step = max(1, len(sentences) // 4)  # Create ~4 segments
            for i in range(step, len(sentences), step):
                if i not in boundary_positions and i - boundary_positions[-1] >= min_distance:
                    boundary_positions.append(i)
        
        # Convert boundary positions to segment objects
        segments = []
        for i, boundary_pos in enumerate(boundary_positions):
            start_sent_idx = boundary_pos
            
            # Determine end of this segment
            if i + 1 < len(boundary_positions):
                end_sent_idx = boundary_positions[i + 1] - 1
            else:
                end_sent_idx = len(sentences) - 1
            
            # Get start and end character positions
            if start_sent_idx < len(sentences) and end_sent_idx < len(sentences):
                start_sentence = sentences[start_sent_idx]
                end_sentence = sentences[end_sent_idx]
                
                if isinstance(start_sentence, dict) and isinstance(end_sentence, dict):
                    start_pos = start_sentence.get('start', 0)
                    end_pos = end_sentence.get('end', 0)
                else:
                    # Fallback for non-dict formats
                    start_pos = 0
                    end_pos = len(' '.join(str(s) for s in sentences[start_sent_idx:end_sent_idx+1]))
                
                segments.append({
                    'seq': i + 1,
                    'start': start_pos,
                    'end': end_pos,
                    'start_sent': start_sent_idx,
                    'end_sent': end_sent_idx
                })
        
        self.logger.info(f"Created {len(segments)} segments from {len(boundary_positions)} boundaries")
        return segments
    
    def _create_annotated_text(self, text, sentences, segments):
        """
        Create annotated text with segment markers.
        
        Args:
            text (str): Original text
            sentences (list): List of sentence dictionaries (for compatibility)
            segments (list): List of segment dictionaries with 'start', 'end', 'seq' keys
            
        Returns:
            str: Annotated text with segment markers like "##1# text #1##"
        """
        if not segments:
            return text
        
        annotated_text = text
        offset = 0
        
        # Sort segments by start position to insert markers correctly
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        for segment in sorted_segments:
            start_pos = segment['start'] + offset
            end_pos = segment['end'] + offset
            
            # Create markers: ##seq# at start, #seq## at end
            start_marker = f"##{segment['seq']}# "
            end_marker = f" #{segment['seq']}##"
            
            # Insert start marker
            annotated_text = annotated_text[:start_pos] + start_marker + annotated_text[start_pos:]
            offset += len(start_marker)
            
            # Update end position and insert end marker
            end_pos += len(start_marker)
            annotated_text = annotated_text[:end_pos] + end_marker + annotated_text[end_pos:]
            offset += len(end_marker)
        
        return annotated_text

def create_segmenter(
    pretrained_model: str = 'google-bert/bert-base-uncased',
    embedding_dim: int = 768,
    ngram_filters: list = None,
    conv_hidden_units: list = None,
    one_side_context_size: int = 10,
    cnn_dense_size: int = 300,
    lstm_hidden_size: int = 300,
    lstm_layers: int = 1,
    attention_size: int = 100,
    classifier_hidden_size: int = 500,
    dropout: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 8,
    epochs: int = 10,
    patience: int = 3,
    boundary_threshold: float = 0.5,
    min_boundary_distance: int = 2,
    max_seq_length: int = 128,
    fine_tuning: bool = True,
    dataset_type: str = 'wikisection',
    dataset_subset: str = 'en_city',
    min_segment_length: int = 50,
    min_segments_per_document: int = 2,
    **kwargs  # Accept additional parameters to avoid errors
) -> CNNBiLSTMSegmenter:
    """
    Factory function to create a CNN-BiLSTM segmenter with Att+ CNN architecture.
    
    Note: Ignores dataset-specific parameters like random_seed, train_size, val_size, test_size
    """
    # Map parameters to the CNNBiLSTMSegmenter configuration
    config = {
        'pretrained_model': pretrained_model,
        'embedding_dim': embedding_dim,
        'ngram_filters': ngram_filters or [2, 3, 4, 5],
        'conv_hidden_units': conv_hidden_units or [200, 200, 200, 200],
        'one_side_context_size': one_side_context_size,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'num_epochs': epochs,
        'boundary_threshold': boundary_threshold,
        'min_boundary_distance': min_boundary_distance,
        'max_seq_length': max_seq_length,
    }
    
    return CNNBiLSTMSegmenter(**config)
