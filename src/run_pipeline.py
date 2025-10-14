#!/usr/bin/env python3
"""
Topic Segmentation Pipeline

This script runs topic segmentation algorithms on datasets and evaluates their performance.

Supported Algorithms:
    - nsp: Next Sentence Prediction (BERT-based)
    - cnn_bilstm: Att+ CNN (Contextual CNN+BiLSTM+Attention)
    - topseg: TopSeg (RoBERTa with coherence loss)
    - lumberchunker: LumberChunker (LLM-based narrative segmentation)

Supported Datasets:
    - wikisection: WikiSection dataset (en_city, en_disease, de_city, de_disease)
    - councilseg: CouncilSeg dataset (Portuguese and English municipal meeting minutes)

Usage:
    python run_pipeline.py --algorithm nsp --dataset wikisection --subset en_city
    python run_pipeline.py --algorithm topseg --dataset councilseg --language pt
    python run_pipeline.py --algorithm lumberchunker --dataset councilseg --language en --max-docs 10
"""

import os
import sys
import json
import logging
import argparse
import random
from datetime import datetime

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NOTE: Algorithm and dataset imports are delayed until after GPU setup
# to ensure CUDA_VISIBLE_DEVICES takes effect before PyTorch initialization

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_available_algorithms_list():
    """Get list of available algorithms without importing them."""
    return ['nsp', 'cnn_bilstm', 'topseg', 'texttiling', 'lumberchunker']


def get_available_datasets_list():
    """Get list of available datasets without importing them."""
    return ['wikisection', 'councilseg']


def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run topic segmentation algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run NSP on WikiSection English city dataset
  python run_pipeline.py --algorithm nsp --dataset wikisection --subset en_city
  
  # Run TopSeg on CouncilSeg Portuguese dataset with fine-tuning
  python run_pipeline.py --algorithm topseg --dataset councilseg --language pt --split test
  
  # Run LumberChunker on CouncilSeg English with limited documents
  python run_pipeline.py --algorithm lumberchunker --dataset councilseg --language en --max-docs 5
        """
    )
    
    parser.add_argument(
        '--algorithm',
        choices=get_available_algorithms_list(),
        required=True,
        help='Segmentation algorithm to use'
    )
    
    parser.add_argument(
        '--dataset',
        choices=get_available_datasets_list(),
        required=True,
        help='Dataset to use for evaluation'
    )
    
    parser.add_argument(
        '--subset',
        default='en_city',
        help='Dataset subset for WikiSection (en_city, en_disease, de_city, de_disease)'
    )
    
    parser.add_argument(
        '--language',
        choices=['pt', 'en'],
        default='pt',
        help='Language for CouncilSeg dataset (pt=Portuguese, en=English)'
    )
    
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test', 'all'],
        default='test',
        help='Dataset split to use'
    )
    
    parser.add_argument(
        '--max-docs',
        type=int,
        default=None,
        help='Maximum number of documents to process'
    )
    
    parser.add_argument(
        '--sample-ratio',
        type=float,
        default=None,
        help='Fraction of dataset to sample (0.0-1.0)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        default=None,
        help='Configuration name from config file'
    )
    
    parser.add_argument(
        '--config-file',
        default='pipeline_configs.json',
        help='JSON file containing algorithm configurations'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID to use (default: 0, -1 for CPU)'
    )
    
    return parser.parse_args()


def load_algorithm_config(config_name, config_file, algorithm):
    """
    Load algorithm configuration from file or use defaults.
    
    Args:
        config_name: Name of specific config to load
        config_file: Path to config file
        algorithm: Algorithm name for default config
        
    Returns:
        dict: Configuration dictionary
    """
    # Try to load from config file
    if config_name and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            all_configs = json.load(f)
        
        if config_name in all_configs:
            logging.info(f"Loaded configuration '{config_name}' from {config_file}")
            return all_configs[config_name]
        else:
            logging.warning(f"Configuration '{config_name}' not found. Available: {list(all_configs.keys())}")
    
    # Try to load algorithm-specific config
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            all_configs = json.load(f)
        if algorithm in all_configs:
            logging.info(f"Loaded default configuration for '{algorithm}'")
            return all_configs[algorithm]
    
    # Default configurations
    default_configs = {
        'nsp': {
            'fine_tuning': True,
            'learning_rate': 2e-5,
            'batch_size': 16,
            'epochs': 3,
            'threshold': 0.5,
            'merge_numbered_sentences': True
        },
        'cnn_bilstm': {
            'fine_tuning': True,
            'learning_rate': 0.001,
            'batch_size': 8,
            'epochs': 10,
            'patience': 3
        },
        'topseg': {
            'fine_tuning': True,
            'learning_rate': 2e-5,
            'batch_size': 16,
            'epochs': 10,
            'threshold': 0.5
        },
        'texttiling': {
            'w': 20,
            'k': 15,
            'similarity_method': 0,
            'stopwords': None,
            'smoothing_method': [0],
            'smoothing_width': 2,
            'smoothing_rounds': 1,
            'cutoff_policy': 1
        },
        'lumberchunker': {
            'model_id': 'gemini-2.5-flash-lite',
            'min_words_per_window': 400,
            'max_words_per_window': 550,
            'temperature': 0.1
        }
    }
    
    logging.info(f"Using default configuration for '{algorithm}'")
    return default_configs.get(algorithm, {})


def setup_dataset_processor(dataset, subset, language, config):
    """
    Create and configure dataset processor.
    
    Args:
        dataset: Dataset name
        subset: Subset for WikiSection
        language: Language for CouncilSeg
        config: Algorithm configuration
        
    Returns:
        DatasetProcessor instance
    """
    from dataset_processors import create_dataset_processor
    
    # Determine dataset path (relative to src directory)
    base_data_path = "../data"
    
    if dataset == 'wikisection':
        dataset_path = os.path.join(base_data_path, "wikisection_dataset")
        processor_params = {
            'dataset_path': dataset_path,
            'subset': subset,
            'min_segment_length': config.get('min_segment_length', 50),
            'min_segments_per_document': config.get('min_segments_per_document', 2),
            'random_seed': config.get('random_seed', 42)
        }
    elif dataset == 'councilseg':
        dataset_path = os.path.join(base_data_path, "councilseg_dataset")
        processor_params = {
            'dataset_path': dataset_path,
            'language': language,
            'min_segment_length': config.get('min_segment_length', 50),
            'min_segments_per_document': config.get('min_segments_per_document', 2),
            'random_seed': config.get('random_seed', 42)
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    processor = create_dataset_processor(dataset, **processor_params)
    logging.info(f"Created dataset processor for {dataset}")
    
    return processor


def prepare_algorithm_config(algorithm, config, dataset, subset, language):
    """
    Prepare algorithm configuration with dataset information.
    
    Args:
        algorithm: Algorithm name
        config: Base configuration
        dataset: Dataset name
        subset: Dataset subset
        language: Dataset language
        
    Returns:
        dict: Prepared algorithm configuration
    """
    algorithm_config = config.copy()
    
    # Add dataset information for neural algorithms that need it
    if algorithm in ['nsp', 'cnn_bilstm', 'topseg']:
        algorithm_config.update({
            'dataset_type': dataset,
            'dataset_subset': subset if dataset == 'wikisection' else None,
            'min_segment_length': config.get('min_segment_length', 50),
            'min_segments_per_document': config.get('min_segments_per_document', 2)
        })
    
    return algorithm_config


def load_and_process_documents(dataset_processor, split, max_docs, sample_ratio):
    """
    Load and process documents from dataset.
    
    Args:
        dataset_processor: Dataset processor instance
        split: Data split to use
        max_docs: Maximum number of documents
        sample_ratio: Sampling ratio
        
    Returns:
        list: Processed documents
    """
    logging.info(f"Loading {split} split...")
    
    # Get documents using modern approach
    if hasattr(dataset_processor, 'get_documents'):
        documents = dataset_processor.get_documents(split=split, max_documents=max_docs)
    else:
        # Fallback for older processors
        dataset = dataset_processor.load_dataset()
        documents = dataset_processor.process_dataset(dataset, split=split)
        if max_docs and len(documents) > max_docs:
            documents = documents[:max_docs]
    
    if not documents:
        raise ValueError(f"No documents loaded for {split} split")
    
    # Apply sampling if requested
    if sample_ratio is not None:
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError(f"Invalid sample ratio: {sample_ratio}")
        
        original_count = len(documents)
        sample_size = max(1, int(len(documents) * sample_ratio))
        
        random.seed(42)
        documents = random.sample(documents, sample_size)
        
        logging.info(f"Sampled {sample_size} documents ({sample_ratio*100:.1f}%) from {original_count}")
    
    logging.info(f"Loaded {len(documents)} documents")
    return documents


def create_segmentation_algorithm(algorithm, algorithm_config, dataset_processor, max_docs=None, sample_ratio=None):
    """
    Create and optionally train segmentation algorithm.
    
    Args:
        algorithm: Algorithm name
        algorithm_config: Algorithm configuration
        dataset_processor: Dataset processor for training data
        max_docs: Maximum number of documents for training (optional)
        sample_ratio: Fraction of dataset to sample for training (optional)
        
    Returns:
        SegmentationAlgorithm instance
    """
    from algorithms import create_algorithm
    import random
    
    logging.info(f"Creating {algorithm} algorithm...")
    segmenter = create_algorithm(algorithm, **algorithm_config)
    logging.info("Algorithm created successfully")
    
    # Train if algorithm supports and requires training
    if hasattr(segmenter, 'training_mode') and segmenter.training_mode:
        logging.info(f"Algorithm requires training...")
        
        # Load training data with max_docs limit
        if hasattr(dataset_processor, 'get_documents'):
            train_documents = dataset_processor.get_documents(split='train', max_documents=max_docs)
            val_documents = dataset_processor.get_documents(split='val', max_documents=max_docs)
        else:
            dataset = dataset_processor.load_dataset()
            train_documents = dataset_processor.process_dataset(dataset, split='train')
            val_documents = dataset_processor.process_dataset(dataset, split='val')
            # Apply max_docs limit for older processors
            if max_docs:
                train_documents = train_documents[:max_docs]
                val_documents = val_documents[:max_docs]
        
        # Apply sample_ratio if specified
        if sample_ratio is not None:
            if not 0.0 < sample_ratio <= 1.0:
                raise ValueError(f"Invalid sample ratio: {sample_ratio}")
            
            # Sample training documents
            original_train_count = len(train_documents)
            train_sample_size = max(1, int(len(train_documents) * sample_ratio))
            random.seed(42)
            train_documents = random.sample(train_documents, train_sample_size)
            logging.info(f"Sampled {train_sample_size} training documents ({sample_ratio*100:.1f}%) from {original_train_count}")
            
            # Sample validation documents
            original_val_count = len(val_documents)
            val_sample_size = max(1, int(len(val_documents) * sample_ratio))
            random.seed(42)
            val_documents = random.sample(val_documents, val_sample_size)
            logging.info(f"Sampled {val_sample_size} validation documents ({sample_ratio*100:.1f}%) from {original_val_count}")
        
        logging.info(f"Training with {len(train_documents)} docs, validating with {len(val_documents)} docs")
        
        try:
            segmenter.train_model(train_documents, val_documents)
            logging.info("Training completed successfully")
        except Exception as e:
            logging.error(f"Training failed: {e}")
            raise
    
    return segmenter


def process_document(doc_info, segmenter, algorithm):
    """
    Process a single document with the segmentation algorithm.
    
    Args:
        doc_info: Document information dictionary
        segmenter: Segmentation algorithm instance
        algorithm: Algorithm name
        
    Returns:
        tuple: (doc_text, predicted_boundaries, gt_boundaries, evaluation_results) or None if failed
    """
    from evaluation import evaluate_segmentation
    
    # Get document ID - check multiple possible field names
    doc_id = doc_info.get('id') or doc_info.get('meeting_id') or doc_info.get('document_id')
    if not doc_id:
        logging.error(f"Document missing ID: {doc_info.keys()}")
        return None
    
    try:
        # Get document text and ground truth segments
        doc_text = doc_info.get('text', '')
        gt_segments = doc_info.get('segments', [])
        
        if not doc_text:
            logging.error(f"Document {doc_id} has no text")
            return None
        
        # Get sentences
        sentences = doc_info.get('sentences', [])
        if not sentences:
            # Fallback: use algorithm's sentence processing or utils
            from algorithms.utils import split_text_into_sentences, merge_numbered_sentences
            
            # Determine language from document or default to English
            language = 'portuguese' if 'pt' in str(doc_info).lower() else 'english'
            sent_texts = split_text_into_sentences(doc_text, language=language)
            
            # Apply numbered sentence merging if algorithm supports it
            if hasattr(segmenter, 'merge_numbered_sentences') and segmenter.merge_numbered_sentences:
                sent_texts = merge_numbered_sentences(sent_texts)
            
            # Convert to sentence dictionaries with spans
            sentences = []
            start = 0
            for sent_text in sent_texts:
                sent_start = doc_text.find(sent_text, start)
                if sent_start != -1:
                    sent_end = sent_start + len(sent_text)
                    sentences.append({
                        'text': sent_text,
                        'start': sent_start,
                        'end': sent_end
                    })
                    start = sent_end
        
        # Perform segmentation
        logging.info(f"  Segmenting with {len(sentences)} sentences...")
        annotated_text, predicted_boundaries = segmenter.segment_text(doc_text, sentences)
        
        # Convert ground truth segments to boundaries format for evaluation
        # Add segment_id field to ground truth segments if missing (use seq field if present)
        gt_boundaries = []
        for i, seg in enumerate(gt_segments):
            seg_copy = seg.copy()
            if 'segment_id' not in seg_copy:
                # Check if seq exists, otherwise generate from index
                seg_copy['segment_id'] = seg_copy.get('seq', str(i + 1))
            gt_boundaries.append(seg_copy)
        
        # Add segment_id field to predicted boundaries if missing
        for i, pred in enumerate(predicted_boundaries):
            if 'segment_id' not in pred:
                pred['segment_id'] = str(i + 1)
        
        # Evaluate segmentation
        evaluation_results = evaluate_segmentation(
            predicted_boundaries,
            gt_boundaries,
            tolerance=1,
            sentences=sentences
        )
        
        return doc_text, predicted_boundaries, gt_boundaries, evaluation_results
        
    except Exception as e:
        logging.error(f"Failed to process document {doc_id}: {e}")
        return None


def save_results(results, output_dir, algorithm, dataset, config):
    """
    Save evaluation results to file.
    
    Args:
        results: Results dictionary
        output_dir: Output directory
        algorithm: Algorithm name
        dataset: Dataset name
        config: Algorithm configuration
    """
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    overall_results = {
        'algorithm': algorithm,
        'dataset': dataset,
        'config': config,
        'total_documents': results['total_documents'],
        'successful_documents': results['successful_documents'],
        'average_f1': results['avg_f1'],
        'average_precision': results['avg_precision'],
        'average_recall': results['avg_recall'],
        'average_pk_score': results['avg_pk'],
        'average_windowdiff': results['avg_windowdiff'],
        'average_bed_fmeasure': results['avg_bed_f'],
        'average_boundary_similarity': results['avg_boundary_sim'],
        'document_results': results['documents']
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")


def print_summary(results, algorithm, dataset):
    """Print evaluation summary."""
    logging.info(f"\n{'='*60}")
    logging.info(f"EVALUATION COMPLETED")
    logging.info(f"{'='*60}")
    logging.info(f"Algorithm: {algorithm}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Documents processed: {results['successful_documents']}/{results['total_documents']}")
    logging.info(f"Average F1 Score: {results['avg_f1']:.3f}")
    logging.info(f"Average Precision: {results['avg_precision']:.3f}")
    logging.info(f"Average Recall: {results['avg_recall']:.3f}")
    logging.info(f"Average Pk: {results['avg_pk']:.3f}")
    logging.info(f"Average WindowDiff: {results['avg_windowdiff']:.3f}")
    logging.info(f"{'='*60}")


def run_pipeline(args):
    """
    Run the segmentation pipeline.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Import algorithms and datasets AFTER GPU setup has been done in main()
    from algorithms import create_algorithm
    from dataset_processors import create_dataset_processor
    from evaluation import evaluate_segmentation
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{args.dataset}_{args.subset if args.dataset == 'wikisection' else args.language}"
    output_dir = os.path.join(
        args.output_dir,
        dataset_name,
        f"{args.algorithm}_{args.split}_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Load configuration
        config = load_algorithm_config(args.config, args.config_file, args.algorithm)
        
        # Setup dataset processor
        dataset_processor = setup_dataset_processor(
            args.dataset,
            args.subset,
            args.language,
            config
        )
        
        # Load documents
        test_documents = load_and_process_documents(
            dataset_processor,
            args.split,
            args.max_docs,
            args.sample_ratio
        )
        
        # Save ground truth as JSON
        ground_truth_data = {
            "documents": []
        }
        
        for doc in test_documents:
            doc_id = doc.get('meeting_id') or doc.get('document_id') or doc.get('id')
            ground_truth_data["documents"].append({
                "document_id": doc_id,
                "full_text": doc.get('text', ''),
                "segments": doc.get('segments', [])
            })
        
        ground_truth_file = os.path.join(output_dir, "ground_truth.json")
        with open(ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved ground truth to: {ground_truth_file}")
        
        # Create algorithm
        algorithm_config = prepare_algorithm_config(
            args.algorithm,
            config,
            args.dataset,
            args.subset,
            args.language
        )
        segmenter = create_segmentation_algorithm(
            args.algorithm,
            algorithm_config,
            dataset_processor,
            max_docs=args.max_docs,  # Pass max_docs to limit training data
            sample_ratio=args.sample_ratio  # Pass sample_ratio to sample training data
        )
        
        # Process documents
        logging.info(f"\nProcessing {len(test_documents)} documents...")
        
        predictions_data = {
            "evaluation": {
                "total_documents": len(test_documents),
                "successful_documents": 0,
                "average_pk_score": 0.0,
                "average_windowdiff": 0.0,
                "average_bed_fmeasure": 0.0,
                "average_boundary_similarity": 0.0
            },
            "documents": []
        }
        
        successful_docs = 0
        total_metrics = {
            'pk': 0.0, 'windowdiff': 0.0, 'bed_f': 0.0, 'boundary_sim': 0.0
        }
        
        for i, doc_info in enumerate(test_documents):
            doc_id = doc_info.get('id') or doc_info.get('meeting_id') or doc_info.get('document_id')
            logging.info(f"\nDocument {i+1}/{len(test_documents)}: {doc_id}")
            
            result = process_document(doc_info, segmenter, args.algorithm)
            
            if result:
                doc_text, predicted_boundaries, gt_boundaries, evaluation_results = result
                
                # Add document to predictions data
                predictions_data["documents"].append({
                    "document_id": doc_id,
                    "full_text": doc_text,
                    "segments": predicted_boundaries,
                    "evaluation": {
                        "pk_score": evaluation_results.get('pk_score', 0.0),
                        "windowdiff": evaluation_results.get('windowdiff', 0.0),
                        "bed_fmeasure": evaluation_results.get('bed_fmeasure', 0.0),
                        "boundary_similarity": evaluation_results.get('boundary_similarity', 0.0),
                        "true_positives": evaluation_results.get('tp', 0),
                        "false_positives": evaluation_results.get('fp', 0),
                        "false_negatives": evaluation_results.get('fn', 0),
                        "num_predicted_segments": len(predicted_boundaries),
                        "num_ground_truth_segments": len(gt_boundaries)
                    }
                })
                
                # Accumulate metrics
                successful_docs += 1
                total_metrics['pk'] += evaluation_results.get('pk_score', 0.0)
                total_metrics['windowdiff'] += evaluation_results.get('windowdiff', 0.0)
                total_metrics['bed_f'] += evaluation_results.get('bed_fmeasure', 0.0)
                total_metrics['boundary_sim'] += evaluation_results.get('boundary_similarity', 0.0)
                
                logging.info(f"  Pk={evaluation_results['pk_score']:.3f}, "
                           f"WinDiff={evaluation_results['windowdiff']:.3f}, "
                           f"BED-F={evaluation_results.get('bed_fmeasure', 0.0):.3f}")
        
        # Calculate overall statistics
        if successful_docs > 0:
            predictions_data["evaluation"]["successful_documents"] = successful_docs
            predictions_data["evaluation"]["average_pk_score"] = total_metrics['pk'] / successful_docs
            predictions_data["evaluation"]["average_windowdiff"] = total_metrics['windowdiff'] / successful_docs
            predictions_data["evaluation"]["average_bed_fmeasure"] = total_metrics['bed_f'] / successful_docs
            predictions_data["evaluation"]["average_boundary_similarity"] = total_metrics['boundary_sim'] / successful_docs
            
            # Save predictions and evaluation to JSON
            predictions_file = os.path.join(output_dir, "predictions_with_evaluation.json")
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(predictions_data, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved predictions and evaluation to: {predictions_file}")
            
            # Print summary
            logging.info(f"\n{'='*60}")
            logging.info(f"EVALUATION COMPLETED")
            logging.info(f"{'='*60}")
            logging.info(f"Algorithm: {args.algorithm}")
            logging.info(f"Dataset: {dataset_name}")
            logging.info(f"Documents processed: {successful_docs}/{len(test_documents)}")
            logging.info(f"Average Pk: {predictions_data['evaluation']['average_pk_score']:.3f}")
            logging.info(f"Average WindowDiff: {predictions_data['evaluation']['average_windowdiff']:.3f}")
            logging.info(f"Average BED F-measure: {predictions_data['evaluation']['average_bed_fmeasure']:.3f}")
            logging.info(f"Average Boundary Similarity: {predictions_data['evaluation']['average_boundary_similarity']:.3f}")
            logging.info(f"{'='*60}")
            
            return True
        else:
            logging.error("No documents processed successfully")
            return False
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    args = setup_arguments()
    
    # Setup GPU BEFORE any imports or operations that might initialize CUDA
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        logging.info(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
        logging.info("Setting CUDA_VISIBLE_DEVICES='' (CPU only)")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = run_pipeline(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
