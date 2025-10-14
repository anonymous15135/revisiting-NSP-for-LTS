#!/usr/bin/env python3
"""
Leave-One-Municipality-Out Cross-Validation (LOMOCV) for Topic Segmentation

This script runs LOMOCV experiments on the CouncilSeg dataset,
training on 5 municipalities and testing on the 6th held-out municipality,
repeating for all 6 municipalities.

The 6 municipalities are:
- M1
- M2
- M3
- M4
- M5
- M6

Usage:
    python run_loocv.py --algorithm nsp --dataset councilseg --language pt
    python run_loocv.py --algorithm topseg --dataset councilseg --language en --gpu 1
    python run_loocv.py --algorithm cnn_bilstm --dataset councilseg --language pt --gpu 1
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# NOTE: Algorithm and dataset imports are delayed until after GPU setup
# to ensure CUDA_VISIBLE_DEVICES takes effect before PyTorch initialization

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_available_algorithms_list():
    """Get list of available algorithms without importing them."""
    return ['nsp', 'cnn_bilstm', 'topseg', 'lumberchunker']


def get_available_datasets_list():
    """Get list of available datasets without importing them."""
    return ['councilseg']  # Only CouncilSeg supported for municipality-based LOOCV


def setup_arguments():
    """Setup command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run Leave-One-Municipality-Out Cross-Validation for CouncilSeg dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run LOMOCV with NSP on CouncilSeg Portuguese
  python run_loocv.py --algorithm nsp --dataset councilseg --language pt
  
  # Run LOMOCV with TopSeg on CouncilSeg English with GPU 1
  python run_loocv.py --algorithm topseg --dataset councilseg --language en --gpu 1
  
  # Run LOMOCV with CNN-BiLSTM on GPU 2
  python run_loocv.py --algorithm cnn_bilstm --dataset councilseg --language pt --gpu 2
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
        default='councilseg',
        help='Dataset to use (only CouncilSeg supported for municipality-based LOOCV)'
    )
    
    parser.add_argument(
        '--subset',
        default='en_city',
        help='Not used for CouncilSeg (kept for compatibility)'
    )
    
    parser.add_argument(
        '--language',
        choices=['pt', 'en'],
        default='pt',
        help='Language for CouncilSeg dataset (pt=Portuguese, en=English)'
    )
    
    parser.add_argument(
        '--max-docs',
        type=int,
        default=None,
        help='Not used for municipality-based LOOCV (all municipalities are used)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results_loocv',
        help='Output directory for LOOCV results'
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
    
    parser.add_argument(
        '--save-iterations',
        action='store_true',
        help='Save individual iteration results (can be large for many documents)'
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


def load_all_documents(dataset_processor, max_docs=None):
    """
    Load all available documents from the CouncilSeg dataset, grouped by municipality.
    
    Args:
        dataset_processor: Dataset processor instance
        max_docs: Not used (kept for compatibility)
        
    Returns:
        dict: Dictionary mapping municipality names to lists of documents
    """
    logging.info("Loading all documents grouped by municipality...")
    
    # Load from all splits and combine
    if hasattr(dataset_processor, 'get_documents'):
        train_docs = dataset_processor.get_documents(split='train')
        val_docs = dataset_processor.get_documents(split='val')
        test_docs = dataset_processor.get_documents(split='test')
        all_documents = train_docs + val_docs + test_docs
    else:
        # Fallback for older processors
        dataset = dataset_processor.load_dataset()
        all_documents = dataset_processor.process_dataset(dataset)
    
    if not all_documents:
        raise ValueError("No documents loaded from dataset")
    
    # Group documents by municipality
    municipalities = {}
    for doc in all_documents:
        # Extract municipality from document_id or meeting_id
        doc_id = doc.get('document_id') or doc.get('meeting_id') or doc.get('id', '')
        
        # Municipality is the first part before underscore (e.g., "M1_cm_001_2024-01-03")
        municipality = doc_id.split('_')[0] if '_' in doc_id else 'Unknown'
        
        if municipality not in municipalities:
            municipalities[municipality] = []
        municipalities[municipality].append(doc)
    
    # Log statistics
    logging.info(f"\nMunicipalities found: {sorted(municipalities.keys())}")
    for municipality, docs in sorted(municipalities.items()):
        logging.info(f"  {municipality}: {len(docs)} documents")
    
    total_docs = sum(len(docs) for docs in municipalities.values())
    logging.info(f"\nTotal: {len(municipalities)} municipalities, {total_docs} documents\n")
    
    return municipalities


def create_loocv_fold(municipalities_dict, test_municipality):
    """
    Create a single LOMOCV fold with train and test municipalities.
    
    Args:
        municipalities_dict: Dictionary mapping municipality names to document lists
        test_municipality: Name of the municipality to use for testing
        
    Returns:
        tuple: (train_documents, test_documents)
    """
    train_documents = []
    test_documents = []
    
    for municipality, docs in municipalities_dict.items():
        if municipality == test_municipality:
            test_documents.extend(docs)
        else:
            train_documents.extend(docs)
    
    return train_documents, test_documents


def create_segmentation_algorithm(algorithm, algorithm_config):
    """
    Create segmentation algorithm instance.
    
    Args:
        algorithm: Algorithm name
        algorithm_config: Algorithm configuration
        
    Returns:
        SegmentationAlgorithm instance
    """
    from algorithms import create_algorithm
    
    segmenter = create_algorithm(algorithm, **algorithm_config)
    return segmenter


def train_algorithm(segmenter, train_documents):
    """
    Train algorithm on training documents.
    
    Args:
        segmenter: Algorithm instance
        train_documents: List of training documents
    """
    # Train if algorithm supports and requires training
    if hasattr(segmenter, 'training_mode') and segmenter.training_mode:
        # Use a small portion for validation (e.g., 20% of training data)
        val_size = max(1, len(train_documents) // 5)
        val_documents = train_documents[:val_size]
        train_only = train_documents[val_size:]
        
        segmenter.train_model(train_only, val_documents)


def process_test_document(doc_info, segmenter, algorithm):
    """
    Process a single test document with the segmentation algorithm.
    
    Args:
        doc_info: Document information dictionary
        segmenter: Segmentation algorithm instance
        algorithm: Algorithm name
        
    Returns:
        dict: Evaluation results or None if failed
    """
    from evaluation import evaluate_segmentation
    
    # Get document ID
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
        annotated_text, predicted_boundaries = segmenter.segment_text(doc_text, sentences)
        
        # Convert ground truth segments to boundaries format
        gt_boundaries = []
        for i, seg in enumerate(gt_segments):
            seg_copy = seg.copy()
            if 'segment_id' not in seg_copy:
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
        
        # Add document info to results
        evaluation_results['document_id'] = doc_id
        evaluation_results['num_predicted_segments'] = len(predicted_boundaries)
        evaluation_results['num_ground_truth_segments'] = len(gt_boundaries)
        
        return evaluation_results
        
    except Exception as e:
        logging.error(f"Failed to process document {doc_id}: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return None


def aggregate_loocv_results(iteration_results):
    """
    Aggregate results across all LOOCV iterations.
    
    Args:
        iteration_results: List of evaluation results from each iteration
        
    Returns:
        dict: Aggregated statistics
    """
    if not iteration_results:
        return {}
    
    # Collect all metrics
    metrics = {
        'pk_scores': [],
        'windowdiff_scores': [],
        'bed_fmeasures': [],
        'boundary_similarities': [],
        'tp_counts': [],
        'fp_counts': [],
        'fn_counts': [],
        'num_predicted': [],
        'num_ground_truth': []
    }
    
    for result in iteration_results:
        metrics['pk_scores'].append(result.get('pk_score', 0.0))
        metrics['windowdiff_scores'].append(result.get('windowdiff', 0.0))
        metrics['bed_fmeasures'].append(result.get('bed_fmeasure', 0.0))
        metrics['boundary_similarities'].append(result.get('boundary_similarity', 0.0))
        metrics['tp_counts'].append(result.get('tp', 0))
        metrics['fp_counts'].append(result.get('fp', 0))
        metrics['fn_counts'].append(result.get('fn', 0))
        metrics['num_predicted'].append(result.get('num_predicted_segments', 0))
        metrics['num_ground_truth'].append(result.get('num_ground_truth_segments', 0))
    
    # Calculate statistics
    aggregated = {
        'total_iterations': len(iteration_results),
        'successful_iterations': len(iteration_results),
        
        # Mean and std for each metric
        'pk_score': {
            'mean': float(np.mean(metrics['pk_scores'])),
            'std': float(np.std(metrics['pk_scores'])),
            'min': float(np.min(metrics['pk_scores'])),
            'max': float(np.max(metrics['pk_scores']))
        },
        'windowdiff': {
            'mean': float(np.mean(metrics['windowdiff_scores'])),
            'std': float(np.std(metrics['windowdiff_scores'])),
            'min': float(np.min(metrics['windowdiff_scores'])),
            'max': float(np.max(metrics['windowdiff_scores']))
        },
        'bed_fmeasure': {
            'mean': float(np.mean(metrics['bed_fmeasures'])),
            'std': float(np.std(metrics['bed_fmeasures'])),
            'min': float(np.min(metrics['bed_fmeasures'])),
            'max': float(np.max(metrics['bed_fmeasures']))
        },
        'boundary_similarity': {
            'mean': float(np.mean(metrics['boundary_similarities'])),
            'std': float(np.std(metrics['boundary_similarities'])),
            'min': float(np.min(metrics['boundary_similarities'])),
            'max': float(np.max(metrics['boundary_similarities']))
        },
        
        # Total counts
        'total_tp': int(np.sum(metrics['tp_counts'])),
        'total_fp': int(np.sum(metrics['fp_counts'])),
        'total_fn': int(np.sum(metrics['fn_counts'])),
        'total_predicted_segments': int(np.sum(metrics['num_predicted'])),
        'total_ground_truth_segments': int(np.sum(metrics['num_ground_truth']))
    }
    
    return aggregated


def save_lomocv_results(results, output_dir, args, save_iterations=False):
    """
    Save LOMOCV results to file.
    
    Args:
        results: Results dictionary with overall_aggregated and per_municipality data
        output_dir: Output directory
        args: Command line arguments
        save_iterations: Whether to save individual document results per municipality
    """
    # Save summary with overall and per-municipality results
    summary_file = os.path.join(output_dir, "lomocv_summary.json")
    summary = {
        'algorithm': args.algorithm,
        'dataset': args.dataset,
        'dataset_config': {
            'language': args.language,
        },
        'overall_aggregated': results['overall_aggregated'],
        'per_municipality': [
            {
                'municipality': muni['municipality'],
                'aggregated': muni['aggregated']
            }
            for muni in results['per_municipality']
        ],
        'municipalities_tested': results['municipalities_tested'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved LOMOCV summary to: {summary_file}")
    
    # Optionally save detailed per-document results per municipality
    if save_iterations:
        iterations_file = os.path.join(output_dir, "lomocv_iterations.json")
        iterations_data = {
            'algorithm': args.algorithm,
            'dataset': args.dataset,
            'municipalities': [
                {
                    'municipality': muni['municipality'],
                    'documents': muni['documents']
                }
                for muni in results['per_municipality']
            ]
        }
        with open(iterations_file, 'w', encoding='utf-8') as f:
            json.dump(iterations_data, f, indent=2)
        logging.info(f"Saved iteration details to: {iterations_file}")


def print_lomocv_summary(overall_results, municipality_results, algorithm, dataset):
    """Print LOMOCV summary statistics."""
    logging.info(f"\n{'='*70}")
    logging.info(f"LEAVE-ONE-MUNICIPALITY-OUT CROSS-VALIDATION RESULTS")
    logging.info(f"{'='*70}")
    logging.info(f"Algorithm: {algorithm}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Total Municipalities: {len(municipality_results)}")
    logging.info(f"Total Documents: {overall_results['total_iterations']}")
    logging.info(f"")
    
    # Per-municipality results
    logging.info(f"Results by Municipality:")
    logging.info(f"{'-'*70}")
    for muni_result in municipality_results:
        muni_name = muni_result['municipality']
        muni_agg = muni_result['aggregated']
        logging.info(f"  {muni_name}:")
        logging.info(f"    Documents: {muni_agg['num_successful_documents']}/{muni_agg['num_test_documents']}")
        logging.info(f"    Pk: {muni_agg['pk_score']['mean']:.4f}")
        logging.info(f"    WindowDiff: {muni_agg['windowdiff']['mean']:.4f}")
        logging.info(f"    BED F-measure: {muni_agg['bed_fmeasure']['mean']:.4f}")
    logging.info(f"")
    
    # Overall results
    logging.info(f"Overall Results (across all municipalities):")
    logging.info(f"{'-'*70}")
    logging.info(f"Pk Score:")
    logging.info(f"  Mean:  {overall_results['pk_score']['mean']:.4f} ± {overall_results['pk_score']['std']:.4f}")
    logging.info(f"  Range: [{overall_results['pk_score']['min']:.4f}, {overall_results['pk_score']['max']:.4f}]")
    logging.info(f"")
    logging.info(f"WindowDiff:")
    logging.info(f"  Mean:  {overall_results['windowdiff']['mean']:.4f} ± {overall_results['windowdiff']['std']:.4f}")
    logging.info(f"  Range: [{overall_results['windowdiff']['min']:.4f}, {overall_results['windowdiff']['max']:.4f}]")
    logging.info(f"")
    logging.info(f"BED F-measure:")
    logging.info(f"  Mean:  {overall_results['bed_fmeasure']['mean']:.4f} ± {overall_results['bed_fmeasure']['std']:.4f}")
    logging.info(f"  Range: [{overall_results['bed_fmeasure']['min']:.4f}, {overall_results['bed_fmeasure']['max']:.4f}]")
    logging.info(f"")
    logging.info(f"Boundary Similarity:")
    logging.info(f"  Mean:  {overall_results['boundary_similarity']['mean']:.4f} ± {overall_results['boundary_similarity']['std']:.4f}")
    logging.info(f"  Range: [{overall_results['boundary_similarity']['min']:.4f}, {overall_results['boundary_similarity']['max']:.4f}]")
    logging.info(f"")
    logging.info(f"Total Statistics:")
    logging.info(f"  True Positives:  {overall_results['total_tp']}")
    logging.info(f"  False Positives: {overall_results['total_fp']}")
    logging.info(f"  False Negatives: {overall_results['total_fn']}")
    logging.info(f"  Predicted Segments:     {overall_results['total_predicted_segments']}")
    logging.info(f"  Ground Truth Segments:  {overall_results['total_ground_truth_segments']}")
    logging.info(f"{'='*70}")


def run_loocv(args):
    """
    Run Leave-One-Municipality-Out Cross-Validation experiment.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Import algorithms and datasets AFTER GPU setup has been done in main()
    from algorithms import create_algorithm
    from dataset_processors import create_dataset_processor
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{args.dataset}_{args.language}"
    output_dir = os.path.join(
        args.output_dir,
        dataset_name,
        f"{args.algorithm}_lomocv_{timestamp}"
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
        
        # Load all documents grouped by municipality
        municipalities = load_all_documents(dataset_processor, args.max_docs)
        municipality_names = sorted(municipalities.keys())
        n_municipalities = len(municipality_names)
        
        total_docs = sum(len(docs) for docs in municipalities.values())
        
        logging.info(f"\n{'='*70}")
        logging.info(f"Starting Leave-One-Municipality-Out Cross-Validation")
        logging.info(f"{'='*70}")
        logging.info(f"Total municipalities: {n_municipalities}")
        logging.info(f"Total documents: {total_docs}")
        logging.info(f"Municipalities: {', '.join(municipality_names)}")
        logging.info(f"{'='*70}\n")
        
        # Prepare algorithm config
        algorithm_config = prepare_algorithm_config(
            args.algorithm,
            config,
            args.dataset,
            args.subset,
            args.language
        )
        
        # Run LOMOCV iterations (one per municipality)
        municipality_results = []
        
        for fold_idx, test_municipality in enumerate(municipality_names):
            logging.info(f"\n{'='*70}")
            logging.info(f"FOLD {fold_idx + 1}/{n_municipalities}: Testing on {test_municipality}")
            logging.info(f"{'='*70}")
            
            # Create fold
            train_documents, test_documents = create_loocv_fold(municipalities, test_municipality)
            
            # Log training municipalities
            train_municipalities = [m for m in municipality_names if m != test_municipality]
            logging.info(f"Training municipalities: {', '.join(train_municipalities)}")
            logging.info(f"Training documents: {len(train_documents)}")
            logging.info(f"Testing documents: {len(test_documents)} (from {test_municipality})")
            
            try:
                # Create fresh algorithm instance for this fold
                logging.info(f"\nCreating {args.algorithm} algorithm...")
                segmenter = create_segmentation_algorithm(args.algorithm, algorithm_config)
                
                # Train on 5 municipalities
                logging.info("Training model...")
                train_algorithm(segmenter, train_documents)
                logging.info("Training completed")
                
                # Test on held-out municipality
                logging.info(f"\nTesting on {len(test_documents)} documents from {test_municipality}...")
                
                fold_results = []
                for doc_idx, test_doc in enumerate(test_documents):
                    doc_id = test_doc.get('id') or test_doc.get('meeting_id') or test_doc.get('document_id')
                    logging.info(f"  Processing document {doc_idx + 1}/{len(test_documents)}: {doc_id}")
                    
                    result = process_test_document(test_doc, segmenter, args.algorithm)
                    
                    if result:
                        fold_results.append(result)
                        logging.info(f"    ✓ Pk={result['pk_score']:.3f}, "
                                   f"WinDiff={result['windowdiff']:.3f}, "
                                   f"BED-F={result['bed_fmeasure']:.3f}")
                    else:
                        logging.warning(f"    ✗ Failed to process document")
                
                # Aggregate results for this municipality
                if fold_results:
                    fold_aggregated = aggregate_loocv_results(fold_results)
                    fold_aggregated['municipality'] = test_municipality
                    fold_aggregated['num_test_documents'] = len(test_documents)
                    fold_aggregated['num_successful_documents'] = len(fold_results)
                    municipality_results.append({
                        'municipality': test_municipality,
                        'aggregated': fold_aggregated,
                        'documents': fold_results
                    })
                    
                    logging.info(f"\n{'='*70}")
                    logging.info(f"Fold {fold_idx + 1} Summary ({test_municipality}):")
                    logging.info(f"  Documents processed: {len(fold_results)}/{len(test_documents)}")
                    logging.info(f"  Average Pk: {fold_aggregated['pk_score']['mean']:.4f}")
                    logging.info(f"  Average WindowDiff: {fold_aggregated['windowdiff']['mean']:.4f}")
                    logging.info(f"  Average BED F-measure: {fold_aggregated['bed_fmeasure']['mean']:.4f}")
                    logging.info(f"{'='*70}")
                else:
                    logging.warning(f"✗ Fold {fold_idx + 1} ({test_municipality}) failed - no successful documents")
                
            except Exception as e:
                logging.error(f"Error in fold {fold_idx + 1} ({test_municipality}): {e}")
                import traceback
                logging.error(f"Traceback: {traceback.format_exc()}")
        
        # Aggregate results across all municipalities
        if not municipality_results:
            logging.error("No successful folds completed")
            return False
        
        logging.info(f"\nCompleted {len(municipality_results)}/{n_municipalities} folds successfully")
        
        # Collect all document results for overall aggregation
        all_document_results = []
        for muni_result in municipality_results:
            all_document_results.extend(muni_result['documents'])
        
        overall_aggregated = aggregate_loocv_results(all_document_results)
        
        # Save results
        results = {
            'overall_aggregated': overall_aggregated,
            'per_municipality': municipality_results,
            'municipalities_tested': municipality_names
        }
        save_lomocv_results(results, output_dir, args, save_iterations=args.save_iterations)
        
        # Print final summary
        print_lomocv_summary(overall_aggregated, municipality_results, args.algorithm, dataset_name)
        
        return True
        
    except Exception as e:
        logging.error(f"LOMOCV failed: {e}", exc_info=True)
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
    
    success = run_loocv(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
