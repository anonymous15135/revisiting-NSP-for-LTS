"""
Evaluation module for topic segmentation.

This module provides evaluation functions for topic segmentation algorithms
using both traditional metrics and advanced segeval metrics.
"""

import logging
import regex
from nltk.metrics.segmentation import pk, windowdiff
import segeval
from decimal import Decimal


def extract_boundaries_from_annotation(annotated_text):
    """
    Extract segment boundary markers from annotated text.
    
    Args:
        annotated_text: Text with ##SEQ# and #SEQ## markers
        
    Returns:
        list: List of boundary dictionaries with start and end positions
    """
    boundaries = []
    
    # Find all topic start markers ##SEQ#
    start_matches = list(regex.finditer(r"##(\d+)#", annotated_text))
    
    # Find all topic end markers #SEQ##
    end_matches = list(regex.finditer(r"#(\d+)##", annotated_text))
    
    # Match start and end markers by their sequence numbers
    for start_match in start_matches:
        seq_num = start_match.group(1)  # Extract the sequence number
        start_pos = start_match.start()
        
        # Find the corresponding end marker with the same sequence number
        for end_match in end_matches:
            if end_match.group(1) == seq_num:
                end_pos = end_match.end()
                boundaries.append({
                    'segment_id': seq_num,
                    'start': start_pos,
                    'end': end_pos,
                    'text': annotated_text[start_pos:end_pos]
                })
                break
    
    # Sort by start position for consistent evaluation
    boundaries.sort(key=lambda x: x['start'])
    
    return boundaries


def convert_boundaries_to_sentence_indices(boundaries, sentences):
    """Convert character-based boundaries to sentence-based boundaries."""
    sent_boundaries = []
    
    for boundary in boundaries:
        # Find sentence indices for start and end positions
        start_sent_idx = 0
        end_sent_idx = len(sentences) - 1
        
        for i, sentence in enumerate(sentences):
            if sentence['start'] <= boundary['start'] <= sentence['end']:
                start_sent_idx = i
            if sentence['start'] <= boundary['end'] <= sentence['end']:
                end_sent_idx = i
                break
        
        sent_boundaries.append({
            'segment_id': boundary.get('segment_id', boundary.get('seq', '')),
            'start_char': boundary['start'],
            'end_char': boundary['end'],
            'start_sent': start_sent_idx,
            'end_sent': end_sent_idx,
            'text': boundary.get('text', '')
        })
    
    return sent_boundaries


def convert_boundaries_to_segeval_masses(boundaries):
    """
    Convert boundary data to segeval masses format using official segeval functions.
    
    Args:
        boundaries: List of boundary dictionaries with 'start' and 'end' keys
        
    Returns:
        tuple: Tuple of segment lengths (masses) for segeval
    """
    if not boundaries:
        return ()
    
    # Sort boundaries by start position to ensure correct order
    sorted_boundaries = sorted(boundaries, key=lambda x: x['start'])
    
    # Calculate segment lengths (masses) - this is the direct approach that segeval expects
    masses = []
    for boundary in sorted_boundaries:
        length = boundary['end'] - boundary['start']
        if length > 0:  # Only include positive lengths
            masses.append(length)
    
    return tuple(masses)


def normalize_boundaries_for_segeval(pred_boundaries, gt_boundaries):
    """
    Normalize boundaries to compatible segeval masses format using official segeval utilities.
    Ensures both segmentations have compatible lengths for segeval metrics.
    
    Args:
        pred_boundaries: List of predicted boundary objects
        gt_boundaries: List of ground truth boundary objects
        
    Returns:
        tuple: (normalized_pred_masses, normalized_gt_masses)
    """
    if not pred_boundaries or not gt_boundaries:
        return (), ()
    
    # Convert boundaries to masses using our conversion function
    pred_raw_masses = convert_boundaries_to_segeval_masses(pred_boundaries)
    gt_raw_masses = convert_boundaries_to_segeval_masses(gt_boundaries)
    
    if not pred_raw_masses or not gt_raw_masses:
        return (), ()
    
    # Normalize both segmentations to have the same total length
    # This is crucial for segeval metrics to work correctly
    pred_total = sum(pred_raw_masses)
    gt_total = sum(gt_raw_masses)
    
    # Choose a common target length (use the maximum to preserve detail)
    target_length = 1000  # Standard length for segeval
    
    # Scale both to the same target length
    pred_scale = target_length / pred_total
    gt_scale = target_length / gt_total
    
    # Scale predicted masses
    pred_scaled = [max(1, round(mass * pred_scale)) for mass in pred_raw_masses]
    gt_scaled = [max(1, round(mass * gt_scale)) for mass in gt_raw_masses]
    
    # Adjust final segments to ensure exact target length
    pred_sum = sum(pred_scaled)
    gt_sum = sum(gt_scaled)
    
    if pred_sum != target_length and pred_scaled:
        diff = target_length - pred_sum
        pred_scaled[-1] = max(1, pred_scaled[-1] + diff)
    
    if gt_sum != target_length and gt_scaled:
        diff = target_length - gt_sum
        gt_scaled[-1] = max(1, gt_scaled[-1] + diff)
    
    # Convert to tuples for segeval compatibility
    pred_masses = tuple(pred_scaled)
    gt_masses = tuple(gt_scaled)
    
    # Verify segeval compatibility by testing conversions
    try:
        # Test that segeval can handle these masses
        pred_positions = segeval.convert_masses_to_positions(pred_masses)
        gt_positions = segeval.convert_masses_to_positions(gt_masses)
        
        # Ensure both have the same total length
        pred_length = len(pred_positions)
        gt_length = len(gt_positions)
        
        if pred_length != gt_length:
            # Adjust to match lengths by modifying the last segment
            if pred_length > gt_length:
                diff = pred_length - gt_length
                if len(gt_masses) > 0:
                    gt_masses = gt_masses[:-1] + (gt_masses[-1] + diff,)
            else:
                diff = gt_length - pred_length
                if len(pred_masses) > 0:
                    pred_masses = pred_masses[:-1] + (pred_masses[-1] + diff,)
        
        # Final validation
        pred_positions_final = segeval.convert_masses_to_positions(pred_masses)
        gt_positions_final = segeval.convert_masses_to_positions(gt_masses)
        
        if len(pred_positions_final) == len(gt_positions_final):
            return pred_masses, gt_masses
        else:
            # If still not matching, use equal-length fallback
            logging.warning(f"Length mismatch after adjustment: {len(pred_positions_final)} vs {len(gt_positions_final)}")
            return create_equal_length_masses(pred_raw_masses, gt_raw_masses)
            
    except Exception as e:
        logging.warning(f"Segeval format validation failed: {e}")
        return create_equal_length_masses(pred_raw_masses, gt_raw_masses)


def create_equal_length_masses(pred_raw_masses, gt_raw_masses):
    """Create equal-length masses for segeval compatibility."""
    # Simple approach: normalize both to same number of segments with equal total
    max_segments = max(len(pred_raw_masses), len(gt_raw_masses))
    target_total = 1000
    
    # For predictions: pad or compress to max_segments
    if len(pred_raw_masses) < max_segments:
        # Pad with small segments
        pred_masses = list(pred_raw_masses) + [1] * (max_segments - len(pred_raw_masses))
    else:
        pred_masses = list(pred_raw_masses[:max_segments])
    
    # For ground truth: pad or compress to max_segments  
    if len(gt_raw_masses) < max_segments:
        # Pad with small segments
        gt_masses = list(gt_raw_masses) + [1] * (max_segments - len(gt_raw_masses))
    else:
        gt_masses = list(gt_raw_masses[:max_segments])
    
    # Scale both to target total
    pred_scale = target_total / sum(pred_masses)
    gt_scale = target_total / sum(gt_masses)
    
    pred_normalized = [max(1, round(m * pred_scale)) for m in pred_masses]
    gt_normalized = [max(1, round(m * gt_scale)) for m in gt_masses]
    
    # Ensure exact target total
    pred_diff = target_total - sum(pred_normalized)
    gt_diff = target_total - sum(gt_normalized)
    
    if pred_normalized:
        pred_normalized[-1] += pred_diff
    if gt_normalized:
        gt_normalized[-1] += gt_diff
    
    return tuple(pred_normalized), tuple(gt_normalized)


def compute_segeval_metrics(pred_boundaries, gt_boundaries):
    """
    Compute segeval-based metrics using official segeval conversion functions.
    
    Args:
        pred_boundaries: List of predicted boundary objects
        gt_boundaries: List of ground truth boundary objects
        
    Returns:
        dict: Dictionary with segeval metrics
    """
    results = {}
    
    try:
        # Convert boundaries to segeval masses format using official functions
        pred_masses, gt_masses = normalize_boundaries_for_segeval(pred_boundaries, gt_boundaries)
        
        if not pred_masses or not gt_masses:
            # Handle empty segmentations
            return {
                "boundary_similarity": 0.0,
                "boundary_edit_distance": None,
                "bed_confusion_matrix": None,
                "bed_precision": 0.0,
                "bed_recall": 0.0,
                "bed_fmeasure": 0.0,
                "segeval_pk": 1.0,
                "segeval_windowdiff": 1.0
            }
        
        logging.info(f"Segeval masses - Predicted: {pred_masses}, Ground Truth: {gt_masses}")
        
        # Boundary Similarity (B) - using explicit boundary format
        try:
            boundary_sim = float(segeval.boundary_similarity(
                pred_masses, gt_masses, 
                boundary_format=segeval.BoundaryFormat.mass
            ))
            results["boundary_similarity"] = boundary_sim
        except Exception as e:
            logging.warning(f"Failed to compute boundary similarity: {e}")
            results["boundary_similarity"] = 0.0
        
        # Boundary Edit Distance (BED) - using official boundary string conversion
        try:
            # Use segeval's official boundary string conversion
            pred_boundary_string = segeval.boundary_string_from_masses(pred_masses)
            gt_boundary_string = segeval.boundary_string_from_masses(gt_masses)
            
            # Compute BED with standard n_t parameter
            bed_result = segeval.boundary_edit_distance(
                pred_boundary_string, gt_boundary_string, n_t=2
            )
            
            # Extract edit operations from BED result
            # bed_result is a tuple: (additions, substitutions, transpositions)
            additions, substitutions, transpositions = bed_result
            
            bed_summary = {
                "additions": len(additions) if additions else 0,
                "substitutions": len(substitutions) if substitutions else 0,
                "transpositions": len(transpositions) if transpositions else 0,
                "total_edits": sum([
                    len(additions) if additions else 0,
                    len(substitutions) if substitutions else 0, 
                    len(transpositions) if transpositions else 0
                ])
            }
            
            results["boundary_edit_distance"] = bed_summary
        except Exception as e:
            logging.warning(f"Failed to compute boundary edit distance: {e}")
            results["boundary_edit_distance"] = None
        
        # BED-based Confusion Matrix (BED-CM) - using masses format
        try:
            confusion_matrix = segeval.boundary_confusion_matrix(
                pred_masses, gt_masses,
                boundary_format=segeval.BoundaryFormat.mass
            )
            
            # Extract precision, recall, and F-measure using segeval's official functions
            bed_precision = float(segeval.precision(confusion_matrix))
            bed_recall = float(segeval.recall(confusion_matrix))
            bed_fmeasure = float(segeval.fmeasure(confusion_matrix))
            
            results["bed_confusion_matrix"] = str(confusion_matrix)  # Convert to string for JSON serialization
            results["bed_precision"] = bed_precision
            results["bed_recall"] = bed_recall
            results["bed_fmeasure"] = bed_fmeasure
            
        except Exception as e:
            logging.warning(f"Failed to compute BED confusion matrix: {e}")
            results["bed_confusion_matrix"] = None
            results["bed_precision"] = 0.0
            results["bed_recall"] = 0.0
            results["bed_fmeasure"] = 0.0
        
        # Traditional segeval metrics for comparison - using masses format explicitly
        try:
            segeval_pk = float(segeval.pk(
                pred_masses, gt_masses,
                boundary_format=segeval.BoundaryFormat.mass
            ))
            segeval_windowdiff = float(segeval.window_diff(
                pred_masses, gt_masses,
                boundary_format=segeval.BoundaryFormat.mass
            ))
            
            results["segeval_pk"] = segeval_pk
            results["segeval_windowdiff"] = segeval_windowdiff
            
        except Exception as e:
            logging.warning(f"Failed to compute segeval traditional metrics: {e}")
            results["segeval_pk"] = 1.0
            results["segeval_windowdiff"] = 1.0
    
    except Exception as e:
        logging.error(f"Failed to compute segeval metrics: {e}")
        # Return default values in case of complete failure
        results = {
            "boundary_similarity": 0.0,
            "boundary_edit_distance": None,
            "bed_confusion_matrix": None,
            "bed_precision": 0.0,
            "bed_recall": 0.0,
            "bed_fmeasure": 0.0,
            "segeval_pk": 1.0,
            "segeval_windowdiff": 1.0
        }
    
    return results


def convert_boundaries_to_sentence_indices(boundaries, sentences):
    """Convert character-based boundaries to sentence-based boundaries."""
    sent_boundaries = []
    
    for boundary in boundaries:
        # Find sentence indices for start and end positions
        start_sent_idx = 0
        end_sent_idx = len(sentences) - 1
        
        for i, sentence in enumerate(sentences):
            if sentence['start'] <= boundary['start'] <= sentence['end']:
                start_sent_idx = i
            if sentence['start'] <= boundary['end'] <= sentence['end']:
                end_sent_idx = i
                break
        
        sent_boundaries.append({
            'segment_id': boundary.get('segment_id', boundary.get('seq', '')),
            'start_char': boundary['start'],
            'end_char': boundary['end'],
            'start_sent': start_sent_idx,
            'end_sent': end_sent_idx,
            'text': boundary.get('text', '')
        })
    
    return sent_boundaries


def evaluate_segmentation(pred_boundaries, gt_boundaries, tolerance=0, use_sentence_boundaries=True, sentences=None):
    """
    Evaluate segmentation quality using multiple metrics with sentence-based tolerance.
    
    Args:
        pred_boundaries: List of predicted boundary objects
        gt_boundaries: List of ground truth boundary objects
        tolerance: Sentence-based tolerance for matching boundaries (0 = exact match)
        use_sentence_boundaries: Whether to use sentence-based evaluation
        sentences: List of sentence objects for sentence-based evaluation
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    results = {}
    
    # Convert to sentence-based boundaries if sentences are provided
    if use_sentence_boundaries and sentences is not None:
        pred_sent_boundaries = convert_boundaries_to_sentence_indices(pred_boundaries, sentences)
        gt_sent_boundaries = convert_boundaries_to_sentence_indices(gt_boundaries, sentences)
        use_sentence_based = True
    else:
        # Fallback to character-based evaluation
        pred_sent_boundaries = pred_boundaries
        gt_sent_boundaries = gt_boundaries
        use_sentence_based = False
    
    # --- F1 Score with sentence-based tolerance ---
    match_scores = []
    
    for p_idx, pred_topic in enumerate(pred_sent_boundaries):
        for gt_idx, gt_topic in enumerate(gt_sent_boundaries):
            if use_sentence_based:
                # Calculate sentence distance for start and end positions
                start_distance = abs(pred_topic['start_sent'] - gt_topic['start_sent'])
                end_distance = abs(pred_topic['end_sent'] - gt_topic['end_sent'])
            else:
                # Fallback to character-based distance
                start_distance = abs(pred_topic.get('start', 0) - gt_topic.get('start', 0))
                end_distance = abs(pred_topic.get('end', 0) - gt_topic.get('end', 0))
            
            # A topic is considered a match if both start and end are within tolerance
            if start_distance <= tolerance and end_distance <= tolerance:
                # Score is inversely proportional to the sum of distances
                score = 1000 - (start_distance + end_distance)
                match_scores.append((score, p_idx, gt_idx))
    
    # Sort by score (higher score = better match)
    match_scores.sort(reverse=True)
    
    # Greedy matching - match best scores first
    matched_pred = set()
    matched_gt = set()
    matches = []
    
    for score, p_idx, gt_idx in match_scores:
        # If either prediction or ground truth is already matched, skip
        if p_idx in matched_pred or gt_idx in matched_gt:
            continue
        
        # Mark as matched
        matched_pred.add(p_idx)
        matched_gt.add(gt_idx)
        
        # Store match information
        pred_topic = pred_sent_boundaries[p_idx]
        gt_topic = gt_sent_boundaries[gt_idx]
        
        if use_sentence_based:
            start_distance = abs(pred_topic['start_sent'] - gt_topic['start_sent'])
            end_distance = abs(pred_topic['end_sent'] - gt_topic['end_sent'])
            
            match_info = {
                'pred_seq': pred_topic.get('segment_id', pred_topic.get('seq', '')), 
                'gt_seq': gt_topic.get('segment_id', gt_topic.get('seq', '')),
                'start_distance_sent': start_distance,
                'end_distance_sent': end_distance,
                'start_distance_char': abs(pred_topic['start_char'] - gt_topic['start_char']),
                'end_distance_char': abs(pred_topic['end_char'] - gt_topic['end_char']),
                'pred_start_sent': pred_topic['start_sent'],
                'pred_end_sent': pred_topic['end_sent'],
                'gt_start_sent': gt_topic['start_sent'],
                'gt_end_sent': gt_topic['end_sent'],
                'pred_start_char': pred_topic['start_char'],
                'pred_end_char': pred_topic['end_char'],
                'gt_start_char': gt_topic['start_char'],
                'gt_end_char': gt_topic['end_char'],
            }
        else:
            start_distance = abs(pred_topic.get('start', 0) - gt_topic.get('start', 0))
            end_distance = abs(pred_topic.get('end', 0) - gt_topic.get('end', 0))
            
            match_info = {
                'pred_seq': pred_topic.get('segment_id', pred_topic.get('seq', '')), 
                'gt_seq': gt_topic.get('segment_id', gt_topic.get('seq', '')),
                'start_distance': start_distance,
                'end_distance': end_distance,
                'pred_start': pred_topic.get('start', 0),
                'pred_end': pred_topic.get('end', 0),
                'gt_start': gt_topic.get('start', 0),
                'gt_end': gt_topic.get('end', 0),
            }
        
        matches.append(match_info)
    
    # Calculate true/false positives/negatives for reference
    true_positives = len(matches)
    false_positives = len(pred_boundaries) - true_positives
    false_negatives = len(gt_boundaries) - true_positives
    
    # Store basic counts (F1/precision/recall removed - use segeval metrics instead)
    results.update({
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "tolerance": tolerance,
        "tolerance_type": "sentences" if use_sentence_based else "characters",
        "matched_topics": matches
    })

    # --- Pk Score and WindowDiff (using original character-based boundaries) ---
    try:
        # Find the maximum position to create vectors of proper length
        max_pos = 0
        for boundary in pred_boundaries + gt_boundaries:
            if use_sentence_based:
                max_pos = max(max_pos, boundary.get('start', 0), boundary.get('end', 0))
            else:
                max_pos = max(max_pos, boundary.get('start', 0), boundary.get('end', 0))
        
        if max_pos == 0:
            # No boundaries found, return default scores
            results.update({
                "pk_score": 1.0,
                "windowdiff": 1.0,
                "window_size": 1
            })
            return results
        
        # Create binary vectors (1 at boundary positions, 0 elsewhere)
        pred_vector = [0] * (max_pos + 1)
        gt_vector = [0] * (max_pos + 1)
        
        for boundary in pred_boundaries:
            start_pos = boundary.get('start', 0)
            if start_pos <= max_pos:
                pred_vector[start_pos] = 1
            
        for boundary in gt_boundaries:
            start_pos = boundary.get('start', 0)
            if start_pos <= max_pos:
                gt_vector[start_pos] = 1
        
        # Calculate window size: half the average segment length
        k = max(int(len(gt_vector) / (2 * (sum(gt_vector) + 1))), 2)
        
        # Convert binary vectors to NLTK-compatible string format
        gt_string = ''.join(str(x) for x in gt_vector)
        pred_string = ''.join(str(x) for x in pred_vector)
        
        # Calculate Pk score and WindowDiff using NLTK
        pk_score = pk(gt_string, pred_string, k)
        window_diff = windowdiff(gt_string, pred_string, k)
        
        # Store window-based metrics
        results.update({
            "pk_score": pk_score,
            "windowdiff": window_diff,
            "window_size": k
        })
        
    except Exception as e:
        logging.warning(f"Failed to compute window-based metrics: {e}")
        results.update({
            "pk_score": 1.0,
            "windowdiff": 1.0,
            "window_size": 1
        })
    
    # --- Add segeval metrics ---
    try:
        segeval_metrics = compute_segeval_metrics(pred_boundaries, gt_boundaries)
        results.update(segeval_metrics)
        
    except Exception as e:
        logging.warning(f"Failed to compute segeval metrics: {e}")
        # Add default segeval metrics if computation fails
        results.update({
            "boundary_similarity": 0.0,
            "boundary_edit_distance": None,
            "bed_confusion_matrix": None,
            "bed_precision": 0.0,
            "bed_recall": 0.0,
            "bed_fmeasure": 0.0,
            "segeval_pk": 1.0,
            "segeval_windowdiff": 1.0
        })
    
    return results
