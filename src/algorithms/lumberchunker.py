"""
LumberChunker: Long-Form Narrative Document Segmentation

Implementation based on the paper:
"LumberChunker: Long-Form Narrative Document Segmentation"
by André V. Duarte, João D.S. Marques, Miguel Graça, Miguel Freire, Lei Li, and Arlindo L. Oliveira
Published in Findings of EMNLP 2024

Paper: https://aclanthology.org/2024.findings-emnlp.377/
Original Repository: https://github.com/joaodsmarques/LumberChunker

This implementation adapts LumberChunker for topic segmentation on municipal meeting minutes,
using Google's Gemini 2.0 Flash model via the Google AI Studio API.

Key differences from original:
- Adapted for Portuguese municipal meeting minutes instead of English narrative books
- Supports both paragraphs (original) and sentences as base units
- Evaluation compares chunks to ground truth segments using ROUGE similarity

Credits:
- Original authors: André V. Duarte, João D.S. Marques, Miguel Graça, Miguel Freire, Lei Li, Arlindo L. Oliveira
- Implementation adapted by: José Miguel Isidro
"""

import os
import re
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import numpy as np

from .base import SegmentationAlgorithm

# Try to import LLM factory
try:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
    from src.data_generation_program.llm_factory import ModelFactory
except ImportError:
    logging.warning("ModelFactory not available. Will try direct Google GenerativeAI import.")
    ModelFactory = None

# Try to import both Google AI libraries
genai_legacy = None
genai = None
USING_VERTEX_AI = False

# Try legacy google.generativeai first (works with API keys)
try:
    import google.generativeai as genai_legacy
except ImportError:
    logging.warning("google.generativeai not available.")
    genai_legacy = None

# Try new Vertex AI SDK (google-genai)
try:
    from google import genai
    from google.genai.types import HttpOptions
    USING_VERTEX_AI = True
except ImportError:
    logging.warning("google-genai not available.")
    USING_VERTEX_AI = False


class RateLimiter:
    """
    Rate limiter for API calls with support for RPM, TPM, and RPD limits.
    
    Tracks:
    - Requests per minute (RPM)
    - Tokens per minute (TPM) 
    - Requests per day (RPD)
    
    Gemini 2.0 Flash limits:
    - RPM: 15
    - TPM: 1,000,000
    - RPD: 200
    """
    
    def __init__(self, 
                 rpm_limit: int = 30,
                 tpm_limit: int = 1_000_000,
                 rpd_limit: int = 200,
                 enable_rpm_limit: bool = True,
                 enable_rpd_limit: bool = True):
        """
        Initialize rate limiter.
        
        Args:
            rpm_limit: Maximum requests per minute
            tpm_limit: Maximum tokens per minute
            rpd_limit: Maximum requests per day
            enable_rpm_limit: Whether to enforce RPM limits (set False for Vertex AI)
            enable_rpd_limit: Whether to enforce RPD limits (set False for Vertex AI)
        """
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.rpd_limit = rpd_limit
        self.enable_rpm_limit = enable_rpm_limit
        self.enable_rpd_limit = enable_rpd_limit
        
        # Track requests and tokens in sliding windows
        self.request_times = deque()  # (timestamp, token_count)
        self.daily_requests = deque()  # (date, count)
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0
        
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Using rough estimate: 1 token ≈ 4 characters (conservative for Gemini)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _clean_old_entries(self):
        """Remove entries outside the time windows."""
        current_time = time.time()
        current_date = datetime.now().date()
        
        # Clean minute window (keep last 60 seconds)
        while self.request_times and current_time - self.request_times[0][0] > 60:
            self.request_times.popleft()
        
        # Clean daily window (keep only today)
        while self.daily_requests and self.daily_requests[0][0] < current_date:
            self.daily_requests.popleft()
    
    def _get_current_counts(self) -> Tuple[int, int, int]:
        """
        Get current request and token counts.
        
        Returns:
            Tuple of (requests_in_minute, tokens_in_minute, requests_today)
        """
        self._clean_old_entries()
        
        # Count requests and tokens in last minute
        requests_in_minute = len(self.request_times)
        tokens_in_minute = sum(tokens for _, tokens in self.request_times)
        
        # Count requests today
        current_date = datetime.now().date()
        requests_today = sum(count for date, count in self.daily_requests if date == current_date)
        
        return requests_in_minute, tokens_in_minute, requests_today
    
    def _calculate_wait_time(self, estimated_tokens: int) -> float:
        """
        Calculate how long to wait before making request.
        
        Args:
            estimated_tokens: Estimated tokens for next request
            
        Returns:
            Wait time in seconds
        """
        requests_in_minute, tokens_in_minute, requests_today = self._get_current_counts()
        
        wait_times = []
        
        # Check RPM limit (if enabled)
        if self.enable_rpm_limit and requests_in_minute >= self.rpm_limit:
            # Wait until oldest request falls outside the window
            oldest_time = self.request_times[0][0]
            wait_time = 61 - (time.time() - oldest_time)
            if wait_time > 0:
                wait_times.append(wait_time)
                logging.debug(f"RPM limit reached ({requests_in_minute}/{self.rpm_limit}), need to wait {wait_time:.1f}s")
        
        # Check TPM limit (always enabled for token management)
        if tokens_in_minute + estimated_tokens > self.tpm_limit:
            # Wait until enough tokens free up
            if self.request_times:
                oldest_time = self.request_times[0][0]
                wait_time = 61 - (time.time() - oldest_time)
                if wait_time > 0:
                    wait_times.append(wait_time)
                    logging.debug(f"TPM limit would be exceeded ({tokens_in_minute + estimated_tokens}/{self.tpm_limit}), need to wait {wait_time:.1f}s")
        
        # Check RPD limit (if enabled)
        if self.enable_rpd_limit and requests_today >= self.rpd_limit:
            # Wait until next day
            now = datetime.now()
            tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time())
            wait_time = (tomorrow - now).total_seconds()
            wait_times.append(wait_time)
            logging.warning(f"RPD limit reached ({requests_today}/{self.rpd_limit}), need to wait until tomorrow ({wait_time/3600:.1f} hours)")
        
        return max(wait_times) if wait_times else 0
    
    def wait_if_needed(self, estimated_tokens: int) -> float:
        """
        Wait if necessary to comply with rate limits.
        
        Args:
            estimated_tokens: Estimated tokens for next request
            
        Returns:
            Time waited in seconds
        """
        wait_time = self._calculate_wait_time(estimated_tokens)
        
        if wait_time > 0:
            logging.info(f"Rate limit: waiting {wait_time:.1f}s before next request...")
            time.sleep(wait_time)
            self.total_wait_time += wait_time
            return wait_time
        
        return 0
    
    def record_request(self, tokens_used: int):
        """
        Record a completed request.
        
        Args:
            tokens_used: Actual tokens used in the request
        """
        current_time = time.time()
        current_date = datetime.now().date()
        
        # Record in minute window
        self.request_times.append((current_time, tokens_used))
        
        # Record in daily count
        if not self.daily_requests or self.daily_requests[-1][0] != current_date:
            self.daily_requests.append([current_date, 1])
        else:
            self.daily_requests[-1][1] += 1
        
        # Update statistics
        self.total_requests += 1
        self.total_tokens += tokens_used
        
        # Clean old entries
        self._clean_old_entries()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics
        """
        requests_in_minute, tokens_in_minute, requests_today = self._get_current_counts()
        
        return {
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'total_wait_time': self.total_wait_time,
            'current_rpm': requests_in_minute,
            'current_tpm': tokens_in_minute,
            'current_rpd': requests_today,
            'rpm_limit': self.rpm_limit,
            'tpm_limit': self.tpm_limit,
            'rpd_limit': self.rpd_limit,
            'rpm_utilization': f"{requests_in_minute}/{self.rpm_limit} ({100*requests_in_minute/self.rpm_limit:.1f}%)",
            'tpm_utilization': f"{tokens_in_minute}/{self.tpm_limit} ({100*tokens_in_minute/self.tpm_limit:.1f}%)",
            'rpd_utilization': f"{requests_today}/{self.rpd_limit} ({100*requests_today/self.rpd_limit:.1f}%)"
        }


class LumberChunkerSegmenter(SegmentationAlgorithm):
    """
    LumberChunker-based text segmentation algorithm.
    
    This algorithm uses an LLM to dynamically segment documents by iteratively
    identifying points where content begins to shift. It processes groups of
    sequential passages (sentences) and asks the LLM to identify semantic boundaries.
    
    Original paper approach:
    - Process ~550 words at a time
    - Assign IDs to each passage (paragraph)
    - Ask LLM: "Find the first paragraph where content clearly changes"
    - LLM returns: "Answer: ID XXXX"
    - Use that as boundary and continue
    
    Our adaptation:
    - Support both paragraphs (original) and sentences as base units
    - Adapt for Portuguese municipal meeting minutes
    - Target word count per window configurable (default: 400-550 words)
    """
    
    def __init__(self, 
                 model_id: str = "gemini-2.5-flash-lite",
                 min_words_per_window: int = 400,
                 max_words_per_window: int = 550,
                 temperature: float = 0.1,
                 max_retries: int = 3,
                 retry_delay: int = 60,
                 rpm_limit: int = 30,
                 tpm_limit: int = 1_000_000,
                 rpd_limit: int = 200,
                 enable_rpm_limit: bool = False,
                 enable_rpd_limit: bool = False,
                 segmentation_unit: str = "sentence"):
        """
        Initialize the LumberChunker segmenter.
        
        Args:
            model_id: Gemini model identifier
            min_words_per_window: Minimum words before asking for boundary
            max_words_per_window: Maximum words to include in window
            temperature: LLM temperature (lower = more deterministic)
            max_retries: Maximum retry attempts on API errors
            retry_delay: Seconds to wait between retries
            rpm_limit: Requests per minute limit (default: 30 for Gemini 2.0 Flash Lite)
            tpm_limit: Tokens per minute limit (default: 1M for Gemini 2.5 Flash Lite)
            rpd_limit: Requests per day limit (default: 200 for Gemini 2.0 Flash Lite)
            enable_rpm_limit: Enable RPM limiting (False by default for Vertex AI)
            enable_rpd_limit: Enable RPD limiting (False by default for Vertex AI)
            segmentation_unit: Base unit for segmentation ('paragraph' or 'sentence')
        """
        self.model_id = model_id
        self.min_words_per_window = min_words_per_window
        self.max_words_per_window = max_words_per_window
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.segmentation_unit = segmentation_unit.lower()
        
        # Validate segmentation unit
        if self.segmentation_unit not in ['sentence', 'paragraph']:
            raise ValueError(f"segmentation_unit must be 'sentence' or 'paragraph', got '{segmentation_unit}'")
        
        logging.info(f"LumberChunker initialized with segmentation_unit='{self.segmentation_unit}'")
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            rpd_limit=rpd_limit,
            enable_rpm_limit=enable_rpm_limit,
            enable_rpd_limit=enable_rpd_limit
        )
        
        if not enable_rpm_limit:
            logging.info("RPM rate limiting disabled (suitable for Vertex AI)")
        if not enable_rpd_limit:
            logging.info("RPD rate limiting disabled (suitable for Vertex AI)")
        
        # Model will be loaded lazily
        self.model = None
        self._using_factory = False
        self._using_vertex_ai = False
        
    def _load_model(self):
        """Load the Gemini model if not already loaded."""
        if self.model is not None:
            return
            
        logging.info(f"Loading Gemini model: {self.model_id}")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it with your Google AI Studio API key."
            )
        
        # Try Vertex AI SDK FIRST (google-genai) - works with Vertex AI
        if USING_VERTEX_AI:
            try:
                # Initialize Vertex AI client with API key
                self.model = genai.Client(api_key=api_key)
                self._using_factory = False
                self._using_vertex_ai = True
                logging.info("Loaded model using Vertex AI SDK (google-genai)")
                return
            except Exception as e:
                logging.warning(f"Failed to load with Vertex AI SDK: {e}. Trying other methods.")
        
        # Try ModelFactory second
        if ModelFactory is not None:
            try:
                self.model = ModelFactory.create_generator(
                    "google_gemini",
                    api_key=api_key,
                    model_name=self.model_id,
                    verbose=False
                )
                self._using_factory = True
                logging.info("Loaded model using ModelFactory")
                return
            except Exception as e:
                logging.warning(f"Failed to load with ModelFactory: {e}. Trying legacy API.")
        
        # Try legacy google.generativeai as fallback (for Google AI Studio keys)
        if genai_legacy is not None:
            try:
                genai_legacy.configure(api_key=api_key)
                self.model = genai_legacy.GenerativeModel(self.model_id)
                self._using_factory = False
                self._using_vertex_ai = False
                logging.info("Loaded model using legacy Google GenerativeAI API (Google AI Studio)")
                return
            except Exception as e:
                logging.error(f"Failed to load with legacy API: {e}")
                raise
        
        # If we get here, nothing worked
        raise ImportError(
            "Could not load any Gemini API library. "
            "Please install: pip install google-generativeai"
        )
    
    def _count_words(self, text: str) -> int:
        """
        Approximate token count by counting words.
        Original implementation: 1 word ~ 1.2 tokens
        
        Args:
            text: Input text
            
        Returns:
            Approximate token count
        """
        words = text.split()
        return round(1.2 * len(words))
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Using rough estimate: 1 token ≈ 4 characters (conservative for Gemini)
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def _create_system_prompt(self) -> str:
        """
        Create the system prompt for LumberChunker.
        
        Adapted from original for Portuguese municipal meetings.
        Uses different prompts for paragraphs vs sentences.
        """
        if self.segmentation_unit == 'paragraph':
            # Original paper prompt adapted for Portuguese
            return """Receberá como entrada um documento em português com parágrafos identificados por 'ID XXXX: <texto>'.

Tarefa: Encontre o primeiro parágrafo (não o primeiro) onde o conteúdo muda claramente em comparação com os parágrafos anteriores. Procure mudanças de tópico, assunto ou contexto.

Resposta: Retorne o ID do parágrafo com a mudança de conteúdo no formato exemplificado: 'Resposta: ID XXXX'.

Considerações Adicionais: 
- Evite grupos muito longos de parágrafos
- Procure um bom equilíbrio entre identificar mudanças de conteúdo e manter grupos gerenciáveis
- Se houver mudanças muito subtis, escolha a mudança mais significativa
- Considere o contexto completo antes de decidir"""
        else:  # sentence
            return """Receberá como entrada um documento em português com frases identificadas por 'ID XXXX: <texto>'.

Tarefa: Encontre a primeira frase (não a primeira) onde o conteúdo muda claramente em comparação com as frases anteriores. Procure mudanças de tópico, assunto ou contexto.

Resposta: Retorne o ID da frase com a mudança de conteúdo no formato exemplificado: 'Resposta: ID XXXX'.

Considerações Adicionais: 
- Evite grupos muito longos de frases
- Procure um bom equilíbrio entre identificar mudanças de conteúdo e manter grupos gerenciáveis
- Se houver mudanças muito subtis, escolha a mudança mais significativa
- Considere o contexto completo antes de decidir"""
    
    def _llm_prompt(self, user_prompt: str) -> Optional[str]:
        """
        Send prompt to LLM with retry logic and rate limiting.
        
        Args:
            user_prompt: The user prompt to send
            
        Returns:
            LLM response text or None on failure
        """
        # Estimate tokens for rate limiting
        estimated_tokens = self._count_tokens(user_prompt)
        
        # Wait if needed to comply with rate limits
        wait_time = self.rate_limiter.wait_if_needed(estimated_tokens)
        if wait_time > 0:
            # Log rate limit stats after waiting
            stats = self.rate_limiter.get_stats()
            logging.info(
                f"Rate limit status: "
                f"RPM: {stats['rpm_utilization']}, "
                f"TPM: {stats['tpm_utilization']}, "
                f"RPD: {stats['rpd_utilization']}"
            )
        
        for attempt in range(self.max_retries):
            try:
                if self._using_factory:
                    # Using ModelFactory
                    response = self.model.generate(
                        prompt=user_prompt,
                        temperature=self.temperature
                    )
                    # Record successful request
                    self.rate_limiter.record_request(estimated_tokens)
                    return response
                elif self._using_vertex_ai:
                    # Using Vertex AI SDK (google-genai)
                    response = self.model.models.generate_content(
                        model=self.model_id,
                        contents=user_prompt,
                        config={
                            "temperature": self.temperature,
                        }
                    )
                    
                    # Record successful request
                    self.rate_limiter.record_request(estimated_tokens)
                    
                    # Extract text from response
                    if hasattr(response, 'text'):
                        return response.text
                    elif hasattr(response, 'candidates') and response.candidates:
                        return response.candidates[0].content.parts[0].text
                    else:
                        logging.warning("Gemini blocked the prompt as unsafe")
                        return "content_flag_increment"
                else:
                    # Using legacy API
                    generation_config = {
                        "temperature": self.temperature,
                    }
                    
                    # Safety settings to avoid content blocks
                    safety_settings = [
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "threshold": "BLOCK_NONE"
                        },
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "threshold": "BLOCK_NONE"
                        }
                    ]
                    
                    response = self.model.generate_content(
                        user_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    
                    # Record successful request
                    self.rate_limiter.record_request(estimated_tokens)
                    
                    if response.candidates:
                        return response.candidates[0].content.parts[0].text
                    else:
                        logging.warning("Gemini blocked the prompt as unsafe")
                        return "content_flag_increment"
                        
            except Exception as e:
                error_msg = str(e)
                
                if "list index out of range" in error_msg:
                    logging.warning("Gemini thinks prompt is unsafe")
                    # Still count as a request for rate limiting
                    self.rate_limiter.record_request(estimated_tokens)
                    return "content_flag_increment"
                
                # Check for daily quota exhaustion
                if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in error_msg or \
                   ("quota exceeded" in error_msg.lower() and "day" in error_msg.lower()):
                    logging.error(f"Daily quota exhausted (RPD limit): {error_msg}")
                    # Re-raise to propagate to caller
                    raise Exception(f"RPD_QUOTA_EXHAUSTED: {error_msg}")
                
                logging.warning(
                    f"Error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                # Don't record failed requests in rate limiter
                if attempt < self.max_retries - 1:
                    logging.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logging.error("Max retries reached. Returning None.")
                    return None
        
        return None
    
    def _extract_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract paragraphs from text.
        
        Paragraphs are detected by double newlines or significant whitespace breaks.
        This follows the original LumberChunker approach.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraph dictionaries with 'text', 'start', 'end' keys
        """
        paragraphs = []
        
        # Split by double newlines (common paragraph separator)
        # Also handle various newline patterns
        para_texts = re.split(r'\n\s*\n+', text)
        
        start_pos = 0
        for para_text in para_texts:
            para_text = para_text.strip()
            if not para_text:
                continue
            
            # Find actual position in original text
            para_start = text.find(para_text, start_pos)
            if para_start == -1:
                # Fallback: use approximated position
                para_start = start_pos
            
            para_end = para_start + len(para_text)
            
            paragraphs.append({
                'text': para_text,
                'start': para_start,
                'end': para_end
            })
            
            start_pos = para_end
        
        # If no paragraphs found (single block of text), treat entire text as one paragraph
        if not paragraphs:
            paragraphs = [{
                'text': text.strip(),
                'start': 0,
                'end': len(text)
            }]
        
        logging.info(f"Extracted {len(paragraphs)} paragraphs from text")
        return paragraphs
    
    def segment(self, text: str, sentences: List[Dict[str, Any]] = None) -> List[int]:
        """
        Segment text using LumberChunker approach.
        
        Args:
            text: Full document text
            sentences: List of sentence dictionaries with 'text', 'start', 'end' keys
                      (only used if segmentation_unit='sentence')
            
        Returns:
            List of unit indices where segments begin
        """
        # Load model if needed
        self._load_model()
        
        # Select segmentation units based on configuration
        if self.segmentation_unit == 'paragraph':
            units = self._extract_paragraphs(text)
            unit_name = "paragraph"
            unit_name_plural = "paragraphs"
        else:  # sentence
            if not sentences or len(sentences) == 0:
                logging.warning("No sentences provided for sentence-level segmentation")
                return [0]
            units = sentences
            unit_name = "sentence"
            unit_name_plural = "sentences"
        
        logging.info(f"Segmenting document with {len(units)} {unit_name_plural} using LumberChunker")
        
        # Add IDs to units
        id_units = []
        for i, unit in enumerate(units):
            unit_text = unit.get('text', unit.get('sentence', ''))
            id_units.append(f"ID {i}: {unit_text}")
        
        # Initialize segmentation
        segment_boundaries = [0]  # Always start with first unit
        current_pos = 0
        
        system_prompt = self._create_system_prompt()
        
        while current_pos < len(id_units) - 5:  # Stop if less than 5 units left
            # Build window of sentences
            word_count = 0
            window_size = 0
            
            # Expand window until we reach target word count
            while (word_count < self.min_words_per_window and 
                   current_pos + window_size < len(id_units) - 1):
                window_size += 1
                window_text = "\n".join(id_units[current_pos:current_pos + window_size])
                word_count = self._count_words(window_text)
            
            # Don't exceed max words
            if word_count > self.max_words_per_window and window_size > 1:
                window_size -= 1
                window_text = "\n".join(id_units[current_pos:current_pos + window_size])
            else:
                window_text = "\n".join(id_units[current_pos:current_pos + window_size])
            
            # Create prompt
            user_prompt = f"\nDocumento:\n{window_text}"
            full_prompt = system_prompt + user_prompt
            
            # Get LLM response
            llm_output = self._llm_prompt(full_prompt)
            
            # Handle unsafe content flag
            if llm_output == "content_flag_increment":
                current_pos += 1
                continue
            
            if llm_output is None:
                # If LLM fails, just increment by window size
                logging.warning(f"LLM failed at position {current_pos}, using window size")
                current_pos += max(1, window_size // 2)
                continue
            
            # Parse response to extract ID
            pattern = r"Resposta: ID \d+"
            match = re.search(pattern, llm_output)
            
            if match is None:
                # Try English pattern as fallback
                pattern = r"Answer: ID \d+"
                match = re.search(pattern, llm_output)
            
            if match is None:
                logging.warning(
                    f"Could not parse LLM response at position {current_pos}: {llm_output[:100]}"
                )
                # Default: move by half window
                current_pos += max(1, window_size // 2)
                continue
            
            # Extract ID from response
            match_text = match.group(0)
            id_pattern = r'\d+'
            id_match = re.search(id_pattern, match_text)
            
            if id_match:
                boundary_id = int(id_match.group())
                
                # Validate boundary
                if boundary_id <= current_pos or boundary_id >= len(id_units):
                    logging.warning(
                        f"Invalid boundary ID {boundary_id} at position {current_pos}"
                    )
                    current_pos += 1
                    continue
                
                # Add boundary if not duplicate
                if boundary_id not in segment_boundaries:
                    segment_boundaries.append(boundary_id)
                    logging.debug(f"Added boundary at {unit_name} {boundary_id}")
                
                # Move to next position
                current_pos = boundary_id
            else:
                logging.warning(f"Could not extract ID number from: {match_text}")
                current_pos += 1
        
        # Sort boundaries
        segment_boundaries = sorted(list(set(segment_boundaries)))
        
        logging.info(
            f"LumberChunker found {len(segment_boundaries)} segment boundaries "
            f"(based on {unit_name_plural}): {segment_boundaries}"
        )
        
        # Log rate limiter statistics
        self._log_rate_limiter_stats()
        
        return segment_boundaries
    
    def _log_rate_limiter_stats(self):
        """Log rate limiter statistics."""
        stats = self.rate_limiter.get_stats()
        logging.info("=" * 60)
        logging.info("Rate Limiter Statistics:")
        logging.info(f"  Total API requests: {stats['total_requests']}")
        logging.info(f"  Total tokens used: {stats['total_tokens']:,}")
        logging.info(f"  Total wait time: {stats['total_wait_time']:.1f}s ({stats['total_wait_time']/60:.1f} min)")
        logging.info(f"  Current RPM: {stats['rpm_utilization']}")
        logging.info(f"  Current TPM: {stats['tpm_utilization']}")
        logging.info(f"  Current RPD: {stats['rpd_utilization']}")
        logging.info("=" * 60)
    
    def segment_text(self, text: str, sentences: List[Dict[str, Any]]) -> Tuple[str, List[int]]:
        """
        Segment text into topically coherent segments.
        
        This method implements the abstract interface from SegmentationAlgorithm.
        It wraps the segment() method to match the expected signature.
        
        Args:
            text: The full text to segment
            sentences: List of sentence dictionaries with 'text', 'start', 'end' keys
                      (only used if segmentation_unit='sentence')
            
        Returns:
            tuple: (annotated_text, segment_boundaries)
                - annotated_text: Text with segment markers (same as input for LumberChunker)
                - segment_boundaries: List of unit indices where segments begin
        """
        # Get segment boundaries
        boundaries = self.segment(text, sentences)
        
        # For LumberChunker, we don't modify the text, so just return it as-is
        # The boundaries indicate where new segments start
        return text, boundaries
    
    def get_name(self) -> str:
        """Get algorithm name."""
        return "lumberchunker"
