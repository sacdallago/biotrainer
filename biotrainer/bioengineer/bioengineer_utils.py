import math
import torch

from typing import Optional, Tuple
# ============================================================================
# Scoring Window Calculation and Constants (from ProteinGym) - WT-Marginals
# Adapted from ProteinGym Source: https://github.com/OATML-Markslab/ProteinGym/blob/37ea726885452197125f841a33320341d665bc3f/proteingym/baselines/esm/compute_fitness.py#L433
# ============================================================================
# ESM models have a maximum context length of 1024 tokens (including BOS/EOS)
MAX_CONTEXT_LENGTH = 1024

# Window configuration for overlapping scoring strategy
WINDOW_SIZE = 1024  # Total tokens per window (including BOS/EOS)
EDGE_REGION_SIZE = 256  # Positions at window edges that receive sigmoid weighting
SIGMOID_CENTER = 128  # Center point for sigmoid transition (half of edge region)
SIGMOID_SLOPE = 16  # Controls smoothness of sigmoid transition (smaller = smoother)
STEP_SIZE = 511  # Window stride (≈ half window for substantial overlap)
MIN_CENTRAL_OVERLAP = 511  # Minimum overlap required; if less, add central window

# Window edge indices (for 1024-token window with 0-indexing)
# Window spans [0, 1023] where 0=BOS, 1022=last amino acid, 1023=EOS
LAST_VALID_POS = 1022  # Last position before EOS in window
EDGE_TRANSITION_START = LAST_VALID_POS - EDGE_REGION_SIZE  # Position 766
WINDOW_END_IDX = 1023  # Last index in window (inclusive)


def _create_window_weights(window_size: int = WINDOW_SIZE) -> torch.Tensor:
    """
    Create sigmoid-based positional weights for overlapping window scoring.

    Weights are 1.0 in the center and smoothly transition to lower values
    at the edges using a sigmoid function. This prevents discontinuities
    when combining overlapping windows.

    Weight profile:
    - Positions [0, 256): sigmoid ramp up from ~0 to 1
    - Positions [256, 766]: constant weight = 1.0
    - Positions (766, 1023]: sigmoid ramp down from 1 to ~0

    Args:
        window_size: Size of scoring window (default: 1024)

    Returns:
        torch.Tensor of shape [window_size] with positional weights
    """
    weights = torch.ones(window_size)

    # Left edge: sigmoid ramp-up (positions 1 to 256)
    for i in range(1, EDGE_REGION_SIZE + 1):
        # Sigmoid centered at SIGMOID_CENTER (128), slope controlled by SIGMOID_SLOPE (16)
        # At i=1: strongly negative input → weight ≈ 0
        # At i=128: zero input → weight = 0.5
        # At i=256: strongly positive input → weight ≈ 1
        weights[i] = 1 / (1 + math.exp(-(i - SIGMOID_CENTER) / SIGMOID_SLOPE))

    # Right edge: sigmoid ramp-down (positions 766 to 1022)
    for i in range(EDGE_TRANSITION_START, WINDOW_END_IDX):
        # Inverted sigmoid for ramp-down
        # At i=766: strongly negative input → weight ≈ 1
        # At i=894 (766+128): zero input → weight = 0.5
        # At i=1022: strongly positive input → weight ≈ 0
        weights[i] = 1 / (1 + math.exp((i - LAST_VALID_POS + SIGMOID_CENTER) / SIGMOID_SLOPE))

    return weights


def compute_windowed_logits(
        sequence_tokens: torch.Tensor,
        model_forward_fn,
        attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute logits for long sequences using overlapping windowed scoring.

    Strategy (from ProteinGym):
    1. Score sequence with overlapping windows of size 1024
    2. Weight each window's predictions using sigmoid edge weights
    3. Combine weighted predictions from all windows
    4. Normalize by total weight at each position

    This prevents edge effects and provides smooth predictions across
    the entire sequence length.

    Args:
        sequence_tokens: Tokenized sequence [1, seq_len] (includes BOS/EOS)
        model_forward_fn: Function that takes tokens and returns logits
        attention_mask: Optional attention mask [1, seq_len]

    Returns:
        torch.Tensor: Weighted average logits [seq_len, vocab_size]
    """
    batch_size, seq_len = sequence_tokens.shape
    assert batch_size == 1, "Windowed scoring only supports batch_size=1"

    # Get vocab size from a single forward pass
    with torch.no_grad():
        sample_output = model_forward_fn(
            sequence_tokens[:, :min(seq_len, WINDOW_SIZE)],
            attention_mask[:, :min(seq_len, WINDOW_SIZE)] if attention_mask is not None else None
        )
        vocab_size = sample_output.shape[-1]

    # Initialize accumulators for weighted averaging
    device = sequence_tokens.device
    token_probs = torch.zeros((batch_size, seq_len, vocab_size), device=device)
    token_weights = torch.zeros((batch_size, seq_len), device=device)

    # Create window weights (sigmoid-weighted positions)
    weights = _create_window_weights(WINDOW_SIZE).to(device)

    # Initialize left and right window boundaries
    start_left = 0
    end_left = WINDOW_END_IDX  # First window: [0, 1023]

    start_right = seq_len - WINDOW_SIZE  # Last window ends at seq_len-1
    end_right = seq_len - 1

    # Score from both ends moving inward with overlapping windows
    with torch.no_grad():
        while True:
            # ===== Score left window =====
            left_tokens = sequence_tokens[:, start_left:end_left + 1]
            left_mask = attention_mask[:, start_left:end_left + 1] if attention_mask is not None else None

            left_logits = model_forward_fn(left_tokens, left_mask)
            left_log_probs = torch.log_softmax(left_logits, dim=-1)

            # Accumulate weighted predictions
            token_probs[:, start_left:end_left + 1] += left_log_probs * weights.unsqueeze(-1)
            token_weights[:, start_left:end_left + 1] += weights

            # ===== Score right window =====
            right_tokens = sequence_tokens[:, start_right:end_right + 1]
            right_mask = attention_mask[:, start_right:end_right + 1] if attention_mask is not None else None

            right_logits = model_forward_fn(right_tokens, right_mask)
            right_log_probs = torch.log_softmax(right_logits, dim=-1)

            # Accumulate weighted predictions
            token_probs[:, start_right:end_right + 1] += right_log_probs * weights.unsqueeze(-1)
            token_weights[:, start_right:end_right + 1] += weights

            # Check if windows overlap (stopping condition)
            if end_left > start_right:
                break

            # Move windows inward by STEP_SIZE (511 tokens)
            start_left += STEP_SIZE
            end_left += STEP_SIZE
            start_right -= STEP_SIZE
            end_right -= STEP_SIZE

        # ===== Add central window if overlap is insufficient =====
        final_overlap = end_left - start_right + 1

        if final_overlap < MIN_CENTRAL_OVERLAP:
            # Center window around sequence midpoint
            center_start = (seq_len // 2) - (WINDOW_SIZE // 2)
            center_end = center_start + WINDOW_END_IDX

            center_tokens = sequence_tokens[:, center_start:center_end + 1]
            center_mask = attention_mask[:, center_start:center_end + 1] if attention_mask is not None else None

            center_logits = model_forward_fn(center_tokens, center_mask)
            center_log_probs = torch.log_softmax(center_logits, dim=-1)

            # Accumulate central window predictions
            token_probs[:, center_start:center_end + 1] += center_log_probs * weights.unsqueeze(-1)
            token_weights[:, center_start:center_end + 1] += weights

    # Normalize by total weights at each position (weighted average)
    token_probs = token_probs / token_weights.unsqueeze(-1)

    return token_probs[0]  # Return [seq_len, vocab_size]

# ============================================================================
# Get Optimal Window (from ProteinGym) - MASKED-Marginals
# Adapted from ProteinGym Source: https://github.com/OATML-Markslab/ProteinGym/blob/37ea726885452197125f841a33320341d665bc3f/proteingym/utils/scoring_utils.py
# ============================================================================

def get_optimal_window(
        masked_position: int,
        seq_len_with_special: int,
        model_window: int = MAX_CONTEXT_LENGTH
) -> Tuple[int, int]:
    """
    Calculate optimal window boundaries for scoring a single masked position.

    Strategy (from ProteinGym):
    - For short sequences (≤ model_window): use entire sequence
    - For positions in first half: window starts at beginning
    - For positions in last half: window ends at sequence end
    - For middle positions: center window on masked position

    This maximizes context around the masked position for accurate prediction.

    Args:
        masked_position: Index of the masked token (0-indexed, including BOS)
        seq_len_with_special: Total sequence length including BOS/EOS tokens
        model_window: Maximum window size (default: 1024)

    Returns:
        Tuple[start, end): Window boundaries as a half-open interval [start, end)

    Example:
        >>> # Sequence of length 2000, masking position 1000
        >>> get_optimal_window(1000, 2000, 1024)
        (488, 1512)  # Centered window: 1000 ± 512 (end-exclusive)

        >>> # Masking position 100 (near start)
        >>> get_optimal_window(100, 2000, 1024)
        (0, 1024)  # Window starts at beginning (end-exclusive)
    """
    half_window = model_window // 2  # 512 tokens on each side for default value

    # Case 1: Sequence fits entirely within model window
    if seq_len_with_special <= model_window:
        return 0, seq_len_with_special

    # Case 2: Masked position is in the first half - align window to start
    if masked_position < half_window:
        return 0, model_window

    # Case 3: Masked position is in the last half - align window to end
    if masked_position >= seq_len_with_special - half_window:
        return seq_len_with_special - model_window, seq_len_with_special

    # Case 4: Masked position is in the middle - center window on it
    start = masked_position - half_window
    end = masked_position + half_window  # end-exclusive

    # Clamp to valid range
    start = max(0, start)
    end = min(seq_len_with_special, end)

    return start, end