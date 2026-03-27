# Fused Linear Cross Entropy Loss
# Adapted from unsloth-zoo (AGPL-3.0) by Daniel Han-Chen & Unsloth team.
# Standalone version — no external dependencies beyond PyTorch.
#
# Core idea: instead of materializing the full logits matrix (seq_len x vocab_size),
# split hidden_states into chunks and compute matmul + cross_entropy per chunk.
# Uses torch.func.grad_and_value() to compute gradients and loss simultaneously,
# so backward pass just returns pre-computed gradients.

import functools
import math

import torch
import torch.nn.functional as F


def _compute_ce_chunk(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    divisor: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor]]:
    """
    Compute cross entropy loss for a single chunk.

    Args:
        hidden_states: (chunk_size, hidden_dim)
        lm_head_weight: (vocab_size, hidden_dim)
        labels: (chunk_size,)
        divisor: scalar tensor for normalization
    Returns:
        (loss, (unscaled_loss_detached,))
    """
    logits = F.linear(
        hidden_states.to(dtype=lm_head_weight.dtype),
        lm_head_weight,
    )
    vocab_size = lm_head_weight.shape[0]

    loss = F.cross_entropy(
        input=logits.view(-1, vocab_size).float().contiguous(),
        target=labels.view(-1).contiguous(),
        ignore_index=-100,
        reduction="sum",
    )
    loss = loss / divisor
    return loss, (loss.detach(),)


def _get_chunk_count(bsz: int, qlen: int, vocab_size: int, device: torch.device | None = None) -> int:
    """
    Determine number of chunks based on available GPU memory.
    Uses 30% of free GPU memory as target (conservative for grad_and_value overhead).
    """
    try:
        dev = device if device is not None and device.type == "cuda" else 0
        free, total = torch.cuda.mem_get_info(dev)
        free_gb = free / (1024 ** 3) * 0.3
    except Exception:
        free_gb = 1.0  # fallback: assume 1 GB available

    if free_gb < 1e-9:
        free_gb = 1.0

    # Each chunk produces logits of size (chunk_tokens x vocab_size x 4 bytes)
    multiplier = (vocab_size * 4 / (1024 ** 3)) / free_gb / 4
    n_splits = (bsz * qlen) * multiplier
    n_splits = max(round(n_splits) * 4, 1)
    return n_splits


class FusedLinearCrossEntropy(torch.autograd.Function):
    """
    Custom autograd function that computes chunked fused linear + cross entropy.

    Forward: splits hidden_states into chunks, for each chunk computes
    matmul(chunk, lm_head.T) -> cross_entropy using torch.func.grad_and_value().
    This simultaneously gives us the loss AND the gradients w.r.t. hidden_states
    and lm_head_weight, without ever materializing the full logits matrix.

    Backward: returns pre-computed gradients from forward (no recomputation needed).
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        labels: torch.Tensor,
        n_chunks: int,
    ) -> torch.Tensor:
        device = lm_head_weight.device
        bsz, qlen, hd = hidden_states.shape

        # Shift labels for causal LM
        shifted_labels = torch.empty_like(labels, device=device)
        shifted_labels[..., :-1] = labels[..., 1:]
        shifted_labels[..., -1] = -100
        shifted_labels = shifted_labels.view(-1)

        # Divisor for loss normalization
        divisor = (shifted_labels != -100).sum().to(dtype=torch.float32, device=device)
        if divisor.numel() != 1:
            divisor = divisor.ravel()[0]

        # Check what needs gradients
        weight_requires_grad = lm_head_weight.requires_grad

        # Pre-allocate gradient tensors
        grad_hidden = torch.empty_like(hidden_states, device=device)
        # Only allocate grad_weight if lm_head needs gradients (LoRA: it doesn't)
        # Use a tiny placeholder tensor for save_for_backward when not needed
        grad_weight = torch.zeros_like(lm_head_weight, device=device) if weight_requires_grad \
            else torch.empty(0, device=device)

        accumulated_loss = torch.zeros(1, device=device)[0]

        # Split into chunks
        flat_hidden = hidden_states.view(-1, hd)
        chunk_labels = torch.chunk(shifted_labels, n_chunks, dim=0)
        chunk_hidden = torch.chunk(flat_hidden, n_chunks, dim=0)
        chunk_grads = torch.chunk(grad_hidden.view(-1, hd), n_chunks, dim=0)

        for labels_j, hidden_j, grad_j in zip(chunk_labels, chunk_hidden, chunk_grads):
            if weight_requires_grad:
                (chunk_grad_hidden, chunk_grad_weight), (chunk_loss, (unscaled_loss,)) = \
                    torch.func.grad_and_value(
                        _compute_ce_chunk,
                        argnums=(0, 1),
                        has_aux=True,
                    )(hidden_j, lm_head_weight, labels_j, divisor)
                grad_weight.add_(chunk_grad_weight)
                del chunk_grad_weight
            else:
                (chunk_grad_hidden,), (chunk_loss, (unscaled_loss,)) = \
                    torch.func.grad_and_value(
                        _compute_ce_chunk,
                        argnums=(0,),
                        has_aux=True,
                    )(hidden_j, lm_head_weight, labels_j, divisor)

            accumulated_loss.add_(unscaled_loss)
            grad_j[:] = chunk_grad_hidden
            del chunk_grad_hidden, chunk_loss, unscaled_loss

        ctx.save_for_backward(grad_hidden, grad_weight)
        ctx.weight_requires_grad = weight_requires_grad
        return accumulated_loss

    @staticmethod
    def backward(ctx, grad_output):
        grad_hidden, grad_weight = ctx.saved_tensors
        return (
            grad_hidden,
            grad_weight if ctx.weight_requires_grad else None,
            None,
            None,
        )


def fused_linear_cross_entropy_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    n_chunks: int | None = None,
) -> torch.Tensor:
    """
    Compute cross entropy loss without materializing the full logits matrix.

    Splits hidden_states into chunks, computes matmul + CE per chunk,
    and accumulates gradients. Peak VRAM: O(chunk_size * vocab_size)
    instead of O(seq_len * vocab_size).

    Args:
        hidden_states: (batch, seq_len, hidden_dim) — last hidden states
        lm_head_weight: (vocab_size, hidden_dim) — lm_head weight matrix
        labels: (batch, seq_len) — token labels (with -100 for ignored)
        n_chunks: number of chunks (auto-computed from GPU memory if None)
    Returns:
        loss: scalar tensor
    """
    bsz, qlen, hd = hidden_states.shape
    vocab_size = lm_head_weight.shape[0]

    if n_chunks is None:
        n_chunks = _get_chunk_count(bsz, qlen, vocab_size, device=hidden_states.device)

    return FusedLinearCrossEntropy.apply(
        hidden_states,
        lm_head_weight,
        labels,
        n_chunks,
    )
