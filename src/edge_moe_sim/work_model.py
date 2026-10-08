"""Integer MoE operation and weight-volume formulae."""

from __future__ import annotations

from .schema import ModelConfig


def ceil_div(numerator: int, denominator: int) -> int:
    """Return integer ceiling division after validating the denominator."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator < 0:
        raise ValueError("numerator must be non-negative")
    return (numerator + denominator - 1) // denominator


def router_ops(model: ModelConfig, token_count: int) -> int:
    """Return Router GEMM work in integer operations."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return 2 * token_count * model.hidden_size * model.num_experts


def gate_up_ops(model: ModelConfig, expert_token_count: int) -> int:
    """Return input projection work (legacy gate_up stage), in integer operations."""
    if expert_token_count <= 0:
        raise ValueError("expert_token_count must be positive")
    return 2 * model.input_projection_count * expert_token_count * model.hidden_size * model.intermediate_size


def down_ops(model: ModelConfig, expert_token_count: int) -> int:
    """Return Down GEMM work in integer operations."""
    if expert_token_count <= 0:
        raise ValueError("expert_token_count must be positive")
    return 2 * expert_token_count * model.intermediate_size * model.hidden_size


def nonlinear_elements(model: ModelConfig, expert_token_count: int) -> int:
    """Return the number of nonlinear elements."""
    if expert_token_count <= 0:
        raise ValueError("expert_token_count must be positive")
    return expert_token_count * model.intermediate_size


def gate_up_weight_bytes(model: ModelConfig) -> int:
    """Return input projection weight bytes, rounded upward per stage."""
    return ceil_div(model.input_projection_count * model.hidden_size * model.intermediate_size * model.weight_bits, 8)


def down_weight_bytes(model: ModelConfig) -> int:
    """Return Down weight storage in bytes, rounded upward."""
    return ceil_div(model.hidden_size * model.intermediate_size * model.weight_bits, 8)


def expert_weight_bytes(model: ModelConfig) -> int:
    """Return total Gate-Up plus Down weight storage in bytes."""
    return gate_up_weight_bytes(model) + down_weight_bytes(model)
