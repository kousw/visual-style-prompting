from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from diffusers.utils import USE_PEFT_BACKEND


class ReferenceSelfAttnProcessor:
    r"""
    processor for attention map
    """

    def __init__(
        self,
        reference_index=None,
    ) -> None:
        self.reference_index = reference_index
        self.enable_swap = True
        # self.swap_layer_start_index = 23
        self.self_attention_count = 0

    def set_reference_index(self, reference_index):
        self.reference_index = reference_index

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        is_cross = encoder_hidden_states is not None

        if not is_cross:
            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, dim = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = attn.to_k(encoder_hidden_states, *args)  # [batch_size, seq_len, dim]
            value = attn.to_v(
                encoder_hidden_states, *args
            )  # [batch_size, seq_len, dim]

            if self.enable_swap:
                b, s, d = key.shape
                if b == 2:
                    key0, key1 = key.chunk(2, dim=0)
                    key = torch.cat([key1, key1], dim=0)

                    value0, value1 = value.chunk(2, dim=0)
                    value = torch.cat([value1, value1], dim=0)

                elif b == 3:
                    key0, key1, key2 = key.chunk(3, dim=0)
                    key = torch.cat([key2, key2, key1], dim=0)

                    value0, value1, value2 = value.chunk(3, dim=0)
                    value = torch.cat([value2, value2, value1], dim=0)

            query = attn.head_to_batch_dim(
                query
            )  # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
            key = attn.head_to_batch_dim(
                key
            )  # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]
            value = attn.head_to_batch_dim(
                value
            )  # [batch_size, seq_len, dim] -> [batch_size * heads, seq_len, dim // heads]

            attention_probs = attn.get_attention_scores(query, key, attention_mask)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

        else:
            # cross attention

            residual = hidden_states

            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = attn.to_q(hidden_states, *args)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    def get_subject_driven_attention_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype

        alpha = query.shape[2] ** (-0.5)

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            # self subject attention mask
            baddbmm_input = torch.log(attention_mask)
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=alpha,
        )
        del baddbmm_input

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
