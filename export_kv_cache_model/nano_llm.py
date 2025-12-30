#!/usr/bin/env python3

from torch import nn

import adaptor
import torch_model


class NanoLLM(nn.Module):
    """
    FunASR-Nano model with LLM decoder.
    
    Architecture:
    - SenseVoice Encoder: audio features extraction
    - Adaptor: converts encoder output to LLM dimension
    - LLM: generates text from fused audio-text embeddings
    """

    def __init__(
        self,
        encoder_output_size: int = 512,
        llm_dim: int = 1024,
        adaptor_config: dict = None,
    ):
        """
        Args:
          encoder_output_size:
            Output size of the audio encoder (default: 512 for SenseVoiceEncoderSmall).
          llm_dim:
            Hidden dimension of the LLM (default: 1024 for Qwen3-0.6B).
          adaptor_config:
            Configuration dict for the adaptor. If None, uses default values.
        """
        super().__init__()
        self.audio_encoder = torch_model.SenseVoiceEncoderSmall()
        self.encoder_output_size = encoder_output_size

        # Adaptor configuration
        if adaptor_config is None:
            adaptor_config = {
                "downsample_rate": 1,
                "encoder_dim": encoder_output_size,
                "llm_dim": llm_dim,
                "ffn_dim": 2048,
                "n_layer": 5,
            }

        self.adaptor = adaptor.Transformer(**adaptor_config)
        self.llm_dim = llm_dim

    def forward(self, x):
        """
        Forward pass for ONNX export (encoder + adaptor only).
        This is the main forward method used for ONNX export.
        
        Args:
          x: (N, T, C) - Audio features
        Returns:
          encoder_out: (N, T, llm_dim) - Audio features in LLM dimension
        """
        encoder_out = self.audio_encoder(x)
        encoder_out = self.adaptor(encoder_out)
        return encoder_out

    def forward_with_fusion(
        self, x, inputs_embeds=None, fbank_beg=None, fake_token_len=None
    ):
        """
        Forward pass with audio-text fusion (for PyTorch inference).
        
        For full LLM inference, you need to:
        1. Get encoder_out from this forward pass
        2. Fuse encoder_out into inputs_embeds at positions specified by fbank_beg
        3. Run LLM with fused inputs_embeds
        
        Args:
          x:
            Audio features of shape (N, T, C)
          inputs_embeds:
            Optional. Text embeddings of shape (N, T_text, llm_dim).
            If provided, audio features will be fused into it.
          fbank_beg:
            Optional. Starting positions in inputs_embeds where audio features should be inserted.
            Shape: (N, num_audio_segments)
          fake_token_len:
            Optional. Length of audio tokens for each segment.
            Shape: (N, num_audio_segments)
            
        Returns:
          - encoder_out: (N, T_audio, llm_dim) - Audio features in LLM dimension
          - If inputs_embeds provided: fused inputs_embeds ready for LLM
        """
        # Audio encoder + adaptor
        encoder_out = self.forward(x)  # (N, T_audio, llm_dim)
        
        # If inputs_embeds is provided, fuse audio features into it
        if inputs_embeds is not None and fbank_beg is not None and fake_token_len is not None:
            batch_size = inputs_embeds.shape[0]
            speech_idx = 0
            
            for batch_idx in range(batch_size):
                for turn_id in range(fbank_beg.shape[1]):
                    fbank_beg_idx = fbank_beg[batch_idx, turn_id].item()
                    if fbank_beg_idx > 0:
                        speech_token_len = fake_token_len[batch_idx, turn_id].item()
                        speech_token = encoder_out[speech_idx, :speech_token_len, :]
                        
                        # Fuse audio features into text embeddings
                        inputs_embeds[
                            batch_idx,
                            fbank_beg_idx : fbank_beg_idx + speech_token_len,
                            :,
                        ] = speech_token
                        
                        speech_idx += 1
            
            return inputs_embeds
        
        return encoder_out

