"""InterACT Policy

Implementation of InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation (https://arxiv.org/abs/2409.07914).
"""

import math
from collections import deque
from itertools import chain
from typing import Callable

import einops
import copy
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.common.policies.interact.configuration_interact import InterACTConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from typing import Optional


class InterACTPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "interact"],
):
    """
    InterACT policy as per InterACT: Inter-dependency Aware Action Chunking with Hierarchical Attention Transformers for Bimanual Manipulation (paper: https://arxiv.org/abs/2409.07914)
    """

    name = "interact"

    def __init__(
        self,
        config: InterACTConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = InterACTConfig()
        self.config: InterACTConfig = config

        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.model = InterACT(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = InterACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)

        # If we are doing temporal ensembling, do online updates where we keep track of the number of actions
        # we are ensembling over.
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.model(batch)  # (batch_size, chunk_size, action_dim)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.model(batch)[:, : self.config.n_action_steps]

            # TODO(rcadene): make _forward return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        actions_hat = self.model(batch)

        l1_loss = (
            F.l1_loss(batch["action"], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        loss_dict["loss"] = l1_loss

        return loss_dict


class InterACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://arxiv.org/abs/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[:i+1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class InterACT(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.config = config
        # BERT style VAE encoder with input tokens [cls, robot_state, *action_sequence].
        # The cls token forms parameters of the latent's distribution (like this [*means, *log_variances]).
        self.use_robot_state = "observation.state" in config.input_shapes
        self.use_images = any(k.startswith("observation.image") for k in config.input_shapes)
        self.use_env_state = "observation.environment_state" in config.input_shapes
        self.cls_input_arm1 = nn.Embedding(1, config.dim_model)
        self.cls_input_arm2 = nn.Embedding(1, config.dim_model)
        self.cls_input_av = nn.Embedding(1, config.dim_model)
        self.cls_input_image = nn.Embedding(1, config.dim_model)

        num_cls_tokens_left = config.num_cls_tokens_arm
        num_cls_tokens_right = config.num_cls_tokens_arm
        num_cls_tokens_av = config.num_cls_tokens_arm
        num_cls_tokens_image = config.num_cls_tokens_image
        dim_model = config.dim_model

        # CLS tokens for left arm, right arm, av arm and images (learnable parameters)
        cls_input_arm1 = self.cls_input_arm1.to(device='cuda:0')
        cls_input_arm2 = self.cls_input_arm2.to(device='cuda:0')
        cls_input_av = self.cls_input_av.to(device='cuda:0')
        cls_input_image = self.cls_input_image.to(device='cuda:0')



        cls_input_arm1 = cls_input_arm1.weight
        self.cls_token_left = cls_input_arm1.repeat(num_cls_tokens_left, 1)  # (num_cls_left, 1, dim_model)
        cls_input_arm2 = cls_input_arm2.weight
        self.cls_token_right = cls_input_arm2.repeat(num_cls_tokens_right, 1)  # (num_cls_right, 1, dim_model)
        cls_input_av = cls_input_av.weight
        self.cls_token_av = cls_input_av.repeat(num_cls_tokens_av, 1)  # (num_cls_right, 1, dim_model)
        cls_input_image = cls_input_image.weight
        self.cls_token_image = cls_input_image.repeat(num_cls_tokens_image, 1)  # (num_cls_image, 1, dim_model)

        # self.cls_token_left = nn.Parameter(torch.randn(num_cls_tokens_left, dim_model))  # (num_cls_left, dim_model)
        # self.cls_token_right = nn.Parameter(torch.randn(num_cls_tokens_right, dim_model))  # (num_cls_right, dim_model)
        # self.cls_token_image = nn.Parameter(torch.randn(num_cls_tokens_image, dim_model))  # (num_cls_image, dim_model)

        # Backbone for image feature extraction.
        if self.use_images:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # Note: The assumption here is that we are using a ResNet model (and hence layer4 is the final
            # feature map).
            # Note: The forward method of this returns a dict: {"feature_map": output}.
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer (acts as VAE decoder when training with the variational objective).
        self.encoder = InterACT_Heirarchical_Attention_Encoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer encoder input projections. The tokens will be structured like
        # [latent, (robot_state), (env_state), (image_feature_map_pixels)].

        self.encoder_robot_state_input_proj = nn.Linear(1, 512)

 # 7 + 7 DoF for left and right arms
        if self.use_env_state:
            self.encoder_env_state_input_proj = nn.Linear(
                config.input_shapes["observation.environment_state"][0], config.dim_model
            )
        # self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.use_images:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer encoder positional embeddings.
        n_1d_tokens = 33  # 7 + 7 + 5 + 7 + 7 ?
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.use_images:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer decoder.
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Final action regression head on the output of the transformer's decoder.
        # self.action_head_arm1 = nn.Linear(config.dim_model, config.output_shapes["action"][0]//3)
        # self.action_head_arm2 = nn.Linear(config.dim_model, config.output_shapes["action"][0]//3)
        # self.action_head_av = nn.Linear(config.dim_model, config.output_shapes["action"][0]//3)
        self.action_head = nn.Linear(config.dim_model, config.output_shapes["action"][0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        batch_size = (
            batch["observation.images"]
            if "observation.images" in batch
            else batch["observation.environment_state"]
        ).shape[0]

        # Number of cls tokens for arms and images from config
        num_cls_tokens_arm = self.config.num_cls_tokens_arm
        num_cls_tokens_image = self.config.num_cls_tokens_image

        # Prepare cls tokens for left arm, right arm, and image segments
        cls_tokens_left = self.cls_token_left.unsqueeze(0).repeat(batch_size, 1, 1).to(device='cuda:0')  # (B, num_cls_left, D)
        cls_tokens_right = self.cls_token_right.unsqueeze(0).repeat(batch_size, 1, 1).to(device='cuda:0')   # (B, num_cls_right, D)
        cle_tokens_av = self.cls_token_middle.unsqueeze(0).repeat(batch_size, 1, 1).to(device='cuda:0')
        cls_tokens_image = self.cls_token_image.unsqueeze(0).repeat(batch_size, 1, 1).to(device='cuda:0')   # (B, num_cls_image, D)

        encoder_in_tokens = []
        encoder_in_pos_embed = []

        # Robot state tokens for left arm
        left_arm_state = batch["observation.state"][:, :7]  # Assuming the first 7 are for the left arm
        left_arm_state_proj = self.encoder_robot_state_input_proj(left_arm_state.unsqueeze(-1))  # (B, 7, 512)
        
        right_arm_state = batch["observation.state"][:, 7:14]  # Assuming the next 7 are for the right arm
        right_arm_state_proj = self.encoder_robot_state_input_proj(right_arm_state.unsqueeze(-1))  # (B, 7, 512)

        av_arm_state =  batch["observation.state"][:, 14:]
        av_arm_state_proj = self.encoder_robot_state_input_proj(av_arm_state.unsqueeze(-1))

        # CLS tokens for the left arm are already in (B, num_cls_tokens_arm, 512)
        encoder_in_tokens.append(torch.cat([cls_tokens_left, left_arm_state_proj, cls_tokens_right, right_arm_state_proj], dim=1))  # (B, num_cls_tokens_arm + 7, 512)
        # encoder_in_tokens.append(torch.cat([], dim=1))

        # Positional embeddings for the left arm (total length is num_cls_tokens_arm + 7)
        encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[:num_cls_tokens_arm + 7].unsqueeze(1))  # (num_cls_tokens_arm + 7, 1, 512)

        # CLS tokens for the right arm are already in (B, num_cls_tokens_arm, 512)
          # (B, num_cls_tokens_arm + 7, 512)

        # Positional embeddings for the right arm (total length is num_cls_tokens_arm + 7)
        start_idx = num_cls_tokens_arm + 7  # Adjust the starting index based on the previous positional embeddings
        encoder_in_pos_embed.append(self.encoder_1d_feature_pos_embed.weight[start_idx:start_idx+num_cls_tokens_arm+7].unsqueeze(1))  # (num_cls_tokens_arm + 7, 1, 512)

    # Environment state token (if applicable)

        # Camera observation features and positional embeddings
        if self.use_images:
            all_cam_features = []
            all_cam_pos_embeds = []

            for cam_index in range(batch["observation.images"].shape[1]):  # Loop over cameras
                cam_features = self.backbone(batch["observation.images"][:, cam_index])["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, C, h, w)
                all_cam_features.append(cam_features)
                all_cam_pos_embeds.append(cam_pos_embed)

            # Concatenate camera observation feature maps and positional embeddings along the width dimension
            all_cam_features = torch.cat(all_cam_features, axis=-1)
            all_cam_features = einops.rearrange(all_cam_features, "b c h w -> (h w) b c")
            all_cam_features = torch.cat([einops.rearrange(cls_tokens_image, "b s d -> s b d"), all_cam_features], dim=0)

            encoder_in_tokens.extend(all_cam_features)
            all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1)
            all_cam_pos_embeds = einops.rearrange(all_cam_pos_embeds, "b c h w -> (h w) b c")
            start_idx = start_idx + num_cls_tokens_arm+7
            encoder_in_pos_embed.extend(torch.cat([self.encoder_1d_feature_pos_embed.weight[start_idx:start_idx+1].unsqueeze(1), all_cam_pos_embeds], dim=0))
            

        # Stack all tokens and positional embeddings along the sequence dimension

        for i in range(len(encoder_in_tokens)):
            if encoder_in_tokens[i].dim() == 2:  # Check if it's [8, 512]
                encoder_in_tokens[i] = encoder_in_tokens[i].unsqueeze(1)

        for i in range(len(encoder_in_pos_embed)):
            if encoder_in_pos_embed[i].dim() == 2:  # If shape is (1, 512)
                encoder_in_pos_embed[i] = encoder_in_pos_embed[i].unsqueeze(1)
                
        encoder_in_tokens = torch.cat(encoder_in_tokens, axis=1)
        encoder_in_pos_embed = torch.cat(encoder_in_pos_embed, axis=0)

        # Forward pass through transformer encoder
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        # Prepare decoder input (zero-initialized)
        # decoder_in = torch.zeros(
        #     (self.config.chunk_size, batch_size, self.config.dim_model),
        #     dtype=encoder_in_pos_embed.dtype,
        #     device=encoder_in_pos_embed.device,
        # )
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

  
        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions



#####################################################################################################################################
# DEFINE ENCODER AND DECODER
#####################################################################################################################################

# class InterACT_Heirarchical_Attention_Encoder(nn.Module):
#     ## COMBINATION OF SEGMENT-WISE and CROSS-SEGMENT ATTENTION

#     def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
#         super().__init__()
#         num_layers = config.n_encoder_blocks

#         self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
#         self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

#     def forward(
#         self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
#     ) -> Tensor:
#         for layer in self.layers:
#             x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
#         x = self.norm(x)
#         return x

class InterACT_Heirarchical_Attention_Encoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.num_blocks = config.num_blocks
        self.pos_encoder = PositionalEncoding(config.dim_model, 5000) 

        # Define segment-wise and cross-segment encoders
        self.segment_encoders = nn.ModuleList([
            SegmentWiseEncoder(config) for _ in range(self.num_blocks)
        ])
        self.cross_segment_encoders = nn.ModuleList([
            CrossSegmentEncoder(config)
            for _ in range(self.num_blocks)
        ])

        self.arm_cls = config.num_cls_tokens_arm
        self.cam_cls = config.num_cls_tokens_image

    def forward(self, segments, pos_embed):
        # Ensure that pos_embed is applied to the input segments
        segments = einops.rearrange(segments, "b s d -> s b d")  # (num_segments, batch_size, dim_model)
        # segments = segments + pos_embed
        segments = self.pos_encoder(segments)
        # Split the segments into left arm, right arm, and camera parts
        segment_1 = segments[:self.arm_cls + 7]  # Left arm (cls tokens + state)
        segment_2 = segments[self.arm_cls + 7:self.arm_cls * 2 + 14]  # Right arm (cls tokens + state)
        segment_3 = segments[self.arm_cls * 2 + 14:]  # Camera (cls tokens + image features)

        # Apply segment-wise and cross-segment encoders block-wise
        for i in range(self.num_blocks):
            # Segment-wise encoding for each segment
            updated_segment_1 = self.segment_encoders[i](segment_1)
            updated_segment_2 = self.segment_encoders[i](segment_2)
            updated_segment_3 = self.segment_encoders[i](segment_3)

            # Cross-segment encoding using the CLS tokens from each segment
            updated_cls_tokens = self.cross_segment_encoders[i](
                torch.cat([
                    updated_segment_1[:self.arm_cls], 
                    updated_segment_2[:self.arm_cls], 
                    updated_segment_3[:self.cam_cls]
                ], dim=0)
            )

            # Update the segments with new CLS tokens
            segment_1 = torch.cat([updated_cls_tokens[:self.arm_cls], updated_segment_1[self.arm_cls:]], dim=0)  # Left arm
            segment_2 = torch.cat([updated_cls_tokens[self.arm_cls:self.arm_cls * 2], updated_segment_2[self.arm_cls:]], dim=0)  # Right arm
            segment_3 = torch.cat([updated_cls_tokens[self.arm_cls * 2:], updated_segment_3[self.cam_cls:]], dim=0)  # Cameras
        
        # segment_1 = torch.cat([updated_cls_tokens[:self.arm_cls], segments[:self.arm_cls + 7]], dim=0)  # Left arm
        # segment_2 = torch.cat([updated_cls_tokens[self.arm_cls:self.arm_cls * 2], segments[self.arm_cls + 7:self.arm_cls * 2 + 14]], dim=0)  # Right arm
        # segment_3 = torch.cat([updated_cls_tokens[self.arm_cls * 2:], segments[self.arm_cls * 2 + 14:]], dim=0)  # Cameras

        # Concatenate all segments again
        segments = torch.cat([segment_1, segment_2, segment_3], dim=0)

        return segments

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=3, device=None):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        # Ensure positional encoding is on the same device as x
        if self.encoding.device != x.device:
            self.encoding = self.encoding.to(x.device)
        return x + self.encoding[:, :x.size(1)].detach()



class SegmentWiseEncoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        # Multi-head self-attention with positional encoding
        if self.pre_norm:
            x = self.norm1(x)
        # q = k = x if pos_embed is None else x + pos_embed
        q = k = x #  self.pos_encoder(x)
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class CrossSegmentEncoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        # Multi-head self-attention with positional encoding
        if self.pre_norm:
            x = self.norm1(x)
        # q = k = x if pos_embed is None else x + pos_embed
        q = k = x #  self.pos_encoder(x)
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x



class ACTEncoderLayer(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.pos_encoder = PositionalEncoding(config.dim_model, 5000)
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        # q = k = x if pos_embed is None else x + pos_embed
        q = k = self.pos_encoder(x)
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: InterACTConfig):
        super().__init__()
        self.pos_encoder = PositionalEncoding(config.dim_model, 5000) 
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        # q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        q = k = self.pos_encoder(x)
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x
    
class Multi_Arm_Decoder(nn.Module):
    def __init__(self, config: InterACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.pre_num_layers = config.n_pre_decoder_layers
        self.post_num_layers = config.n_post_decoder_layers
        self.num_layers = self.pre_num_layers + self.post_num_layers
        self.arm1_layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(self.num_layers)])
        self.arm2_layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(self.num_layers)])

        self.num_cls_tokens_arm = config.num_cls_tokens_arm
        self.num_cls_tokens_image = config.num_cls_tokens_image

        self.info_sharing_block = ACTEncoderLayer(config)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        
        arm1_cls = encoder_out[:self.num_cls_tokens_arm]  # Memory for arm1's cls tokens and state
        arm1_chunk = encoder_out[self.num_cls_tokens_arm:self.num_cls_tokens_arm + 7]  # Memory for arm1's chunk
        arm2_cls = encoder_out[self.num_cls_tokens_arm + 7:self.num_cls_tokens_arm*2 + 7]  # Memory for arm2's cls tokens and state
        arm2_chunk = encoder_out[self.num_cls_tokens_arm*2 + 7:self.num_cls_tokens_arm*2 + 14]  # Memory for arm2's chunk
        obs_chunk = encoder_out[self.num_cls_tokens_arm*2 + 14:]  # Memory for the observation tokens (e.g., images)

        # Memory1 for arm1 decoder: arm1_cls, arm1_chunk, and obs_chunk
        memory1 = torch.cat([arm2_cls, arm1_chunk, obs_chunk], dim=0)

        # Memory2 for arm2 decoder: arm2_cls, arm2_chunk, and obs_chunk
        memory2 = torch.cat([arm1_cls, arm2_chunk, obs_chunk], dim=0)
        
        for i in range(self.pre_num_layers):
            x1 = self.arm1_layers[i](
                x1, memory1, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )


            x2 = self.arm2_layers[i](
                x2, memory2, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        
        # Info sharing block
            combined = torch.cat((x1, x2), dim=0)
            combined = self.info_sharing_block(
                combined
            )

            x1, x2 = torch.split(combined, x1.size(0), dim=0)

        if self.norm1 is not None:
            x1 = self.norm1(x1)
            x2 = self.norm1(x2)
            
            
        for i in range(self.post_num_layers):
            x1 = self.arm1_layers[self.pre_num_layers + i](
                x1, memory1, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
            x2 = self.arm2_layers[self.pre_num_layers + i](
                x2, memory2, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )

        if self.norm2 is not None:
            x1 = self.norm2(x1)
            x2 = self.norm2(x2)
        
        return x1, x2

class InterACT_Multi_Arm_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Unpack config parameters
        num_layers = config.n_decoder_layers
        self.num_layers = num_layers
        dim_model = config.dim_model
        nhead = config.n_heads
        dim_feedforward = config.dim_feedforward
        dropout = config.dropout
        activation = config.feedforward_activation

        self.norm = nn.LayerNorm(dim_model)

        # Create a transformer decoder layer using the unpacked config
        decoder_layer = ACTDecoderLayer(
            config
        )
        
        # Define decoders for arm1 and arm2
        self.arm1_decoder = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.arm2_decoder = nn.ModuleList([decoder_layer for _ in range(num_layers)])

        # Self-attention layer for combining the arms' decoder outputs
        self.self_attn = nn.MultiheadAttention(dim_model, num_heads=nhead)

        # Feed-forward layers for the combined output
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def forward(self, tgt1, tgt2, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        arm1_cls = memory[:3]  # Memory for arm1's cls tokens and state
        arm1_chunk = memory[3:10]  # Memory for arm1's chunk
        arm2_cls = memory[10:13]  # Memory for arm2's cls tokens and state
        arm2_chunk = memory[13:20]  # Memory for arm2's chunk
        obs_chunk = memory[20:]  # Memory for the observation tokens (e.g., images)

        # Memory1 for arm1 decoder: arm1_cls, arm1_chunk, and obs_chunk
        memory1 = torch.cat([arm2_cls, arm1_cls, arm1_chunk, obs_chunk], dim=0)

        # Memory2 for arm2 decoder: arm2_cls, arm2_chunk, and obs_chunk
        memory2 = torch.cat([arm1_cls, arm2_cls, arm2_chunk, obs_chunk], dim=0)

        # Initialize intermediate outputs for storing decoder outputs
        intermediate_outputs1 = []
        intermediate_outputs2 = []

        for layer in range(self.num_layers):
            # Apply the arm1 decoder
            tgt1 = self.arm1_decoder[layer](
                tgt1, memory1
            )

            # Apply the arm2 decoder
            tgt2 = self.arm2_decoder[layer](
                tgt2, memory2
            )

            if layer == self.num_layers - 1:
                # Apply self-attention to combine the outputs from both arms
                combined = torch.cat((tgt1, tgt2), dim=0)
                q = k = combined
                combined2, _ = self.self_attn(q, k, value=combined)

                # Residual connections and normalization
                combined = combined + self.dropout1(combined2)
                combined = self.norm1(combined)

                combined2 = self.linear2(self.dropout(self.activation(self.linear1(combined))))
                combined = combined + self.dropout2(combined2)
                combined = self.norm2(combined)

                # Split the combined output back into arm1 and arm2 parts
                tgt1, tgt2 = torch.split(combined, tgt1.size(0), dim=0)

        return tgt1, tgt2



def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need.

    Args:
        num_positions: Number of token positions required.
    Returns: (num_positions, dimension) position embeddings (the first dimension is the batch dimension).

    """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need.

    The variation is that the position indices are normalized in [0, 2π] (not quite: the lower bound is 1/H
    for the vertical direction, and 1/W for the horizontal direction.
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: The desired dimension of the embeddings.
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # Inverse "common ratio" for the geometric progression in sinusoid frequencies.
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        # Note: These are like range(1, H+1) and range(1, W+1) respectively, but in most implementations
        # they would be range(0, H) and range(0, W). Keeping it at as is to match the original code.
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "Normalize" the position index such that it ranges in [0, 2π].
        # Note: Adding epsilon on the denominator should not be needed as all values of y_embed and x_range
        # are non-zero by construction. This is an artifact of the original code.
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        # pos_embed_x and pos_embed_y are (1, H, W, C // 2).
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
