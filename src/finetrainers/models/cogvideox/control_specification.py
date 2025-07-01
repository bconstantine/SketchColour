import functools
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import safetensors
import torch
from accelerate import init_empty_weights
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from PIL.Image import Image
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer

from finetrainers.data import VideoArtifact
from finetrainers.logging import get_logger
from finetrainers.models.modeling_utils import ControlModelSpecification
from finetrainers.models.modeling_utils import ModelSpecification
from finetrainers.models.utils import DiagonalGaussianDistribution, _expand_conv2d_with_zeroed_weights
from finetrainers.processors import ProcessorMixin, T5Processor
from finetrainers.typing import ArtifactType, SchedulerType
from finetrainers.utils import _enable_vae_memory_optimizations, get_non_null_items, safetensors_torch_save_function
from finetrainers.patches.dependencies.diffusers.control import control_channel_concat
from .utils import prepare_rotary_positional_embeddings
from .base_specification import CogVideoXLatentEncodeProcessor
#from finetrainers.trainer.control_trainer.config import FrameConditioningType

if TYPE_CHECKING:
    from finetrainers.trainer.control_trainer.config import FrameConditioningType

logger = get_logger()

def check_tensor_range(tensor: torch.Tensor, name: str):
    """
    Prints the min/max of `tensor`. 
    Flags if there are values < 0 or > 1.
    If tensor is None, simply notifies that fact.
    """
    if tensor is None:
        print(f"{name} is None")
        return

    # Compute min and max
    min_val = float(tensor.min().item())
    max_val = float(tensor.max().item())
    print(f"{name} → min: {min_val:.6f}, max: {max_val:.6f}")

    # Check if any values are out of [0,1]
    if min_val < 0:
        print(f"  ↳ {name} has values < 0")
    if max_val > 1:
        print(f"  ↳ {name} has values > 1")
    if 0.0 <= min_val and max_val <= 1.0:
        print(f"  ↳ {name} is fully within [0, 1]\n")
    else:
        print(f"  ↳ {name} is not normalized to [0, 1]\n")

class CogVideoXControlModelSpecification(ControlModelSpecification):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "THUDM/CogVideoX-5b", #THUDM/CogVideoX-5b-I2V
        tokenizer_id: Optional[str] = None,
        text_encoder_id: Optional[str] = None,
        transformer_id: Optional[str] = None,
        vae_id: Optional[str] = None,
        text_encoder_dtype: torch.dtype = torch.bfloat16,
        transformer_dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        condition_model_processors: List[ProcessorMixin] = None,
        latent_model_processors: List[ProcessorMixin] = None,
        image_model_processors: List[ProcessorMixin] = None, 
        control_model_processors: List[ProcessorMixin] = None,
        image_pipeline: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            tokenizer_id=tokenizer_id,
            text_encoder_id=text_encoder_id,
            transformer_id=transformer_id,
            vae_id=vae_id,
            text_encoder_dtype=text_encoder_dtype,
            transformer_dtype=transformer_dtype,
            vae_dtype=vae_dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

        if condition_model_processors is None:
            condition_model_processors = [T5Processor(["encoder_hidden_states", "prompt_attention_mask"])]
        if latent_model_processors is None:
            latent_model_processors = [CogVideoXLatentEncodeProcessor(["latents"])]
        
        if image_pipeline and image_model_processors is None:
            image_model_processors = [CogVideoXLatentEncodeProcessor(["image_latents"])]

        if control_model_processors is None:
            control_model_processors = [CogVideoXLatentEncodeProcessor(["control_latents"])]


        self.condition_model_processors = condition_model_processors
        self.latent_model_processors = latent_model_processors
        self.image_model_processors = image_model_processors
        self.control_model_processors = control_model_processors
        self.image_pipeline = image_pipeline

    @property
    def control_injection_layer_name(self) -> str:
        return "patch_embed.proj"

    @property
    def _resolution_dim_keys(self):
        return {"latents": (1, 3, 4)}

    def load_condition_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.tokenizer_id is not None:
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, **common_kwargs)
        else:
            tokenizer = T5Tokenizer.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="tokenizer", **common_kwargs
            )

        if self.text_encoder_id is not None:
            text_encoder = AutoModel.from_pretrained(
                self.text_encoder_id, torch_dtype=self.text_encoder_dtype, **common_kwargs
            )
        else:
            text_encoder = T5EncoderModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="text_encoder",
                torch_dtype=self.text_encoder_dtype,
                **common_kwargs,
            )

        return {"tokenizer": tokenizer, "text_encoder": text_encoder}

    def load_latent_models(self) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.vae_id is not None:
            vae = AutoencoderKLCogVideoX.from_pretrained(self.vae_id, torch_dtype=self.vae_dtype, **common_kwargs)
        else:
            vae = AutoencoderKLCogVideoX.from_pretrained(
                self.pretrained_model_name_or_path, subfolder="vae", torch_dtype=self.vae_dtype, **common_kwargs
            )

        return {"vae": vae}

    def load_diffusion_models(self, new_in_features) -> Dict[str, torch.nn.Module]:
        common_kwargs = {"revision": self.revision, "cache_dir": self.cache_dir}

        if self.transformer_id is not None:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.transformer_id, torch_dtype=self.transformer_dtype, **common_kwargs
            )
        else:
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=self.transformer_dtype,
                **common_kwargs,
            )

        transformer.patch_embed.proj = _expand_conv2d_with_zeroed_weights(
            transformer.patch_embed.proj, new_in_channels=new_in_features
        )

        scheduler = CogVideoXDDIMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="scheduler", **common_kwargs
        )

        return {"transformer": transformer, "scheduler": scheduler}

    def load_pipeline(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        transformer: Optional[CogVideoXTransformer3DModel] = None,
        vae: Optional[AutoencoderKLCogVideoX] = None,
        scheduler: Optional[CogVideoXDDIMScheduler] = None,
        enable_slicing: bool = False,
        enable_tiling: bool = False,
        enable_model_cpu_offload: bool = False,
        training: bool = False,
        #TODO (Bryan): Remove the last args
        force_every_validation_to_be_bfloat16:bool=False,
        **kwargs,
    ) -> Union[CogVideoXPipeline, CogVideoXImageToVideoPipeline]:
        components = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
        }
        components = get_non_null_items(components)
        if self.image_pipeline:
            pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                self.pretrained_model_name_or_path, 
                **components, 
                revision=self.revision,
                low_cpu_mem_usage=True,
                cache_dir=self.cache_dir, 
            )
        else:
            pipe = CogVideoXPipeline.from_pretrained(
                self.pretrained_model_name_or_path, **components, 
                revision=self.revision, cache_dir=self.cache_dir
            )
        pipe.text_encoder.to(self.text_encoder_dtype)
        pipe.vae.to(self.vae_dtype)

        _enable_vae_memory_optimizations(pipe.vae, enable_slicing, enable_tiling)
        if not training:
            #TODO (Bryan): Uncomment this line for stability test
            if force_every_validation_to_be_bfloat16:
                logger.info("Entering transforming transformers dtype to: ")
                logger.info(self.transformer_dtype)
                pipe.transformer.to(self.transformer_dtype)
            pass
        if enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        return pipe

    @torch.no_grad()
    def prepare_conditions(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        caption: str,
        max_sequence_length: int = 226,
        **kwargs,
    ) -> Dict[str, Any]:
        conditions = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "caption": caption,
            "max_sequence_length": max_sequence_length,
            **kwargs,
        }
        input_keys = set(conditions.keys())
        conditions = super().prepare_conditions(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}
        conditions.pop("prompt_attention_mask", None)
        return conditions

    @torch.no_grad()
    def prepare_latents(
        self,
        vae: AutoencoderKLCogVideoX,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        control_image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        common_kwargs = {
            "vae": vae,
            "generator": generator,
            "compute_posterior": compute_posterior,
            **kwargs,
        }
        ##TODO(Bryan): Make sure the Video is not None, hardcode the image to be None
        #check_tensor_range(image, "image")
        #check_tensor_range(video, "video")
        assert video is not None
        assert image is not None
        
        conditions = {"image": None, "video": video, **common_kwargs}
        input_keys = set(conditions.keys())
        conditions = super().prepare_latents(**conditions)
        conditions = {k: v for k, v in conditions.items() if k not in input_keys}

        #TODO(Bryan): This only applies to I2v
        image_conditions = {"image": image, "video": None, **common_kwargs}
        input_keys = set(image_conditions.keys())
        image_conditions = ControlModelSpecification.prepare_latents(
            self, self.image_model_processors, **image_conditions
        )
        image_conditions = {k: v for k, v in image_conditions.items() if k not in input_keys}

        ##TODO(Bryan): Change the Control for multiple control support, not only videos
        if control_video ==None:
            control_video = kwargs["control"]

        control_conditions = {"image": control_image, "video": control_video, **common_kwargs}
        input_keys = set(control_conditions.keys())
        control_conditions = ControlModelSpecification.prepare_latents(
            self, self.control_model_processors, **control_conditions
        )
        control_conditions = {k: v for k, v in control_conditions.items() if k not in input_keys}

        return {**control_conditions, **image_conditions, **conditions}

    def forward(
        self,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDDIMScheduler,
        condition_model_conditions: Dict[str, torch.Tensor],
        latent_model_conditions: Dict[str, torch.Tensor],
        sigmas: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        compute_posterior: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents
        # Just hardcode for now. In Diffusers, we will refactor such that RoPE would be handled within the model itself.
        VAE_SPATIAL_SCALE_FACTOR = 8
        rope_base_height = self.transformer_config.sample_height * VAE_SPATIAL_SCALE_FACTOR
        rope_base_width = self.transformer_config.sample_width * VAE_SPATIAL_SCALE_FACTOR
        patch_size = self.transformer_config.patch_size
        patch_size_t = getattr(self.transformer_config, "patch_size_t", None)

        if compute_posterior:
            latents = latent_model_conditions.pop("latents")
            control_latents = latent_model_conditions.pop("control_latents")
            #TODO (Bryan): This is only for I2V
            image_latents = latent_model_conditions.pop("image_latents")
        else:
            posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("latents"), _dim=2)
            latents = posterior.sample(generator=generator)
            control_posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("control_latents"), _dim=2)
            control_latents = control_posterior.sample(generator=generator)
            #TODO (Bryan): This is only for I2V
            image_posterior = DiagonalGaussianDistribution(latent_model_conditions.pop("image_latents"), _dim=2)
            image_latents = image_posterior.sample(generator=generator)
            del posterior, control_posterior, image_posterior

        if not getattr(self.vae_config, "invert_scale_latents", False):
            #Cog Video enter here with 0.7 scaling factor
            latents = latents * self.vae_config.scaling_factor
            control_latents = control_latents * self.vae_config.scaling_factor
            #TODO (Bryan): This is only for I2V
            image_latents = image_latents * self.vae_config.scaling_factor


        if patch_size_t is not None:
            #CogVideoX Do not enter here
            latents = self._pad_frames(latents, patch_size_t)
            control_latents = self._pad_frames(control_latents, patch_size_t)
            #TODO (Bryan): This is only for I2V
            image_latents = self._pad_frames(image_latents, patch_size_t)

        timesteps = (sigmas.flatten() * 1000.0).long()
        
        #TODO (Bryan): Make the image latents shape the same as the control latents shape
        control_latents_frames = control_latents.shape[1]
        B, F, C, H, W = image_latents.shape
        if F < control_latents_frames:
            # how many frames to pad
            pad_frames = control_latents_frames - F
            # create zeros of shape (B, pad_frames, C, H, W)
            padding = torch.zeros(
                B, pad_frames, C, H, W,
                device=image_latents.device,
                dtype=image_latents.dtype
            )
            # concatenate along the frame dimension
            image_latents = torch.cat([image_latents, padding], dim=1)
        
        #TODO(Bryan): This is only for I2V, make sure every size is the same
        assert image_latents.shape == latents.shape == control_latents.shape

        noise = torch.zeros_like(latents).normal_(generator=generator)
        noisy_latents = scheduler.add_noise(latents, noise, timesteps)

        
        batch_size, num_frames, num_channels, height, width = latents.shape
        masked_image_latents = apply_frame_conditioning_on_latents(
            image_latents, 
            num_frames, 
            channel_dim=2,
            frame_dim=1,
            frame_conditioning_type="index", 
            frame_conditioning_index=0, 
            concatenate_mask=self.frame_conditioning_concatenate_mask,
        )
        control_latents = apply_frame_conditioning_on_latents(
            control_latents,
            num_frames,
            channel_dim=2,
            frame_dim=1,
            frame_conditioning_type=self.frame_conditioning_type,
            frame_conditioning_index=self.frame_conditioning_index,
            concatenate_mask=self.frame_conditioning_concatenate_mask,
        )

        transformer_latents = torch.cat([noisy_latents, masked_image_latents, control_latents], dim=2) #num channels dimension
        ofs_emb = (
            None
            if getattr(self.transformer_config, "ofs_embed_dim", None) is None
            else latents.new_full((batch_size,), fill_value=2.0)
        )
        image_rotary_emb = (
            prepare_rotary_positional_embeddings(
                height=height * VAE_SPATIAL_SCALE_FACTOR,
                width=width * VAE_SPATIAL_SCALE_FACTOR,
                num_frames=num_frames,
                vae_scale_factor_spatial=VAE_SPATIAL_SCALE_FACTOR,
                patch_size=patch_size,
                patch_size_t=patch_size_t,
                attention_head_dim=self.transformer_config.attention_head_dim,
                device=transformer.device,
                base_height=rope_base_height,
                base_width=rope_base_width,
            )
            if self.transformer_config.use_rotary_positional_embeddings
            else None
        )

        latent_model_conditions["hidden_states"] = transformer_latents.to(latents)
        latent_model_conditions["image_rotary_emb"] = image_rotary_emb
        latent_model_conditions["ofs"] = ofs_emb
        velocity = transformer(
            **latent_model_conditions,
            **condition_model_conditions,
            timestep=timesteps,
            return_dict=False,
        )[0]
        # For CogVideoX, the transformer predicts the velocity. The denoised output is calculated by applying the same
        # code paths as scheduler.get_velocity(), which can be confusing to understand.
        pred = scheduler.get_velocity(velocity, noisy_latents, timesteps)
        target = latents

        return pred, target, sigmas

    def validation(
        self,
        pipeline: CogVideoXPipeline,
        prompt: str,
        image: Optional[torch.Tensor] = None,
        control_video: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        generator: Optional[torch.Generator] = None,
        frame_conditioning_type: "FrameConditioningType"= "full", 
        frame_conditioning_index: int = 0, 
        **kwargs,
    ) -> List[ArtifactType]:
        logger.info("Previous numframes")
        logger.info(num_frames)
        logger.info("Previous height")
        logger.info(height)
        logger.info("Previous width")
        logger.info(width)
        #TODO(Bryan): erase hardcode of height width and num_frames
        inner_num_frames = 5
        height = 480
        width = 720

        # TODO(aryan): add support for more parameters
        
        from finetrainers.trainer.control_trainer.data import apply_frame_conditioning_on_latents
        use_i2v = image is not None           # any time an image is given, we need the I2V head
        assert image is not None
        assert control_video is not None
        if use_i2v:
            pipeline = CogVideoXImageToVideoPipeline.from_pipe(pipeline)
            
        with torch.no_grad():
            dtype = pipeline.vae.dtype
            device = pipeline._execution_device
            #in_channels = self.transformer_config.in_channels  # We need to use the original in_channels
            #latents = pipeline.prepare_latents(1, in_channels, height, width, num_frames, dtype, device, generator)

            control_video = pipeline.video_processor.preprocess_video(control_video, height=height, width=width)
            control_video = control_video.to(device=device, dtype=dtype)

            #OLD METHOD:
            control_latents = pipeline.vae.encode(control_video).latent_dist.mode()
            # control_latents = pipeline.vae.encode(control_video).latent_dist.sample(generator=generator)
            #TODO (Bryan): CogVideoX has latent scaling factor for I2V
            control_latents = self.vae_config.scaling_factor * control_latents

            control_latents = apply_frame_conditioning_on_latents(
                control_latents,
                inner_num_frames,
                channel_dim=1,
                frame_dim=2,
                frame_conditioning_type=frame_conditioning_type,
                frame_conditioning_index=frame_conditioning_index,
                concatenate_mask=self.frame_conditioning_concatenate_mask,
            )
            


        generation_kwargs = {
            "image": image, 
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "return_dict": True,
            "output_type": "pil",
        }
        generation_kwargs = get_non_null_items(generation_kwargs)

        #TODO (Bryan): They did classifier free guidance, duplicate control_latents on the batch part
        control_latents = torch.concat([control_latents]*2, dim=0)
        #control latents are BCFHW, convert to BFCHW
        control_latents = control_latents.permute(0, 2, 1, 3, 4)

        with control_channel_concat(pipeline.transformer, ["hidden_states"], [control_latents], dims=[2]):
            video = pipeline(**generation_kwargs).frames[0]
        return [VideoArtifact(value=video)]

    def _save_lora_weights(
        self,
        directory: str,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        norm_state_dict: Optional[Dict[str, torch.Tensor]] = None, 
        scheduler: Optional[SchedulerType] = None,
        metadata: Optional[Dict[str, str]] = None,
        *args,
        **kwargs,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            if self.image_pipeline:
                CogVideoXImageToVideoPipeline.save_lora_weights(
                    directory,
                    transformer_state_dict,
                    save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                    safe_serialization=True,
                )
            else:
                CogVideoXPipeline.save_lora_weights(
                    directory,
                    transformer_state_dict,
                    save_function=functools.partial(safetensors_torch_save_function, metadata=metadata),
                    safe_serialization=True,
                )
        if norm_state_dict is not None:
            safetensors.torch.save_file(norm_state_dict, os.path.join(directory, "norm_state_dict.safetensors"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    def _save_model(
        self,
        directory: str,
        transformer: CogVideoXTransformer3DModel,
        transformer_state_dict: Optional[Dict[str, torch.Tensor]] = None,
        scheduler: Optional[SchedulerType] = None,
    ) -> None:
        # TODO(aryan): this needs refactoring
        if transformer_state_dict is not None:
            with init_empty_weights():
                transformer_copy = CogVideoXTransformer3DModel.from_config(transformer.config)
            transformer_copy.load_state_dict(transformer_state_dict, strict=True, assign=True)
            transformer_copy.save_pretrained(os.path.join(directory, "transformer"))
        if scheduler is not None:
            scheduler.save_pretrained(os.path.join(directory, "scheduler"))

    @staticmethod
    def _pad_frames(latents: torch.Tensor, patch_size_t: int) -> torch.Tensor:
        num_frames = latents.size(1)
        additional_frames = patch_size_t - (num_frames % patch_size_t)
        if additional_frames > 0:
            last_frame = latents[:, -1:]
            padding_frames = last_frame.expand(-1, additional_frames, -1, -1, -1)
            latents = torch.cat([latents, padding_frames], dim=1)
        return latents

    @property
    def _original_control_layer_in_features(self):
        return self.transformer_config.in_channels
    
    @property
    def _vae_output_channel(self):
        return self.vae_config.latent_channels

    @property
    def _original_control_layer_out_features(self):
        return self.transformer_config.num_attention_heads * self.transformer_config.attention_head_dim

    @property
    def _qk_norm_identifiers(self):
        return ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]        