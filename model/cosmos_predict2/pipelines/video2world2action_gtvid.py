import torch
from einops import rearrange
from torch import nn

from cosmos_predict2.pipelines.video2world import DeviceMesh, Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline


class Video2World2ActionPipeline(nn.Module):
    def __init__(
        self,
        video2world_pipeline: Video2WorldPipeline,
        world2action_pipeline: World2ActionPipeline,
    ) -> None:
        super().__init__()

        self.video2world_pipeline = video2world_pipeline
        self.world2action_pipeline = world2action_pipeline

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.video2world_pipeline.apply_fsdp(dp_mesh)
        self.world2action_pipeline.apply_fsdp(dp_mesh)

    def apply_cp(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(
        self,
        input_vid: torch.Tensor,
        gt_future_vid: torch.Tensor,
        state_B_HO_O: torch.Tensor,
        prompt: str,
        num_sampling_step: int = 35,
        stop_after_step: int | None = None,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor:
        if self.video2world_pipeline.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            if not guardrail_presets.run_text_guardrail(prompt, self.video2world_pipeline.text_guardrail_runner):
                msg = "Text guardrail error on prompt."
                raise RuntimeError(msg)

        B, _C, T, _H, _W = input_vid.shape
        assert T in {1, 5}
        total_frames = self.video2world_pipeline.tokenizer.get_pixel_num_frames(self.video2world_pipeline.config.state_t)
        assert gt_future_vid.shape[2] == total_frames - T, f"{gt_future_vid.shape=}, {input_vid.shape=}"

        data_batch = {
            "obs/workspace_rgb": input_vid,
            "action/workspace_rgb": gt_future_vid,
            "obs/language_embedding": self.video2world_pipeline.encode_prompt(prompt).to(
                self.video2world_pipeline.torch_dtype
            ),
            "num_conditional_frames": self.video2world_pipeline.tokenizer.get_latent_num_frames(T),
            "is_preprocessed": True,
        }
        _, video_B_C_T_H_W, condition = self.video2world_pipeline.get_mimic_data_and_condition(data_batch)
        video_epsilon_B_C_T_H_W = torch.randn(video_B_C_T_H_W.size(), dtype=torch.bfloat16, device="cuda")

        self.video2world_pipeline.scheduler.set_timesteps(num_sampling_step, device="cuda")
        video_sigma_B_1 = self.video2world_pipeline.scheduler.sigmas[stop_after_step].repeat(B).unsqueeze(1)

        world_pred = self.video2world_pipeline.denoise(
            video_B_C_T_H_W + video_epsilon_B_C_T_H_W * rearrange(video_sigma_B_1, "b t -> b 1 t 1 1"),
            video_sigma_B_1,
            condition,
            use_cuda_graphs=False,
            return_only_hidden_states_up_to=self.world2action_pipeline.config.xattn_layer_idx,
            return_decoded_video=False,
        )
        crossattn_emb = world_pred.hidden_states[self.world2action_pipeline.config.xattn_layer_idx]
        hidden_state_shape = crossattn_emb.shape
        crossattn_emb = crossattn_emb.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])

        return self.world2action_pipeline(
            state_B_HO_O=state_B_HO_O,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma_B_1,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
