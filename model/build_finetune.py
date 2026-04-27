import torch
import torch.nn as nn

from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights


class IRRA(nn.Module):
    """Minimal AERI finetune baseline.

    The only supported training objective is:
        base = aerial-text SDM + ground-text SDM
    """

    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(
            args.pretrain_choice,
            args.img_size,
            args.stride_size,
        )
        self.embed_dim = base_cfg["embed_dim"]
        self.logit_scale = torch.ones([]) * (1 / args.temperature)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [token.strip() for token in loss_names.split("+") if token.strip()]
        if self.current_task != ["base"]:
            raise ValueError("This baseline branch only supports LOSS_NAMES='base'.")
        print(f"Training Model with {self.current_task} tasks")

    def encode_image(self, image):
        image_feats = self.base_model.encode_image(image)
        return image_feats[:, 0, :].float()

    def encode_text(self, text):
        text_feats = self.base_model.encode_text(text.long())
        return text_feats[torch.arange(text_feats.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        images = batch["images"]
        ground_images = batch["ground_imgs"]
        caption_ids = batch["caption_ids"]

        device_type = images.device.type
        use_amp = device_type == "cuda"
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=use_amp):
            image_feats, ground_image_feats, text_feats = self.base_model(
                images,
                ground_images,
                caption_ids,
            )

        aerial_feats = image_feats[:, 0, :].float()
        ground_feats = ground_image_feats[:, 0, :].float()
        text_feats = text_feats[
            torch.arange(text_feats.shape[0], device=text_feats.device),
            caption_ids.argmax(dim=-1),
        ].float()

        base_terms = objectives.compute_aeri_base_sdm_terms(
            aerial_feats,
            ground_feats,
            text_feats,
            batch["pids"],
            self.logit_scale,
        )
        base_loss = (
            base_terms["base_aerial_text"]
            + base_terms["base_ground_text"]
        )

        return {
            **base_terms,
            "base_loss": base_loss,
        }


def build_finetune_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    return model
