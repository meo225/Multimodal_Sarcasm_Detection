"""Test CrossAttentionFusion: ba chế độ direction, không cần GPU hay tải model."""
import pytest
import torch

from vimmsd.models.fusion import CrossAttentionFusion
from vimmsd.utils.config import load_config

B, LT, LI, D = 2, 12, 17, 32


def _batch():
    mask = torch.ones(B, LT, dtype=torch.bool)
    mask[1, 8:] = False
    text = {"tokens": torch.randn(B, LT, D), "mask": mask, "pooled": torch.randn(B, D)}
    image = {"tokens": torch.randn(B, LI, D), "pooled": torch.randn(B, D)}
    return text, image


def _fusion(direction, **kw):
    return CrossAttentionFusion(D, D, hidden_dim=16, num_heads=4, num_layers=kw.pop("num_layers", 1),
                                dropout=0.0, direction=direction, **kw).eval()


@pytest.mark.parametrize("direction", CrossAttentionFusion.DIRECTIONS)
def test_output_dim_is_same_for_every_direction(direction):
    f = _fusion(direction)
    out = f(*_batch())
    assert out.shape == (B, f.out_dim) == (B, 16 * 4)
    assert torch.isfinite(out).all()


def test_attention_maps_per_direction():
    text, image = _batch()
    both = _fusion("both")
    both(text, image)
    assert both.last_attn["text_to_image"].shape == (B, LT, LI)
    assert both.last_attn["image_to_text"].shape == (B, LI, LT)

    t2i = _fusion("text_to_image")
    t2i(text, image)
    assert t2i.last_attn["text_to_image"].shape == (B, LT, LI)
    assert t2i.last_attn["image_to_text"] is None
    assert len(t2i.image_blocks) == 0

    i2t = _fusion("image_to_text")
    i2t(text, image)
    assert i2t.last_attn["image_to_text"].shape == (B, LI, LT)
    assert i2t.last_attn["text_to_image"] is None
    assert len(i2t.text_blocks) == 0


def test_image_to_text_ignores_padded_text_tokens():
    text, image = _batch()
    f = _fusion("image_to_text")
    f(text, image)
    assert f.last_attn["image_to_text"][1, :, 8:].abs().max() == 0


def test_text_to_image_does_not_depend_on_image_to_text_params():
    f = _fusion("text_to_image")
    assert not any(n.startswith("image_blocks") for n, _ in f.named_parameters())


def test_both_equals_original_two_way_behaviour():
    f = _fusion("both", num_layers=2)
    assert len(f.text_blocks) == len(f.image_blocks) == 2
    assert f(*_batch()).shape == (B, 64)


def test_one_way_changes_only_the_attending_side():
    torch.manual_seed(0)
    text, image = _batch()
    t2i = _fusion("text_to_image")
    out = t2i(text, image)
    t_pool, i_pool = out[:, :16], out[:, 16:32]
    i_proj = t2i.image_proj(image["tokens"]).mean(1)
    assert torch.allclose(i_pool, i_proj, atol=1e-6)  # ảnh không được attend nên chỉ qua phép chiếu
    assert not torch.allclose(t_pool, t2i.text_proj(text["tokens"]).mean(1))


def test_invalid_direction_is_rejected():
    with pytest.raises(ValueError, match="direction"):
        CrossAttentionFusion(D, D, direction="sideways")


@pytest.mark.parametrize("name,direction", [("fusion_cross_attn", "both"),
                                            ("fusion_cross_attn_t2i", "text_to_image"),
                                            ("fusion_cross_attn_i2t", "image_to_text")])
def test_configs_select_direction(name, direction):
    cfg = load_config(f"configs/{name}.yaml")
    assert cfg.model.modality == "cross_attn"
    assert cfg.model.fusion.direction == direction
    assert cfg.name == name
