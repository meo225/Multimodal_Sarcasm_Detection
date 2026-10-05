"""Đọc ảnh an toàn và pad thành hình vuông.

Dùng chung cho đường A (ảnh đưa vào image encoder) và đường B (OCR, mô tả VLM), để mọi bước nhìn thấy
cùng một ảnh: cùng frame, cùng chiều xoay EXIF, cùng nền cho vùng trong suốt."""
from PIL import Image, ImageOps

WHITE = (255, 255, 255)


def open_image_rgb(path) -> Image.Image:
    """Mở ảnh, lấy frame đầu (GIF), sửa hướng theo EXIF, ghép vùng trong suốt lên nền trắng, chuyển RGB.
    Nền trắng thay vì đen: vùng trong suốt chuyển thẳng sang RGB thường thành đen và che mất chữ đen.
    File lỗi thì raise (OSError, ValueError), để nơi gọi tự quyết định thay thế hay bỏ qua."""
    with Image.open(path) as img:
        img.seek(0)
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            out = Image.new("RGB", rgba.size, WHITE)
            out.paste(rgba, mask=rgba.getchannel("A"))
            return out
        return img.convert("RGB")


class PadToSquare:
    """Thêm viền vào cạnh ngắn để ảnh thành hình vuông (letterbox), giữ nguyên toàn bộ nội dung.
    Sau bước này, center-crop của processor CLIP không cắt gì và resize của ViT không làm méo ảnh.
    `fill` nên là màu mean của encoder: sau khi normalize, viền có giá trị gần 0."""

    def __init__(self, fill=(0, 0, 0)):
        self.fill = tuple(fill)

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        out = Image.new("RGB", (side, side), self.fill)
        out.paste(img, ((side - w) // 2, (side - h) // 2))
        return out

    def __repr__(self):
        return f"PadToSquare(fill={self.fill})"


def encoder_mean_color(encoder_name):
    """Màu mean (0-255) của processor đi kèm checkpoint encoder, dùng làm màu viền và màu ảnh thay thế."""
    from transformers import AutoImageProcessor

    mean = AutoImageProcessor.from_pretrained(encoder_name).image_mean
    return tuple(round(m * 255) for m in mean)
