from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

class FatLabels2:
    def __init__(self, device="cpu"):
        self.device = device

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "Hello"}),
                "font_size": ("INT", {"default": 36, "min": 1}),  # Font size in pixels
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_fat_label_with_cv2"
    CATEGORY = "image/text"

    def create_fat_label_with_cv2(self, text, font_size):
        # Create a blank grayscale image as canvas with a fixed background color
        bg_color = 0  # Black background (grayscale)

        # Create a drawing context to calculate text size
        text_width, text_height = self.calculate_text_size(text, font_size)

        # Calculate canvas dimensions with padding
        canvas_width = text_width + 40  # Add 20px padding on each side
        canvas_height = text_height + 40  # Add 20px padding on each side

        canvas = Image.new("L", (canvas_width, canvas_height), bg_color)

        # Font color is always white
        font_color = 255  # White (grayscale)

        # Create an ImageFont object with the desired font size
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)

        # Create a drawing context
        draw = ImageDraw.Draw(canvas)

        # Calculate text position
        x = (canvas_width - text_width) // 2
        y = (canvas_height - text_height) // 2

        # Draw text on the image with the specified font size
        draw.text((x, y), text, fill=font_color, font=font)

        # Convert the image to a PyTorch tensor
        image_tensor_out = torch.tensor(np.array(canvas).astype(np.float32) / 255.0).unsqueeze(0)

        return (image_tensor_out,)

    def calculate_text_size(self, text, font_size):
        # Create a temporary canvas to calculate text size
        canvas = Image.new("L", (1, 1), 0)  # Create a blank 1x1 grayscale image
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        text_width, text_height = draw.textsize(text, font=font)
        return text_width, text_height

NODE_CLASS_MAPPINGS = {
    "FatLabels": FatLabels2,
}
