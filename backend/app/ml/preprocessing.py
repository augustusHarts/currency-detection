from pathlib import Path
from PIL import Image as PILImage

class PreProcessing:

    def __init__(self):
        pass

    def remove_background_white(
        self, 
        input_path: Path, 
        output_path: Path
    ) -> Path:
        """
        Uses rembg (U2Net) to remove the background, then composites
        the subject onto a solid white canvas.

        Returns the path to the cleaned image.
        """
        
        with open(input_path, "rb") as f:
            raw = f.read()

        # Import here to avoid heavy startup imports (Render port binding).
        from rembg import remove as rembg_remove

        # rembg returns a PNG with transparent background (RGBA)
        removed = rembg_remove(raw)
        rgba = PILImage.open(__import__("io").BytesIO(removed)).convert("RGBA")

        # Paste onto white background using the alpha channel as mask
        white_bg = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
        white_bg.paste(rgba, mask=rgba.split()[3])          # alpha channel as mask
        white_bg = white_bg.convert("RGB")

        white_bg.save(output_path)
        
        return output_path