from PIL import Image, ImageOps
import io

class ImageProcessor:
    def __init__(self, max_dimension=2048):
        self.max_dimension = max_dimension

    def process_image(self, uploaded_file) -> dict:
        """
        Process an uploaded image file.
        Returns a dictionary with the processed PIL Image or an error status.
        """
        try:
            image = Image.open(uploaded_file)
            
            # Correct orientation based on EXIF
            image = ImageOps.exif_transpose(image)
            
            # Convert to RGB to ensure compatibility
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            width, height = image.size
            
            # Check for extremely low resolution
            if width < 50 or height < 50:
                return {"success": False, "error": "Image resolution is too low to be reliably analyzed."}
                
            # Resize if too large
            if width > self.max_dimension or height > self.max_dimension:
                image.thumbnail((self.max_dimension, self.max_dimension), Image.Resampling.LANCZOS)
                
            return {"success": True, "image": image}
            
        except Exception as e:
            return {"success": False, "error": f"Failed to process image: {str(e)}"}
