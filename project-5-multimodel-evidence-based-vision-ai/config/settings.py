import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configurations
# We use Gemini 3.6 Flash for balanced multimodal capabilities
VISION_MODEL_NAME = "gemini-3.6-flash"
REASONING_MODEL_NAME = "gemini-3.6-flash"

# System behavior
REQUIRE_CONFIDENCE_THRESHOLD = 0.7

# Image limits
MAX_IMAGE_SIZE_MB = 10
SUPPORTED_IMAGE_FORMATS = ["PNG", "JPEG", "JPG", "WEBP"]
MAX_IMAGES_PER_TURN = 5
