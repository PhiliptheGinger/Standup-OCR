# Minimal scaffold: this test is a placeholder demonstrating initialization can succeed/fail.
import pytest
from pathlib import Path
from src.handwriting_ocr_client import HandwritingOCRClient, HandwritingOCRError

def test_init_requires_token():
    with pytest.raises(ValueError):
        HandwritingOCRClient(token="")

# The rest of integration tests require a real token and network access; keep them out of CI.
# To test transcribe_image locally, set HANDWRITING_OCR_TOKEN env var and run a manual test.