"""Real-model checks. Needs internet + torch/transformers. Run with:
   RUN_MODEL_TESTS=1 pytest tests/test_models_integration.py -s
Results are printed, not asserted strictly: detectors are probabilistic, so use
this to eyeball behaviour on your own samples."""
import io
import os

import pytest

pytestmark = pytest.mark.skipif(not os.getenv("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1")

HUMAN = ("ok so I finally tried the new ramen place on 5th, and honestly? The broth was fine, nothing wild, "
         "but the guy behind the counter remembered my order from two weeks ago which is kind of why I'll go back. "
         "My sister hates it there, says it's too salty, but she also puts ketchup on rice so I don't trust her.") * 2
AI = ("Artificial intelligence is transforming modern industries by enabling organizations to automate processes, "
      "enhance decision-making, and unlock new opportunities for innovation. Furthermore, it is important to note "
      "that responsible deployment requires robust governance, transparency, and continuous monitoring. "
      "In conclusion, AI offers significant benefits while also presenting challenges that must be addressed.") * 2


def test_text_samples():
    from modules.text_detector import detect_text
    for name, t in [("human", HUMAN), ("ai", AI)]:
        r = detect_text(t)
        print(f"{name}: AI={r.ai_prob:.0%} -> {r.label}")


def test_image_sample():
    from PIL import Image
    import numpy as np
    from modules.image_detector import detect_image
    buf = io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((256, 256, 3)) * 255).astype("uint8")).save(buf, "PNG")
    r = detect_image(buf.getvalue())
    print(f"noise image: AI={r.ai_prob:.0%} -> {r.label} ({r.confidence})")
