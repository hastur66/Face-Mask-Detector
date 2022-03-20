import pytest
import script_webcam

def test_detect_and_predict_mask():
    assert type(script_webcam.detect_and_predict_mask).__name__ == 'function'