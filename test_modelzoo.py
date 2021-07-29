from modelzoo_iitm import models

def test_modelzoo_no_params():
    assert models() == "No Models loaded"

def test_modelzoo_with_params():
    assert models("PointNet") == "PointNet model loaded successfully"