import pytest

from src.pipeline import OUT_OF_SCOPE, build_pipeline, load, predict_with_fallback

DATA = "data/clinc150.json"


@pytest.fixture(scope="module")
def model():
    X, y = load(DATA, "train")
    return build_pipeline().fit(X, y)          # trained once, shared by both tests


def test_pipeline_learns_intents(model):
    X_val, y_val = load(DATA, "val")
    assert model.score(X_val, y_val) > 0.85


def test_low_confidence_becomes_out_of_scope(model):
    assert predict_with_fallback(model, ["blorp fizzle wug"], threshold=0.99) == [OUT_OF_SCOPE]