from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predicts_an_alarm_request():
    r = client.post("/predict", json={"text": "set an alarm for 7 am"})
    assert r.status_code == 200
    assert r.json()["intent"] == "alarm"


def test_rejects_empty_text():
    assert client.post("/predict", json={"text": ""}).status_code == 422