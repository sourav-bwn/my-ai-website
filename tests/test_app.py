"""Offline tests for app.py. OpenAI is never contacted: requests.post is stubbed."""
import importlib
import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as app_module


def make_client(api_key="test-key"):
    app_module.API_KEY = api_key
    app_module.app.config.update(TESTING=True)
    return app_module.app.test_client()


def ok_response(text="Hello!"):
    return {"choices": [{"message": {"content": text}}]}


class FakePost:
    """Configurable stand-in for requests.post."""

    def __init__(self, payload=None, raises=None):
        self.payload = payload if payload is not None else ok_response()
        self.raises = raises
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.raises:
            raise self.raises
        response = mock.Mock()
        response.json.return_value = self.payload
        return response


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_non_json_body_is_a_400_not_a_server_error(self):
        r = self.client.post("/chat", data="hello", content_type="text/plain")
        self.assertEqual(r.status_code, 400)
        self.assertIn("JSON", r.get_json()["reply"])

    def test_missing_message_is_a_400(self):
        r = self.client.post("/chat", json={})
        self.assertEqual(r.status_code, 400)

    def test_empty_message_is_a_400(self):
        r = self.client.post("/chat", json={"message": "   "})
        self.assertEqual(r.status_code, 400)

    def test_overlong_message_is_rejected(self):
        r = self.client.post("/chat", json={"message": "x" * 4001})
        self.assertEqual(r.status_code, 400)
        self.assertIn("too long", r.get_json()["reply"])

    def test_validate_message_trims(self):
        message, error = app_module.validate_message({"message": "  hi  "})
        self.assertIsNone(error)
        self.assertEqual(message, "hi")


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client()

    def test_success_returns_reply_and_200(self):
        stub = FakePost()
        with mock.patch.object(app_module.requests, "post", stub):
            r = self.client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["reply"], "Hello!")
        _, kwargs = stub.calls[0]
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(kwargs["json"]["messages"][0]["content"], "hi")
        self.assertIsNotNone(kwargs.get("timeout"), "the OpenAI call must have a timeout")

    def test_missing_api_key_is_a_500(self):
        client = make_client(api_key=None)
        r = client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 500)

    def test_openai_error_payload_is_a_502(self):
        stub = FakePost(payload={"error": {"message": "bad key"}})
        with mock.patch.object(app_module.requests, "post", stub):
            r = self.client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 502)
        self.assertIn("API Error", r.get_json()["reply"])

    def test_timeout_is_a_504(self):
        stub = FakePost(raises=requests.Timeout())
        with mock.patch.object(app_module.requests, "post", stub):
            r = self.client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 504)

    def test_unreachable_service_is_a_502(self):
        stub = FakePost(raises=requests.ConnectionError())
        with mock.patch.object(app_module.requests, "post", stub):
            r = self.client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 502)

    def test_malformed_upstream_payload_is_a_502(self):
        stub = FakePost(payload={"unexpected": True})
        with mock.patch.object(app_module.requests, "post", stub):
            r = self.client.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 502)

    def test_home_route(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
