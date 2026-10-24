# My AI Website - Backend

A small Flask backend that proxies chat messages from the website to OpenAI
(`gpt-4o-mini`) and returns the reply as JSON.

## API

### `POST /chat`

Request body:

```json
{ "message": "Hello" }
```

Responses:

| Status | Meaning |
|--------|---------|
| 200 | `{ "reply": "..." }` from the model |
| 400 | Missing or empty `message`, non-JSON body, or message over 4000 chars |
| 500 | Server has no `API_KEY` configured |
| 502 | OpenAI returned an error or an unexpected payload |
| 504 | OpenAI took too long (30s read timeout) |

## Run locally

```bash
pip install -r requirements.txt
export API_KEY=sk-...
python app.py   # listens on port 10000
```

## Tests

No OpenAI calls are made (HTTP is stubbed):

```bash
python3 -m unittest discover -s tests
```
