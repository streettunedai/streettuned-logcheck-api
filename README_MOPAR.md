# StreetTunedAI Mopar API

This repo now contains a standalone Mopar-only API in `app_mopar/main.py`.

It does not modify or import the existing `app/main.py` API.

## Render Mopar Service

The Mopar Render service should use this start command:

```bash
uvicorn app_mopar.main:app --host 0.0.0.0 --port $PORT
```

The health check URL should be:

```text
/health
```

Expected public health URL:

```text
https://streettuned-mopar-logcheck-api.onrender.com/health
```

Expected health response includes:

```json
{
  "service": "streettuned-mopar-logcheck-api"
}
```

## GPT Action Schema

Use `openapi.mopar.actions.json` for the Mopar GPT action.

The Mopar GPT should call:

1. `POST /validate`
2. `POST /analyze` only if validate returns `ok: true`

Always send:

```json
{
  "platform_hint": "mopar"
}
```
