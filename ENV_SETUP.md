# Environment Configuration

This workspace relies on a shared `.env` file at the repository root. It centralizes the credentials needed by the FastAPI backend, the LiveKit worker, and any CLI tooling.

## Active Secrets

| Variable | Service | Purpose |
| --- | --- | --- |
| `LIVEKIT_URL` | LiveKit Cloud | WebSocket endpoint for realtime sessions |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | LiveKit Cloud | Signs room tokens for voice sessions |
| `ANTHROPIC_API_KEY` | Anthropic Claude | LLM generations (recipes, RAG answers) |
| `DEEPGRAM_API_KEY` | Deepgram | Speech-to-text and text-to-speech for the cooking companion |
| `OPENAI_API_KEY` | OpenAI | Embeddings (LlamaIndex recipe vectorization) |

All values are recorded as plain key/value pairs inside `.env`. The file is ignored by Git, so the secrets remain local by default.

## Usage

1. **Load locally**
   ```bash
   source .env
   ```
   or rely on tools like `python-dotenv` / `dotenv` which read automatically.

2. **Backend (`CalHacks-Agents`)**
   - The FastAPI app pulls these keys at startup (`python main.py` or `uvicorn` run).

3. **LiveKit worker (`Cooking-Companion`)**
   - The worker imports `dotenv` at launch (`python cooking_companion.py`) and uses the same `.env`.

4. **iOS client**
   - The app should request LiveKit tokens via `/session/create`; do **not** hardcode secrets in Swift code.

## Security Notes

- Keep `.env` out of source control (see `.gitignore` entry).
- Rotate keys periodically, especially for demo environments.
- When sharing the repo externally, supply a sanitized `.env.example` instead of the live file.
