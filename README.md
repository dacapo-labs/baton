# Baton

AI proxy gateway with multi-model fan-out, Bitwarden auth, and zone-aware routing.

Part of the LifeMaestro ecosystem.

## Features

- **LiteLLM Foundation**: Support for 100+ AI providers through a single API
- **Bitwarden Auth**: Fetch API keys from Bitwarden vault, auto-TOTP support
- **Fan-out Queries**: Query multiple models in parallel with aggregation
- **Judge Mode**: Use PAI rate_content pattern to select best response
- **Zone-Aware**: Per-zone model defaults, rate limits, and allowed providers
- **Comprehensive Logging**: JSONL logs to LUKS-encrypted storage
- **Adaptive Routing**: Learn from judge decisions to auto-route queries
- **SMS Notifications**: Twilio integration for MFA and approval gates

## Installation

```bash
# From lifemaestro repo
cd baton
pip install -e .

# Or with uv
uv pip install -e .
```

## Configuration

Copy the example config:

```bash
mkdir -p ~/.config/lifemaestro
cp baton.example.toml ~/.config/lifemaestro/baton.toml
```

## Usage

### Start Server

```bash
baton serve
# or
baton serve --host 0.0.0.0 --port 4000
```

### Check Health

```bash
baton health
```

### Auth Management

```bash
# Check provider status
baton auth status

# Refresh credentials from Bitwarden
baton auth refresh

# Save session for later restore
baton auth save

# Restore saved session
baton auth restore
```

### Stats

```bash
# Summary
baton-stats summary

# Judge analysis
baton-stats judge --by-query-type

# Model win rates
baton-stats winners
```

## API

OpenAI-compatible API at `http://localhost:4000/v1/chat/completions`:

```bash
# Simple request
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smart",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Fan-out with judge
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["claude-sonnet-4-20250514", "gpt-4o"],
    "fanout": "judge",
    "messages": [{"role": "user", "content": "Explain recursion"}]
  }'
```

### Fan-out Modes

- `first`: Return first successful response (default)
- `all`: Return all responses
- `race`: Return fastest response
- `vote`: Majority vote (for classification)
- `judge`: Use judge model to select best response

### Zone Headers

Pass zone context via headers:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "X-Maestro-Zone: work" \
  -H "X-Maestro-Session: 2024-01-15-project" \
  ...
```

## Systemd Service

Install user service:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/baton.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable baton
systemctl --user start baton
```

## aws-devbox Integration

Copy hooks to aws-devbox:

```bash
# Post-boot hooks
cp hooks/40-bw-unlock.sh ~/.config/aws-devbox/hooks/post-boot/
cp hooks/50-baton.sh ~/.config/aws-devbox/hooks/post-boot/

# Pre-shutdown hook
cp hooks/pre-shutdown.sh ~/.config/aws-devbox/hooks/pre-shutdown/50-baton.sh

chmod +x ~/.config/aws-devbox/hooks/**/*.sh
```

## Architecture

```
┌──────────────┐     ┌─────────────────────────────────────┐
│   Client     │────▶│              Baton                  │
│ (Claude CLI) │     │  ┌─────────┐  ┌─────────────────┐  │
└──────────────┘     │  │  Auth   │  │    Fan-out      │  │
                     │  │ (BW)    │  │  ┌───┐ ┌───┐    │  │
                     │  └─────────┘  │  │ M1│ │ M2│... │  │
                     │               │  └───┘ └───┘    │  │
                     │  ┌─────────┐  │       ▼         │  │
                     │  │ Zones   │  │    ┌─────┐      │  │
                     │  │         │  │    │Judge│      │  │
                     │  └─────────┘  │    └─────┘      │  │
                     │               └─────────────────┘  │
                     │  ┌─────────────────────────────┐   │
                     │  │        Logger (JSONL)       │   │
                     │  └─────────────────────────────┘   │
                     └─────────────────────────────────────┘
                                      │
                                      ▼
                     ┌─────────────────────────────────────┐
                     │     /data/baton/logs (LUKS)         │
                     └─────────────────────────────────────┘
```

## License

MIT
