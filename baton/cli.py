"""Baton CLI - Command line interface for Baton proxy."""

from __future__ import annotations

import argparse
import sys


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="baton",
        description="AI proxy gateway with multi-model fan-out and zone-aware routing",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    serve_parser = subparsers.add_parser("serve", help="Start the Baton server")
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=4000,
        help="Port to bind to (default: 4000)",
    )
    serve_parser.add_argument(
        "--config",
        help="Path to config file",
    )
    serve_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    auth_parser = subparsers.add_parser("auth", help="Authentication management")
    auth_sub = auth_parser.add_subparsers(dest="auth_command")
    auth_sub.add_parser("refresh", help="Refresh credentials from Bitwarden")
    auth_sub.add_parser("save", help="Save BW session for later restore")
    auth_sub.add_parser("restore", help="Restore BW session from saved file")
    auth_sub.add_parser("status", help="Show authentication status")

    subparsers.add_parser("health", help="Check server health")

    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_sub = config_parser.add_subparsers(dest="config_command")
    config_sub.add_parser("show", help="Show current configuration")
    config_sub.add_parser("path", help="Show config file path")

    args = parser.parse_args()

    if args.command == "serve":
        run_server(args)
    elif args.command == "auth":
        run_auth(args)
    elif args.command == "health":
        run_health()
    elif args.command == "config":
        run_config(args)
    else:
        parser.print_help()
        sys.exit(1)


def run_server(args):
    """Start the Baton server."""
    import os
    import uvicorn

    from .config import load_config

    if args.config:
        os.environ["BATON_CONFIG"] = args.config

    config = load_config()

    host = args.host or config.get("server", {}).get("host", "127.0.0.1")
    port = args.port or config.get("server", {}).get("port", 4000)
    debug = args.debug or config.get("debug", False)

    print(f"Starting Baton on {host}:{port}")
    uvicorn.run(
        "baton.server:app",
        host=host,
        port=port,
        reload=debug,
    )


def run_auth(args):
    """Handle auth commands."""
    import httpx

    base_url = "http://127.0.0.1:4000"

    if args.auth_command == "refresh":
        try:
            resp = httpx.post(f"{base_url}/auth/refresh", timeout=30)
            print(resp.json())
        except httpx.ConnectError:
            print("Error: Baton server not running")
            sys.exit(1)

    elif args.auth_command == "save":
        try:
            resp = httpx.post(f"{base_url}/auth/save", timeout=10)
            print(resp.json())
        except httpx.ConnectError:
            print("Error: Baton server not running")
            sys.exit(1)

    elif args.auth_command == "restore":
        try:
            resp = httpx.post(f"{base_url}/auth/restore", timeout=10)
            print(resp.json())
        except httpx.ConnectError:
            print("Error: Baton server not running")
            sys.exit(1)

    elif args.auth_command == "status":
        from .config import load_config
        from .plugins.auth import BatonAuth

        config = load_config()
        auth = BatonAuth(config)

        providers = ["anthropic", "openai", "google", "deepseek", "mistral", "groq"]
        print("Provider Status:")
        for provider in providers:
            key = auth.get_api_key(provider)
            status = "configured" if key else "not configured"
            print(f"  {provider}: {status}")

    else:
        print("Usage: baton auth [refresh|save|restore|status]")
        sys.exit(1)


def run_health():
    """Check server health."""
    import httpx

    try:
        resp = httpx.get("http://127.0.0.1:4000/health", timeout=5)
        health = resp.json()
        print(f"Status: {health.get('status')}")
        print(f"Zone: {health.get('zone') or 'not set'}")
        print(f"Session: {health.get('session') or 'not set'}")
        print("Providers:")
        for provider, status in health.get("providers", {}).items():
            print(f"  {provider}: {status}")
    except httpx.ConnectError:
        print("Error: Baton server not running")
        sys.exit(1)


def run_config(args):
    """Handle config commands."""
    from .config import find_config, load_config

    if args.config_command == "path":
        path = find_config()
        if path:
            print(path)
        else:
            print("No config file found")
            print("Search paths:")
            from .config import CONFIG_PATHS
            for p in CONFIG_PATHS:
                if p:
                    print(f"  {p}")

    elif args.config_command == "show":
        import json
        config = load_config()
        print(json.dumps(config, indent=2))

    else:
        print("Usage: baton config [show|path]")
        sys.exit(1)


if __name__ == "__main__":
    main()
