"""Baton plugins for LiteLLM proxy."""

from .auth import BatonAuth
from .cli_auth import CLIAuthManager, ClaudeCliAuth, CodexCliAuth, GeminiCliAuth
from .fanout import BatonFanout
from .guardrails import BatonGuardrails
from .judge import BatonJudge
from .keepalive import KeepaliveDaemon, init_keepalive_daemon, get_keepalive_daemon
from .logger import BatonLogger
from .playwright_auth import PlaywrightAuthManager, OAuthSession
from .rate_limits import RateLimitTracker, RateLimitConfig, init_rate_limiter, get_rate_limiter
from .model_checker import (
    ModelAvailabilityChecker,
    BedrockModelChecker,
    VertexModelChecker,
    ModelInfo,
    get_model_checker,
    init_model_checker,
)
from .model_monitor import (
    ModelMonitor,
    MonitorConfig,
    ModelChange,
    get_model_monitor,
    init_model_monitor,
    start_model_monitor,
    stop_model_monitor,
)
from .version_checker import (
    VersionChecker,
    VersionCheckerConfig,
    VersionInfo,
    get_version_checker,
    init_version_checker,
    start_version_checker,
    stop_version_checker,
)
from .repo_updater import (
    RepoUpdater,
    RepoUpdaterConfig,
    RepoInfo,
    get_repo_updater,
    init_repo_updater,
    start_repo_updater,
    stop_repo_updater,
    clone_repo,
    KNOWN_REPOS,
)
from .skills_cache import (
    SkillsCache,
    SkillsCacheConfig,
    SkillInfo,
    get_skills_cache,
    init_skills_cache,
    start_skills_cache,
    stop_skills_cache,
)
from .router import BatonRouter
from .twilio import BatonTwilio
from .zones import BatonZones

__all__ = [
    # Core
    "BatonAuth",
    "BatonFanout",
    "BatonGuardrails",
    "BatonJudge",
    "BatonLogger",
    "BatonRouter",
    "BatonTwilio",
    "BatonZones",
    # CLI Auth
    "CLIAuthManager",
    "ClaudeCliAuth",
    "CodexCliAuth",
    "GeminiCliAuth",
    # Playwright Auth
    "PlaywrightAuthManager",
    "OAuthSession",
    # Rate Limits
    "RateLimitTracker",
    "RateLimitConfig",
    "init_rate_limiter",
    "get_rate_limiter",
    # Keepalive
    "KeepaliveDaemon",
    "init_keepalive_daemon",
    "get_keepalive_daemon",
    # Model Checker
    "ModelAvailabilityChecker",
    "BedrockModelChecker",
    "VertexModelChecker",
    "ModelInfo",
    "get_model_checker",
    "init_model_checker",
    # Model Monitor
    "ModelMonitor",
    "MonitorConfig",
    "ModelChange",
    "get_model_monitor",
    "init_model_monitor",
    "start_model_monitor",
    "stop_model_monitor",
    # Version Checker
    "VersionChecker",
    "VersionCheckerConfig",
    "VersionInfo",
    "get_version_checker",
    "init_version_checker",
    "start_version_checker",
    "stop_version_checker",
    # Repo Updater
    "RepoUpdater",
    "RepoUpdaterConfig",
    "RepoInfo",
    "get_repo_updater",
    "init_repo_updater",
    "start_repo_updater",
    "stop_repo_updater",
    "clone_repo",
    "KNOWN_REPOS",
    # Skills Cache
    "SkillsCache",
    "SkillsCacheConfig",
    "SkillInfo",
    "get_skills_cache",
    "init_skills_cache",
    "start_skills_cache",
    "stop_skills_cache",
]
