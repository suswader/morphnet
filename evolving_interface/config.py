"""Central configuration and site profile management."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ── paths ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MCPS_DIR = ROOT_DIR / "mcps"
MCPS_DIR.mkdir(exist_ok=True)
SITES_DIR = ROOT_DIR / "sites"

# ── Gemini models ────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_COMPUTER_USE_MODEL = os.getenv(
    "GEMINI_COMPUTER_USE_MODEL", "gemini-3-flash-preview",
)
GEMINI_REASONING_MODEL = os.getenv(
    "GEMINI_REASONING_MODEL", "gemini-3.1-pro-preview",
)
GEMINI_FAST_MODEL = os.getenv(
    "GEMINI_FAST_MODEL", "gemini-3.1-flash-lite-preview",
)

# ── Task autonomy ───────────────────────────────────────────────
# "benchmark" — auto-submit everything, auto-acknowledge safety
# "cautious"  — pause before irreversible actions
# "autonomous" — submit without hesitation
TASK_AUTONOMY_LEVEL = os.getenv("TASK_AUTONOMY_LEVEL", "benchmark")


# ── Site profiles ────────────────────────────────────────────────

class AuthConfig(BaseModel):
    username: str
    password: str
    login_url: str = ""


class SiteProfile(BaseModel):
    name: str
    url: str
    description: str = ""
    auth: AuthConfig | None = None
    agent_hints: str = ""
    extra: dict[str, str] = Field(default_factory=dict)

    @property
    def origin(self) -> str:
        parsed = urlparse(self.url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @property
    def netloc(self) -> str:
        return urlparse(self.url).netloc


def load_profiles(sites_dir: Path = SITES_DIR) -> dict[str, SiteProfile]:
    profiles: dict[str, SiteProfile] = {}
    if not sites_dir.exists():
        return profiles
    for path in sorted(sites_dir.glob("*.yaml")):
        with open(path) as f:
            data = yaml.safe_load(f)
        if data:
            profile = SiteProfile(**data)
            profiles[profile.name] = profile
    return profiles


def get_profile_for_url(
    url: str,
    profiles: dict[str, SiteProfile] | None = None,
) -> SiteProfile | None:
    if profiles is None:
        profiles = load_profiles()
    netloc = urlparse(url).netloc
    for profile in profiles.values():
        if profile.netloc == netloc:
            return profile
    return None


def build_auth_hint(profile: SiteProfile | None) -> str:
    if profile is None or profile.auth is None:
        return ""
    auth = profile.auth
    return (
        f"LOGIN CREDENTIALS — username: {auth.username}  password: {auth.password}\n"
        f"If you need to log in, go to {profile.url}{auth.login_url} — "
        f"click the username field, type the username, click the password "
        f"field, type the password, then click the Sign In / Login button."
    )


def build_site_context(profile: SiteProfile | None) -> str:
    if profile is None:
        return ""
    parts: list[str] = []
    auth = build_auth_hint(profile)
    if auth:
        parts.append(auth)
    if profile.agent_hints:
        parts.append(profile.agent_hints)
    return "\n".join(parts)


def get_allowed_origins(
    profiles: dict[str, SiteProfile] | None = None,
) -> set[str]:
    if profiles is None:
        profiles = load_profiles()
    return {p.netloc for p in profiles.values()}


# ── Derived site data ────────────────────────────────────────────
SITE_PROFILES: dict[str, SiteProfile] = load_profiles()

SITES: dict[str, str] = {
    name: os.getenv(f"WEBARENA_{name.upper()}_URL", profile.url)
    for name, profile in SITE_PROFILES.items()
}
