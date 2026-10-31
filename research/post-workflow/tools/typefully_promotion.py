"""Validate private draft-specific AutoPlug/results artifacts without network IO.

Hashes bind retained files. This check does not interpret screenshot semantics
or establish remote provenance. It never reads the results registry itself.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def validate_promotion(package, plan_relative, errors):
    """Append schema-1 plan errors. Preserve the caller's errors and all files."""
    package = Path(package).resolve()

    def fail(message):
        errors.append(f"Typefully promotion: {message}")

    def text(value):
        return isinstance(value, str) and bool(value.strip())

    def identifier(value):
        return not isinstance(value, bool) and re.fullmatch(r"[1-9][0-9]*", str(value)) is not None

    def path_for(value, label):
        if not text(value) or "\\" in value or "\x00" in value:
            fail(f"{label} must be a safe package-relative path")
            return None
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"{label} must be a safe package-relative path")
            return None
        target = package / relative
        if any(current.is_symlink() for current in (target, *target.parents) if current != package and package in current.parents):
            fail(f"{label} must not use symlinks")
            return None
        try:
            target.resolve().relative_to(package)
            return target
        except (OSError, RuntimeError, ValueError):
            fail(f"{label} escapes the package")
            return None

    def artifact(value, label):
        if not isinstance(value, dict):
            fail(f"{label} requires path and sha256")
            return None
        path = path_for(value.get("path"), label)
        if path is None:
            return None
        try:
            content = path.read_bytes()
            if not content or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))) or hashlib.sha256(content).hexdigest() != value.get("sha256"):
                fail(f"{label} must be nonempty and match its sha256")
            return path
        except OSError as exc:
            fail(f"cannot read {label}: {exc}")
            return None

    def read_json(path, label):
        try:
            value = json.loads(path.read_text()) if path else None
            if isinstance(value, dict):
                return value
            fail(f"{label} must be a JSON object")
        except (OSError, UnicodeError, ValueError) as exc:
            fail(f"cannot read {label}: {exc}")
        return {}

    def timestamp(value, label):
        try:
            if not isinstance(value, str) or "T" not in value:
                raise ValueError
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.utcoffset() is None:
                raise ValueError
            return parsed
        except ValueError:
            fail(f"{label} must be an ISO timestamp with a timezone")
            return None

    data = read_json(path_for(plan_relative, "plan"), "plan")
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        fail("schema_version must equal 1")
    for field in ("draft_id", "social_set_id"):
        if not identifier(data.get(field)):
            fail(f"{field} must be a positive integer ID")
    raw = read_json(artifact(data.get("raw_readback"), "raw_readback"), "raw_readback")
    for field in ("id", "draft_id", "social_set_id"):
        if field == "draft_id" and field not in raw:
            continue
        expected = data.get("social_set_id" if field == "social_set_id" else "draft_id")
        if not identifier(raw.get(field)) or str(raw[field]) != str(expected):
            fail(f"raw_readback.{field} does not match plan")
    if raw.get("status") != "draft":
        fail("raw_readback.status must equal draft")
    for field in {"scheduled_date", "published_at", *[key for key in raw if key in {"scheduled_at", "publish_state"} or key.endswith(("_published_at", "_published_url"))]}:
        if field not in raw or raw[field] is not None:
            fail(f"raw_readback.{field} must be null")
    auto = data.get("auto_plug") if isinstance(data.get("auto_plug"), dict) else {}
    checked = timestamp(auto.get("checked_at"), "auto_plug.checked_at")
    updated = timestamp(raw["updated_at"], "raw_readback.updated_at") if raw.get("updated_at") else None
    if checked and updated and checked < updated:
        fail("auto_plug.checked_at is older than the draft readback")
    artifact(auto.get("evidence"), "auto_plug.evidence")
    x = raw.get("platforms", {}).get("x", {}) if isinstance(raw.get("platforms"), dict) else {}
    x = x if isinstance(x, dict) else {}
    settings = x.get("settings") if isinstance(x.get("settings"), dict) else {}
    reply_keys = ("reply_to", "reply_to_id", "reply_to_url", "reply_to_post_url", "reply_to_tweet_id")
    is_reply = any(container.get(key) for container in (raw, x, settings) for key in reply_keys)
    if auto.get("state") == "configured":
        if x.get("enabled") is not True or is_reply:
            fail("configured AutoPlug requires an enabled original X post")
        reply = auto.get("reply_text")
        if not text(reply) or len(reply) > 280:
            fail("auto_plug.reply_text must contain 1 to 280 characters")
        else:
            urls = re.findall(r"https?://[^\s]+", reply)
            body = re.sub(r"https?://[^\s]+", "", reply)
            generic = re.search(r"(?i)follow me for more|if you (?:enjoyed|liked) this|check out my (?:profile|website)|subscribe to my newsletter", body)
            words = re.findall(r"[A-Za-z]{3,}", body)
            if len(words) < 5 or generic or (not urls and not text(auto.get("destination_reason"))):
                fail("auto_plug.reply_text must be specific useful copy with a URL, not default copy or a bare URL")
            for url in urls:
                try:
                    parsed = urlsplit(url.rstrip(".,;!?)"))
                    valid = parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password and not parsed.port
                    bare_repository = parsed.hostname and parsed.hostname.lower() in {"github.com", "www.github.com"} and len(parsed.path.strip("/").split("/")) <= 2
                except ValueError:
                    valid, bare_repository = False, False
                if not valid or bare_repository:
                    fail("auto_plug.reply_text URL must be HTTPS and link beyond a bare repository")
        if auto.get("metric") != "likes" or type(auto.get("threshold")) is not int or auto["threshold"] <= 0:
            fail("AutoPlug requires metric likes and a positive integer threshold")
        artifact(auto.get("threshold_decision"), "auto_plug.threshold_decision")
    elif auto.get("state") == "not_applicable":
        if x.get("enabled") is True and not is_reply:
            fail("AutoPlug not_applicable requires no X post or an explicit reply target")
    else:
        fail("auto_plug.state must equal configured or not_applicable")
    tracking = data.get("results_tracking") if isinstance(data.get("results_tracking"), dict) else {}
    if tracking.get("mode") != "read_only" or not text(tracking.get("registry_path")) or "\x00" in tracking.get("registry_path", ""):
        fail("results_tracking requires read_only mode and a registry_path")
    if tracking.get("milestones_hours") != [24, 168] or tracking.get("state") not in ("awaiting_publication", "tracking"):
        fail("results_tracking requires milestones_hours [24, 168] and a valid state")
    for field in ("draft_id", "social_set_id"):
        if not identifier(tracking.get(field)) or str(tracking[field]) != str(data.get(field)):
            fail(f"results_tracking.{field} does not match plan")
