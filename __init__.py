"""Neuralwatt model provider for Hermes. See README.md."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.neuralwatt.com/v1"
KEY_ENV = "NEURALWATT_API_KEY"
BASE_URL_ENV = "NEURALWATT_BASE_URL"
USER_AGENT = "hermes-neuralwatt-provider/0.1.0"

CATALOG_TTL_S = 3600.0
BILLING_TTL_S = 3600.0
RETRY_AFTER_FAILURE_S = 300.0
HOT_PATH_TIMEOUT_S = 5.0

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_MILLION = Decimal(1_000_000)


def _secret(name: str) -> str:
    try:
        from agent.secret_scope import get_secret

        return (get_secret(name) or "").strip()
    except Exception:
        return ""


def _api_base(base_url: str | None = None) -> str:
    return (base_url or _secret(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def _get_json(url: str, api_key: str = "", timeout: float = HOT_PATH_TIMEOUT_S) -> Any:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
    if api_key:
        request.add_unredirected_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except (OSError, ValueError, urllib.error.URLError) as exc:
        logger.debug("neuralwatt: GET %s failed: %s", url, exc)
        return None


def _billing_key(base: str, api_key: str) -> str:
    return f"{base}|{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _kwh(value: float) -> str:
    return f"{value:.3f} kWh" if value < 10 else f"{value:.1f} kWh"


class _Cache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> tuple[bool, Any]:
        with self._lock:
            entry = self._entries.get(key)
        if entry is None or entry[0] < time.monotonic():
            return False, None
        return True, entry[1]

    def put(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic() + ttl, value)

    def put_failure(self, key: str, ttl: float) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry[1] is None or entry[0] < time.monotonic():
                self._entries[key] = (time.monotonic() + ttl, None)


_catalogs = _Cache()
_billing = _Cache()


class NeuralwattProfile(ProviderProfile):
    def _catalog(self, base_url: str | None = None, api_key: str = "", timeout: float = HOT_PATH_TIMEOUT_S) -> dict[str, dict] | None:
        base = _api_base(base_url)
        hit, catalog = _catalogs.get(base)
        if hit:
            return catalog
        payload = _get_json(f"{base}/models", api_key or _secret(KEY_ENV), timeout)
        items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            _catalogs.put_failure(base, RETRY_AFTER_FAILURE_S)
            return None
        catalog = {item["id"]: item for item in items if isinstance(item, dict) and isinstance(item.get("id"), str)}
        _catalogs.put(base, catalog, CATALOG_TTL_S)
        return catalog

    def _model(self, model: str | None, base_url: str | None = None) -> dict | None:
        if not model:
            return None
        return (self._catalog(base_url) or {}).get(model)

    def fetch_models(self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0) -> list[str] | None:
        catalog = self._catalog(base_url, api_key or "", timeout)
        if catalog is None:
            return None
        return [model_id for model_id, item in catalog.items() if not (item.get("metadata") or {}).get("deprecated")]

    def live_context_length(self, model: str, *, api_key: str = "", base_url: str = "") -> int | None:
        item = (self._catalog(base_url or None, api_key) or {}).get(model)
        return _context_length(item)

    def get_model_context_length(self, model: str) -> int | None:
        hit, catalog = _catalogs.get(_api_base())
        return _context_length((catalog or {}).get(model)) if hit else None

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, **context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(reasoning_config, dict):
            return {}, {}
        if reasoning_config.get("enabled") is False:
            effort = _off_effort(self._model(context.get("model"), context.get("base_url")))
        else:
            effort = str(reasoning_config.get("effort") or "").strip().lower()
            effort = "max" if effort == "ultra" else effort
        return ({}, {"reasoning_effort": effort}) if effort in EFFORTS else ({}, {})

    def build_extra_body(self, *, session_id: str | None = None, **context: Any) -> dict[str, Any]:
        conversation_id = (session_id or "").strip()
        if not conversation_id or len(conversation_id) > 256 or not conversation_id.isprintable():
            return {}
        return {"metadata": {"conversation_id": conversation_id}}

    def _accounting_method(self) -> str | None:
        api_key = _secret(KEY_ENV)
        if not api_key:
            return None
        base = _api_base()
        key = _billing_key(base, api_key)
        hit, method = _billing.get(key)
        if hit:
            return method
        quota = _get_json(f"{base}/quota", api_key)
        method = _accounting_method_of(quota)
        if method:
            _billing.put(key, method, BILLING_TTL_S)
        else:
            _billing.put_failure(key, RETRY_AFTER_FAILURE_S)
        return method

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        from agent.usage_pricing import CostResult

        raw = getattr(usage, "raw_usage", None) or {}
        reported = _reported_cost(raw.get("cost")) if isinstance(raw, dict) else None
        if reported is not None:
            return CostResult(amount_usd=Decimal(str(reported)), status="actual", source="provider_cost_api",
                              label=_cost_label(Decimal(str(reported))), fetched_at=datetime.now(timezone.utc))
        method = self._accounting_method()
        if method != "token":
            note = "Billed by energy, not tokens." if method == "energy" else "Billing method unknown."
            return CostResult(amount_usd=None, status="unknown", source="none", label="n/a",
                              notes=(f"{note} /usage shows actual spend.",))
        pricing = (self._model(model) or {}).get("metadata", {}).get("pricing") or {}
        amount = _token_cost(pricing, usage)
        if amount is None:
            return None
        return CostResult(amount_usd=amount, status="estimated", source="provider_models_api",
                          label=_cost_label(amount), fetched_at=datetime.now(timezone.utc),
                          pricing_version="neuralwatt-v1-models")

    def fetch_account_usage(self, *, base_url: str | None = None, api_key: str | None = None) -> Any | None:
        from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow

        api_key = (api_key or _secret(KEY_ENV)).strip()
        if not api_key:
            return None
        base = _api_base(base_url)
        now = datetime.now(timezone.utc)
        quota = _get_json(f"{base}/quota", api_key, timeout=4.0)
        if not isinstance(quota, dict):
            return AccountUsageSnapshot(provider=self.name, source="neuralwatt_quota_api", fetched_at=now,
                                        title="Neuralwatt account", unavailable_reason="Neuralwatt account API unavailable")
        method = _accounting_method_of(quota)
        if method:
            _billing.put(_billing_key(base, api_key), method, BILLING_TTL_S)
        energy = _get_json(f"{base}/usage/energy", api_key, timeout=4.0)
        windows = [AccountUsageWindow(**window) for window in _quota_windows(quota)]
        details = _quota_details(quota) + _energy_details(energy)
        return AccountUsageSnapshot(provider=self.name, source="neuralwatt_quota_api", fetched_at=now,
                                    title="Neuralwatt account", plan=_plan_label(quota),
                                    windows=tuple(windows), details=tuple(details))


def _context_length(item: dict | None) -> int | None:
    if not item:
        return None
    limits = (item.get("metadata") or {}).get("limits") or {}
    for value in (item.get("max_model_len"), limits.get("max_context_length")):
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _reported_cost(cost: Any) -> float | None:
    return _number(cost.get("request_cost_usd")) if isinstance(cost, dict) else _number(cost)


def _off_effort(item: dict | None) -> str:
    """``none`` where accepted, else the lightest level. Unknown models get ``low``, which all accept."""
    if not item:
        return "low"
    reasoning = (item.get("metadata") or {}).get("reasoning") or {}
    accepted = reasoning.get("accepted_efforts") or ()
    if "none" in accepted:
        return "none"
    supported = [effort for effort in reasoning.get("supported_efforts") or () if effort in EFFORTS]
    return min(supported, key=EFFORTS.index) if supported else ""


def _token_cost(pricing: dict, usage: Any) -> Decimal | None:
    if pricing.get("pricing_tbd"):
        return None
    rates = {key: _number(pricing.get(key)) for key in ("input_per_million", "output_per_million", "cached_input_per_million")}
    if rates["input_per_million"] is None or rates["output_per_million"] is None:
        return None
    cached_rate = rates["cached_input_per_million"] if rates["cached_input_per_million"] is not None else rates["input_per_million"]
    tokens = {name: int(getattr(usage, name, 0) or 0) for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}
    total = (
        Decimal(str(rates["input_per_million"])) * (tokens["input_tokens"] + tokens["cache_write_tokens"])
        + Decimal(str(cached_rate)) * tokens["cache_read_tokens"]
        + Decimal(str(rates["output_per_million"])) * tokens["output_tokens"]
    )
    return total / _MILLION


def _cost_label(amount: Decimal) -> str:
    if amount == 0:
        return "$0.00"
    if amount >= Decimal("0.01"):
        return f"~${amount:.2f}"
    label = f"~${amount:.4f}"
    return label if label != "~$0.0000" else "~$<0.0001"


def _accounting_method_of(quota: Any) -> str | None:
    if not isinstance(quota, dict):
        return None
    method = (quota.get("balance") or {}).get("accounting_method")
    return method if method in ("token", "energy") else None


def _plan_label(quota: dict) -> str | None:
    subscription = quota.get("subscription") or {}
    if subscription.get("plan"):
        return f"{str(subscription['plan']).title()} plan"
    method = _accounting_method_of(quota)
    return {"energy": "Pay as you go, billed by energy", "token": "Pay as you go, billed by tokens"}.get(method or "")


def _quota_windows(quota: dict) -> list[dict]:
    windows = []
    subscription = quota.get("subscription") or {}
    included, used = _number(subscription.get("kwh_included")), _number(subscription.get("kwh_used"))
    if included and used is not None:
        windows.append({
            "label": "Plan energy",
            "used_percent": min(100.0, used / included * 100),
            "reset_at": _parse_time(subscription.get("current_period_end")),
            "detail": f"{_kwh(used)} of {_kwh(included)} used" + (", in overage" if subscription.get("in_overage") else ""),
        })
    free = quota.get("daily_free_energy") or {}
    included, used = _number(free.get("kwh_included")), _number(free.get("kwh_used"))
    if included and used is not None:
        models = ", ".join(str(label) for label in free.get("model_labels") or ())
        windows.append({
            "label": "Daily free energy",
            "used_percent": min(100.0, used / included * 100),
            "reset_at": _parse_time(free.get("reset_at")),
            "detail": f"{_kwh(used)} of {_kwh(included)} used" + (f" (covers {models})" if models else ""),
            "scope": "model",
        })
    allowance = (quota.get("key") or {}).get("allowance") or {}
    limit, spent = _number(allowance.get("limit_usd")), _number(allowance.get("spent_usd"))
    if limit and spent is not None:
        windows.append({
            "label": f"Key allowance ({allowance.get('period') or 'period'})",
            "used_percent": min(100.0, spent / limit * 100),
            "detail": f"${spent:.2f} of ${limit:.2f} spent" + (", key blocked" if allowance.get("blocked") else ""),
        })
    return windows


def _quota_details(quota: dict) -> list[str]:
    details = []
    balance = quota.get("balance") or {}
    remaining, total = _number(balance.get("credits_remaining_usd")), _number(balance.get("total_credits_usd"))
    if remaining is not None:
        details.append(f"Credits: ${remaining:.2f} left" + (f" of ${total:.2f}" if total else ""))
    month = (quota.get("usage") or {}).get("current_month") or {}
    parts = []
    if (cost := _number(month.get("cost_usd"))) is not None:
        parts.append(f"${cost:.2f}")
    if (requests := _number(month.get("requests"))) is not None:
        parts.append(f"{int(requests):,} requests")
    if (kwh := _number(month.get("energy_kwh"))) is not None:
        parts.append(_kwh(kwh))
    if parts:
        details.append("This month: " + ", ".join(parts))
    return details


def _energy_details(energy: Any) -> list[str]:
    totals = (energy or {}).get("totals") if isinstance(energy, dict) else None
    if not isinstance(totals, dict):
        return []
    kwh, carbon = _number(totals.get("energy_kwh")), _number(totals.get("carbon_g_co2eq"))
    if kwh is None:
        return []
    line = f"Last 30 days: {_kwh(kwh)}"
    if carbon is not None:
        line += f", {carbon:,.1f} g CO2e" if carbon < 1000 else f", {carbon / 1000:,.2f} kg CO2e"
    return [line]


neuralwatt = NeuralwattProfile(
    name="neuralwatt",
    aliases=("neural", "neural-watt", "neuralwatt-ai"),
    display_name="Neuralwatt",
    description="Neuralwatt: energy-transparent inference (GLM, Kimi, Qwen, DeepSeek)",
    signup_url="https://portal.neuralwatt.com",
    env_vars=(KEY_ENV, BASE_URL_ENV),
    base_url=DEFAULT_BASE_URL,
    default_aux_model="nw-flash",
    fallback_models=("nw-large", "nw-flash", "nw-small"),
)

register_provider(neuralwatt)
