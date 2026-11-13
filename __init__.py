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

ENERGY_MATRIX_URL = "https://portal.neuralwatt.com/api/energy-pricing/matrix"
LEGACY_CREDIT_RATE_PER_KWH = Decimal("5")

CATALOG_TTL_S = 3600.0
MATRIX_TTL_S = 900.0
BILLING_TTL_S = 600.0
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
_matrix = _Cache()


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

    def _billing_profile(self) -> dict | None:
        api_key = _secret(KEY_ENV)
        if not api_key:
            return None
        base = _api_base()
        key = _billing_key(base, api_key)
        hit, billing = _billing.get(key)
        if hit:
            return billing
        billing = _billing_of(_get_json(f"{base}/quota", api_key))
        if billing:
            _billing.put(key, billing, BILLING_TTL_S)
        else:
            _billing.put_failure(key, RETRY_AFTER_FAILURE_S)
        return billing

    def _energy_matrix(self) -> dict | None:
        hit, matrix = _matrix.get(ENERGY_MATRIX_URL)
        if hit:
            return matrix
        matrix = _get_json(ENERGY_MATRIX_URL)
        if isinstance(matrix, dict) and isinstance(matrix.get("models"), list):
            _matrix.put(ENERGY_MATRIX_URL, matrix, MATRIX_TTL_S)
            return matrix
        _matrix.put_failure(ENERGY_MATRIX_URL, RETRY_AFTER_FAILURE_S)
        return None

    def _energy_cost(self, model: str, usage: Any, billing: dict) -> Any:
        from agent.usage_pricing import CostResult

        catalog = self._catalog() or {}
        item = catalog.get(model) or {}
        if billing["plan_active"]:
            return _included_cost(CostResult, "Drawn from your plan's kWh allowance.")
        if billing["free_remaining_kwh"] > 0 and _free_energy_covers(model, item, billing["free_labels"]):
            return _included_cost(CostResult, "Covered by today's free energy.")
        matrix = self._energy_matrix()
        row, multiplier = _matrix_row(model, catalog, matrix)
        wh = _wh_for_prompt(row, matrix, _prompt_tokens(usage))
        rate = LEGACY_CREDIT_RATE_PER_KWH if billing["legacy_credits"] else _energy_rate(matrix)
        if wh is None or rate is None:
            return CostResult(amount_usd=None, status="unknown", source="none", label="n/a",
                              notes=("No published energy figure for this model and prompt size. /usage shows actual spend.",))
        amount = Decimal(str(wh)) / 1000 * rate * multiplier
        if row.get("lane") != "speed":
            ceiling = _token_cost((item.get("metadata") or {}).get("pricing") or {}, usage)
            if ceiling is not None:
                amount = min(amount, ceiling)
        return CostResult(amount_usd=amount, status="estimated", source="provider_models_api",
                          label=_cost_label(amount), fetched_at=datetime.now(timezone.utc),
                          pricing_version=f"neuralwatt-energy-matrix@{matrix.get('generated_at', '')}",
                          notes=("Estimated from Neuralwatt's average energy per request. /usage shows actual spend.",))

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        from agent.usage_pricing import CostResult

        raw = getattr(usage, "raw_usage", None) or {}
        reported = _reported_cost(raw.get("cost")) if isinstance(raw, dict) else None
        if reported is not None:
            return CostResult(amount_usd=Decimal(str(reported)), status="actual", source="provider_cost_api",
                              label=_cost_label(Decimal(str(reported))), fetched_at=datetime.now(timezone.utc))
        billing = self._billing_profile()
        if billing and billing["method"] == "energy":
            return self._energy_cost(model, usage, billing)
        if not billing or billing["method"] != "token":
            return CostResult(amount_usd=None, status="unknown", source="none", label="n/a",
                              notes=("Billing method unknown. /usage shows actual spend.",))
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
        if billing := _billing_of(quota):
            _billing.put(_billing_key(base, api_key), billing, BILLING_TTL_S)
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


def _billing_of(quota: Any) -> dict | None:
    method = _accounting_method_of(quota)
    if not method:
        return None
    subscription = quota.get("subscription") or {}
    free = quota.get("daily_free_energy") or {}
    reset = _parse_time(free.get("reset_at"))
    free_remaining = _number(free.get("kwh_remaining")) or 0.0
    if reset and reset <= datetime.now(timezone.utc):
        free_remaining = _number(free.get("kwh_included")) or 0.0
    plan_remaining = _number(subscription.get("kwh_remaining"))
    if plan_remaining is None and _number(subscription.get("kwh_included")) is not None:
        plan_remaining = _number(subscription.get("kwh_included")) - (_number(subscription.get("kwh_used")) or 0.0)
    return {
        "method": method,
        "plan_active": subscription.get("status", "active") in ("active", "trialing", "canceling")
        and not subscription.get("in_overage") and (plan_remaining or 0) > 0,
        "free_remaining_kwh": free_remaining,
        "free_labels": [str(label) for label in free.get("model_labels") or ()],
        "legacy_credits": (_number((quota.get("balance") or {}).get("legacy_credits_usd")) or 0) > 0,
    }


def _included_cost(cost_result: Any, note: str) -> Any:
    return cost_result(amount_usd=Decimal(0), status="included", source="none", label="included",
                       fetched_at=datetime.now(timezone.utc), notes=(note,))


def _free_energy_covers(model: str, item: dict, labels: list[str]) -> bool:
    if model.endswith("-speed"):
        return False
    name = str((item.get("metadata") or {}).get("huggingface_id") or "").rsplit("/", 1)[-1].lower()
    return bool(name) and any(name == label.lower() or name.startswith(label.lower() + "-") for label in labels)


def _matrix_row(model: str, catalog: dict, matrix: dict | None) -> tuple[dict, Decimal]:
    """Flex ids use their standard row and the flex multiplier; aliases use their target's row."""
    rows = {row.get("id"): row for row in (matrix or {}).get("models") or () if isinstance(row, dict)}
    item = catalog.get(model) or {}
    pricing = (item.get("metadata") or {}).get("pricing") or {}
    multiplier = Decimal("1")
    if pricing.get("service_tier") == "flex" or model.endswith("-flex"):
        multiplier = Decimal(str(_number(pricing.get("flex_discount_multiplier")) or 0.65))
        model = model.removesuffix("-flex")
    if model in rows:
        return rows[model], multiplier
    target = ((catalog.get(model) or item).get("metadata") or {}).get("huggingface_id")
    for candidate, candidate_item in catalog.items():
        row = rows.get(candidate)
        if row and row.get("lane") == "standard" and (candidate_item.get("metadata") or {}).get("huggingface_id") == target:
            return row, multiplier
    return {}, multiplier


def _prompt_tokens(usage: Any) -> int:
    return sum(int(getattr(usage, name, 0) or 0) for name in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))


def _wh_for_prompt(row: dict, matrix: dict | None, prompt_tokens: int) -> float | None:
    for bucket, figure in zip((matrix or {}).get("prompt_size_buckets") or (), row.get("buckets") or ()):
        low, high = bucket.get("min_prompt_tokens") or 0, bucket.get("max_prompt_tokens")
        if prompt_tokens >= low and (high is None or prompt_tokens < high):
            return _number((figure or {}).get("wh_per_request"))
    return None


def _energy_rate(matrix: dict | None) -> Decimal | None:
    rate = (matrix or {}).get("energy_rate") or {}
    amount = _number(rate.get("amount"))
    return Decimal(str(amount)) if amount and rate.get("per") == "kWh" and rate.get("currency") == "USD" else None


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
