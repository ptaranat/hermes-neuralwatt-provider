import copy
import dataclasses
import importlib.util
import json
import sys
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PROVIDER_PATH = Path(__file__).resolve().parents[1] / "__init__.py"
BASE = "https://api.neuralwatt.com/v1"


def _stub_hermes():
    """Minimal stand-ins for the Hermes modules the provider imports, used when
    Hermes itself is not installed."""
    try:
        import providers.base  # noqa: F401
        import agent.account_usage  # noqa: F401
        return
    except ImportError:
        pass

    @dataclasses.dataclass
    class ProviderProfile:
        name: str
        aliases: tuple = ()
        display_name: str = ""
        description: str = ""
        signup_url: str = ""
        env_vars: tuple = ()
        base_url: str = ""
        default_aux_model: str = ""
        fallback_models: tuple = ()

    @dataclasses.dataclass(frozen=True)
    class CostResult:
        amount_usd: object
        status: str
        source: str
        label: str
        fetched_at: object = None
        pricing_version: object = None
        notes: tuple = ()

    @dataclasses.dataclass(frozen=True)
    class AccountUsageWindow:
        label: str
        used_percent: object = None
        reset_at: object = None
        detail: object = None
        scope: str = "account"

    @dataclasses.dataclass(frozen=True)
    class AccountUsageSnapshot:
        provider: str
        source: str
        fetched_at: object
        title: str = "Account limits"
        plan: object = None
        windows: tuple = ()
        details: tuple = ()
        unavailable_reason: object = None

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    module("providers", register_provider=lambda profile: None)
    module("providers.base", ProviderProfile=ProviderProfile)
    module("agent")
    module("agent.secret_scope", get_secret=lambda name, default=None: default)
    module("agent.usage_pricing", CostResult=CostResult)
    module("agent.account_usage", AccountUsageSnapshot=AccountUsageSnapshot, AccountUsageWindow=AccountUsageWindow)


_stub_hermes()
SPEC = importlib.util.spec_from_file_location("neuralwatt_provider", PROVIDER_PATH)
nw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nw)


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


@dataclasses.dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    raw_usage: dict | None = None


class ProviderTests(unittest.TestCase):
    def setUp(self):
        nw._catalogs = nw._Cache()
        nw._billing = nw._Cache()
        self.responses = {
            f"{BASE}/models": fixture("models.json"),
            f"{BASE}/quota": fixture("quota-energy.json"),
            f"{BASE}/usage/energy": fixture("energy.json"),
        }
        self.calls = []
        self.secrets = {nw.KEY_ENV: "sk-test"}

        def fake_get(url, api_key="", timeout=0):
            self.calls.append(url)
            return copy.deepcopy(self.responses.get(url))

        for target, replacement in (("_get_json", fake_get), ("_secret", lambda name: self.secrets.get(name, ""))):
            patcher = mock.patch.object(nw, target, side_effect=replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def token_billed(self):
        quota = fixture("quota-energy.json")
        quota["balance"]["accounting_method"] = "token"
        self.responses[f"{BASE}/quota"] = quota

    def test_profile_keeps_recipe_identity(self):
        self.assertEqual(nw.neuralwatt.name, "neuralwatt")
        self.assertEqual(nw.neuralwatt.aliases, ("neural", "neural-watt", "neuralwatt-ai"))
        self.assertEqual(nw.neuralwatt.env_vars, ("NEURALWATT_API_KEY", "NEURALWATT_BASE_URL"))
        self.assertEqual(nw.neuralwatt.base_url, BASE)

    def test_fallbacks_are_live_tracking_aliases(self):
        ids = {item["id"] for item in fixture("models.json")["data"]}
        self.assertLessEqual(set(nw.neuralwatt.fallback_models) | {nw.neuralwatt.default_aux_model}, ids)

    def test_fetch_models_lists_catalog_without_deprecated(self):
        catalog = fixture("models.json")
        catalog["data"][0]["metadata"]["deprecated"] = True
        self.responses[f"{BASE}/models"] = catalog
        models = nw.neuralwatt.fetch_models(api_key="sk-test")
        self.assertEqual(len(models), len(catalog["data"]) - 1)
        self.assertNotIn(catalog["data"][0]["id"], models)
        self.assertIn("nw-large", models)

    def test_fetch_models_failure_returns_none_and_backs_off(self):
        self.responses.pop(f"{BASE}/models")
        self.assertIsNone(nw.neuralwatt.fetch_models())
        self.assertIsNone(nw.neuralwatt.fetch_models())
        self.assertEqual(self.calls.count(f"{BASE}/models"), 1)

    def test_custom_base_url_is_used(self):
        self.responses["https://proxy.example/v1/models"] = fixture("models.json")
        self.assertIn("nw-small", nw.neuralwatt.fetch_models(base_url="https://proxy.example/v1/"))
        self.assertEqual(self.calls, ["https://proxy.example/v1/models"])

    def test_context_length(self):
        self.assertIsNone(nw.neuralwatt.get_model_context_length("nw-small"))
        self.assertEqual(nw.neuralwatt.live_context_length("nw-small"), 262128)
        self.assertEqual(nw.neuralwatt.get_model_context_length("glm-5.3"), 1048560)
        self.assertIsNone(nw.neuralwatt.live_context_length("no-such-model"))
        self.assertEqual(self.calls.count(f"{BASE}/models"), 1)

    def test_reasoning_effort(self):
        extras = nw.neuralwatt.build_api_kwargs_extras
        self.assertEqual(extras(reasoning_config=None, model="nw-small"), ({}, {}))
        self.assertEqual(extras(reasoning_config={"enabled": True, "effort": "Medium"}, model="nw-small"),
                         ({}, {"reasoning_effort": "medium"}))
        self.assertEqual(extras(reasoning_config={"enabled": True, "effort": "turbo"}, model="nw-small"), ({}, {}))
        self.assertEqual(extras(reasoning_config={"enabled": True, "effort": "ultra"}, model="nw-small"),
                         ({}, {"reasoning_effort": "max"}))
        self.assertEqual(extras(reasoning_config={"enabled": True}, model="nw-small"), ({}, {}))

    def test_reasoning_off(self):
        extras = nw.neuralwatt.build_api_kwargs_extras
        off = {"enabled": False}
        self.assertEqual(extras(reasoning_config=off, model="nw-small"), ({}, {"reasoning_effort": "none"}))
        self.assertEqual(extras(reasoning_config=off, model="glm-5.3"), ({}, {"reasoning_effort": "low"}))
        self.assertEqual(extras(reasoning_config=off, model="unknown"), ({}, {"reasoning_effort": "low"}))

    def test_reasoning_off_without_catalog_sends_low(self):
        self.responses.pop(f"{BASE}/models")
        extras = nw.neuralwatt.build_api_kwargs_extras
        self.assertEqual(extras(reasoning_config={"enabled": False}, model="glm-5.3"), ({}, {"reasoning_effort": "low"}))

    def test_every_model_accepts_the_unknown_model_fallback(self):
        for item in fixture("models.json")["data"]:
            self.assertIn("low", item["metadata"]["reasoning"]["accepted_efforts"], item["id"])

    def test_session_id_becomes_conversation_id(self):
        body = nw.neuralwatt.build_extra_body
        self.assertEqual(body(session_id="20261113_101500_ab12"), {"metadata": {"conversation_id": "20261113_101500_ab12"}})
        self.assertEqual(body(session_id=None), {})
        self.assertEqual(body(session_id="x" * 257), {})
        self.assertEqual(body(session_id="bad\nid"), {})

    def test_energy_billed_cost_is_unknown(self):
        cost = nw.neuralwatt.get_usage_cost("nw-small", Usage(input_tokens=1000, output_tokens=100))
        self.assertEqual(cost.status, "unknown")
        self.assertIsNone(cost.amount_usd)
        self.assertIn("/usage", cost.notes[0])

    def test_unknown_billing_cost_is_unknown(self):
        self.responses.pop(f"{BASE}/quota")
        cost = nw.neuralwatt.get_usage_cost("nw-small", Usage(input_tokens=59, output_tokens=8))
        self.assertEqual(cost.status, "unknown")
        self.assertIn("Billing method unknown", cost.notes[0])

    def test_failure_does_not_clobber_fresh_value(self):
        cache = nw._Cache()
        cache.put("k", {"ok": True}, 60)
        cache.put_failure("k", 60)
        self.assertEqual(cache.get("k"), (True, {"ok": True}))
        cache.put("k", None, 60)
        cache.put_failure("k", 60)
        self.assertEqual(cache.get("k"), (True, None))

    def test_token_billed_cost_is_estimated_from_catalog(self):
        self.token_billed()
        usage = Usage(input_tokens=1_000_000, cache_read_tokens=2_000_000, output_tokens=500_000)
        cost = nw.neuralwatt.get_usage_cost("nw-small", usage)
        self.assertEqual(cost.status, "estimated")
        self.assertEqual(cost.source, "provider_models_api")
        self.assertEqual(cost.amount_usd, Decimal("0.45") + Decimal("0.50") + Decimal("1.6"))

    def test_quota_lookup_is_cached(self):
        self.token_billed()
        for _ in range(3):
            nw.neuralwatt.get_usage_cost("nw-small", Usage(input_tokens=10))
        self.assertEqual(self.calls.count(f"{BASE}/quota"), 1)

    def test_reported_cost_in_neuralwatt_shape(self):
        cost = nw.neuralwatt.get_usage_cost("nw-small", Usage(raw_usage={"cost": {"request_cost_usd": 0.000007}}))
        self.assertEqual(cost.amount_usd, Decimal("7e-06"))

    def test_billing_cache_is_per_key(self):
        nw.neuralwatt.fetch_account_usage(api_key="sk-other")
        self.token_billed()
        cost = nw.neuralwatt.get_usage_cost("nw-small", Usage(input_tokens=10))
        self.assertEqual(cost.status, "estimated")

    def test_cost_labels(self):
        self.assertEqual(nw._cost_label(Decimal("0")), "$0.00")
        self.assertEqual(nw._cost_label(Decimal("1.234")), "~$1.23")
        self.assertEqual(nw._cost_label(Decimal("0.0046")), "~$0.0046")
        self.assertEqual(nw._cost_label(Decimal("0.000007")), "~$<0.0001")
        self.assertEqual(nw._cost_label(Decimal("0.00005")), "~$<0.0001")

    def test_auth_header_is_not_forwarded_on_redirect(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            raise OSError("offline")

        mock.patch.stopall()
        with mock.patch.object(nw.urllib.request, "urlopen", side_effect=fake_urlopen):
            self.assertIsNone(nw._get_json(f"{BASE}/quota", "sk-test"))
        request = captured["request"]
        self.assertEqual(request.unredirected_hdrs.get("Authorization"), "Bearer sk-test")
        self.assertNotIn("Authorization", request.headers)

    def test_reported_cost_wins(self):
        cost = nw.neuralwatt.get_usage_cost("nw-small", Usage(raw_usage={"cost": 0.000007}))
        self.assertEqual(cost.status, "actual")
        self.assertEqual(cost.amount_usd, Decimal("0.000007"))

    def test_cost_without_prices_defers_to_hermes(self):
        self.token_billed()
        self.assertIsNone(nw.neuralwatt.get_usage_cost("no-such-model", Usage(input_tokens=10)))

    def test_account_usage(self):
        snapshot = nw.neuralwatt.fetch_account_usage(api_key="sk-test")
        self.assertEqual(snapshot.provider, "neuralwatt")
        self.assertEqual(snapshot.plan, "Pay as you go, billed by energy")
        self.assertEqual(snapshot.details, (
            "Credits: $4.25 left of $10.00",
            "This month: $1.20, 410 requests, 0.130 kWh",
            "Last 30 days: 0.130 kWh, 35.2 g CO2e",
        ))
        (free,) = snapshot.windows
        self.assertEqual(free.label, "Daily free energy")
        self.assertAlmostEqual(free.used_percent, 40.0)
        self.assertEqual(free.scope, "model")
        self.assertIn("GLM-5.3-Flash", free.detail)

    def test_account_usage_with_subscription_and_key_allowance(self):
        quota = fixture("quota-energy.json")
        quota["subscription"] = {"plan": "standard", "current_period_end": "2026-12-11T05:05:25Z",
                                 "kwh_included": 20.0, "kwh_used": 13.9, "in_overage": False}
        quota["key"]["allowance"] = {"limit_usd": 5.0, "period": "monthly", "spent_usd": 1.25, "blocked": False}
        self.responses[f"{BASE}/quota"] = quota
        snapshot = nw.neuralwatt.fetch_account_usage(api_key="sk-test")
        self.assertEqual(snapshot.plan, "Standard plan")
        self.assertEqual([w.label for w in snapshot.windows], ["Plan energy", "Daily free energy", "Key allowance (monthly)"])
        self.assertAlmostEqual(snapshot.windows[0].used_percent, 69.5)
        self.assertEqual(snapshot.windows[2].detail, "$1.25 of $5.00 spent")

    def test_account_usage_feeds_billing_cache(self):
        nw.neuralwatt.fetch_account_usage(api_key="sk-test")
        nw.neuralwatt.get_usage_cost("nw-small", Usage(input_tokens=10))
        self.assertEqual(self.calls.count(f"{BASE}/quota"), 1)

    def test_account_usage_without_energy_endpoint(self):
        self.responses.pop(f"{BASE}/usage/energy")
        snapshot = nw.neuralwatt.fetch_account_usage(api_key="sk-test")
        self.assertEqual(len(snapshot.details), 2)

    def test_account_usage_unavailable(self):
        self.responses.pop(f"{BASE}/quota")
        snapshot = nw.neuralwatt.fetch_account_usage(api_key="sk-test")
        self.assertEqual(snapshot.unavailable_reason, "Neuralwatt account API unavailable")

    def test_account_usage_needs_a_key(self):
        self.secrets.clear()
        self.assertIsNone(nw.neuralwatt.fetch_account_usage())
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
