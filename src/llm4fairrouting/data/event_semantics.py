"""Domain semantics for simulated demand-event generation.

This module is the current source of truth for how raw simulated demand events
are lifted into `EventCore`.

Important design note:
- today these rules are still heuristic priors plus random sampling
- they are centralized here on purpose so the policy is easy to inspect
- this is the intended seam for a future move to external YAML/JSON rule config

In other words, the generation policy is not "learned" or hidden. It is a
legible rule table that can be discussed, audited, and revised.
"""

from __future__ import annotations

import random
from typing import Sequence

MEDICAL_SUPPLY_TYPE = "medical"
COMMERCIAL_SUPPLY_TYPE = "commercial"
DIALOGUE_STYLE_VARIANTS = ("direct", "technical", "conversational", "dispatch_brief")

PRIORITY_TO_TIER = {
    1: "life_support",
    2: "critical",
    3: "regular",
    4: "consumer",
}

PRIORITY_TO_DEADLINE = {
    1: 15,
    2: 30,
    3: 60,
    4: 120,
}

MATERIAL_LABELS = {
    "aed": "AED defibrillator",
    "blood_product": "blood product",
    "cardiac_drug": "cardiac emergency drug",
    "thrombolytic": "thrombolytic agent",
    "ventilator": "ventilator",
    "icu_drug": "ICU medication",
    "vaccine": "vaccine",
    "medicine": "medication",
    "protective_suit": "protective suit",
    "mask": "face mask",
    "disinfectant": "disinfectant solution",
    "food": "food/meal",
    "otc_drug": "OTC medication",
    "daily_supply": "daily supplies",
}

TEMPERATURE_SENSITIVE_MATERIALS = {"blood_product", "thrombolytic", "vaccine"}

UNIT_BY_MATERIAL = {
    "aed": "unit",
    "blood_product": "pack",
    "cardiac_drug": "dose",
    "thrombolytic": "dose",
    "ventilator": "unit",
    "icu_drug": "kit",
    "vaccine": "dose",
    "medicine": "box",
    "protective_suit": "set",
    "mask": "pack",
    "disinfectant": "bottle",
    "food": "meal",
    "otc_drug": "box",
    "daily_supply": "package",
}

# NOTE: The following tables are intentionally explicit. They document the
# current EventCore construction policy in a configuration-like form, even
# though they still live in Python today.
SUPPLY_PRIORITY_MATERIAL_CANDIDATES = {
    MEDICAL_SUPPLY_TYPE: {
        1: ("aed", "blood_product", "cardiac_drug", "thrombolytic"),
        2: ("ventilator", "icu_drug"),
        3: ("vaccine", "medicine", "protective_suit"),
        4: ("medicine", "vaccine"),
    },
    COMMERCIAL_SUPPLY_TYPE: {
        1: ("medicine", "protective_suit", "disinfectant"),
        2: ("medicine", "mask", "disinfectant"),
        3: ("food", "mask", "daily_supply"),
        4: ("food", "daily_supply", "otc_drug"),
    },
}

DESTINATION_OPTIONS_BY_PRIORITY = {
    1: ("hospital", "public_space"),
    2: ("hospital", "clinic"),
    3: ("clinic", "community_health_center", "residential_area"),
    4: ("residential_area", "office", "community_locker"),
}
VACCINE_DESTINATION_OPTIONS = ("clinic", "community_health_center")

REQUESTER_ROLE_OPTIONS_BY_PRIORITY = {
    1: ("emergency_doctor", "paramedic", "triage_nurse"),
    2: ("icu_nurse", "clinical_pharmacist", "ward_coordinator"),
    3: ("community_health_worker", "clinic_manager", "pharmacy_staff"),
    4: ("consumer", "family_caregiver", "office_administrator"),
}

RECEIVER_READY_PROBABILITY = {
    1: 0.95,
    2: 0.80,
    3: 0.60,
    4: 0.45,
}

VULNERABILITY_CHILD_PRIORITY_PROB = 0.25
VULNERABILITY_ELDERLY_RANDOM_PROB = 0.30
ELDERLY_RATIO_RANGE = (0.15, 0.62)
COMMUNITY_POPULATION_RANGE = (600, 9000)

SCENARIO_CONTEXT_OVERRIDES = {
    (1, "aed"): "CPR is already in progress and the local AED cabinet is empty.",
    (1, "blood_product"): "An active transfusion case needs immediate blood product support.",
    (1, "cardiac_drug"): "The emergency cart is short on the cardiac rescue dose.",
    (1, "thrombolytic"): "The stroke treatment window is closing quickly.",
    (2, "ventilator"): "A critical patient needs a backup ventilator before the next procedure.",
    (2, "icu_drug"): "The ICU team needs a refill before the next administration round.",
    (3, "vaccine"): "The afternoon vaccination block starts soon and stock is running low.",
    (4, "otc_drug"): "A same-day home-care order needs symptom relief medication.",
}


def material_candidates(supply_type: str, priority: int) -> tuple[str, ...]:
    """Choose material candidates from transparent supply/priority lookup tables."""
    mapping = SUPPLY_PRIORITY_MATERIAL_CANDIDATES.get(
        supply_type,
        SUPPLY_PRIORITY_MATERIAL_CANDIDATES[COMMERCIAL_SUPPLY_TYPE],
    )
    return mapping.get(int(priority), mapping[4])


def destination_type_for_priority(priority: int, material_type: str, rng: random.Random) -> str:
    """Sample destination type from explicit priority-dependent options."""
    options = DESTINATION_OPTIONS_BY_PRIORITY.get(int(priority), DESTINATION_OPTIONS_BY_PRIORITY[4])
    if material_type == "vaccine":
        options = VACCINE_DESTINATION_OPTIONS
    return rng.choice(options)


def requester_role_for_priority(priority: int, rng: random.Random) -> str:
    """Sample requester role from explicit priority-dependent options."""
    return rng.choice(
        REQUESTER_ROLE_OPTIONS_BY_PRIORITY.get(
            int(priority),
            REQUESTER_ROLE_OPTIONS_BY_PRIORITY[4],
        )
    )


def special_handling_for_material(material_type: str) -> list[str]:
    special_handling: list[str] = []
    if material_type in TEMPERATURE_SENSITIVE_MATERIALS:
        special_handling.append("cold_chain")
    if material_type in {"aed", "ventilator"}:
        special_handling.append("shock_protection")
    return special_handling


def population_vulnerability(
    material_type: str,
    priority: int,
    destination_type: str,
    rng: random.Random,
) -> dict[str, object]:
    """Construct a transparent vulnerability profile for the receiving population.

    Current policy:
    - vaccines imply child-serving contexts more often
    - higher-priority clinical requests are more likely to involve children
    - clinic-like destinations are more likely to involve elderly recipients
    - population size and elderly ratio are sampled from bounded ranges

    This is still heuristic, but the heuristic is intentionally exposed here so
    it can be reviewed and later moved into configuration without changing the
    rest of the pipeline.
    """
    children = material_type == "vaccine" or (
        priority <= 2 and rng.random() < VULNERABILITY_CHILD_PRIORITY_PROB
    )
    elderly = destination_type in {"clinic", "community_health_center"} or (
        rng.random() < VULNERABILITY_ELDERLY_RANDOM_PROB
    )
    elderly_ratio = round(rng.uniform(*ELDERLY_RATIO_RANGE), 2)
    population = rng.randint(*COMMUNITY_POPULATION_RANGE)
    return {
        "elderly_ratio": elderly_ratio,
        "population": population,
        "elderly_involved": elderly,
        "children_involved": children,
        "vulnerable_community": bool(children or elderly or elderly_ratio >= 0.45),
    }


def receiver_ready(priority: int, rng: random.Random) -> bool:
    """Sample operational readiness from a transparent priority-weighted prior."""
    return rng.random() < RECEIVER_READY_PROBABILITY.get(int(priority), RECEIVER_READY_PROBABILITY[4])


def operational_readiness(receiver_ready_value: bool, priority: int) -> str:
    if receiver_ready_value:
        if priority <= 2:
            return "Landing zone cleared; team waiting for immediate handoff"
        return "Receiver confirmed and ready for handoff"
    return "Standard handoff readiness"


def scenario_context(priority: int, material_type: str, destination_type: str) -> str:
    """Generate concise scenario context from explicit overrides and safe fallbacks."""
    if (priority, material_type) in SCENARIO_CONTEXT_OVERRIDES:
        return SCENARIO_CONTEXT_OVERRIDES[(priority, material_type)]
    if destination_type in {"hospital", "clinic", "community_health_center"}:
        return "The local care team has a time-bound replenishment need."
    return "The receiver requested a same-day drone delivery."


def unique_keywords(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value).lower() for value in values if value))
