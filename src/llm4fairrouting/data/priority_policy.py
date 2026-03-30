"""Transparent human-aligned priority policy shared across data generation and training."""

from __future__ import annotations

PRIORITY_POLICY_VERSION = "human_aligned_priority_v1"

# These principles document the intended behavior behind contribution 2:
# scarce logistics capacity should be allocated according to concrete need,
# not surface urgency wording alone.
HUMAN_ALIGNED_PRIORITY_PRINCIPLES = {
    "clinical_severity": "Life-support and clinically critical requests outrank routine demand.",
    "time_criticality": "Hard and short deadlines increase urgency when resources are scarce.",
    "requester_context": "Emergency and critical-care roles provide reliable urgency evidence.",
    "population_vulnerability": "Children, elderly groups, and vulnerable communities deserve stronger protection.",
    "special_handling": "Cold-chain or fragile medical cargo requires more careful prioritization.",
    "operational_readiness": "Ready receivers and explicit handoff preparation improve actionability for dispatch.",
}

PRIORITY_REASON_CODE_DOCS = {
    "tier_life_support": "The request belongs to the life-support tier.",
    "tier_critical": "The request belongs to the critical-care tier.",
    "tier_regular": "The request belongs to the regular clinical tier.",
    "tier_consumer": "The request belongs to the consumer tier.",
    "deadline_le_15m": "The request has an extreme 15-minute deadline.",
    "deadline_le_30m": "The request has a hard 30-minute deadline.",
    "deadline_le_60m": "The request has a same-window 60-minute deadline.",
    "emergency_requester": "The requester role is an emergency role.",
    "critical_requester": "The requester role is a critical-care role.",
    "special_handling_cold_chain": "The request requires cold-chain handling.",
    "special_handling_shock_protection": "The request requires shock protection.",
    "vulnerable_population": "The receiving population includes vulnerable groups.",
    "children_involved": "Children are involved in the receiving population.",
    "elderly_involved": "Elderly recipients are involved in the receiving population.",
    "receiver_ready": "The receiver is operationally ready for handoff.",
    "destination_hospital_like": "The destination is a hospital-like care facility.",
    "scenario_context_present": "The scenario context provides direct situational evidence.",
}
