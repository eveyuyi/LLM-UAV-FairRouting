# S3 Case Study: LLM vs Rule-Based Priority Inference

Showing top 3 cases where LLM outperformed rule-based by ≥ 1 priority levels.

## Case 1: `DEM_000_04` (window `2024-03-15T00:00-00:05::direct`)

| Field | Value |
|-------|-------|
| Ground truth priority | P1 |
| Rule-based prediction | P2 (error=1) |
| LLM prediction        | P1 (error=0) |
| Priority gain         | 1 levels |
| Event ID              | `DEM_000_04` |

**Dialogue snippet:**
```
[00:00] ER Physician (DEM_11531): Clinical dispatch request. Please route 368 doses of cardiac emergency drug (18.4 kg) from Medical Supply Center MED_531. The emergency cart is short on the cardiac rescue dose. Delivery is needed within 15 minutes. The request comes from the emergency doctor. Landing zone cleared; team waiting for immediate handoff This is the overnight shift and local stock is already depleted. The receiving point is a public space. The receiver serves a vulnerable population.
[00:00] Emergency Dispatch: Request acknowledged. Please follow standard packaging and handoff checks. Departure from Medical Supply Center MED_531 is being prioritized. Planned ETA is 15 min. Please use the closest emergency drop point for handoff.
[00:00] ER Physician: Confirmed. A resuscitation 
```

## Case 2: `DEM_000_08` (window `2024-03-15T00:00-00:05::technical`)

| Field | Value |
|-------|-------|
| Ground truth priority | P1 |
| Rule-based prediction | P2 (error=1) |
| LLM prediction        | P1 (error=0) |
| Priority gain         | 1 levels |
| Event ID              | `DEM_000_08` |

**Dialogue snippet:**
```
[00:00] ER Physician (DEM_14909): Clinical dispatch request. Please route 478 doses of cardiac emergency drug (23.9 kg) from Medical Supply Center MED_136. The emergency cart is short on the cardiac rescue dose. Delivery is needed within 15 minutes. The request comes from the emergency doctor. Landing zone cleared; team waiting for immediate handoff It is an overnight request, so the on-site team is working with a reduced backup stock. The receiving point is a public space. The receiver serves a vulnerable population.
[00:00] Emergency Dispatch: Request acknowledged. Please follow standard packaging and handoff checks. Departure from Medical Supply Center MED_136 is being prioritized. Planned ETA is 15 min. Please route straight to the emergency receiving pad.
[00:00] ER Physician: Confirm
```

## Case 3: `DEM_002_02` (window `2024-03-15T00:10-00:15::direct`)

| Field | Value |
|-------|-------|
| Ground truth priority | P1 |
| Rule-based prediction | P2 (error=1) |
| LLM prediction        | P1 (error=0) |
| Priority gain         | 1 levels |
| Event ID              | `DEM_002_02` |

**Dialogue snippet:**
```
[00:10] Paramedic: We’re in a critical situation! The emergency cart is short on the cardiac rescue dose. We need 7.8 kg of cardiac emergency drug delivered to DEM_10550 immediately.
[00:11] Dispatcher: Acknowledged. A drone is being dispatched from Medical Supply Center MED_531. Delivery is required within 15 minutes. Is the landing zone clear?
[00:12] Paramedic: Yes, the team is standing by at the landing zone, ready for handoff.
[00:13] Dispatcher: Great! Please keep the standard packaging protocol in mind. The drone will head straight for the emergency receiving pad.
```
