# S3 Case Study: LLM vs Rule-Based Priority Inference

Showing top 3 cases where LLM outperformed rule-based by ≥ 1 priority levels.

## Case 1: `DEM_005_06` (window `2024-03-15T00:25-00:30::direct`)

| Field | Value |
|-------|-------|
| Ground truth priority | P4 |
| Rule-based prediction | P2 (error=2) |
| LLM prediction        | P4 (error=0) |
| Priority gain         | 2 levels |
| Event ID              | `DEM_005_06` |

**Dialogue snippet:**
```
[00:25] Office Administrator: I placed an urgent order for 8.8 kg of OTC medication for same-day delivery to DEM_25495. The delivery deadline is 120 minutes.
[00:26] Dispatcher: That’s confirmed. We need to make sure the recipient is ready to receive it as a household member will collect it after notification. Is that correct?
[00:27] Office Administrator: Yes, exactly. They’re prepared to sign for it upon arrival.
[00:28] Dispatcher: Great. I’ll pack the medication in a standard tamper-evident pharmacy bag and dispatch it immediately from Commercial Distribution Hub COM_5.
```

## Case 2: `DEM_013_04` (window `2024-03-15T01:05-01:10::direct`)

| Field | Value |
|-------|-------|
| Ground truth priority | P4 |
| Rule-based prediction | P2 (error=2) |
| LLM prediction        | P4 (error=0) |
| Priority gain         | 2 levels |
| Event ID              | `DEM_013_04` |

**Dialogue snippet:**
```
[01:05] Family Caregiver: Requesting same-day delivery of 24.1 kg of food from Commercial Distribution Hub COM_52 to DEM_6303. It's essential for our family. 
[01:06] Dispatcher: Delivery is required within 120 minutes, and is the landing zone ready? 
[01:07] Family Caregiver: Yes, a household member will collect it right after the notification. Can it be dropped off at a community locker? 
[01:08] Dispatcher: Absolutely, we’ll handle it with care and keep the thermal bag closed until handoff.
```

## Case 3: `DEM_013_00` (window `2024-03-15T01:05-01:10::direct`)

| Field | Value |
|-------|-------|
| Ground truth priority | P4 |
| Rule-based prediction | P2 (error=2) |
| LLM prediction        | P4 (error=0) |
| Priority gain         | 2 levels |
| Event ID              | `DEM_013_00` |

**Dialogue snippet:**
```
[01:05] Office Administrator (DEM_4281): Clinical dispatch request. Please route 220 boxs of OTC medication (22.0 kg) from Commercial Distribution Hub COM_52. A same-day home-care order needs symptom relief medication. Delivery is needed within 120 minutes. The request comes from the office administrator. Landing zone cleared; team waiting for immediate handoff It is an overnight request, so the on-site team is working with a reduced backup stock. The receiving point is a office. The receiver serves a vulnerable population.
[01:05] Delivery Platform: Request acknowledged. Pack the order in a standard tamper-evident pharmacy bag. Departure from Commercial Distribution Hub COM_52 is being prioritized. Planned ETA is 120 min. A locker drop-off is preferred.
[01:05] Office Administrator: Confi
```
