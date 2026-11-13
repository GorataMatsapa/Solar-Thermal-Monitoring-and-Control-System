# STMaCS HIL test matrix (software HIL)

All data below is **HIL SIMULATED** (digital twin + injected faults). EXPERIMENTAL HIL TEST SETTING - not a validated HOTSET safety limit.

| Test | Scenario | Events (type / severity / detection delay) | Peak S1 (°C) | Result |
|---|---|---|---|---|
| T1 | Normal operation | none | 56.9 | PASS |
| T2 | Pump failure / no circulation | PUMP_FAILURE major 40.0 s; STAGNATION major 420.0 s | 120.0 | PASS |
| T3 | Low-flow condition | LOW_FLOW major 70.0 s | 77.8 | PASS |
| T4A | Stagnation - NO mitigation (Run A) | PUMP_FAILURE critical 40.0 s; STAGNATION critical 380.0 s | 146.2 | PASS |
| T4B | Stagnation - WITH simulated mitigation (Run B) | PUMP_FAILURE major 40.0 s; STAGNATION major 380.0 s | 120.2 | PASS |
| T5 | Sensor failure (validated inputs) | SENSOR_FAULT/S3 warning 30.0 s; SENSOR_FAULT/S1 major 30.0 s; SENSOR_FAULT/S4 major 30.0 s | 75.1 | PASS |
| T5R | Sensor failure - RAW inputs (comparison only) | SENSOR_FAULT/S3 warning 30.0 s; SENSOR_FAULT/S1 major 30.0 s; SENSOR_FAULT/S4 major 30.0 s | 80.4 | PASS |
| T6 | Fault recovery | PUMP_FAILURE major 40.0 s; STAGNATION warning 420.0 s | 90.3 | PASS |

## Checks

### T1 - Normal operation
*Purpose:* Baseline operation, no faults  
*Run id:* `T1-20261103-091033`

- PASS - no faults logged (0 events)
- PASS - pump runs under sun (pump ON 100% of the time after 5 min)
- PASS - pump command == actual (command and actual never differ)
- PASS - collector stays below warning level (max S1 56.9 C)

### T2 - Pump failure / no circulation
*Purpose:* Pump commanded ON, actuator forced OFF, flow zero  
*Run id:* `T2-20261103-091034`

- PASS - pump failure event logged (1 event(s))
- PASS - source recorded as injected (injected)
- PASS - detected within persistence + 2 steps (40.0 s)
- PASS - command ON / actual OFF / flow ~0 recorded ({'expected_flow_lpm': 2.0, 'measured_flow_lpm': 0.0, 'flow_deviation_lpm': 2.0, 'flow_deviation_pct': 100.0, 'pump_command': True, 'pump_actual': False, 'collector_c': 56.43})
- PASS - SMS request logged (never falsely 'sent') (unavailable)
- PASS - collector temperature rises without circulation (S1 53.3 -> 108.5 C)

### T3 - Low-flow condition
*Purpose:* Pump ON but injected flow far below expected  
*Run id:* `T3-20261103-091035`

- PASS - low-flow event logged (1 event(s))
- PASS - no pump-failure event (pump is really ON)
- PASS - detected only after the persistence time (70.0 s (persistence 60 s))
- PASS - severity classified from deviation (69.8 % deviation -> major)
- PASS - pump command and actual stay ON
- PASS - deviation values recorded ({'expected_flow_lpm': 2.0, 'measured_flow_lpm': 0.6, 'flow_deviation_lpm': 1.4, 'flow_deviation_pct': 69.8, 'pump_command': True, 'pump_actual': True})

### T4A - Stagnation - NO mitigation (Run A)
*Purpose:* Pump disabled under high solar input, detection and logging only  
*Run id:* `T4A-20261103-091036`

- PASS - stagnation event logged (1 event(s))
- PASS - severity reaches major or higher (critical)
- PASS - no protective actions in the unmitigated run (0 actions)

### T4B - Stagnation - WITH simulated mitigation (Run B)
*Purpose:* Same fault, autonomous simulated protection  
*Run id:* `T4B-20261103-091037`

- PASS - stagnation event logged (1 event(s))
- PASS - severity reaches major or higher (major)
- PASS - simulated protective actions applied (SIM_SOLAR_INPUT_REDUCED, SIM_SOLENOID_OPEN)
- PASS - protective actions are SIMULATION domain only

### T5 - Sensor failure (validated inputs)
*Purpose:* Disconnected / missing / implausible temperatures  
*Run id:* `T5-20261103-091038`

- PASS - S3 fault detected and logged (30.0 s, out_of_range)
- PASS - S1 fault detected and logged (30.0 s, missing)
- PASS - S4 fault detected and logged (30.0 s, out_of_range)
- PASS - invalid readings never reached the controllers (0 samples)
- PASS - backup heater not switched on by the bad S3 reading (heater ON 0 s)
- PASS - heater fail-safe OFF used when S3 is invalid beyond the hold time
- PASS - pump fallback used when S1/S4 invalid beyond the hold time

### T5R - Sensor failure - RAW inputs (comparison only)
*Purpose:* Same faults but controllers fed unvalidated readings  
*Run id:* `T5R-20261103-091039`

- PASS - S3 fault detected and logged (30.0 s, out_of_range)
- PASS - S1 fault detected and logged (30.0 s, missing)
- PASS - S4 fault detected and logged (30.0 s, out_of_range)
- PASS - (raw run) bad readings DID reach the controllers (108 samples)

### T6 - Fault recovery
*Purpose:* Pump failure injected then removed; recovery monitored  
*Run id:* `T6-20261103-091040`

- PASS - pump failure event logged
- PASS - event marked RESOLVED, history retained (RESOLVED)
- PASS - recovery time recorded (fault removed at 1200 s, resolved at 1270.0 s)
- PASS - recovery within persistence + 2 steps
- PASS - pump command == actual at the end
- PASS - no active fault at the end
- PASS - collector cooling after recovery (90.3 -> 57.6 C)
- PASS - recovery SMS request logged

### Cross-run checks

- PASS - Run B peak collector temperature lower than Run A (A 146.2 C vs B 120.2 C)
- PASS - validated inputs used less heater energy than raw inputs after the S3 fault (validated 0 s vs raw 600 s)
