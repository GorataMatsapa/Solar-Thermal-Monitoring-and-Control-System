import math
import time
from datetime import datetime, timedelta, time as dtime
import numpy as np
import streamlit as st

from utils.digital_twin_model import (
    HOTSETDigitalTwin,
    run_simulation,
    PUMP_AUTO,
    PUMP_MANUAL_ON,
    PUMP_MANUAL_OFF,
)
import streamlit.components.v1 as components

from utils import site_data as sd


# Simulation playback: one simulated minute per tick, 120 simulated minutes in total.
TICK_SECONDS = 1.0
SIM_MINUTES = 120

# "Demo day" (predictive heating demo): a full day 04:00-22:00 on a virtual clock. Each tick
# advances STEPS_PER_TICK simulated minutes, so the whole day plays in about two minutes.
DAY_START_HOUR = 4
DAY_MINUTES = 18 * 60
STEPS_PER_TICK = 10
STRAT_THERMOSTAT = "Thermostat baseline"
STRAT_PREDICTIVE = "Predictive control"
FORCE_ON_SETPOINT_C = 95.0     # twin thermostat setpoint that never cuts the element (as in ai_closed_loop)

# Defaults for the interactive controls
DEFAULT_AMBIENT = 20.0        # °C
DEFAULT_PEAK_IRRADIANCE = 500  # W/m²
DEFAULT_DEMAND_LPM = 0.0      # L/min of hot water drawn off
DEFAULT_HEATER_POWER = 0      # W (0 = backup heater disabled)
DEFAULT_HEATER_SETPOINT = 35  # °C

# Cold / hot domestic water lines only carry water while hot water is being
# drawn off, so their dashes follow the "Hot-water draw" slider. Set this to
# True to keep them moving whenever the twin is in SIMULATION mode, even with
# zero draw-off (purely cosmetic).
WATER_LINES_ALWAYS_FLOW = False

# Heat-meter positions on the diagram (centre x, centre y).
# HM1: solar loop, on the cooled return line (vertical run below the pump station)
# HM2: domestic side, on the cold-water inlet line
HM1_POS = (225, 540)
HM2_POS = (985, 390)

# Temperature-sensor nodes on the diagram (centre x, centre y).
# S1: collector fluid, hot line      S2: tank middle
# S3: tank top (hot water)           S4: tank bottom
# S2-S4 sit on the right-hand side of the tank (mirror of the left, about x=530).
SENSOR_POS = {
    "S1": (245, 335),
    "S2": (635, 370),
    "S3": (635, 262),
    "S4": (635, 500),
}


def _as_bool(v):
    """
    Normalise a state flag to True / False / None. The model may return numpy
    booleans, 0/1 or "ON"/"OFF" - none of which pass an `is True` check.
    """
    if v is None:
        return None
    if isinstance(v, str):
        return v.strip().upper() in ("ON", "TRUE", "1", "YES")
    return bool(v)


def value(value, unit="", digits=1):
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}{unit}"


# =========================================================
# DIGITAL TWIN STATE
# =========================================================
def _apply_controls(model):
    """Push the current slider values into the (persistent) model."""
    ss = st.session_state
    model.set_ambient_temperature(ss.get("ambient_slider", DEFAULT_AMBIENT))
    model.set_peak_irradiance(ss.get("irradiance_slider", DEFAULT_PEAK_IRRADIANCE))
    model.set_hot_water_demand(ss.get("demand_slider", DEFAULT_DEMAND_LPM) / 60.0)
    model.set_heater(
        ss.get("heater_power_slider", DEFAULT_HEATER_POWER),
        ss.get("heater_setpoint_slider", DEFAULT_HEATER_SETPOINT),
    )


def _demo_on():
    return bool(st.session_state.get("demo_mode"))


def _sim_total():
    return DAY_MINUTES if _demo_on() else SIM_MINUTES


def _day_start():
    return datetime.combine(datetime.now().date(), dtime(DAY_START_HOUR))


def _new_model():
    model = HOTSETDigitalTwin(
        initial_tank_temperature=25.0,
        ambient_temperature=DEFAULT_AMBIENT,
        cold_water_temperature=18.0,
        solar_irradiance=DEFAULT_PEAK_IRRADIANCE,
        hot_water_demand=0.0,
        heat_loss_coefficient=5.0,
        heater_power=0.0,
        simulation_duration_minutes=SIM_MINUTES,
    )
    if _demo_on():
        # stratified start: warm top, cool bottom (S3 sits just under the comfort limit by default)
        top = float(st.session_state.get("demo_start_top", 42.0))
        model.node_temperatures = np.linspace(top, top - 14.0, model.tank_nodes)
        model.collector_temperature = 20.0
        model.update_derived_state()
    else:
        _apply_controls(model)
    return model


def get_simulation_state(simulation_minute):
    """
    Return the HOTSET system state at the current point in the simulation.

    The model lives in st.session_state and is advanced one step per tick,
    so changing a slider or the pump mode only affects the future - it does
    not rewrite the history.
    """
    state = st.session_state.twin_model.get_state()
    return {
        "collector_temperature": state["collector"]["temperature"],
        "tank_temperature": state["tank"]["temperature"],
        "cold_water_temperature": state["water"]["cold_temperature"],
        "hot_water_temperature": state["water"]["hot_water_temperature"],
        "flow_rate": state["hydraulics"]["flow_rate"],
        "pump_state": state["hydraulics"]["pump_on"],
        "mixer_state": "PHYSICAL / NOT MODELLED",
        "backup_heater_state": state["heater"]["on"],
        "system_status": "SIMULATION",
        "solar_thermal_power": state["collector"]["solar_power"],
        "stored_energy": state["tank"]["stored_energy_kwh"],
        "solar_irradiance": state["collector"]["solar_irradiance"],
        "simulation_time": simulation_minute,
        # new
        "pump_mode": state["hydraulics"]["pump_mode"],
        "high_limit_active": state["hydraulics"]["high_limit_active"],
        "ambient_temperature": state["environment"]["ambient_temperature"],
        "hot_water_demand": state["water"]["demand"],          # L/s
        "heater_power": state["heater"]["power"],
        "heater_setpoint": state["heater"]["setpoint"],
        # virtual heat meters (simulation only)
        "heat_meters": state["heat_meters"],
        # temperature sensors S1..S4
        "sensors": state["sensors"],
    }


# =========================================================
# SVG HELPERS (semi-3D look)
# =========================================================
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(c)))) for c in rgb)


def shade(hex_color, factor):
    """factor > 0 lightens toward white, factor < 0 darkens toward black."""
    r, g, b = _hex_to_rgb(hex_color)
    target = 255 if factor > 0 else 0
    f = abs(factor)
    return _rgb_to_hex((r + (target - r) * f, g + (target - g) * f, b + (target - b) * f))


_TEMP_STOPS = [
    (15, "#38bdf8"),
    (35, "#2dd4bf"),
    (50, "#fbbf24"),
    (65, "#fb923c"),
    (85, "#ef4444"),
]


def temp_to_color(t):
    """Map a water temperature (°C) to a colour so the tank visibly heats up."""
    if t <= _TEMP_STOPS[0][0]:
        return _TEMP_STOPS[0][1]
    if t >= _TEMP_STOPS[-1][0]:
        return _TEMP_STOPS[-1][1]
    for (t0, c0), (t1, c1) in zip(_TEMP_STOPS, _TEMP_STOPS[1:]):
        if t0 <= t <= t1:
            k = (t - t0) / (t1 - t0)
            r0, g0, b0 = _hex_to_rgb(c0)
            r1, g1, b1 = _hex_to_rgb(c1)
            return _rgb_to_hex((r0 + (r1 - r0) * k, g0 + (g1 - g0) * k, b0 + (b1 - b0) * k))
    return _TEMP_STOPS[-1][1]


def pipe(d, color, width=5, flow=False):
    """
    Draw a pipe as a layered tube: drop shadow, dark rim, body, top-left
    highlight. If flow=True, animated dashes travel along the path direction.
    """
    attrs = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'
    dark = shade(color, -0.45)
    light = shade(color, 0.55)
    parts = [
        f'<path d="{d}" transform="translate(2 4)" {attrs} stroke="#020617" '
        f'stroke-opacity="0.45" stroke-width="{width + 2}"/>',
        f'<path d="{d}" {attrs} stroke="{dark}" stroke-width="{width + 2}"/>',
        f'<path d="{d}" {attrs} stroke="{color}" stroke-width="{width}"/>',
        f'<path d="{d}" transform="translate(-0.8 -1)" {attrs} stroke="{light}" '
        f'stroke-opacity="0.55" stroke-width="{max(1.0, width * 0.3):.1f}"/>',
    ]
    if flow:
        parts.append(
            f'<path class="flow" d="{d}" {attrs} stroke="#ffffff" stroke-opacity="0.6" '
            f'stroke-width="{width * 0.45:.1f}" stroke-dasharray="3 13"/>'
        )
    return "".join(parts)


def flange(x, y, w=12, h=16):
    """Small metal pipe fitting centred on (x, y)."""
    return (
        f'<rect x="{x - w / 2}" y="{y - h / 2}" width="{w}" height="{h}" rx="3" '
        f'fill="url(#metal)" stroke="#e2e8f0" stroke-opacity="0.6" stroke-width="1"/>'
    )


def vessel(x, y, w, h):
    """Expansion vessel drawn as a shaded capsule with a specular strip."""
    cx = x + w / 2
    return f"""
      <ellipse cx="{cx + 4}" cy="{y + h + 5}" rx="{w * 0.6:.1f}" ry="6"
               fill="#000" opacity="0.45" filter="url(#softBlur)"/>
      <rect x="{cx - 4}" y="{y + h - 3}" width="8" height="8" rx="1.5" fill="#475569"/>
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{w * 0.45:.1f}"
            fill="url(#vesselBody)" stroke="#94a3b8" stroke-width="1.5"/>
      <path d="M{x + 2} {y + h * 0.52:.1f} H{x + w - 2}"
            stroke="#0f172a" stroke-opacity="0.55" stroke-width="2"/>
      <rect x="{x + w * 0.16:.1f}" y="{y + 7}" width="{w * 0.12:.1f}" height="{h - 14}"
            rx="3" fill="#ffffff" opacity="0.2"/>
      <ellipse cx="{cx}" cy="{y + 5}" rx="{w * 0.36:.1f}" ry="3.5"
               fill="#cbd5e1" opacity="0.55"/>
    """


def heat_meter(cx, cy, tag, sub, active, vertical=False):
    """
    Heat meter (flow sensor + calculator) sitting on a pipe, centred on (cx, cy).
    active=False -> greyed out (installed / not used for live telemetry).
    """
    colour = "#22d3ee" if active else "#64748b"
    led = "#34d399" if active else "#475569"
    opacity = 1.0 if active else 0.4
    w, h = (30, 40) if vertical else (44, 30)
    x, y = cx - w / 2, cy - h / 2

    if vertical:
        label = (
            f'<text x="{x + w + 6}" y="{cy - 2}" fill="{colour}" font-size="10" '
            f'font-weight="700" text-anchor="start">{tag}</text>'
            f'<text x="{x + w + 6}" y="{cy + 10}" fill="{colour}" font-size="8" '
            f'text-anchor="start">{sub}</text>'
        )
    else:
        label = (
            f'<text x="{cx}" y="{y - 6}" fill="{colour}" font-size="10" '
            f'font-weight="700" text-anchor="middle">{tag} · {sub}</text>'
        )

    return f"""
      <!-- opaque backing so the pipe does not show through a greyed-out meter -->
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#0b1220"/>
      <g opacity="{opacity}" filter="url(#shadow3D)">
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6"
              fill="#0f172a" stroke="{colour}" stroke-width="2"/>
        <rect x="{x + 5}" y="{y + 6}" width="{w - 10}" height="12" rx="2"
              fill="#06222b" stroke="{colour}" stroke-opacity="0.5"/>
        <text x="{cx}" y="{y + 15.2}" fill="{colour}" font-size="8" font-weight="700"
              text-anchor="middle">kWh</text>
        <circle cx="{x + 9}" cy="{y + h - 8}" r="2.5" fill="{led}"/>
        <path d="M{x + 16} {y + h - 8} H{x + w - 7}" stroke="{colour}"
              stroke-opacity="0.6" stroke-width="2" stroke-linecap="round"/>
        {label}
      </g>
    """


def heat_meter_card(x, y, w, title, sub, data, active):
    """Dashboard card with one heat meter's readings (greyed out + N/A when inactive)."""
    opacity = 1.0 if active else 0.45
    accent = "#22d3ee" if active else "#94a3b8"
    badge = "ACTIVE · SIMULATED" if active else "NOT USED IN LIVE TELEMETRY"

    columns = [
        ("FLOW", "flow_rate", " L/min", 1),
        ("SUPPLY", "supply_temperature", " °C", 1),
        ("RETURN", "return_temperature", " °C", 1),
        ("ΔT", "delta_t", " K", 1),
        ("POWER", "power", " W", 0),
        ("ENERGY", "energy_kwh", " kWh", 2),
    ]
    step = (w - 32) / len(columns)
    cells = ""
    for i, (label, key, unit, digits) in enumerate(columns):
        cx = x + 16 + i * step
        text = value(data[key], unit, digits) if data else "N/A"
        cells += (
            f'<text x="{cx:.1f}" y="{y + 54}" fill="#94a3b8" font-size="10">{label}</text>'
            f'<text x="{cx:.1f}" y="{y + 74}" fill="{accent}" font-size="14" '
            f'font-weight="700">{text}</text>'
        )

    return f"""
      <g opacity="{opacity}" filter="url(#shadow3D)">
        <rect x="{x}" y="{y}" width="{w}" height="84" rx="10"
              fill="url(#cardGrad)" stroke="#334155"/>
        <path d="M{x + 10} {y + 1.5} H{x + w - 10}" stroke="#fff"
              stroke-opacity="0.12" stroke-width="1.5"/>
        <text x="{x + 16}" y="{y + 20}" fill="#f8fafc" font-size="12"
              font-weight="700">{title}</text>
        <text x="{x + 16}" y="{y + 34}" fill="#94a3b8" font-size="10">{sub}</text>
        <text x="{x + w - 16}" y="{y + 20}" fill="{accent}" font-size="10"
              font-weight="700" text-anchor="end">{badge}</text>
        {cells}
      </g>
    """


def sensor_node(tag, temp):
    """
    Temperature-sensor node (S1..S4) drawn on the schematic. The fill follows
    the temperature; before live data exists (temp is None) it is slate grey.
    """
    cx, cy = SENSOR_POS[tag]
    if temp is None:
        fill, text = "#334155", "#94a3b8"
    else:
        fill, text = temp_to_color(temp), "#0b1220"
    return f"""
      <g filter="url(#shadow3D)">
        <circle cx="{cx}" cy="{cy}" r="9.5" fill="{fill}" stroke="#f8fafc"
                stroke-opacity="0.85" stroke-width="1.5"/>
        <text x="{cx}" y="{cy + 3}" fill="{text}" font-size="8" font-weight="700"
              text-anchor="middle">{tag}</text>
      </g>
    """


def sensor_card(x, y, w, sensors):
    """One wide card with S1..S4 and the tank average (N/A until data exists)."""
    columns = [
        ("S1", "COLLECTOR FLUID (HOT LINE)", "s1_collector_fluid"),
        ("S2", "TANK · MIDDLE", "s2_tank_middle"),
        ("S3", "TANK · TOP (HOT WATER)", "s3_tank_top"),
        ("S4", "TANK · BOTTOM (COLD)", "s4_tank_bottom"),
        ("AVG", "TANK AVERAGE (S2-S4)", "tank_average"),
    ]
    step = (w - 32) / len(columns)
    cells = ""
    for i, (tag, label, key) in enumerate(columns):
        cx = x + 16 + i * step
        reading = sensors[key] if sensors else None
        cells += (
            f'<text x="{cx:.1f}" y="{y + 26}" fill="#94a3b8" font-size="10">'
            f'<tspan fill="#f8fafc" font-weight="700">{tag}</tspan> · {label}</text>'
            f'<text x="{cx:.1f}" y="{y + 55}" fill="#fb923c" font-size="17" '
            f'font-weight="700">{value(reading, " °C")}</text>'
        )
    return f"""
      <g filter="url(#shadow3D)">
        <rect x="{x}" y="{y}" width="{w}" height="70" rx="10"
              fill="url(#cardGrad)" stroke="#334155"/>
        <path d="M{x + 10} {y + 1.5} H{x + w - 10}" stroke="#fff"
              stroke-opacity="0.12" stroke-width="1.5"/>
        {cells}
      </g>
    """


# ---------------------------------------------------------
# Flat-plate collector geometry
# The collector face is a skewed parallelogram. cp(u, v) maps face
# coordinates (u across, v down, both 0..1) to SVG coordinates, so the
# absorber, risers and headers all follow the same perspective.
# ---------------------------------------------------------
def cp(u, v):
    return 66 + 110 * u + 16 * v, 160 + 80 * v


def _poly(uv):
    return " ".join(f"{cp(u, v)[0]:.1f},{cp(u, v)[1]:.1f}" for u, v in uv)


def _line(u0, v0, u1, v1):
    x0, y0 = cp(u0, v0)
    x1, y1 = cp(u1, v1)
    return f"M{x0:.1f} {y0:.1f} L{x1:.1f} {y1:.1f}"


def tube(d, paint, width=4, flow=False):
    """Thin copper tube: dark rim, body (solid colour or gradient), highlight, optional flow dashes."""
    attrs = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'
    parts = [
        f'<path d="{d}" {attrs} stroke="#020617" stroke-opacity="0.7" stroke-width="{width + 1.6}"/>',
        f'<path d="{d}" {attrs} stroke="{paint}" stroke-width="{width}"/>',
        f'<path d="{d}" transform="translate(-0.5 -0.6)" {attrs} stroke="#ffffff" '
        f'stroke-opacity="0.35" stroke-width="{max(0.8, width * 0.28):.1f}"/>',
    ]
    if flow:
        parts.append(
            f'<path class="flow" d="{d}" {attrs} stroke="#ffffff" stroke-opacity="0.65" '
            f'stroke-width="{width * 0.4:.1f}" stroke-dasharray="2 6"/>'
        )
    return "".join(parts)


def collector_markup(solar_color, pump_on, collector, irradiance, irradiance_raw):
    """
    Realistic flat-plate collector: aluminium frame, selective absorber plate,
    seven riser tubes between a lower (cool) and an upper (hot) header, glass
    cover with reflections, and a sun with incoming beams.
    Cooled fluid enters at the lower-left; hot fluid leaves at the upper-right.
    """
    face = _poly([(0, 0), (1, 0), (1, 1), (0, 1)])
    glass = _poly([(0.04, 0.05), (0.96, 0.05), (0.96, 0.95), (0.04, 0.95)])
    plate = _poly([(0.07, 0.10), (0.93, 0.10), (0.93, 0.90), (0.07, 0.90)])
    streak = _poly([(0.18, 0.05), (0.32, 0.05), (0.16, 0.95), (0.04, 0.95)])

    # risers (gradient: cool at the bottom, hot at the top) and headers
    risers = "".join(
        tube(
            _line(0.13 + i * 0.74 / 6, 0.825, 0.13 + i * 0.74 / 6, 0.175),
            "url(#riserFluid)", 3.4, pump_on,
        )
        for i in range(7)
    )
    x0, y0 = cp(0, 0.825)
    x1, y1 = cp(0.89, 0.825)
    bottom_header = tube(f"M{x0:.1f} {y0:.1f} L{x1:.1f} {y1:.1f}", "#38bdf8", 5, pump_on)
    x0, y0 = cp(0.11, 0.175)
    x1, y1 = cp(1.0, 0.175)
    top_header = tube(f"M{x0:.1f} {y0:.1f} L{x1:.1f} {y1:.1f} L190 174", solar_color, 5, pump_on)

    # sun + beams (beam strength follows irradiance)
    beam_op = 0.35 if irradiance_raw is None else min(0.9, 0.25 + irradiance_raw / 1000 * 0.65)
    sun_rays = "".join(
        f'<line x1="{44 + 10 * math.cos(math.radians(a)):.1f}" y1="{146 + 10 * math.sin(math.radians(a)):.1f}" '
        f'x2="{44 + 14 * math.cos(math.radians(a)):.1f}" y2="{146 + 14 * math.sin(math.radians(a)):.1f}"/>'
        for a in range(0, 360, 45)
    )
    sun = f"""
      <g stroke="#fbbf24" stroke-linecap="round" stroke-width="2" opacity="{beam_op:.2f}">
        {sun_rays}
        <path d="M54 152 L84 170 M52 162 L78 184 M50 172 L72 198"
              stroke-width="2" stroke-dasharray="5 5" fill="none"/>
      </g>
      <circle cx="44" cy="146" r="6.5" fill="#fbbf24" filter="url(#glow3D)"/>
    """

    screws = "".join(
        f'<circle cx="{cp(u, v)[0]:.1f}" cy="{cp(u, v)[1]:.1f}" r="1.6" fill="#94a3b8"/>'
        for u, v in [(0.02, 0.03), (0.98, 0.03), (0.02, 0.97), (0.98, 0.97)]
    )

    return f"""
      {sun}
      <g>
        <!-- mounting feet -->
        <rect x="96" y="246" width="14" height="7" rx="2" fill="#475569"/>
        <rect x="164" y="246" width="14" height="7" rx="2" fill="#475569"/>
        <!-- frame depth: top edge, right side, bottom edge -->
        <path d="M66 160 L176 160 L186 154 L76 154 Z" fill="#9aa7b8" stroke="#cbd5e1" stroke-width="1"/>
        <path d="M176 160 L186 154 L202 234 L192 240 Z" fill="#475569" stroke="#cbd5e1" stroke-width="1"/>
        <path d="M82 240 L192 240 L192 246 L82 246 Z" fill="#1e293b"/>
        <!-- aluminium frame face -->
        <polygon points="{face}" fill="url(#frameMetal)" stroke="#e2e8f0" stroke-width="1.5"
                 filter="url(#shadow3D)"/>
        <polygon points="{glass}" fill="#0a1020" stroke="#0f172a" stroke-width="2"/>
        <!-- selective absorber plate with heat tint (hotter towards the outlet) -->
        <polygon points="{plate}" fill="url(#absorberPlate)"/>
        <polygon points="{plate}" fill="url(#absorberHeat)"/>
        <!-- riser tubes and headers -->
        {risers}
        {bottom_header}
        {top_header}
        <!-- glass cover: reflection + edge -->
        <polygon points="{glass}" fill="url(#glare)"/>
        <polygon points="{streak}" fill="#ffffff" opacity="0.08"/>
        <polygon points="{glass}" fill="none" stroke="#bfdbfe" stroke-opacity="0.5" stroke-width="1"/>
        {screws}

        <text x="60" y="274" fill="#f8fafc" font-size="15" font-weight="700">SOLAR COLLECTOR</text>
        <text x="60" y="299" fill="#fbbf24" font-size="21" font-weight="700">{collector}</text>
        <text x="60" y="321" fill="#94a3b8" font-size="12">Irradiance: {irradiance}</text>
      </g>
      <text x="36" y="215" fill="#7dd3fc" font-size="10" font-weight="700">COOL IN</text>
    """


def collector_fittings_markup():
    """Brass compression fittings where the loop pipes meet the collector (drawn above the pipes)."""
    return """
      <g filter="url(#shadow3D)">
        <rect x="182" y="168" width="10" height="12" rx="2" fill="url(#brass)"
              stroke="#fde68a" stroke-opacity="0.6"/>
        <rect x="73" y="220" width="10" height="12" rx="2" fill="url(#brass)"
              stroke="#fde68a" stroke-opacity="0.6"/>
      </g>
    """


SVG_CSS = """
@keyframes flow { to { stroke-dashoffset: -16; } }
.flow { animation: flow 0.9s linear infinite; animation-delay: var(--phase-flow, 0s); }
@keyframes pulse { 0%, 100% { opacity: 0.15; } 50% { opacity: 0.45; } }
.pulse { animation: pulse 1.6s ease-in-out infinite; animation-delay: var(--phase-pulse, 0s); }
@media (prefers-reduced-motion: reduce) { .flow, .pulse { animation: none; } }
"""


# =========================================================
# DIGITAL TWIN VISUAL (SEMI-3D)
# =========================================================
def render_svg_twin(state):
    """Render the HOTSET 300 L schematic with a semi-3D (pseudo-isometric) look."""

    collector = value(state.get("collector_temperature"), " °C")
    tank = value(state.get("tank_temperature"), " °C")
    cold = value(state.get("cold_water_temperature"), " °C")
    hot = value(state.get("hot_water_temperature"), " °C")
    flow = value(state.get("flow_rate"), " L/min")
    power = value(state.get("solar_thermal_power"), " W")
    irradiance = value(state.get("solar_irradiance"), " W/m²")
    energy = value(state.get("stored_energy"), " kWh")

    pump_state = _as_bool(state.get("pump_state"))
    pump_on = pump_state is True
    pump_off = pump_state is False
    simulation = state.get("simulation_time") is not None

    # Each tick rebuilds the iframe, which would restart every animation from zero.
    # Offsetting by the elapsed time keeps the motion continuous between ticks.
    elapsed = (state.get("simulation_time") or 0) * TICK_SECONDS
    phase_flow = elapsed % 0.9
    phase_pulse = elapsed % 1.6
    phase_spin = elapsed % 1.2

    pump_label = "ON" if pump_on else "OFF" if pump_off else "N/A"
    if state.get("pump_mode") in (PUMP_MANUAL_ON, PUMP_MANUAL_OFF):
        pump_label += " · MANUAL"
    pump_color = "#34d399" if pump_on else "#94a3b8"
    # Colour = what the fluid is (hot line amber whenever the twin has data);
    # moving dashes = whether it is actually circulating (pump ON).
    solar_color = "#f59e0b" if (simulation or pump_on) else "#64748b"

    heater_state = _as_bool(state.get("backup_heater_state"))
    if simulation and heater_state is not None:
        heater_label = "ON" if heater_state else "OFF"
    elif simulation:
        heater_label = "N/A"
    else:
        heater_label = "NOT INSTALLED"

    hil = state.get("mode") == "HIL"
    mode_label = "HIL MODE" if hil else ("SIMULATION MODE" if simulation else "REAL-TIME MODE")
    mode_color = "#f59e0b" if hil else ("#a78bfa" if simulation else "#38bdf8")

    # Water colour follows tank temperature (hotter at the top, cooler at the bottom).
    tank_raw = state.get("tank_temperature")
    if tank_raw is None:
        water_top, water_bottom = "#475569", "#1e293b"
    else:
        water_top = temp_to_color(tank_raw + 6)
        water_bottom = temp_to_color(tank_raw - 6)

    # ---- virtual heat meters ----
    # REAL-TIME: greyed out (not used for live telemetry).
    # SIMULATION: active, readings come from the model.
    heat_meters = state.get("heat_meters")
    hm_active = simulation and heat_meters is not None
    hm_solar = heat_meters["solar_loop"] if hm_active else None
    hm_domestic = heat_meters["domestic"] if hm_active else None

    hm1_svg = heat_meter(*HM1_POS, "HM1", "SOLAR LOOP", hm_active, vertical=True)
    hm2_svg = heat_meter(*HM2_POS, "HM2", "DOMESTIC", hm_active, vertical=False)
    hm_cards = (
        heat_meter_card(
            35, 960, 560,
            "HM1 · SOLAR LOOP HEAT METER",
            "On the cooled return line, below the pump station",
            hm_solar, hm_active,
        )
        + heat_meter_card(
            610, 960, 555,
            "HM2 · DOMESTIC HEAT METER",
            "On the cold-water inlet (hot-out vs cold-in)",
            hm_domestic, hm_active,
        )
    )

    # ---- temperature sensors ----
    # S1 collector fluid (hot line), S2/S3/S4 tank middle/top/bottom.
    # Used in both modes: N/A in REAL-TIME until the live feed is connected.
    sensors = state.get("sensors")
    sensor_nodes = "".join(
        sensor_node(tag, sensors[key] if sensors else None)
        for tag, key in (
            ("S1", "s1_collector_fluid"),
            ("S2", "s2_tank_middle"),
            ("S3", "s3_tank_top"),
            ("S4", "s4_tank_bottom"),
        )
    )
    sensor_card_svg = sensor_card(35, 848, 1130, sensors)
    if sensors:
        # the tank temperature readout is the average of S2, S3 and S4
        tank = value(sensors["tank_average"], " °C")

    # ---- virtual electric backup heater (immersion element inside the tank) ----
    # REAL-TIME: greyed out (not installed on the physical rig).
    # SIMULATION: grey when OFF, glowing orange with a pulsing halo when ON.
    heater_on = simulation and heater_state is True
    if heater_on:
        h_color, h_opacity, h_led, h_text = "#f97316", 1.0, "#ef4444", "#fdba74"
    elif simulation and heater_state is not None:
        h_color, h_opacity, h_led, h_text = "#94a3b8", 1.0, "#7f1d1d", "#cbd5e1"
    else:
        h_color, h_opacity, h_led, h_text = "#64748b", 0.4, "#475569", "#94a3b8"

    heater_rods = pipe("M402 530 H590 A9 9 0 0 1 590 548 H402", h_color, 6)
    if heater_on:
        heater_rods = f'<g filter="url(#glow3D)">{heater_rods}</g>'
        heater_halo = (
            '<ellipse class="pulse" cx="500" cy="539" rx="115" ry="24" '
            'fill="#f97316" filter="url(#softBlur)"/>'
        )
    else:
        heater_halo = ""

    heater_element = f"""
      <g opacity="{h_opacity}">
        {heater_halo}
        {heater_rods}
        {flange(402, 539, 12, 30)}
        <!-- thermostat / terminal head outside the tank -->
        <g filter="url(#shadow3D)">
          <rect x="362" y="527" width="34" height="24" rx="4"
                fill="#1e293b" stroke="#64748b" stroke-width="1.5"/>
          <path d="M366 528.5 H392" stroke="#fff" stroke-opacity="0.15" stroke-width="1.5"/>
          <circle cx="372" cy="539" r="3.2" fill="{h_led}"/>
          <path d="M385 531 L380 540 H385 L382 548 L390 537 H385 Z" fill="{h_text}"/>
        </g>
        <text x="506" y="521" fill="{h_text}" font-size="10" font-weight="700"
              text-anchor="middle">BACKUP HEATER · {heater_label}</text>
      </g>"""

    # ---- collector (realistic flat-plate) ----
    collector_raw = state.get("collector_temperature")
    irradiance_raw = state.get("solar_irradiance")
    if collector_raw is None:
        absorber_hot, absorber_hot_op, absorber_cool_op = "#38bdf8", 0.0, 0.0
    else:
        absorber_hot, absorber_hot_op, absorber_cool_op = temp_to_color(collector_raw), 0.5, 0.12
    collector_svg = collector_markup(solar_color, pump_on, collector, irradiance, irradiance_raw)
    collector_fittings = collector_fittings_markup()

    # ---- pipes (drawn as layered tubes) ----
    # Solar loop: collector -> pump station (hot line) -> coil section 2 (top to bottom)
    #             -> pump station (cooled return line, pump) -> collector
    solar_supply = pipe("M190 174 H245 V393 H402", solar_color, 5, flow=pump_on)
    solar_return = pipe("M402 459 H225 V570 H30 V226 H80", "#38bdf8", 5, flow=pump_on)
    solar_coil_hot = pipe("M402 393 H632 V415 H428", solar_color, 6, flow=pump_on)
    solar_coil_cool = pipe("M428 415 V437 H632 V459 H402", "#38bdf8", 6, flow=pump_on)
    solar_vessel_pipe = pipe("M225 475 H181", "#38bdf8", 3)
    # Radiator-circuit coil (section 1) - greyed out, outside project scope
    heat_coil = pipe("M658 276 H428 V298 H632 V320 H428 V342 H658", "#64748b", 6)
    # Domestic water side: dashes run in the path direction
    #   cold in -> manifold -> (tank | mixing valve);  tank -> mixing valve -> warm water out
    # These lines only carry water while hot water is drawn off, so the dashes follow
    # "hot_water_demand" (flow only while demand > 0). Live mode has no demand signal,
    # so they stay still there. Set WATER_LINES_ALWAYS_FLOW = True to animate them
    # whenever the twin is in SIMULATION mode.
    demand = state.get("hot_water_demand")
    if WATER_LINES_ALWAYS_FLOW:
        water_flow = simulation
    else:
        water_flow = (demand > 0) if demand is not None else False

    if simulation and demand is not None:
        draw_text = f"Draw-off: {demand * 60:.1f} L/min" + ("" if water_flow else " (lines idle)")
    else:
        draw_text = ""

    hot_out = pipe("M663 239 H745 V193 H835", "#fb7185", 5, flow=water_flow)
    mixed_out = pipe("M851 193 H1090 V236", "#fb7185", 5, flow=water_flow)
    cold_main = pipe("M1090 390 H930 V465", "#38bdf8", 5, flow=water_flow)
    cold_to_tank = pipe("M930 465 H700 V535 H663", "#38bdf8", 5, flow=water_flow)
    cold_to_mixer = pipe("M930 465 V209 H835", "#38bdf8", 5, flow=water_flow)
    heat_supply = pipe("M658 276 H770 V510 H875", "#64748b", 4)
    heat_return = pipe("M875 555 H740 V342 H658", "#64748b", 4)
    heat_vessel_pipe = pipe("M740 555 V646 H794", "#64748b", 3)

    radiator_fins = "".join(
        f'<rect x="{886 + i * 23}" y="496" width="15" height="52" rx="6" '
        f'fill="url(#fin)" stroke="#94a3b8" stroke-width="1"/>'
        f'<ellipse cx="{893.5 + i * 23}" cy="498" rx="7" ry="2.6" fill="#cbd5e1" opacity="0.5"/>'
        for i in range(10)
    )

    impeller_anim = (
        '<animateTransform attributeName="transform" type="rotate" '
        f'from="0 336 459" to="360 336 459" dur="1.2s" begin="-{phase_spin:.2f}s" repeatCount="indefinite"/>'
        if pump_on
        else ""
    )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg"
         viewBox="0 0 1200 1085" width="100%"
         style="--phase-flow:-{phase_flow:.2f}s; --phase-pulse:-{phase_pulse:.2f}s"
         role="img"
         aria-label="HOTSET 300 litre solar thermal digital twin">

      <defs>
        <radialGradient id="bgGlow" cx="50%" cy="38%" r="75%">
          <stop offset="0%" stop-color="#16243f"/>
          <stop offset="100%" stop-color="#0b1220"/>
        </radialGradient>

        <!-- Cylinder shading for the tank shell -->
        <linearGradient id="tankShell3D" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#0f172a"/>
          <stop offset="12%" stop-color="#334155"/>
          <stop offset="35%" stop-color="#7c8ca3"/>
          <stop offset="55%" stop-color="#475569"/>
          <stop offset="88%" stop-color="#1e293b"/>
          <stop offset="100%" stop-color="#0b1220"/>
        </linearGradient>

        <linearGradient id="tankLid" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#cbd5e1"/>
          <stop offset="100%" stop-color="#64748b"/>
        </linearGradient>

        <linearGradient id="waterBody" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{water_top}"/>
          <stop offset="100%" stop-color="{water_bottom}"/>
        </linearGradient>

        <!-- Horizontal light/dark overlay that turns a flat shape into a cylinder -->
        <linearGradient id="cylShade" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#000" stop-opacity="0.55"/>
          <stop offset="28%" stop-color="#fff" stop-opacity="0.20"/>
          <stop offset="50%" stop-color="#fff" stop-opacity="0.02"/>
          <stop offset="100%" stop-color="#000" stop-opacity="0.60"/>
        </linearGradient>

        <radialGradient id="metal" cx="35%" cy="30%" r="80%">
          <stop offset="0%" stop-color="#f1f5f9"/>
          <stop offset="55%" stop-color="#94a3b8"/>
          <stop offset="100%" stop-color="#334155"/>
        </radialGradient>

        <linearGradient id="brass" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#fde68a"/>
          <stop offset="50%" stop-color="#b45309"/>
          <stop offset="100%" stop-color="#78350f"/>
        </linearGradient>

        <linearGradient id="vesselBody" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#1e293b"/>
          <stop offset="35%" stop-color="#64748b"/>
          <stop offset="100%" stop-color="#0f172a"/>
        </linearGradient>

        <linearGradient id="motor" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#1e293b"/>
          <stop offset="40%" stop-color="#64748b"/>
          <stop offset="100%" stop-color="#0f172a"/>
        </linearGradient>

        <radialGradient id="pumpBody" cx="35%" cy="30%" r="85%">
          <stop offset="0%" stop-color="#3b5ea8"/>
          <stop offset="60%" stop-color="#172554"/>
          <stop offset="100%" stop-color="#0b132b"/>
        </radialGradient>

        <radialGradient id="sphereBlue" cx="35%" cy="30%" r="80%">
          <stop offset="0%" stop-color="#7dd3fc"/>
          <stop offset="55%" stop-color="#0c4a6e"/>
          <stop offset="100%" stop-color="#082f49"/>
        </radialGradient>

        <linearGradient id="fin" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#1e293b"/>
          <stop offset="40%" stop-color="#475569"/>
          <stop offset="100%" stop-color="#1e293b"/>
        </linearGradient>

        <linearGradient id="cardGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#1e2a44"/>
          <stop offset="100%" stop-color="#121a2c"/>
        </linearGradient>

        <linearGradient id="collector3D" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#1e3a8a"/>
          <stop offset="60%" stop-color="#172554"/>
          <stop offset="100%" stop-color="#0b132b"/>
        </linearGradient>

        <linearGradient id="glare" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#fff" stop-opacity="0.30"/>
          <stop offset="42%" stop-color="#fff" stop-opacity="0"/>
          <stop offset="58%" stop-color="#fff" stop-opacity="0.08"/>
          <stop offset="100%" stop-color="#fff" stop-opacity="0"/>
        </linearGradient>

        <pattern id="panelGrid" width="17" height="17" patternUnits="userSpaceOnUse">
          <path d="M17 0 H0 V17" fill="none"
                stroke="#93c5fd" stroke-width="1.2" stroke-opacity="0.5"/>
        </pattern>

        <linearGradient id="frameMetal" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#e2e8f0"/>
          <stop offset="45%" stop-color="#64748b"/>
          <stop offset="100%" stop-color="#94a3b8"/>
        </linearGradient>

        <linearGradient id="absorberPlate" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#1b2a4a"/>
          <stop offset="100%" stop-color="#0a1020"/>
        </linearGradient>

        <!-- heat tint: cool blue at the inlet side, collector temperature colour at the outlet side -->
        <linearGradient id="absorberHeat" x1="0" y1="1" x2="0" y2="0">
          <stop offset="0%" stop-color="#38bdf8" stop-opacity="{absorber_cool_op}"/>
          <stop offset="100%" stop-color="{absorber_hot}" stop-opacity="{absorber_hot_op}"/>
        </linearGradient>

        <!-- fluid in the risers: cooled at the bottom header, hot at the top header -->
        <linearGradient id="riserFluid" gradientUnits="userSpaceOnUse" x1="0" y1="226" x2="0" y2="174">
          <stop offset="0%" stop-color="#38bdf8"/>
          <stop offset="100%" stop-color="{solar_color}"/>
        </linearGradient>

        <filter id="shadow3D" x="-10%" y="-10%" width="130%" height="130%">
          <feDropShadow dx="3" dy="6" stdDeviation="5"
                        flood-color="#000000" flood-opacity="0.6"/>
        </filter>

        <filter id="softBlur" x="-50%" y="-100%" width="200%" height="300%">
          <feGaussianBlur stdDeviation="6"/>
        </filter>

        <filter id="glow3D" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur"/>
          <feComposite in="SourceGraphic" in2="blur" operator="over"/>
        </filter>

        <marker id="solarArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0 0 L7 4 L0 8 Z" fill="{solar_color}"/>
        </marker>
        <marker id="returnArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0 0 L7 4 L0 8 Z" fill="#38bdf8"/>
        </marker>
        <marker id="hotArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0 0 L7 4 L0 8 Z" fill="#fb7185"/>
        </marker>
        <marker id="coldArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0 0 L7 4 L0 8 Z" fill="#38bdf8"/>
        </marker>
        <marker id="heatArrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0 0 L7 4 L0 8 Z" fill="#64748b"/>
        </marker>
      </defs>

      <style>{SVG_CSS}</style>

      <rect width="1200" height="1085" rx="18" fill="url(#bgGlow)"/>

      <!-- Header -->
      <text x="35" y="43" fill="#f8fafc" font-size="25" font-weight="700">HOTSET 300 L</text>
      <text x="35" y="69" fill="#94a3b8" font-size="14">
        SMART SOLAR THERMAL SYSTEM · DIGITAL TWIN
      </text>

      <rect x="930" y="25" width="235" height="36" rx="18"
            fill="{mode_color}" fill-opacity="0.15"
            stroke="{mode_color}" stroke-opacity="0.65"/>
      <circle cx="950" cy="43" r="5" fill="{mode_color}"/>
      <text x="964" y="48" fill="{mode_color}" font-size="13" font-weight="700">{mode_label}</text>
      <path d="M30 88 H1170" stroke="#263449" stroke-width="1.5"/>

      <text x="42" y="122" fill="#fbbf24" font-size="14" font-weight="700">SOLAR CIRCUIT</text>
      <text x="430" y="155" fill="#e2e8f0" font-size="14" font-weight="700">THERMAL STORAGE</text>
      <text x="835" y="122" fill="#cbd5e1" font-size="14" font-weight="700">WATER &amp; HEATING</text>

      <!-- ===================================================== -->
      <!-- STORAGE TANK (drawn first so pipes connect on top)      -->
      <!-- ===================================================== -->
      <ellipse cx="542" cy="596" rx="165" ry="24" fill="#000" opacity="0.55" filter="url(#softBlur)"/>

      <!-- feet -->
      <rect x="425" y="594" width="36" height="20" rx="3" fill="#1e293b" stroke="#475569"/>
      <rect x="599" y="594" width="36" height="20" rx="3" fill="#1e293b" stroke="#475569"/>

      <!-- cylindrical insulation shell -->
      <path d="M397 202 V584 A133 26 0 0 0 663 584 V202 Z"
            fill="url(#tankShell3D)" stroke="#cbd5e1" stroke-opacity="0.5" stroke-width="2"/>
      <!-- lid -->
      <ellipse cx="530" cy="202" rx="133" ry="26"
               fill="url(#tankLid)" stroke="#e2e8f0" stroke-opacity="0.6" stroke-width="2"/>
      <ellipse cx="530" cy="202" rx="100" ry="18" fill="none"
               stroke="#0f172a" stroke-opacity="0.25" stroke-width="2"/>

      <!-- cut-away window showing the water -->
      <path d="M411 240 V550 A119 22 0 0 0 649 550 V240 Z"
            fill="url(#waterBody)" stroke="#0f172a" stroke-width="3"/>
      <path d="M411 240 V550 A119 22 0 0 0 649 550 V240 Z" fill="url(#cylShade)"/>
      <!-- water surface -->
      <ellipse cx="530" cy="240" rx="119" ry="14" fill="{water_top}" fill-opacity="0.95"
               stroke="#ffffff" stroke-opacity="0.35" stroke-width="1.5"/>
      <ellipse cx="505" cy="238" rx="55" ry="4" fill="#ffffff" opacity="0.18"/>

      <!-- Section 1 (upper): radiator-circuit coil, greyed out / outside scope -->
      <g opacity="0.5">{heat_coil}</g>
      <!-- Section 2 (lower): solar-circuit coil, hot fluid in at the top, cooled fluid out at the bottom -->
      {solar_coil_hot}
      {solar_coil_cool}

      <!-- Virtual electric backup heater (lower part of the tank) -->
      {heater_element}

      <text x="530" y="263" fill="#f8fafc" font-size="16"
            font-weight="700" text-anchor="middle">HOT WATER TANK</text>
      <text x="530" y="283" fill="#e2e8f0" font-size="13" text-anchor="middle"
            opacity="0">300 L STORAGE</text>
      <text x="530" y="364" fill="#94a3b8" font-size="11"
            font-weight="700" text-anchor="middle">HEAT-TRANSFER SECTION 1</text>
      <text x="530" y="377" fill="#64748b" font-size="9"
            text-anchor="middle">Radiator circuit · outside scope</text>
      <text x="530" y="480" fill="#fde68a" font-size="11"
            font-weight="700" text-anchor="middle">HEAT-TRANSFER SECTION 2</text>
      <text x="530" y="493" fill="#fbbf24" font-size="9"
            text-anchor="middle">Solar circuit</text>
      <text x="530" y="176" fill="#0f172a" font-size="12" font-weight="700"
            text-anchor="middle" opacity="0.75">300 L STORAGE</text>

      <!-- tank temperature probe + readout (below the tank) -->
      <path d="M530 611 V634" stroke="#64748b" stroke-width="2" stroke-dasharray="3 3"/>
      <g filter="url(#shadow3D)">
        <rect x="427" y="634" width="206" height="43" rx="9"
              fill="#0b1220" fill-opacity="0.9" stroke="#64748b"/>
        <text x="530" y="652" fill="#94a3b8" font-size="11" text-anchor="middle">TANK TEMPERATURE · AVG S2-S4</text>
        <text x="530" y="671" fill="#fb923c" font-size="19"
              font-weight="700" text-anchor="middle">{tank}</text>
      </g>

      <!-- ===================================================== -->
      <!-- FLAT-PLATE SOLAR COLLECTOR                              -->
      <!-- ===================================================== -->
      {collector_svg}

      <!-- ===================================================== -->
      <!-- SOLAR PUMP STATION: hot line and cooled return line     -->
      <!-- both pass through the station                           -->
      <!-- ===================================================== -->
      <ellipse cx="324" cy="524" rx="74" ry="9" fill="#000" opacity="0.5" filter="url(#softBlur)"/>
      <g filter="url(#shadow3D)">
        <rect x="262" y="338" width="124" height="180" rx="10"
              fill="#0f172a" fill-opacity="0.92" stroke="#64748b" stroke-width="2"/>
        <rect x="268" y="344" width="112" height="168" rx="7"
              fill="none" stroke="#334155" stroke-width="1"/>
        <path d="M272 340 H376" stroke="#fff" stroke-opacity="0.15" stroke-width="1.5"/>
      </g>
      <text x="324" y="357" fill="#cbd5e1" font-size="9" font-weight="700"
            text-anchor="middle">SOLAR PUMP STATION</text>

      <!-- pipework (drawn over the station housing) -->
      {solar_supply}
      <path d="M214 174 H238" fill="none" stroke="{solar_color}" stroke-width="3"
            marker-end="url(#solarArrow)"/>
      <path d="M245 250 V290" fill="none" stroke="{solar_color}" stroke-width="3"
            marker-end="url(#solarArrow)"/>
      <text x="206" y="160" fill="#fbbf24" font-size="12" font-weight="700">HOT SOLAR FLUID</text>
      {collector_fittings}

      {solar_return}
      <path d="M258 459 H236" fill="none" stroke="#38bdf8" stroke-width="3"
            marker-end="url(#returnArrow)"/>
      <path d="M120 570 H90" fill="none" stroke="#38bdf8" stroke-width="3"
            marker-end="url(#returnArrow)"/>
      <path d="M30 282 V247" fill="none" stroke="#38bdf8" stroke-width="3"
            marker-end="url(#returnArrow)"/>
      <text x="42" y="600" fill="#7dd3fc" font-size="12" font-weight="700">
        COOLED SOLAR FLUID RETURN
      </text>

      <!-- HM1: solar-loop heat meter on the cooled return line -->
      {hm1_svg}

      <!-- hot line (y=393): ball valve + hot-side thermometer -->
      <g filter="url(#shadow3D)">
        <rect x="281" y="387" width="14" height="12" rx="3" fill="url(#metal)" stroke="#e2e8f0" stroke-opacity="0.5"/>
        <rect x="287" y="379" width="3" height="8" fill="#ef4444"/>
        <rect x="283" y="377" width="11" height="3" rx="1.5" fill="#ef4444"/>
        <rect x="343" y="362" width="6" height="28" rx="3" fill="#0f172a" stroke="#e2e8f0" stroke-opacity="0.7"/>
        <rect x="344.5" y="372" width="3" height="18" rx="1.5" fill="#ef4444"/>
        <circle cx="346" cy="391" r="4.5" fill="#ef4444" stroke="#fecaca" stroke-opacity="0.6"/>
      </g>

      <!-- return line (y=459): check valve, pump, cooled-side thermometer -->
      <g filter="url(#shadow3D)">
        <rect x="281" y="453" width="14" height="12" rx="3" fill="url(#metal)" stroke="#e2e8f0" stroke-opacity="0.5"/>
        <path d="M292 455 L286 459 L292 463 Z" fill="#1e293b"/>

        <!-- pump: motor above, volute body on the return line -->
        <rect x="331" y="428" width="10" height="9" fill="#475569"/>
        <rect x="322" y="408" width="28" height="22" rx="4" fill="url(#motor)" stroke="#94a3b8"/>
        <path d="M327 412 V426 M332 412 V426 M337 412 V426 M342 412 V426 M346 412 V426"
              stroke="#0f172a" stroke-opacity="0.55" stroke-width="1.5"/>
        <rect x="313" y="452" width="7" height="14" rx="2" fill="url(#metal)"/>
        <rect x="352" y="452" width="7" height="14" rx="2" fill="url(#metal)"/>
        <circle cx="336" cy="459" r="20" fill="url(#pumpBody)" stroke="{pump_color}" stroke-width="3"/>
        <circle cx="336" cy="459" r="11" fill="#0b1220"/>
        <g>
          <path d="M343 451 L326 459 L343 467 Z" fill="{pump_color}" filter="url(#glow3D)"/>
          {impeller_anim}
        </g>
        <ellipse cx="329" cy="448" rx="8" ry="3.5" fill="#fff" opacity="0.15"/>

        <rect x="369" y="462" width="6" height="26" rx="3" fill="#0f172a" stroke="#e2e8f0" stroke-opacity="0.7"/>
        <rect x="370.5" y="473" width="3" height="15" rx="1.5" fill="#38bdf8"/>
        <circle cx="372" cy="461" r="4.5" fill="#38bdf8" stroke="#bae6fd" stroke-opacity="0.6"/>
      </g>

      <text x="322" y="496" fill="{pump_color}" font-size="14" font-weight="700"
            text-anchor="middle">{pump_label}</text>
      <text x="322" y="510" fill="#94a3b8" font-size="11" text-anchor="middle">{flow}</text>

      <!-- tank connections -->
      {flange(402, 393)}
      {flange(402, 459)}
      {flange(658, 239)}
      {flange(658, 276)}
      {flange(658, 342)}
      {flange(658, 535)}

      <!-- temperature sensors S1-S4 -->
      {sensor_nodes}

      <!-- ===================================================== -->
      <!-- SOLAR EXPANSION VESSEL (tee off the cooled return)      -->
      <!-- ===================================================== -->
      {solar_vessel_pipe}
      <circle cx="225" cy="475" r="4" fill="#7dd3fc"/>
      <g filter="url(#shadow3D)">
        {vessel(139, 449, 42, 55)}
      </g>
      <text x="42" y="526" fill="#cbd5e1" font-size="12" font-weight="700">SOLAR EXPANSION</text>
      <text x="42" y="544" fill="#cbd5e1" font-size="12">VESSEL</text>

      <!-- ===================================================== -->
      <!-- DOMESTIC HOT WATER & MIXING VALVE                       -->
      <!-- ===================================================== -->
      {hot_out}
      <path d="M690 239 H725" fill="none" stroke="#fb7185" stroke-width="3"
            marker-end="url(#hotArrow)"/>
      <text x="680" y="166" fill="#fda4af" font-size="12" font-weight="700">HOT WATER</text>
      <text x="680" y="185" fill="#f8fafc" font-size="13">{hot}</text>

      {cold_to_mixer}
      <path d="M865 209 H835" fill="none" stroke="#38bdf8" stroke-width="3"
            marker-end="url(#coldArrow)"/>

      {mixed_out}
      <path d="M930 193 H970" fill="none" stroke="#fb7185" stroke-width="3"
            marker-end="url(#hotArrow)"/>
      <path d="M1090 213 V231" fill="none" stroke="#fb7185" stroke-width="3"
            marker-end="url(#hotArrow)"/>

      <!-- brass mixing valve with handwheel -->
      <ellipse cx="840" cy="226" rx="26" ry="5" fill="#000" opacity="0.45" filter="url(#softBlur)"/>
      <g filter="url(#shadow3D)">
        <rect x="811" y="187" width="9" height="12" rx="2" fill="url(#brass)"/>
        <rect x="850" y="187" width="9" height="12" rx="2" fill="url(#brass)"/>
        <rect x="850" y="203" width="9" height="12" rx="2" fill="url(#brass)"/>
        <rect x="817" y="179" width="36" height="36" rx="9" fill="url(#brass)"
              stroke="#fde68a" stroke-opacity="0.6"/>
        <circle cx="835" cy="197" r="8" fill="#451a03" stroke="#fde68a" stroke-opacity="0.7"/>
        <rect x="833" y="171" width="4" height="9" fill="#94a3b8"/>
        <ellipse cx="835" cy="171" rx="15" ry="5" fill="#e2e8f0" stroke="#475569"/>
        <ellipse cx="835" cy="170" rx="9" ry="2.2" fill="#ffffff" opacity="0.6"/>
      </g>
      <text x="835" y="238" fill="#cbd5e1" font-size="11" font-weight="700"
            text-anchor="middle">MIXING VALVE</text>

      <!-- faucet -->
      <g transform="translate(1090 256)" filter="url(#shadow3D)">
        <path d="M-21 -13 H-7 V-3 H7 V-13 H21" fill="none" stroke="#e2e8f0" stroke-width="4"
              stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M0 -3 V15" stroke="#e2e8f0" stroke-width="4" stroke-linecap="round"/>
        <path d="M-8 19 Q0 29 8 19" fill="none" stroke="#38bdf8" stroke-width="3"/>
      </g>
      <text x="1000" y="304" fill="#f8fafc" font-size="13" font-weight="700">WARM WATER OUT</text>
      <text x="1000" y="324" fill="#94a3b8" font-size="12">After mixing</text>

      <!-- ===================================================== -->
      <!-- COLD WATER INLET & MANIFOLD                             -->
      <!-- ===================================================== -->
      {cold_main}
      {cold_to_tank}
      <circle cx="930" cy="465" r="6" fill="url(#metal)"/>

      <path d="M735 535 H700" fill="none" stroke="#38bdf8" stroke-width="3"
            marker-end="url(#coldArrow)"/>

      <!-- HM2: domestic heat meter on the cold-water inlet -->
      {hm2_svg}

      <g filter="url(#shadow3D)">
        <circle cx="1090" cy="390" r="23" fill="url(#sphereBlue)" stroke="#38bdf8" stroke-width="2"/>
        <path d="M1090 376 V404 M1076 390 H1104 M1080 380 L1100 400 M1100 380 L1080 400"
              stroke="#bae6fd" stroke-width="2"/>
        <ellipse cx="1082" cy="378" rx="9" ry="4" fill="#fff" opacity="0.25"/>
      </g>
      <path d="M1060 390 H1035" fill="none" stroke="#38bdf8" stroke-width="3"
            marker-end="url(#coldArrow)"/>
      <text x="990" y="433" fill="#7dd3fc" font-size="12" font-weight="700">COLD WATER IN</text>
      <text x="990" y="451" fill="#f8fafc" font-size="13">{cold}</text>
      <text x="990" y="469" fill="#7dd3fc" font-size="11">{draw_text}</text>

      <!-- ===================================================== -->
      <!-- SPACE HEATING CIRCUIT (muted)                           -->
      <!-- ===================================================== -->
      <g opacity="0.6">
        {heat_supply}
        {heat_return}

        <path d="M738 276 H762" fill="none" stroke="#64748b" stroke-width="3"
              marker-end="url(#heatArrow)"/>
        <path d="M725 342 H690" fill="none" stroke="#64748b" stroke-width="3"
              marker-end="url(#heatArrow)"/>

        <!-- heating pump -->
        <g filter="url(#shadow3D)">
          <rect x="708" y="247" width="14" height="10" fill="#475569"/>
          <circle cx="715" cy="276" r="19" fill="url(#metal)" stroke="#94a3b8" stroke-width="2"/>
          <path d="M707 266 L725 276 L707 286 Z" fill="#1e293b"/>
        </g>
        <text x="715" y="308" fill="#cbd5e1" font-size="10" font-weight="700"
              text-anchor="middle">HEATING PUMP</text>
        <text x="715" y="321" fill="#94a3b8" font-size="9" text-anchor="middle">Not modelled</text>

        <!-- radiator: finned panel with depth -->
        <ellipse cx="1010" cy="566" rx="140" ry="8" fill="#000" opacity="0.45" filter="url(#softBlur)"/>
        <g filter="url(#shadow3D)">
          <rect x="875" y="488" width="258" height="68" rx="7"
                fill="#1e293b" stroke="#64748b" stroke-width="2"/>
          {radiator_fins}
          <rect x="875" y="548" width="258" height="8" rx="4" fill="#000" opacity="0.3"/>
        </g>
        <text x="1004" y="582" fill="#cbd5e1" font-size="13" font-weight="700"
              text-anchor="middle">RADIATOR CIRCUIT</text>
        <text x="1004" y="600" fill="#94a3b8" font-size="11" text-anchor="middle">
          Outside STMaCS scope
        </text>

        <!-- heating expansion vessel -->
        {heat_vessel_pipe}
        <circle cx="740" cy="555" r="4" fill="#94a3b8"/>
        <g filter="url(#shadow3D)">
          {vessel(794, 623, 42, 53)}
        </g>
        <text x="849" y="642" fill="#cbd5e1" font-size="11" font-weight="700">HEATING-CIRCUIT</text>
        <text x="849" y="658" fill="#cbd5e1" font-size="11">EXPANSION VESSEL</text>
      </g>

      <!-- ===================================================== -->
      <!-- INDICATORS / DASHBOARD CARDS                            -->
      <!-- ===================================================== -->
      <path d="M30 698 H1170" stroke="#263449" stroke-width="1.5"/>
      <text x="38" y="729" fill="#94a3b8" font-size="12" font-weight="700">
        CURRENT SYSTEM INDICATORS
      </text>

      <g filter="url(#shadow3D)">
        <rect x="35" y="746" width="560" height="65" rx="10" fill="url(#cardGrad)" stroke="#334155"/>
        <path d="M45 747.5 H585" stroke="#fff" stroke-opacity="0.12" stroke-width="1.5"/>
        <text x="51" y="768" fill="#94a3b8" font-size="12">SOLAR THERMAL POWER</text>
        <text x="51" y="793" fill="#fbbf24" font-size="19" font-weight="700">{power}</text>
      </g>

      <g filter="url(#shadow3D)">
        <rect x="610" y="746" width="555" height="65" rx="10" fill="url(#cardGrad)" stroke="#334155"/>
        <path d="M620 747.5 H1155" stroke="#fff" stroke-opacity="0.12" stroke-width="1.5"/>
        <text x="626" y="768" fill="#94a3b8" font-size="12">STORED THERMAL ENERGY</text>
        <text x="626" y="793" fill="#fb923c" font-size="19" font-weight="700">{energy}</text>
      </g>

      <!-- Temperature sensors S1-S4 -->
      <text x="38" y="836" fill="#94a3b8" font-size="12" font-weight="700">
        TEMPERATURE SENSORS
      </text>
      {sensor_card_svg}

      <!-- Heat meters: greyed out in REAL-TIME mode, active in SIMULATION -->
      <text x="38" y="948" fill="#94a3b8" font-size="12" font-weight="700">
        HEAT METERS
      </text>
      {hm_cards}

      <text x="38" y="1068" fill="#64748b" font-size="11">
        Schematic representation · Secondary circuit states are not modelled.
      </text>
    </svg>
    """

    # st.components.v1.html is deprecated in newer Streamlit; st.iframe replaces it.
    if hasattr(st, "iframe"):
        st.iframe(svg.strip(), height=1105)
    else:
        components.html(svg, height=1105, scrolling=False)


# =========================================================
# DEMO DAY: the twin is driven by "actual" weather and hot-water use, and its backup
# heater is run by Predictive Control (or by a plain thermostat for comparison).
# Streamlit only runs the page that is on screen, so the controller is called from here.
# =========================================================
def _advance_demo_day(steps):
    from views import predictive_control as pcm

    ss = st.session_state
    cfg = ss.get("pc_cfg")
    if cfg is None:
        ss.simulation_running = False
        return
    twin = ss.twin_model
    for _ in range(steps):
        if ss.simulation_minute >= DAY_MINUTES:
            break
        now = _day_start() + timedelta(minutes=ss.simulation_minute)
        ss.sim_clock = now                                  # Predictive Control reads this clock
        irr, amb = pcm.actual_weather(cfg, now, ss.get("actual_cloud", 20))
        twin.set_irradiance_override(irr)
        twin.set_ambient_temperature(amb)
        twin.set_hot_water_demand(pcm.demand_lps(cfg, now) * ss.get("actual_use_scale", 1.0))

        if ss.get("demo_strategy", STRAT_PREDICTIVE) == STRAT_PREDICTIVE:
            cmd = pcm.controller_step_for_twin(twin, now)
            on = bool(cmd and cmd.on)
            twin.set_heater(cfg.heater_power_w if on else 0.0,
                            FORCE_ON_SETPOINT_C if on else cfg.thermostat_c)
        else:                                               # plain thermostat on S2
            hot = twin.top_temperature >= cfg.heater_inhibit_c
            twin.set_heater(0.0 if hot else cfg.heater_power_w, cfg.thermostat_c)

        twin.step(timestep_seconds=60)
        ss.simulation_minute += 1
        _log_sim_sample()


def _demo_metrics():
    ss = st.session_state
    log, cfg = ss.get("sim_log"), ss.get("pc_cfg")
    if not log or cfg is None:
        return None
    start = _day_start()
    below, switches, prev = 0, 0, False
    for r in log:
        t = start + timedelta(minutes=r["min"] - 1)
        in_window = (cfg.morning[0] <= t.hour < cfg.morning[1]) or (cfg.evening[0] <= t.hour < cfg.evening[1])
        if in_window and r["s3_tank_top"] < cfg.comfort_min_c:
            below += 1
        on = r["heater_power_w"] > 0
        switches += int(on and not prev)
        prev = on
    kwh = sum(r["heater_power_w"] for r in log) / 60000.0
    return {
        "Backup energy (kWh)": round(kwh, 2),
        "Backup cost": round(kwh * cfg.price_per_kwh, 2),
        "Minutes S3 below comfort (demand windows)": below,
        "Heater switch-ons": switches,
        "Simulated until": (start + timedelta(minutes=len(log))).strftime("%H:%M"),
    }


def _render_demo_panel():
    ss = st.session_state
    metrics = _demo_metrics()
    if metrics:
        ss.setdefault("demo_scores", {})[ss.get("demo_strategy", STRAT_PREDICTIVE)] = metrics
    scores = ss.get("demo_scores", {})
    if scores:
        st.markdown("**Scoreboard** (last run of each strategy; run the same scenario twice to compare)")
        st.dataframe([{"Strategy": k, **v} for k, v in scores.items()], hide_index=True, use_container_width=True)
    log = ss.get("pc_log") or []
    if log and ss.get("demo_strategy", STRAT_PREDICTIVE) == STRAT_PREDICTIVE:
        st.markdown("**Predictive controller decisions**")
        st.dataframe([r for r in log if r["event"] != "Replanned"][:8], hide_index=True, use_container_width=True)


# =========================================================
# CALLBACKS
# =========================================================
def _log_sim_sample():
    """Record the current simulated state so the run can be analysed on Historical Trends."""
    ss = st.session_state
    if "sim_run_id" not in ss:
        ss.sim_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        # demo day: timestamps follow the virtual clock (04:00...), so Historical Trends reads like a real day
        ss.sim_run_start = (_day_start() if _demo_on() else datetime.now()).timestamp()
        ss.sim_log = []
    state = get_simulation_state(ss.simulation_minute)
    sensors = state["sensors"]
    heater_w = state["heater_power"] if state["backup_heater_state"] else 0.0
    ss.sim_log.append({
        "ts": ss.sim_run_start + ss.simulation_minute * 60,
        "min": ss.simulation_minute,
        "s1_collector": sensors["s1_collector_fluid"],
        "s2_tank_mid": sensors["s2_tank_middle"],
        "s3_tank_top": sensors["s3_tank_top"],
        "s4_tank_bottom": sensors["s4_tank_bottom"],
        "flow_lpm": state["flow_rate"],
        "pump": bool(_as_bool(state["pump_state"])),
        "heater_power_w": heater_w,
        "ambient_temp": state["ambient_temperature"],
        "irradiance": state["solar_irradiance"],
        "pump_mode": state["pump_mode"],
    })
    if len(ss.sim_log) % (60 if _demo_on() else 10) == 0:
        _persist_sim_run()


def _persist_sim_run():
    """Save the run so far to Firebase sim/runs (same run id is overwritten, never duplicated)."""
    ss = st.session_state
    log = ss.get("sim_log")
    if not log:
        return
    sd.save_sim_run(
        sd.get_db(), ss.sim_run_id,
        {"label": (f"Demo day - {ss.get('demo_strategy', STRAT_PREDICTIVE)}" if _demo_on()
                   else "Digital Twin run"),
         "started": ss.sim_run_start,
         "note": (f"Simulated {len(log)} min. " + (
             f"Planner forecast cloud {ss.pc_cfg.cloud_today:.0f}%, actual {ss.get('actual_cloud', 20):.0f}%, "
             f"hot-water use x{ss.get('actual_use_scale', 1.0):.1f}. " if _demo_on() and ss.get("pc_cfg") else "")
             + f"Pump mode at end: {log[-1]['pump_mode']}.")},
        log,
    )
    sd.list_sim_runs.clear()


def _end_sim_run():
    """Save and forget the current run (next Run starts a new one)."""
    _persist_sim_run()
    for k in ("sim_run_id", "sim_run_start", "sim_log"):
        st.session_state.pop(k, None)


def _sim_run():
    if st.session_state.simulation_minute >= _sim_total():
        _sim_reset()
    st.session_state.simulation_running = True


def _sim_pause():
    st.session_state.simulation_running = False
    _persist_sim_run()


def _sim_reset():
    _end_sim_run()
    st.session_state.simulation_running = False
    st.session_state.simulation_minute = 0
    st.session_state.pump_mode = PUMP_AUTO
    for k in ("sim_clock", "pc_bundle", "pc_sig", "pc_cmd_on", "pc_last_change", "pc_log", "pc_override",
              "pc_ai_layer", "pc_ai_key", "pc_ai_status"):
        st.session_state.pop(k, None)
    st.session_state.twin_model = _new_model()


def _set_pump_mode(mode):
    st.session_state.pump_mode = mode
    st.session_state.twin_model.set_pump_mode(mode)


def _render_hil_details(live, state):
    """Below the diagram in HIL mode: what is real, what is simulated, and the fault list."""
    from hil import dashboard as hp
    if not live or not live.get("twin_view"):
        return
    st.divider()
    st.subheader("HIL test status")
    st.caption("🟢 PHYSICAL = real sensor/relay on the ESP32 · 🟣 SIMULATED = digital twin of the HOTSET 300 L · "
               "🟠 INJECTED = test fault applied on purpose. All values here are TEST data.")
    hp.render_live()
    st.caption("Thresholds are EXPERIMENTAL HIL test settings. The valve is a bench demonstration of a simulated "
               "heat-dump action; it is not connected to HOTSET pipework.")


def _render_twin_panel(data_source):
    """
    Everything that changes while the simulation plays. It is wrapped in a
    fragment (see show()) so only this part of the page refreshes on each tick.
    """

    if data_source == "HIL":
        from hil import dashboard as hp
        live, err = hp.read_node("hil/live")
        bstate, age = hp.bridge_status(live)
        state = hp.twin_state_from_live(live) if bstate != "none" else None
        if state is None:
            if err and live is None:
                st.error(f"Could not read Firebase: {err}")
            else:
                st.info("No HIL test running. On your computer run `python run_hil_live.py`, "
                        "pick a test above and press Start. Until then the twin has no data.")
            state = {k: None for k in (
                "collector_temperature", "tank_temperature", "cold_water_temperature", "hot_water_temperature",
                "flow_rate", "pump_state", "backup_heater_state", "solar_thermal_power", "stored_energy",
                "solar_irradiance", "simulation_time", "heat_meters", "sensors")}
            state.update({"mixer_state": "PHYSICAL", "system_status": "HIL: WAITING FOR TEST"})
        elif bstate == "offline":
            st.error(f"The HIL bridge stopped updating {age:.0f} s ago. Showing the last known state.")
    elif data_source == "REAL-TIME":
        raw, err = sd.read_latest(sd.get_db())
        empty_state = {
            "collector_temperature": None, "tank_temperature": None, "cold_water_temperature": None,
            "hot_water_temperature": None, "flow_rate": None, "pump_state": None, "mixer_state": "PHYSICAL",
            "backup_heater_state": None, "system_status": "WAITING FOR LIVE DATA", "solar_thermal_power": None,
            "stored_energy": None, "solar_irradiance": None, "simulation_time": None, "heat_meters": None,
            "sensors": None,
        }
        if raw:
            d = sd.parse_record(raw)
            s1, s2, s3, s4 = d["s1"], d["s2"], d["s3"], d["s4"]
            tank_vals = [v for v in (s2, s3, s4) if v is not None]
            tank_avg = sum(tank_vals) / len(tank_vals) if tank_vals else None
            age = time.time() - d["epoch"] if d["epoch"] else None
            if age is not None and age > 180:
                status = f"REAL-TIME (OFFLINE - last reading {sd.fmt_local(d['epoch'])})"
                st.error(f"Installation offline. The twin shows the last recorded reading from "
                         f"{sd.fmt_local(d['epoch'])}.")
            else:
                status = "REAL-TIME (LIVE)"
            if d["bad_sensors"]:
                st.warning(f"Invalid reading from {', '.join(d['bad_sensors'])} - shown as N/A.")
            wx = sd.read_weather_latest(sd.get_db())
            state = dict(empty_state)
            state.update({
                "solar_irradiance": wx.get("irradiance"),   # weather-service estimate, not measured on site
                "collector_temperature": s1,
                "tank_temperature": tank_avg,
                "hot_water_temperature": s3,
                "flow_rate": d["flow"],
                "pump_state": d["pump"],
                "system_status": status,
                "sensors": {
                    "s1_collector_fluid": s1,
                    "s2_tank_middle": s2,
                    "s3_tank_top": s3,
                    "s4_tank_bottom": s4,
                    "tank_average": tank_avg,
                },
            })
        else:
            state = empty_state
            st.info(f"No live data yet. {err or ''}")
    else:
        if (
            st.session_state.simulation_running
            and st.session_state.simulation_minute < _sim_total()
        ):
            if _demo_on():
                _advance_demo_day(STEPS_PER_TICK)
            else:
                # Advance the persistent model by one simulated minute.
                st.session_state.twin_model.step(timestep_seconds=60)
                st.session_state.simulation_minute += 1
                _log_sim_sample()

        state = get_simulation_state(st.session_state.simulation_minute)
        if _demo_on():
            clock = st.session_state.get("sim_clock") or _day_start()
            st.caption(f"Simulation clock: **{clock:%H:%M}** (demo day)")
        else:
            st.caption(f"Simulation time: **{state['simulation_time']} min**")

    st.subheader("System Representation")
    render_svg_twin(state)

    if data_source == "HIL":
        _render_hil_details(live, state)
        return

    # =====================================================
    # SYSTEM STATUS
    # =====================================================

    st.divider()

    st.subheader("Digital Twin Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Data source**")
        st.write(data_source)

    with col2:
        st.write("**System state**")
        st.write(state["system_status"])

    with col3:
        st.write("**Backup heater**")
        st.write(
            "Simulation only"
            if data_source == "REAL-TIME"
            else ("—" if state["backup_heater_state"] is None else ("ON" if state["backup_heater_state"] else "OFF"))
        )

    if data_source == "SIMULATION":

        heater_text = (
            f"{state['heater_power']:.0f} W, setpoint {state['heater_setpoint']:.0f} °C"
            if state["heater_power"] > 0
            else "disabled"
        )
        st.caption(
            f"Pump mode: **{state['pump_mode']}** · "
            f"Ambient: **{state['ambient_temperature']:.1f} °C** · "
            f"Hot-water draw: **{state['hot_water_demand'] * 60:.1f} L/min** · "
            f"Heater: **{heater_text}**"
        )

        if state["high_limit_active"]:
            st.warning("Tank high-limit reached: pump forced OFF.")

        if _demo_on():
            _render_demo_panel()

        st.caption(
            "Simulation values are generated by the HOTSETDigitalTwin "
            "model. They represent simulated system behaviour and are "
            "not physical measurements from the HOTSET installation. "
            "The heat meters are virtual: they are only active in SIMULATION mode."
        )

    else:

        st.caption(
            "Live values come from Firebase site/latest (N/A until the ESP32 reports). "
            "integration is connected. Heat meters are shown greyed out because "
            "they are not used for live telemetry."
        )

    # End of the run: stop the timer with a full-app rerun so the fragment
    # is no longer scheduled to refresh.
    if (
        data_source == "SIMULATION"
        and st.session_state.simulation_running
        and st.session_state.simulation_minute >= _sim_total()
    ):
        st.session_state.simulation_running = False
        _persist_sim_run()
        st.rerun()


def show():

    # =====================================================
    # PAGE HEADER
    # =====================================================

    st.title("🧩 HOTSET 300 L Digital Twin")

    st.markdown(
        """
        **Digital representation of the HOTSET 300 L solar
        thermal system.**

        The Digital Twin represents the physical system structure
        and displays its current operating state from either
        real-time measurements or simulated system data.
        """
    )

    # =====================================================
    # DATA SOURCE
    # =====================================================

    st.subheader("Data Source")

    data_source = st.radio(
        "Select Digital Twin operating mode:",
        ["REAL-TIME", "SIMULATION", "HIL"],
        format_func=lambda m: {"HIL": "HIL (hardware-in-the-loop)"}.get(m, m),
        horizontal=True,
        label_visibility="collapsed",
    )

    st.session_state.digital_twin_data_source = data_source

    if data_source == "REAL-TIME":

        st.info(
            "REAL-TIME mode selected. "
            "Live measurements will be supplied by the "
            "physical HOTSET sensors through Firebase."
        )

    elif data_source == "HIL":

        st.info(
            "HIL mode: the real ESP32 sensors and relays work together with the digital twin "
            "(HOTSET 300 L model) and the fault engine, which run in `run_hil_live.py`. "
            "Values not measured by real hardware are simulated and labelled."
        )

    else:

        st.info(
            "SIMULATION mode selected. "
            "System states will be supplied by the Python "
            "simulation engine."
        )

    # =====================================================
    # SESSION STATE
    # =====================================================

    # keep widget values when the person visits another page and comes back
    for k in ("demo_mode", "demo_strategy", "actual_cloud", "actual_use_scale", "demo_start_top"):
        if k in st.session_state:
            st.session_state[k] = st.session_state[k]

    if "simulation_minute" not in st.session_state:
        st.session_state.simulation_minute = 0

    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False

    if "pump_mode" not in st.session_state:
        st.session_state.pump_mode = PUMP_AUTO

    if "twin_model" not in st.session_state:
        st.session_state.twin_model = _new_model()

    # =====================================================
    # SIMULATION CONTROLS (outside the fragment, so a click
    # triggers a full rerun and the timer below is updated)
    # =====================================================

    if data_source == "SIMULATION":
        col1, col2, col3 = st.columns(3)

        with col1:
            st.button("▶ Run", on_click=_sim_run, use_container_width=True)

        with col2:
            st.button("⏸ Pause", on_click=_sim_pause, use_container_width=True)

        with col3:
            st.button("↺ Reset", on_click=_sim_reset, use_container_width=True)

        # ---- manual pump control ----
        st.markdown("**Manual pump control**")
        pump_cols = st.columns(3)
        for col, mode, label in zip(
            pump_cols,
            (PUMP_AUTO, PUMP_MANUAL_ON, PUMP_MANUAL_OFF),
            ("🤖 Auto", "⚡ Pump ON", "⛔ Pump OFF"),
        ):
            with col:
                st.button(
                    label,
                    key=f"pump_btn_{mode}",
                    on_click=_set_pump_mode,
                    args=(mode,),
                    type="primary" if st.session_state.pump_mode == mode else "secondary",
                    use_container_width=True,
                )

        st.caption(
            "Auto = differential-temperature controller. A manual command is "
            "still overridden by the tank high-limit cut-out (90 °C)."
        )

        st.checkbox(
            "🌤️ Predictive demo day (04:00-22:00, backup heater run by Predictive Control)",
            key="demo_mode", on_change=_sim_reset,
        )

    if data_source == "SIMULATION" and _demo_on():
        pcfg = st.session_state.get("pc_cfg")
        if pcfg is None:
            st.warning("Open the Predictive Control page once first, so its settings (comfort limit, "
                       "heater, forecast) exist. Then come back and press Run.")
        else:
            st.caption(
                f"The planner BELIEVES: cloud today {pcfg.cloud_today:.0f}% · comfort S3 ≥ {pcfg.comfort_min_c:.0f} °C · "
                f"heater {pcfg.heater_power_w:.0f} W (change these on the Predictive Control page). "
                "The sliders below set what REALLY happens, so the forecast can be wrong."
            )
        st.radio("Backup heater controlled by", [STRAT_PREDICTIVE, STRAT_THERMOSTAT],
                 key="demo_strategy", horizontal=True, on_change=_sim_reset)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.slider("Actual cloud cover (%)", 0, 100, 20, key="actual_cloud")
        with d2:
            st.slider("Actual hot-water use vs plan (×)", 0.5, 2.5, 1.0, 0.1, key="actual_use_scale")
        with d3:
            st.slider("Starting tank top S3 (°C)", 30, 60, 42, key="demo_start_top", on_change=_sim_reset)

    elif data_source == "SIMULATION":
        # ---- what-if scenario sliders ----
        st.markdown("**Scenario controls**")
        s1, s2 = st.columns(2)
        with s1:
            st.slider(
                "Ambient temperature (°C)",
                -10.0, 45.0, DEFAULT_AMBIENT, 0.5,
                key="ambient_slider",
            )
        with s2:
            st.slider(
                "Peak solar irradiance (W/m²)",
                0, 1000, DEFAULT_PEAK_IRRADIANCE, 10,
                key="irradiance_slider",
            )

        s3, s4, s5 = st.columns(3)
        with s3:
            st.slider(
                "Hot-water draw (L/min)",
                0.0, 15.0, DEFAULT_DEMAND_LPM, 0.5,
                key="demand_slider",
                help="The cold and hot water pipes only animate while this is above 0.",
            )
        with s4:
            st.slider(
                "Backup heater power (W, 0 = off)",
                0, 3000, DEFAULT_HEATER_POWER, 500,
                key="heater_power_slider",
            )
        with s5:
            st.slider(
                "Heater setpoint (°C)",
                25, 70, DEFAULT_HEATER_SETPOINT, 1,
                key="heater_setpoint_slider",
            )

        # Apply to the persistent model; affects the future only.
        _apply_controls(st.session_state.twin_model)

    # =====================================================
    # TWIN PANEL (auto-refreshing fragment while the simulation runs)
    # =====================================================

    playing = (
        data_source == "SIMULATION"
        and st.session_state.simulation_running
        and st.session_state.simulation_minute < _sim_total()
    )

    if data_source == "HIL":
        from hil import dashboard as hp
        hp.render_controls()
        playing_every = 2
    else:
        playing_every = TICK_SECONDS if playing else None

    twin_panel = st.fragment(run_every=playing_every)(
        _render_twin_panel
    )
    twin_panel(data_source)

    # =====================================================
    # NAVIGATION
    # =====================================================

    if st.button("← Back to Home"):
        st.session_state.page = "Home"
        st.rerun()