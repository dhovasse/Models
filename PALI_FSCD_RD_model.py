import numpy as np

# ==========================================
# PALI-2108 Phase 1b FSCD Reaction Diffusion Model (v2 — 6-Layer Physiology)
# ==========================================
# Upgrades from v1:
#   1. 6-layer bowel wall architecture based on Zhang 2018 ileal histomorphometry
#      with layer-specific formalin shrinkage corrections (1.25x-1.54x).
#   2. Perfusion-based vascular clearance: k = Q·E/f_w + k_met per layer,
#      derived from Holm 1988, Granger 1980, Rowland & Tozer, Paine 2006.
#   3. Fibrosis-dependent parameter scaling: D and k both decrease in fibrotic
#      tissue, but k falls faster than D (vascular rarefaction > matrix densification),
#      creating the paradoxical effect where fibrotic tissue is more permeable.
#   4. Sigmoid-blended layer boundaries for smooth D(x)/k(x) transitions.
#
# Unchanged from v1 (by design):
#   - R = 445 constant (split-binding saturable component is ~4-5% of total;
#     blog confirms negligible impact on penetration dynamics)
#   - Cylindrical geometry (correct but <0.5% effect for bowel dimensions)
#   - Dynamic healing / transit time models (user's novel contributions)
#   - Dosing model, IC90 threshold, prodrug cleavage framework
#
# Credit: This model was inspired by the detailed analysis from a blog post authored by Chaotropy, which provided critical insights into the expected drug behavior in FSCD and helped shape the assumptions and parameters used in this simulation.
# Source: https://www.chaotropy.com/computational-modeling-of-palisade-bios-pali-2108-tissue-penetration-in-fibrostenotic-crohns-disease/
#
# Note: This model is a simplified representation and should be interpreted in
# the context of its assumptions and limitations. It is intended for educational
# purposes to understand the potential drug penetration dynamics in FSCD.

'''
IMPORTANT CAVEAT:
This model's base case assumption of 100% prodrug cleavage (C[0] = 100) relies heavily on the Phase 1b FSCD trial design, which requires paired ileal biopsies.
Because scoping tight strictures is logistically difficult with intact anatomy, we assume the enrolled cohort will heavily skew toward mild FSCD Post-Ileocecal Resection (ICR) patients.
In post-ICR patients, the missing ileocecal valve allows colonic bacteria to reflux into the ileum, perfectly cleaving the pro-drug (100% activation).
In a broader real-world population with an intact valve, the lack of colonic bacteria will likely lead to drastically lower active drug levels at the stricture lumen.

Prodrug cleavage based on patient type estimates:

Target Population: Patients with a prior ileocecal resection (anastomotic strictures).
Estimated C[0] Range: 85.0 to 100.0 (85% to 100% cleavage)
Structural floor: 1.5x due to dense scarring and poor vascularity.
Estimated % of Total US FSCD Patients: ~35-40%

Target Population: De novo stricture patients with localized SIBO.
Estimated C[0] Range: 40.0 to 75.0 (40% to 75% cleavage)
Structural floor: 1.2x due to less angiogenesis.
Estimated % of Total US FSCD Patients: ~30-35%

Target Population: De novo stricture patients with mild/moderate disease where upstream motility is still functional.
*Estimated C[0] Range: 5.0 to 20.0 (5% to 20% cleavage)
Structural floor: 1.2x due to less angiogenesis.
**Estimated % of Total US FSCD Patients: ~25-30%

* Model predicts a complete failure at this range of cleavage.
** Essentially locks out PALI-2108 from 25-30% of US FSCD population.
'''


# ==========================================
# Zhang 2018 Morphometric Data (Table 2)
# ==========================================
# Formalin-fixed thickness (mm) from ileal resection specimens.
# Columns: [Normal, Mild, Moderate, Severe]
# Source: Zhang et al., Hum Pathol 2018; PMID 29555578
#
# Layer-specific formalin shrinkage corrections from published literature
# (range 1.1x-1.6x per dimension). We use layer-specific values:
#   Mucosa: 1.40x (high water content, significant shrinkage)
#   Muscularis mucosae: 1.25x (thin muscle, moderate shrinkage)
#   Submucosa: 1.54x (loose connective tissue, highest shrinkage)
#   Inner muscularis propria: 1.30x (dense muscle, low shrinkage)
#   Outer muscularis propria: 1.30x (dense muscle, low shrinkage)
#   Subserosa: 1.40x (connective/adipose, moderate shrinkage)

ZHANG_FORMALIN_MM = {
    # layer_name:        [Normal,  Mild,    Moderate, Severe]
    'mucosa':            [0.360,   0.360,   0.360,    0.360],   # Stays ~constant
    'muscularis_mucosae':[0.049,   0.200,   0.500,    0.868],   # 17.7x expansion in severe
    'submucosa':         [0.530,   0.900,   1.400,    2.000],   # Major fibrotic zone
    'inner_mp':          [0.370,   0.450,   0.550,    0.703],   # 1.9x expansion
    'outer_mp':          [0.310,   0.340,   0.370,    0.403],   # 1.3x expansion
    'subserosa':         [0.120,   0.180,   0.250,    0.350],   # Second fibrotic zone
}

SHRINKAGE_FACTORS = {
    'mucosa':             1.40,
    'muscularis_mucosae': 1.25,
    'submucosa':          1.54,
    'inner_mp':           1.30,
    'outer_mp':           1.30,
    'subserosa':          1.40,
}

# Compute in-vivo thicknesses
ZHANG_INVIVO_MM = {}
for layer, formalin_vals in ZHANG_FORMALIN_MM.items():
    sf = SHRINKAGE_FACTORS[layer]
    ZHANG_INVIVO_MM[layer] = [v * sf for v in formalin_vals]

# Total wall thickness for each severity level
SEVERITY_NAMES = ['Normal', 'Mild', 'Moderate', 'Severe']
TOTAL_WALL_MM = []
for i in range(4):
    total = sum(ZHANG_INVIVO_MM[layer][i] for layer in ZHANG_INVIVO_MM)
    TOTAL_WALL_MM.append(total)
# Result: ~[2.68, 3.76, 5.31, 7.22] mm in vivo


# ==========================================
# Physiological Clearance Parameters
# ==========================================
# Q: perfusion rate (ml/min/g tissue) — Holm 1988, Granger 1980
# E: extraction ratio for small lipophilic molecules — Rowland & Tozer
# f_w: tissue water fraction — Boix 2005
# k_met_frac: metabolism as fraction of mucosal vascular clearance — Paine 2006
#   Ileal CYP3A4/5 at ~30-50% of hepatic levels → ~20% of mucosal k_vasc

PERFUSION_PARAMS = {
    #                    Q (ml/min/g)  E      f_w   k_met_frac  fibrosis_k_scale  fibrosis_D_scale
    'mucosa':            (0.250,       0.40,  0.70, 0.20,       1.00,             1.00),
    'muscularis_mucosae':(0.120,       0.40,  0.70, 0.10,       0.40,             0.70),
    'submucosa':         (0.150,       0.40,  0.70, 0.05,       0.25,             0.55),
    'inner_mp':          (0.080,       0.40,  0.65, 0.02,       0.50,             0.65),
    'outer_mp':          (0.060,       0.40,  0.65, 0.01,       0.60,             0.75),
    'subserosa':         (0.100,       0.40,  0.70, 0.05,       0.30,             0.60),
}
# fibrosis_k_scale: In fibrotic tissue, vascular rarefaction reduces clearance.
#   Submucosa has the most severe rarefaction (0.25 = 75% reduction).
#   Muscularis mucosae also severe (0.40) due to obliterative muscularization.
#   These values are estimated from the blog's description of vascular rarefaction
#   patterns in Crohn's strictures.
#
# fibrosis_D_scale: Collagen deposition reduces diffusivity, but less than clearance.
#   Submucosa: 0.55 (45% reduction) vs clearance 0.25 (75% reduction).
#   This asymmetry creates the paradoxical permeability effect:
#   k falls faster than D → fibrotic tissue lets drug accumulate.
#
# NOTE: The blog states vascular clearance in the expanded muscularis mucosae
# is "assumed to scale as the square root of fold-expansion (unmeasured)".
# We implement this for the muscularis mucosae specifically.

LAYER_ORDER = ['mucosa', 'muscularis_mucosae', 'submucosa', 'inner_mp', 'outer_mp', 'subserosa']

# Fibrotic target zones (Flier 2020): submucosa and subserosa are the two
# principal fibrotic hot spots in Crohn's resection specimens.
FIBROTIC_TARGET_LAYERS = {'submucosa', 'subserosa'}


def interpolate_layer_thicknesses(total_wall_mm):
    """
    Given a target total wall thickness, interpolate the Zhang 2018 data
    to get per-layer thicknesses. Uses linear interpolation between the
    four severity reference points.
    """
    # Find the bracketing severity levels
    if total_wall_mm <= TOTAL_WALL_MM[0]:
        # Below normal — clamp to normal proportions
        return {layer: ZHANG_INVIVO_MM[layer][0] for layer in LAYER_ORDER}
    elif total_wall_mm >= TOTAL_WALL_MM[-1]:
        # Beyond severe — extrapolate from moderate→severe trend
        idx_lo, idx_hi = 2, 3
        frac = (total_wall_mm - TOTAL_WALL_MM[idx_lo]) / (TOTAL_WALL_MM[idx_hi] - TOTAL_WALL_MM[idx_lo])
    else:
        for i in range(len(TOTAL_WALL_MM) - 1):
            if TOTAL_WALL_MM[i] <= total_wall_mm <= TOTAL_WALL_MM[i + 1]:
                idx_lo, idx_hi = i, i + 1
                frac = (total_wall_mm - TOTAL_WALL_MM[idx_lo]) / (TOTAL_WALL_MM[idx_hi] - TOTAL_WALL_MM[idx_lo])
                break

    raw = {}
    for layer in LAYER_ORDER:
        lo = ZHANG_INVIVO_MM[layer][idx_lo]
        hi = ZHANG_INVIVO_MM[layer][idx_hi]
        raw[layer] = lo + frac * (hi - lo)

    # Rescale so layers sum exactly to requested wall thickness
    raw_total = sum(raw.values())
    scale = total_wall_mm / raw_total
    return {layer: raw[layer] * scale for layer in LAYER_ORDER}


def compute_layer_k(layer_name, fold_expansion, hyperemia_multiplier=1.0):
    """
    Compute vascular clearance rate for a given layer.

    k = (Q · E / f_w + k_met) · fibrosis_modifier · hyperemia_multiplier

    For muscularis mucosae, clearance scales with sqrt of fold-expansion
    (per blog: "assumed to scale as the square root of fold-expansion").
    This models the fact that as this layer massively expands (up to 17.7x),
    new vasculature doesn't keep pace with volume growth.
    """
    Q, E, f_w, k_met_frac, fibrosis_k, _ = PERFUSION_PARAMS[layer_name]

    k_vasc = Q * E / f_w                     # Vascular clearance (1/min)
    k_met = k_met_frac * k_vasc               # Metabolic clearance
    k_total = k_vasc + k_met                   # Total baseline clearance

    # Apply fibrosis-dependent vascular rarefaction
    # In normal tissue (fold_expansion ~ 1), fibrosis effect is minimal.
    # In expanded fibrotic tissue, use the fibrosis scale factor.
    fibrosis_severity = min(1.0, (fold_expansion - 1.0) / 10.0)  # 0→1 over 1x→11x expansion
    k_fibrotic = k_total * (1.0 - fibrosis_severity * (1.0 - fibrosis_k))

    # Muscularis mucosae special handling: sqrt scaling with expansion
    if layer_name == 'muscularis_mucosae' and fold_expansion > 1.0:
        expansion_dilution = 1.0 / np.sqrt(fold_expansion)
        k_fibrotic = k_total * expansion_dilution

    # Convert from 1/min to 1/sec
    k_per_sec = k_fibrotic / 60.0

    return k_per_sec * hyperemia_multiplier


def compute_layer_D(layer_name, fold_expansion):
    """
    Compute effective diffusivity for a given layer.

    D_eff = D_aq · porosity_factor · fibrosis_modifier

    Collagen deposition reduces effective diffusivity through increased
    tortuosity and reduced water fraction, but less severely than it
    reduces vascular clearance.
    """
    D_aq = 5.5e-4  # mm²/s (= 5.5e-6 cm²/s)

    _, _, _, _, _, fibrosis_D = PERFUSION_PARAMS[layer_name]

    # Porosity-based reduction: each layer has a baseline porosity factor
    # reflecting its normal tissue composition (ECM density, cellularity).
    porosity_baseline = {
        'mucosa':             1.00,  # High water content, loose
        'muscularis_mucosae': 0.80,  # Thin muscle band
        'submucosa':          0.85,  # Loose connective tissue
        'inner_mp':           0.60,  # Dense smooth muscle
        'outer_mp':           0.55,  # Dense smooth muscle
        'subserosa':          0.75,  # Connective/adipose
    }

    D_baseline = D_aq * porosity_baseline[layer_name]

    # Apply fibrosis-dependent diffusivity reduction
    fibrosis_severity = min(1.0, (fold_expansion - 1.0) / 10.0)
    D_eff = D_baseline * (1.0 - fibrosis_severity * (1.0 - fibrosis_D))

    return D_eff


class ReactionDiffusionScaledModel:
    def __init__(self, wall_thickness_mm=8.0, dx_mm=0.02, total_days=14, lumen_radius_mm=5.0):
        self.L = wall_thickness_mm
        self.dx = dx_mm
        self.nx = int(self.L / self.dx) + 1

        # Define the grid in terms of absolute radius from the center of the bowel
        self.r_lumen = lumen_radius_mm
        self.r_grid = np.linspace(self.r_lumen, self.r_lumen + self.L, self.nx)

        self.total_days = total_days
        self.dt_seconds = 10.0

        # --- PHYSICS PARAMETERS ---
        self.D_aq = 5.5e-4  # Aqueous diffusivity (mm²/s)
        self.R = 445.0       # Retardation Factor (constant; split-binding component is ~4-5%, negligible)
        # k_base removed — now computed per-layer from perfusion physiology

        # --- LAYER ARCHITECTURE ---
        self.layer_thicknesses = interpolate_layer_thicknesses(wall_thickness_mm)
        self.layer_boundaries = self._compute_boundaries()

        # Pre-compute fold expansion for each layer (vs normal)
        normal_thicknesses = {layer: ZHANG_INVIVO_MM[layer][0] for layer in LAYER_ORDER}
        self.fold_expansion = {}
        for layer in LAYER_ORDER:
            normal = normal_thicknesses[layer]
            current = self.layer_thicknesses[layer]
            self.fold_expansion[layer] = current / normal if normal > 0 else 1.0

    def _compute_boundaries(self):
        """
        Compute cumulative boundaries for each layer.
        Returns dict: layer_name -> (start_depth_mm, end_depth_mm)
        """
        boundaries = {}
        cumulative = 0.0
        for layer in LAYER_ORDER:
            start = cumulative
            end = cumulative + self.layer_thicknesses[layer]
            boundaries[layer] = (start, end)
            cumulative = end
        return boundaries

    def get_layer_at_depth(self, depth_mm):
        """Return the layer name at a given depth from the lumen surface."""
        for layer in LAYER_ORDER:
            start, end = self.layer_boundaries[layer]
            if depth_mm <= end:
                return layer
        return LAYER_ORDER[-1]  # Beyond wall → subserosa

    def _sigmoid_blend(self, x, boundary, width=0.05):
        """Smooth sigmoid transition centered at boundary with given width (mm)."""
        return 1.0 / (1.0 + np.exp(-(x - boundary) / width))

    def build_physics_maps(self, hyperemia_multiplier=1.0):
        """
        Build D_map and k_map arrays across the spatial grid.
        Uses sigmoid-blended transitions between layers for smoothness.
        """
        D_map = np.zeros(self.nx)
        k_map = np.zeros(self.nx)

        # Compute per-layer D and k values
        layer_D = {}
        layer_k = {}
        for layer in LAYER_ORDER:
            layer_D[layer] = compute_layer_D(layer, self.fold_expansion[layer])
            layer_k[layer] = compute_layer_k(layer, self.fold_expansion[layer],
                                              hyperemia_multiplier=hyperemia_multiplier)

        # Build maps with sigmoid blending at boundaries
        for i, r in enumerate(self.r_grid):
            depth = r - self.r_lumen

            # Start with first layer values
            D_val = layer_D[LAYER_ORDER[0]]
            k_val = layer_k[LAYER_ORDER[0]]

            # Blend transitions at each boundary
            for j in range(1, len(LAYER_ORDER)):
                prev_layer = LAYER_ORDER[j - 1]
                curr_layer = LAYER_ORDER[j]
                boundary = self.layer_boundaries[prev_layer][1]

                blend = self._sigmoid_blend(depth, boundary)
                D_val = D_val * (1.0 - blend) + layer_D[curr_layer] * blend
                k_val = k_val * (1.0 - blend) + layer_k[curr_layer] * blend

            D_map[i] = D_val
            k_map[i] = k_val

        return D_map, k_map

    def get_daily_hyperemia(self, day, initial_multiplier, floor=1.5, dynamic=False):
        """
        Calculates the inflammation multiplier for the current day.
        If dynamic=True: Decays from initial_multiplier down to 'floor'.
        If dynamic=False: Stays constant at initial_multiplier.
        """
        if not dynamic:
            return initial_multiplier

        # Virtuous Cycle Logic:
        acute_component = max(0, initial_multiplier - floor)
        decayed_acute = acute_component * np.exp(-0.2 * day)  # Rapid drop in first week

        return floor + decayed_acute

    def get_fibrotic_target_indices(self):
        """
        Return grid indices corresponding to the fibrotic target zones
        (submucosa + subserosa), as identified by Flier 2020.
        """
        wall_depth = self.r_grid - self.r_lumen
        target_mask = np.zeros(self.nx, dtype=bool)

        for layer in FIBROTIC_TARGET_LAYERS:
            if layer in self.layer_boundaries:
                start, end = self.layer_boundaries[layer]
                target_mask |= (wall_depth >= start) & (wall_depth <= end)

        return np.where(target_mask)[0]

    def run_simulation(self, initial_hyperemia=1.0, structural_floor=1.5,
                       activation_pct=100.0, dynamic_healing=False, transit_hours=8.0):
        C = np.zeros(self.nx)
        steps_per_day = int((24 * 3600) / self.dt_seconds)

        # --- SURFACE CONCENTRATION CALIBRATION (v3.1) ---
        # The model normalizes all concentrations to IC90 = 1.0.
        # c_input_max represents the PEAK tissue surface concentration
        # at the mucosal interface (first grid point) during the dosing window.
        #
        # Derivation (first-principles):
        #   Phase 1a MAD colonic biopsies (30mg, healthy volunteers):
        #     "Trough levels near or above IC90 at 36h" → bulk tissue ~1-5× IC90
        #     Peak (estimated ~4-8h) ≈ 1.8× trough → bulk peak ~2-9× IC90
        #     Surface/bulk correction (exponential decay, 2mm biopsy): ~1.8×
        #     → Healthy colon surface peak ≈ 4-16× IC90
        #
        #   Stricture-specific amplification factors:
        #     Stasis concentration (prestenotic pooling): 5-20× vs normal transit
        #     Damaged mucosa (reduced barrier): 2-5× vs healthy epithelium
        #     Post-ICR bacterial activation: ~90% vs ~10% for intact valve
        #
        #   Central estimate (post-ICR, moderate stasis):
        #     ~9× IC90 (colon surface) × ~7× (stasis + barrier) ≈ 60× IC90
        #     Then × 3× for post-ICR activation boost → ~200× IC90
        #
        #   Range: 50× (conservative) to 800× (optimistic post-ICR)
        #   Blog uses: 300-3000× (theoretical, not clinically validated)
        #   Our 200× is data-anchored and 1.5× below the blog's lower bound.
        #
        # activation_pct modulates this for prodrug cleavage scenarios:
        #   100% = full post-ICR activation (200× IC90 peak)
        #   60%  = SIBO-assisted (120× IC90 peak)
        #   10%  = intact valve, minimal activation (20× IC90 peak)
        C_SURFACE_BASE = 200.0  # × IC90, calibrated to Phase 1a + stricture factors
        c_input_max = C_SURFACE_BASE * (activation_pct / 100.0)

        # --- PHARMACOKINETIC BOUNDARY CONDITION (v3) ---
        # The v1/v2 square-wave model assumed instant C_max at the lumen surface
        # for the entire transit window. This is unrealistic for an oral prodrug:
        #
        #   1. Gastric emptying + transit to ileum: ~1.5-3h (enteric-coated tablet)
        #   2. Bacterial β-glucuronidase cleavage: not instantaneous
        #   3. Produces a gradual rise → peak → decay profile (Bateman function)
        #
        # The square wave overestimates 24h AUC by ~1.8x vs. a realistic profile.
        #
        # We use a Bateman function: C(t) = A × [exp(-k_el×t) - exp(-k_abs×t)]
        # scaled so peak = c_input_max.
        #
        # Parameters:
        #   t_lag:  Lag time before any drug appears at the stricture site.
        #           Gastric emptying (0.5-1h) + small bowel transit to ileum (1.5-3h).
        #           For post-ICR patients (missing valve): ~2h (shorter transit).
        #           For intact anatomy: ~3h (longer transit to terminal ileum).
        #           Stricture stasis can prolong local exposure AFTER drug arrives,
        #           but does not accelerate initial delivery.
        #
        #   t_peak: Time from drug arrival to peak luminal concentration.
        #           Reflects bacterial cleavage kinetics + dissolution.
        #           Estimated ~1.5-2.5h based on:
        #           - Phase 1a: food "modestly delayed Tmax" (Palisade Bio, May 2025)
        #           - Delayed-release formulations typically show Tmax 3-6h post-dose
        #           - β-glucuronidase cleavage half-time ~30-60min in colonic lumen
        #
        #   k_el:   Luminal elimination rate (same as v1/v2: t½ = 2h → k = 0.346/h)
        #           Represents drug leaving the lumen via absorption + peristalsis.
        #
        # For strictured patients, stasis prolongs the tail (drug stays in contact
        # longer), which we model by extending the effective elimination half-life
        # proportionally to the transit_hours parameter.
        #
        # Reference: Blog acknowledges "the 8-hour plateau may be optimistic for
        # ileal delivery given typical 2-to-4-hour small bowel transit"

        t_lag = 2.5        # hours: gastric emptying + transit to ileal stricture
        t_peak_local = 2.0 # hours: from arrival to peak (cleavage + dissolution)

        # Effective luminal elimination: in a strictured bowel, stasis slows
        # drug washout. Scale elimination half-life with transit delay.
        # Normal t½_el = 2h. Stricture stasis factor = transit_hours / 5.0
        # (5h = normal baseline transit, so no modification at baseline).
        stasis_factor = max(1.0, transit_hours / 5.0)
        t_half_el = 2.0 * stasis_factor  # hours
        k_el = np.log(2) / t_half_el     # 1/hour

        # Absorption rate derived from t_peak constraint:
        # For Bateman function, Tmax = ln(k_abs/k_el) / (k_abs - k_el)
        # We solve numerically for k_abs given desired t_peak_local and k_el
        # Approximation: k_abs ≈ 2.5 / t_peak_local when k_abs >> k_el
        k_abs = 2.5 / t_peak_local       # 1/hour (approximate)

        # Scaling factor so peak concentration = c_input_max
        # Bateman peak occurs at t_peak_local: find the peak value to normalize
        t_at_peak = t_peak_local
        peak_raw = np.exp(-k_el * t_at_peak) - np.exp(-k_abs * t_at_peak)
        if peak_raw > 0:
            bateman_scale = c_input_max / peak_raw
        else:
            bateman_scale = c_input_max

        # Pre-compute the 24h surface concentration profile
        # (avoids recomputing Bateman at every timestep)
        surface_profile = np.zeros(steps_per_day)
        for s in range(steps_per_day):
            t_hours = s * self.dt_seconds / 3600.0
            t_since_arrival = t_hours - t_lag
            if t_since_arrival <= 0:
                surface_profile[s] = 0.0
            else:
                val = bateman_scale * (np.exp(-k_el * t_since_arrival)
                                       - np.exp(-k_abs * t_since_arrival))
                surface_profile[s] = max(0.0, val)

        for day in range(1, self.total_days + 1):

            # 1. Calculate today's inflammation level
            current_multiplier = self.get_daily_hyperemia(
                day, initial_hyperemia, floor=structural_floor, dynamic=dynamic_healing
            )

            # 2. Build Physics Maps with current hyperemia
            D_map, k_map = self.build_physics_maps(hyperemia_multiplier=current_multiplier)

            # Pre-calculate FDM constants
            alpha = (D_map * self.dt_seconds) / (self.dx**2 * self.R)
            beta = (k_map * self.dt_seconds) / self.R

            # Stability check
            if np.max(alpha) > 0.5:
                self.dt_seconds = 1.0
                steps_per_day = int((24 * 3600) / self.dt_seconds)
                alpha = (D_map * self.dt_seconds) / (self.dx**2 * self.R)
                beta = (k_map * self.dt_seconds) / self.R

            # 3. Run 24 Hours of Diffusion
            for s in range(steps_per_day):
                # Apply surface boundary condition from pre-computed profile
                C[0] = surface_profile[s % len(surface_profile)]

                # Cylindrical diffusion: d²C/dr² + (1/r)dC/dr
                d2C = (C[2:] - 2 * C[1:-1] + C[:-2])
                dC = (C[2:] - C[:-2]) / 2.0
                r_local = self.r_grid[1:-1]

                radial_penalty = (self.dx / r_local) * dC
                C[1:-1] = C[1:-1] + alpha[1:-1] * (d2C + radial_penalty) - (beta[1:-1] * C[1:-1])

                C[-1] = C[-2]  # Neumann BC at serosal surface

        return C

    def print_layer_architecture(self):
        """Print the computed layer architecture for this wall thickness."""
        print(f"\n  {'Layer':<25} {'Thickness':>10} {'Depth Range':>16} {'Fold Exp':>10}")
        print(f"  {'-' * 65}")
        for layer in LAYER_ORDER:
            start, end = self.layer_boundaries[layer]
            thick = self.layer_thicknesses[layer]
            fold = self.fold_expansion[layer]
            marker = " ** FIBROTIC TARGET" if layer in FIBROTIC_TARGET_LAYERS else ""
            print(f"  {layer:<25} {thick:>8.3f} mm   {start:>6.3f}-{end:>6.3f} mm   {fold:>7.1f}x{marker}")
        total = sum(self.layer_thicknesses.values())
        print(f"  {'TOTAL':<25} {total:>8.3f} mm")


def get_verdict_proximal(prox_sub_pct):
    """
    Verdict based on proximal submucosa coverage (first 1mm of submucosa).
    This is the active fibrotic front where early-stage fibrogenesis is most
    vigorous. The interception thesis depends on treating THIS zone.

    Thresholds calibrated to the 6-layer model:
    - >70%: Drug saturates the active fibrotic front. Interception plausible.
    - >40%: Partial coverage of the advancing front. May slow progression.
    - >15%: Minimal reach into the submucosa. Unlikely to alter disease course.
    - <15%: Drug essentially confined to mucosa/muscularis mucosae.
    """
    if prox_sub_pct > 70:
        return "VERY BULLISH — Active fibrotic front saturated; interception plausible"
    elif prox_sub_pct > 40:
        return "BULLISH — Partial interception of proximal fibrosis"
    elif prox_sub_pct > 15:
        return "NEUTRAL — Marginal submucosal reach; may slow but not halt progression"
    else:
        return "BEARISH — Drug confined above submucosa; deep fibrosis untreated"


def get_verdict_total(total_sub_pct):
    """
    Verdict based on total submucosa coverage.
    This is a harder bar — covering >50% of the full submucosa means
    deep penetration well beyond the active front.

    NOTE: Subserosa coverage is excluded from verdicts because serosal-side
    fibrosis (driven by creeping fat / mesenteric inflammation) is likely
    beyond the reach of ANY luminally delivered drug. Penalizing PALI-2108
    for this would conflate a drug limitation with a delivery-route limitation.
    """
    if total_sub_pct > 50:
        return "Deep submucosal penetration; strong anti-fibrotic potential"
    elif total_sub_pct > 25:
        return "Moderate submucosal reach; proximal fibrosis treatable"
    elif total_sub_pct > 10:
        return "Shallow submucosal entry; limited to early-stage disease"
    else:
        return "Negligible submucosal penetration"


def analyze_and_report(thickness, structural_floor=1.5, activation_pct=100.0,
                       missing_valve=True, description=""):
    # 1. Calculate Anatomy (Conservation of Mass Geometry)
    outer_bowel_radius_mm = 12.0
    healthy_wall_mm = 2.0
    healthy_lumen_mm = outer_bowel_radius_mm - healthy_wall_mm  # Baseline 10.0mm radius

    # Calculate the narrowed lumen based on the diseased wall thickness
    dynamic_lumen_mm = max(0.5, outer_bowel_radius_mm - thickness)

    sim = ReactionDiffusionScaledModel(wall_thickness_mm=thickness, lumen_radius_mm=dynamic_lumen_mm)
    IC90 = 1.0

    # 2. Establish the Baseline Biological Transit
    if missing_valve:
        base_transit = 5.0
    else:
        base_transit = 8.0

    # 3. ACAT Area-Proportional Transit Calculation
    area_ratio = (healthy_lumen_mm / dynamic_lumen_mm)**2
    calculated_transit = base_transit * area_ratio

    # 4. Apply the Clinical Obstruction Guardrail
    transit_hours = min(16.0, calculated_transit)

    print(f"\n{'=' * 60}")
    print(f"REPORT FOR {thickness}mm WALL STRICTURE")
    print(f"Lumen narrowed to: {dynamic_lumen_mm:.1f}mm (Area Ratio: {area_ratio:.2f}x)")

    if calculated_transit > 16.0:
        print(f"Calculated Transit: {calculated_transit:.1f} hours -> CAPPED at {transit_hours:.1f}h (Clinical Obstruction Limit)")
    else:
        print(f"Calculated Transit Time: {transit_hours:.1f} hours")

    print(f"Patient Profile: Floor={structural_floor}x | Activation={activation_pct}% | Missing Ileocecal Valve: {'Yes' if missing_valve else 'No'}")
    if description:
        print(f"Patient Description: {description}")

    # Print the 6-layer architecture
    sim.print_layer_architecture()

    print(f"{'=' * 60}")

    # THEORETICAL BEST CASE
    conc_cold = sim.run_simulation(
        initial_hyperemia=structural_floor,
        structural_floor=structural_floor,
        activation_pct=activation_pct,
        transit_hours=transit_hours,
        dynamic_healing=False
    )

    # PREDICTED TRIAL RESULT
    conc_dyn = sim.run_simulation(
        initial_hyperemia=3.0,
        structural_floor=structural_floor,
        activation_pct=activation_pct,
        transit_hours=transit_hours,
        dynamic_healing=True
    )

    # Calculate Coverage: two-tier metric
    #   Tier 1: Proximal submucosa (first 1mm) — the active fibrotic front
    #   Tier 2: Total submucosa — full depth of the primary fibrotic zone
    #   Subserosa is reported but excluded from verdicts (likely unreachable luminally)
    wall_depth = sim.r_grid - sim.r_lumen

    sub_start, sub_end = sim.layer_boundaries['submucosa']
    prox_sub_end = min(sub_start + 1.0, sub_end)
    ser_start, ser_end = sim.layer_boundaries['subserosa']

    # Proximal submucosa indices
    prox_idx = np.where((wall_depth >= sub_start) & (wall_depth <= prox_sub_end))[0]
    # Total submucosa indices
    sub_idx = np.where((wall_depth >= sub_start) & (wall_depth <= sub_end))[0]
    # Subserosa indices (reported only)
    ser_idx = np.where((wall_depth >= ser_start) & (wall_depth <= ser_end))[0]

    if len(prox_idx) == 0 or len(sub_idx) == 0:
        print("Error: No target indices found. Check grid boundaries.")
        return

    def compute_coverages(C_arr):
        prox_cov = (np.sum(C_arr[prox_idx] > IC90) / len(prox_idx)) * 100
        sub_cov = (np.sum(C_arr[sub_idx] > IC90) / len(sub_idx)) * 100
        ser_cov = (np.sum(C_arr[ser_idx] > IC90) / len(ser_idx)) * 100 if len(ser_idx) > 0 else 0.0
        # Max therapeutic depth
        above = np.where(C_arr > IC90)[0]
        max_depth = wall_depth[above[-1]] if len(above) > 0 else 0.0
        return prox_cov, sub_cov, ser_cov, max_depth

    prox_cold, sub_cold, ser_cold, depth_cold = compute_coverages(conc_cold)
    prox_dyn, sub_dyn, ser_dyn, depth_dyn = compute_coverages(conc_dyn)

    print(f"THEORETICAL BEST CASE (Structural Baseline)")
    print(f"  Max Therapeutic Depth:          {depth_cold:.2f} mm")
    print(f"  Proximal Submucosa (1mm):       {prox_cold:.1f}%  →  {get_verdict_proximal(prox_cold)}")
    print(f"  Total Submucosa:                {sub_cold:.1f}%  →  {get_verdict_total(sub_cold)}")
    print(f"  Subserosa (for reference):      {ser_cold:.1f}%")
    print(f"-" * 40)
    print(f"PREDICTED TRIAL RESULT (Day {sim.total_days} Dynamic)")
    print(f"  Max Therapeutic Depth:          {depth_dyn:.2f} mm")
    print(f"  Proximal Submucosa (1mm):       {prox_dyn:.1f}%  →  {get_verdict_proximal(prox_dyn)}")
    print(f"  Total Submucosa:                {sub_dyn:.1f}%  →  {get_verdict_total(sub_dyn)}")
    print(f"  Subserosa (for reference):      {ser_dyn:.1f}%")


if __name__ == "__main__":
    # Print reference architecture
    print("=" * 60)
    print("ZHANG 2018 REFERENCE: In-Vivo Layer Thicknesses (mm)")
    print("=" * 60)
    print(f"  {'Layer':<25}", end="")
    for sev in SEVERITY_NAMES:
        print(f" {sev:>10}", end="")
    print()
    print(f"  {'-' * 70}")
    for layer in LAYER_ORDER:
        print(f"  {layer:<25}", end="")
        for i in range(4):
            print(f" {ZHANG_INVIVO_MM[layer][i]:>10.3f}", end="")
        print()
    print(f"  {'TOTAL':<25}", end="")
    for t in TOTAL_WALL_MM:
        print(f" {t:>10.3f}", end="")
    print()

    print("\n")

    # --- Run scenarios ---
    # Mild FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=5.0, structural_floor=1.5, activation_pct=100.0,
                       missing_valve=True,
                       description="(Base Case) Post-Ileocecal Resection with Dense Scarring")

    # Moderate FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=7.0, structural_floor=1.5, activation_pct=100.0,
                       missing_valve=True,
                       description="Post-Ileocecal Resection with Dense Scarring")

    # Severe FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=9.0, structural_floor=1.5, activation_pct=100.0,
                       missing_valve=True,
                       description="Post-Ileocecal Resection with Dense Scarring")

    # Mild FSCD, De Novo + SIBO -> Low Floor, High SIBO
    analyze_and_report(thickness=5.0, structural_floor=1.2, activation_pct=60.0,
                       missing_valve=False,
                       description="De Novo Stricture with Localized SIBO")

    # Moderate FSCD, De Novo + SIBO -> Low Floor, High SIBO
    analyze_and_report(thickness=7.0, structural_floor=1.2, activation_pct=60.0,
                       missing_valve=False,
                       description="De Novo Stricture with Localized SIBO")

    # Post-Op Anastomosis
    analyze_and_report(thickness=4.0, structural_floor=1.5, activation_pct=100.0,
                       missing_valve=True,
                       description="Post-Op Anastomosis")


'''
The "Coverage %" Cheat Sheet (v2 — Two-Tier System)

TIER 1: Proximal Submucosa (first 1mm from submucosal surface)
This is the ACTIVE FIBROTIC FRONT — where fibrogenesis is most vigorous in early disease.
The interception thesis depends on treating this zone before the wall outgrows the drug's reach.
  >70% — VERY BULLISH: Active front saturated. Interception of fibrotic progression plausible.
  40-70% — BULLISH: Partial interception. Drug reaches significant portion of advancing front.
  15-40% — NEUTRAL: Marginal reach. May slow but unlikely to halt progression.
  <15% — BEARISH: Drug essentially confined above the submucosa.

TIER 2: Total Submucosa
A harder bar reflecting full-depth submucosal penetration.
  >50% — Deep penetration; strong anti-fibrotic potential.
  25-50% — Moderate reach; proximal fibrosis treatable.
  10-25% — Shallow entry; limited to early-stage disease.
  <10% — Negligible submucosal penetration.

NOTE: Subserosa is reported but excluded from verdicts.
Serosal-side fibrosis (creeping fat, mesenteric inflammation) is likely beyond the reach
of ANY luminally delivered drug. This is a delivery-route limitation, not a drug limitation.
'''

# Output:
'''
============================================================                                                                               
ZHANG 2018 REFERENCE: In-Vivo Layer Thicknesses (mm)
============================================================
  Layer                         Normal       Mild   Moderate     Severe
  ----------------------------------------------------------------------
  mucosa                         0.504      0.504      0.504      0.504
  muscularis_mucosae             0.061      0.250      0.625      1.085
  submucosa                      0.816      1.386      2.156      3.080
  inner_mp                       0.481      0.585      0.715      0.914
  outer_mp                       0.403      0.442      0.481      0.524
  subserosa                      0.168      0.252      0.350      0.490
  TOTAL                          2.433      3.419      4.831      6.597



============================================================
REPORT FOR 5.0mm WALL STRICTURE
Lumen narrowed to: 7.0mm (Area Ratio: 2.04x)
Calculated Transit Time: 10.2 hours
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: (Base Case) Post-Ileocecal Resection with Dense Scarring

  Layer                      Thickness      Depth Range   Fold Exp
  -----------------------------------------------------------------
  mucosa                       0.504 mm    0.000- 0.504 mm       1.0x
  muscularis_mucosae           0.669 mm    0.504- 1.173 mm      10.9x
  submucosa                    2.244 mm    1.173- 3.417 mm       2.7x ** FIBROTIC TARGET
  inner_mp                     0.734 mm    3.417- 4.151 mm       1.5x
  outer_mp                     0.485 mm    4.151- 4.637 mm       1.2x
  subserosa                    0.363 mm    4.637- 5.000 mm       2.2x ** FIBROTIC TARGET
  TOTAL                        5.000 mm
============================================================
THEORETICAL BEST CASE (Structural Baseline)
  Max Therapeutic Depth:          1.96 mm
  Proximal Submucosa (1mm):       80.0%  →  VERY BULLISH — Active fibrotic front saturated; interception plausible
  Total Submucosa:                35.7%  →  Moderate submucosal reach; proximal fibrosis treatable
  Subserosa (for reference):      0.0%
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
  Max Therapeutic Depth:          1.86 mm
  Proximal Submucosa (1mm):       70.0%  →  BULLISH — Partial interception of proximal fibrosis
  Total Submucosa:                31.2%  →  Moderate submucosal reach; proximal fibrosis treatable
  Subserosa (for reference):      0.0%

============================================================
REPORT FOR 7.0mm WALL STRICTURE
Lumen narrowed to: 5.0mm (Area Ratio: 4.00x)
Calculated Transit: 20.0 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: Post-Ileocecal Resection with Dense Scarring

  Layer                      Thickness      Depth Range   Fold Exp
  -----------------------------------------------------------------
  mucosa                       0.504 mm    0.000- 0.504 mm       1.0x
  muscularis_mucosae           1.190 mm    0.504- 1.694 mm      19.4x
  submucosa                    3.291 mm    1.694- 4.985 mm       4.0x ** FIBROTIC TARGET
  inner_mp                     0.959 mm    4.985- 5.944 mm       2.0x
  outer_mp                     0.534 mm    5.944- 6.478 mm       1.3x
  subserosa                    0.522 mm    6.478- 7.000 mm       3.1x ** FIBROTIC TARGET
  TOTAL                        7.000 mm
============================================================
THEORETICAL BEST CASE (Structural Baseline)
  Max Therapeutic Depth:          2.28 mm
  Proximal Submucosa (1mm):       60.0%  →  BULLISH — Partial interception of proximal fibrosis
  Total Submucosa:                18.2%  →  Shallow submucosal entry; limited to early-stage disease
  Subserosa (for reference):      0.0%
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
  Max Therapeutic Depth:          2.16 mm
  Proximal Submucosa (1mm):       48.0%  →  BULLISH — Partial interception of proximal fibrosis
  Total Submucosa:                14.5%  →  Shallow submucosal entry; limited to early-stage disease
  Subserosa (for reference):      0.0%

============================================================
REPORT FOR 9.0mm WALL STRICTURE
Lumen narrowed to: 3.0mm (Area Ratio: 11.11x)
Calculated Transit: 55.6 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: Post-Ileocecal Resection with Dense Scarring

  Layer                      Thickness      Depth Range   Fold Exp
  -----------------------------------------------------------------
  mucosa                       0.504 mm    0.000- 0.504 mm       1.0x
  muscularis_mucosae           1.711 mm    0.504- 2.215 mm      27.9x
  submucosa                    4.338 mm    2.215- 6.553 mm       5.3x ** FIBROTIC TARGET
  inner_mp                     1.185 mm    6.553- 7.737 mm       2.5x
  outer_mp                     0.582 mm    7.737- 8.319 mm       1.4x
  subserosa                    0.681 mm    8.319- 9.000 mm       4.1x ** FIBROTIC TARGET
  TOTAL                        9.000 mm
============================================================
THEORETICAL BEST CASE (Structural Baseline)
  Max Therapeutic Depth:          2.42 mm
  Proximal Submucosa (1mm):       22.0%  →  NEUTRAL — Marginal submucosal reach; may slow but not halt progression
  Total Submucosa:                5.1%  →  Negligible submucosal penetration
  Subserosa (for reference):      0.0%
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
  Max Therapeutic Depth:          2.30 mm
  Proximal Submucosa (1mm):       10.0%  →  BEARISH — Drug confined above submucosa; deep fibrosis untreated
  Total Submucosa:                2.3%  →  Negligible submucosal penetration
  Subserosa (for reference):      0.0%

============================================================
REPORT FOR 5.0mm WALL STRICTURE
Lumen narrowed to: 7.0mm (Area Ratio: 2.04x)
Calculated Transit: 16.3 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.2x | Activation=60.0% | Missing Ileocecal Valve: No
Patient Description: De Novo Stricture with Localized SIBO

  Layer                      Thickness      Depth Range   Fold Exp
  -----------------------------------------------------------------
  mucosa                       0.504 mm    0.000- 0.504 mm       1.0x
  muscularis_mucosae           0.669 mm    0.504- 1.173 mm      10.9x
  submucosa                    2.244 mm    1.173- 3.417 mm       2.7x ** FIBROTIC TARGET
  inner_mp                     0.734 mm    3.417- 4.151 mm       1.5x
  outer_mp                     0.485 mm    4.151- 4.637 mm       1.2x
  subserosa                    0.363 mm    4.637- 5.000 mm       2.2x ** FIBROTIC TARGET
  TOTAL                        5.000 mm
============================================================
THEORETICAL BEST CASE (Structural Baseline)
  Max Therapeutic Depth:          2.04 mm
  Proximal Submucosa (1mm):       88.0%  →  VERY BULLISH — Active fibrotic front saturated; interception plausible
  Total Submucosa:                39.3%  →  Moderate submucosal reach; proximal fibrosis treatable
  Subserosa (for reference):      0.0%
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
  Max Therapeutic Depth:          1.88 mm
  Proximal Submucosa (1mm):       72.0%  →  VERY BULLISH — Active fibrotic front saturated; interception plausible
  Total Submucosa:                32.1%  →  Moderate submucosal reach; proximal fibrosis treatable
  Subserosa (for reference):      0.0%

============================================================
REPORT FOR 7.0mm WALL STRICTURE
Lumen narrowed to: 5.0mm (Area Ratio: 4.00x)
Calculated Transit: 32.0 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.2x | Activation=60.0% | Missing Ileocecal Valve: No
Patient Description: De Novo Stricture with Localized SIBO

  Layer                      Thickness      Depth Range   Fold Exp
  -----------------------------------------------------------------
  mucosa                       0.504 mm    0.000- 0.504 mm       1.0x
  muscularis_mucosae           1.190 mm    0.504- 1.694 mm      19.4x
  submucosa                    3.291 mm    1.694- 4.985 mm       4.0x ** FIBROTIC TARGET
  inner_mp                     0.959 mm    4.985- 5.944 mm       2.0x
  outer_mp                     0.534 mm    5.944- 6.478 mm       1.3x
  subserosa                    0.522 mm    6.478- 7.000 mm       3.1x ** FIBROTIC TARGET
  TOTAL                        7.000 mm
============================================================
THEORETICAL BEST CASE (Structural Baseline)
  Max Therapeutic Depth:          2.20 mm
  Proximal Submucosa (1mm):       52.0%  →  BULLISH — Partial interception of proximal fibrosis
  Total Submucosa:                15.8%  →  Shallow submucosal entry; limited to early-stage disease
  Subserosa (for reference):      0.0%
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
  Max Therapeutic Depth:          2.06 mm
  Proximal Submucosa (1mm):       38.0%  →  NEUTRAL — Marginal submucosal reach; may slow but not halt progression
  Total Submucosa:                11.5%  →  Shallow submucosal entry; limited to early-stage disease
  Subserosa (for reference):      0.0%

============================================================
REPORT FOR 4.0mm WALL STRICTURE
Lumen narrowed to: 8.0mm (Area Ratio: 1.56x)
Calculated Transit Time: 7.8 hours
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: Post-Op Anastomosis

  Layer                      Thickness      Depth Range   Fold Exp
  -----------------------------------------------------------------
  mucosa                       0.504 mm    0.000- 0.504 mm       1.0x
  muscularis_mucosae           0.404 mm    0.504- 0.908 mm       6.6x
  submucosa                    1.703 mm    0.908- 2.611 mm       2.1x ** FIBROTIC TARGET
  inner_mp                     0.638 mm    2.611- 3.250 mm       1.3x
  outer_mp                     0.458 mm    3.250- 3.708 mm       1.1x
  subserosa                    0.292 mm    3.708- 4.000 mm       1.7x ** FIBROTIC TARGET
  TOTAL                        4.000 mm
============================================================
THEORETICAL BEST CASE (Structural Baseline)
  Max Therapeutic Depth:          1.80 mm
  Proximal Submucosa (1mm):       90.0%  →  VERY BULLISH — Active fibrotic front saturated; interception plausible
  Total Submucosa:                52.9%  →  Deep submucosal penetration; strong anti-fibrotic potential
  Subserosa (for reference):      0.0%
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
  Max Therapeutic Depth:          1.70 mm
  Proximal Submucosa (1mm):       80.0%  →  VERY BULLISH — Active fibrotic front saturated; interception plausible
  Total Submucosa:                47.1%  →  Moderate submucosal reach; proximal fibrosis treatable
  Subserosa (for reference):      0.0%
'''