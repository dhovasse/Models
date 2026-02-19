import numpy as np

# ==========================================
# PALI-2108 Phase 1b FSCD Reaction Diffusion Model 
# ==========================================
# This model simulates drug penetration in mild, moderate, and severe FSCD scenarios.
# It incorporates:
# - Binding kinetics (Retardation Factor)
# - Dosing regimen (QD with accumulation)
# - PK data from Phase 1b UC Abstract (Tmax, Half-life)
# - Scaled Anatomy: Layers expand as the wall gets thicker.
# - Hyperemia: Simulates the effect of inflammation washing the drug away.
# - Dynamic Healing: Models the reduction in inflammation over time as the drug takes effect.
# - ACAT Area-Proportional Transit Calculation: Transit time scales with the loss of cross-sectional area due to stricture.
#
# Credit: This model was inspired by the detailed analysis from a blog post authored by Chaotropy, which provided critical insights into the expected drug behavior in FSCD and helped shape the assumptions and parameters used in this simulation.
# Source: https://www.chaotropy.com/computational-modeling-of-palisade-bios-pali-2108-tissue-penetration-in-fibrostenotic-crohns-disease/
#
# Note: This model is a simplified representation and should be interpreted in the context of its assumptions and limitations. It is intended for educational purposes to understand the potential drug penetration dynamics in FSCD.

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
        self.D_aq = 5.5e-4 # Diffusivity
        self.R = 445.0     # Retardation Factor
        self.k_base = 0.0015 # Clearance Rate

    def get_layer_boundaries(self):
        # Mucosa constant ~0.8mm
        mucosa_end = 0.8
        # Target Zone scales (40% of wall)
        target_end = self.L * 0.40 
        return mucosa_end, target_end

    def get_layer_properties(self, x, m_end, t_end):
        if x < m_end: return 1.0, 1.0 
        elif x < t_end: return 0.6, 0.5 
        else: return 0.4, 0.3 

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
        decayed_acute = acute_component * np.exp(-0.2 * day) # Rapid drop in first week
        
        return floor + decayed_acute

    def run_simulation(self, initial_hyperemia=1.0, structural_floor=1.5, activation_pct=100.0, dynamic_healing=False, transit_hours=8.0):
        C = np.zeros(self.nx)
        steps_per_day = int((24 * 3600) / self.dt_seconds)
        
        m_end, t_end = self.get_layer_boundaries()
        c_input_max = 100.0 * (activation_pct / 100.0)

        for day in range(1, self.total_days + 1):
            
            # 1. Calculate today's inflammation level
            current_multiplier = self.get_daily_hyperemia(day, initial_hyperemia, floor=structural_floor, dynamic=dynamic_healing)
            
            # 2. Build Physics Maps
            D_map = np.zeros(self.nx)
            k_map = np.zeros(self.nx)
            
            for i, r in enumerate(self.r_grid):
                d_f, k_f = self.get_layer_properties(r - self.r_lumen, m_end, t_end)
                D_map[i] = self.D_aq * d_f
                k_map[i] = self.k_base * k_f * current_multiplier

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
                day_fraction = (s * self.dt_seconds / 3600.0) % 24
                
                # Dosing
                if day_fraction <= transit_hours: 
                    C[0] = c_input_max
                else: 
                    C[0] = c_input_max * np.exp(-0.346 * (day_fraction - transit_hours))

                # --- NEW CYLINDRICAL CODE ---
                d2C = (C[2:] - 2*C[1:-1] + C[:-2])
                dC = (C[2:] - C[:-2]) / 2.0 
                r_local = self.r_grid[1:-1] 

                radial_penalty = (self.dx / r_local) * dC
                C[1:-1] = C[1:-1] + alpha[1:-1] * (d2C + radial_penalty) - (beta[1:-1] * C[1:-1])
                
                C[-1] = C[-2] 

        return C, self.r_grid, m_end, t_end

def get_verdict(coverage_pct):
    if coverage_pct > 80:
        return "VERY BULLISH (Interception Likely)"
    elif coverage_pct > 60:
        return "BULLISH (Effective Treatment)"
    elif coverage_pct > 30:
        return "NEUTRAL (Slowing Progression)"
    else:
        return "BEARISH (Ineffective for Deep Disease)"

def analyze_and_report(thickness, structural_floor=1.5, activation_pct=100.0, missing_valve=True, description=""):
    # 1. Calculate Anatomy (Conservation of Mass Geometry)
    outer_bowel_radius_mm = 12.0
    healthy_wall_mm = 2.0
    healthy_lumen_mm = outer_bowel_radius_mm - healthy_wall_mm # Baseline 10.0mm radius
    
    # Calculate the narrowed lumen based on the diseased wall thickness
    dynamic_lumen_mm = max(0.5, outer_bowel_radius_mm - thickness) 
    
    sim = ReactionDiffusionScaledModel(wall_thickness_mm=thickness, lumen_radius_mm=dynamic_lumen_mm)
    IC90 = 1.0

    # 2. Establish the Baseline Biological Transit
    if missing_valve:
        base_transit = 5.0  # Missing valve baseline
    else:
        base_transit = 8.0  # Intact valve baseline
        
    # 3. ACAT Area-Proportional Transit Calculation
    # Delay scales proportionally with the loss of cross-sectional area (r^2)
    area_ratio = (healthy_lumen_mm / dynamic_lumen_mm)**2
    calculated_transit = base_transit * area_ratio
    
    # 4. Apply the Clinical Obstruction Guardrail
    # Small bowel stasis beyond 16 hours typically results in acute clinical obstruction (vomiting, NPO).
    # We cap the model here, as a patient beyond this would not be taking daily oral medication.
    transit_hours = min(16.0, calculated_transit)
    
    print(f"\n==========================================")
    print(f"REPORT FOR {thickness}mm WALL STRICTURE")
    print(f"Lumen narrowed to: {dynamic_lumen_mm:.1f}mm (Area Ratio: {area_ratio:.2f}x)")
    
    if calculated_transit > 16.0:
        print(f"Calculated Transit: {calculated_transit:.1f} hours -> CAPPED at {transit_hours:.1f}h (Clinical Obstruction Limit)")
    else:
        print(f"Calculated Transit Time: {transit_hours:.1f} hours")
        
    print(f"Patient Profile: Floor={structural_floor}x | Activation={activation_pct}% | Missing Ileocecal Valve: {'Yes' if missing_valve else 'No'}")
    if description:
        print(f"Patient Description: {description}")
    print(f"==========================================")
    
    # THEORETICAL BEST CASE
    conc_cold, x, m_end, t_end = sim.run_simulation(
        initial_hyperemia=structural_floor, 
        structural_floor=structural_floor,
        activation_pct=activation_pct,
        transit_hours=transit_hours,
        dynamic_healing=False
    )
    
    # PREDICTED TRIAL RESULT
    conc_dyn, _, _, _ = sim.run_simulation(
        initial_hyperemia=3.0, 
        structural_floor=structural_floor,
        activation_pct=activation_pct,
        transit_hours=transit_hours,
        dynamic_healing=True
    )
    
    # Calculate Coverage
    wall_depth = x - sim.r_lumen 
    target_indices = np.where((wall_depth >= m_end) & (wall_depth <= t_end))[0]
    
    if len(target_indices) == 0:
        print("Error: No target indices found. Check grid boundaries.")
        return

    cov_cold = (np.sum(conc_cold[target_indices] > IC90) / len(target_indices)) * 100
    cov_dyn = (np.sum(conc_dyn[target_indices] > IC90) / len(target_indices)) * 100
    
    print(f"THEORETICAL BEST CASE (Structural Baseline)")
    print(f" - Target Coverage: {cov_cold:.1f}%")
    print(f" - Verdict:  {get_verdict(cov_cold)}")
    print(f"-" * 40)
    print(f"PREDICTED TRIAL RESULT (Day {sim.total_days} Dynamic)")
    print(f" - Target Coverage: {cov_dyn:.1f}%")
    print(f" - Verdict:  {get_verdict(cov_dyn)}")

if __name__ == "__main__":
    # Mild FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=5.0, structural_floor=1.5, activation_pct=100.0, missing_valve=True, description="(Base Case) Post-Ileocecal Resection with Dense Scarring")

    # Moderate FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=7.0, structural_floor=1.5, activation_pct=100.0, missing_valve=True, description="Post-Ileocecal Resection with Dense Scarring")

    # Severe FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=9.0, structural_floor=1.5, activation_pct=100.0, missing_valve=True, description="Post-Ileocecal Resection with Dense Scarring")

    # Mild FSCD, De Novo + SIBO -> Low Floor, High SIBO
    analyze_and_report(thickness=5.0, structural_floor=1.2, activation_pct=60.0, missing_valve=False, description="De Novo Stricture with Localized SIBO")

    # Moderate FSCD, De Novo + SIBO -> Low Floor, High SIBO
    analyze_and_report(thickness=7.0, structural_floor=1.2, activation_pct=60.0, missing_valve=False, description="De Novo Stricture with Localized SIBO")

    # Post-Op Anastomosis
    analyze_and_report(thickness=4.0, structural_floor=1.5, activation_pct=100.0, missing_valve=True, description="Post-Op Anastomosis")

'''
The "Coverage %" Cheat Sheet
The "Target Zone" is the Fibrotic Submucosa. This is the engine room where the stricture grows.
0% - 20% Coverage (FAIL): The drug barely scratches the surface. The deep fibrosis is completely untreated and will continue to thicken the wall.
20% - 60% Coverage (NEUTRAL / PARTIAL): The drug treats the "front half" of the stricture. This effectively slows down the disease (a clinical benefit), but it likely won't stop the stricture from eventually requiring surgery.
60% - 80% Coverage (BULLISH): The drug reaches the majority of the problem area. This is the "Sweet Spot" for an interception therapy.
80% - 100% Coverage (VERY BULLISH): Complete saturation. The drug effectively "turns off" the fibrosis signal across the entire wall.
'''

# Output from Running Base Model:
'''
==========================================
REPORT FOR 5.0mm WALL STRICTURE
Lumen narrowed to: 7.0mm (Area Ratio: 2.04x)
Calculated Transit Time: 10.2 hours
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: (Base Case) Post-Ileocecal Resection with Dense Scarring
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Target Coverage: 98.3%
 - Verdict:  VERY BULLISH (Interception Likely)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Target Coverage: 90.0%
 - Verdict:  VERY BULLISH (Interception Likely)

==========================================
REPORT FOR 7.0mm WALL STRICTURE
Lumen narrowed to: 5.0mm (Area Ratio: 4.00x)
Calculated Transit: 20.0 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: Post-Ileocecal Resection with Dense Scarring
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Target Coverage: 67.7%
 - Verdict:  BULLISH (Effective Treatment)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Target Coverage: 61.6%
 - Verdict:  BULLISH (Effective Treatment)

==========================================
REPORT FOR 9.0mm WALL STRICTURE
Lumen narrowed to: 3.0mm (Area Ratio: 11.11x)
Calculated Transit: 55.6 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: Post-Ileocecal Resection with Dense Scarring
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Target Coverage: 46.4%
 - Verdict:  NEUTRAL (Slowing Progression)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Target Coverage: 42.1%
 - Verdict:  NEUTRAL (Slowing Progression)

==========================================
REPORT FOR 5.0mm WALL STRICTURE
Lumen narrowed to: 7.0mm (Area Ratio: 2.04x)
Calculated Transit: 16.3 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.2x | Activation=60.0% | Missing Ileocecal Valve: No
Patient Description: De Novo Stricture with Localized SIBO
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Target Coverage: 100.0%
 - Verdict:  VERY BULLISH (Interception Likely)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Target Coverage: 93.3%
 - Verdict:  VERY BULLISH (Interception Likely)

==========================================
REPORT FOR 7.0mm WALL STRICTURE
Lumen narrowed to: 5.0mm (Area Ratio: 4.00x)
Calculated Transit: 32.0 hours -> CAPPED at 16.0h (Clinical Obstruction Limit)
Patient Profile: Floor=1.2x | Activation=60.0% | Missing Ileocecal Valve: No
Patient Description: De Novo Stricture with Localized SIBO
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Target Coverage: 64.6%
 - Verdict:  BULLISH (Effective Treatment)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Target Coverage: 56.6%
 - Verdict:  NEUTRAL (Slowing Progression)

==========================================
REPORT FOR 4.0mm WALL STRICTURE
Lumen narrowed to: 8.0mm (Area Ratio: 1.56x)
Calculated Transit Time: 7.8 hours
Patient Profile: Floor=1.5x | Activation=100.0% | Missing Ileocecal Valve: Yes
Patient Description: Post-Op Anastomosis
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Target Coverage: 100.0%
 - Verdict:  VERY BULLISH (Interception Likely)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Target Coverage: 100.0%
 - Verdict:  VERY BULLISH (Interception Likely)
'''