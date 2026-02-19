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
#
# Credit: This model was inspired by the detailed analysis from a blog post authored by Chaotropy, which provided critical insights into the expected drug behavior in FSCD and helped shape the assumptions and parameters used in this simulation.
# Source: https://chaotropy.com/pali-2108-fscd-prediction-model/
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
        # Acute inflammation (swelling) is the part above the structural floor.
        # This decays over time, but can't go lower than the patient's structural vascularity.
        acute_component = max(0, initial_multiplier - floor)
        decayed_acute = acute_component * np.exp(-0.2 * day) # Rapid drop in first week
        
        return floor + decayed_acute

    def run_simulation(self, initial_hyperemia=1.0, structural_floor=1.5, activation_pct=100.0, dynamic_healing=False):
        """
        Runs the simulation with specific patient parameters.
        initial_hyperemia: Starting inflammation (e.g., 3.0 for flare).
        structural_floor: The best-case vascularity for this patient (1.5 for scar, 1.2 for de novo).
        activation_pct: % of drug activated (100 = Colon-like, 60 = SIBO, 20 = Sterile Ileum).
        dynamic_healing: If True, inflammation drops from initial to floor over time.
        """
        C = np.zeros(self.nx)
        steps_per_day = int((24 * 3600) / self.dt_seconds)
        
        m_end, t_end = self.get_layer_boundaries()
        
        # Calculate C_input based on activation percentage
        c_input_max = 100.0 * (activation_pct / 100.0)

        # Loop Day by Day
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
                if day_fraction <= 8.0: 
                    C[0] = c_input_max
                else: 
                    C[0] = c_input_max * np.exp(-0.346 * (day_fraction - 8.0))

                # --- OLD CARTESIAN CODE ---
                # d2C = (C[2:] - 2*C[1:-1] + C[:-2])
                # C[1:-1] = C[1:-1] + (alpha[1:-1] * d2C) - (beta[1:-1] * C[1:-1])

                # --- NEW CYLINDRICAL CODE ---
                # Calculate 2nd derivative (diffusion across gradient)
                d2C = (C[2:] - 2*C[1:-1] + C[:-2])

                # Calculate 1st derivative (the geometric expansion penalty)
                dC = (C[2:] - C[:-2]) / 2.0 

                # The local radius for each point in the inner grid
                r_local = self.r_grid[1:-1] 

                # Apply the radial update
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

def analyze_and_report(thickness, structural_floor=1.5, activation_pct=100.0, description=""):
    """
    Analyzes a specific patient profile defined by structural_floor and activation_pct.
    """
    sim = ReactionDiffusionScaledModel(wall_thickness_mm=thickness)
    IC90 = 1.0
    
    print(f"\n==========================================")
    print(f"REPORT FOR {thickness}mm WALL STRICTURE")
    print(f"Patient Profile: Floor={structural_floor}x | Activation={activation_pct}%")
    if description:
        print(f"Patient Description: {description}")
    print(f"==========================================")
    
    # THEORETICAL BEST CASE: Structural Baseline
    # Assumes no acute inflammation, just the structural reality (floor).
    conc_cold, x, m_end, t_end = sim.run_simulation(
        initial_hyperemia=structural_floor, 
        structural_floor=structural_floor,
        activation_pct=activation_pct,
        dynamic_healing=False
    )
    
    # PREDICTED TRIAL RESULT: Active Inflammation (Dynamic)
    # Starts at 3.0x (Flare) and heals down to the structural_floor.
    conc_dyn, _, _, _ = sim.run_simulation(
        initial_hyperemia=3.0, 
        structural_floor=structural_floor,
        activation_pct=activation_pct,
        dynamic_healing=True
    )
    
    # Calculate Coverage
    # Subtract the lumen radius from the grid to get the relative wall depth
    wall_depth = x - sim.r_lumen 
    target_indices = np.where((wall_depth >= m_end) & (wall_depth <= t_end))[0]
    
    # Add a safety check to prevent future divide-by-zero errors
    if len(target_indices) == 0:
        print("Error: No target indices found. Check grid boundaries.")
        return

    cov_cold = (np.sum(conc_cold[target_indices] > IC90) / len(target_indices)) * 100
    cov_dyn = (np.sum(conc_dyn[target_indices] > IC90) / len(target_indices)) * 100
    
    print(f"THEORETICAL BEST CASE (Structural Baseline)")
    print(f" - Max potential if acute inflammation is fully resolved.")
    print(f" - Target Coverage: {cov_cold:.1f}%")
    print(f" - Verdict:  {get_verdict(cov_cold)}")
    
    print(f"-" * 40)
    
    print(f"PREDICTED TRIAL RESULT (Day {sim.total_days} Dynamic)")
    print(f" - Starts with flare (3.0x), heals to baseline ({structural_floor}x).")
    print(f" - Target Coverage: {cov_dyn:.1f}%")
    print(f" - Verdict:  {get_verdict(cov_dyn)}")

if __name__ == "__main__":
    # Mild FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=5.0, structural_floor=1.5, activation_pct=100.0, description="(Base Case) Post-Ileocecal Resection with Dense Scarring")

    # Moderate FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=7.0, structural_floor=1.5, activation_pct=100.0, description="Post-Ileocecal Resection with Dense Scarring")

    # Severe FSCD, Anastomotic Scar -> High Floor, prior ileocecal resection
    analyze_and_report(thickness=9.0, structural_floor=1.5, activation_pct=100.0, description="Post-Ileocecal Resection with Dense Scarring")

    # Mild FSCD, De Novo + SIBO -> Low Floor, High SIBO
    analyze_and_report(thickness=5.0, structural_floor=1.2, activation_pct=60.0, description="De Novo Stricture with Localized SIBO")

    # Moderate FSCD, De Novo + SIBO -> Low Floor, High SIBO
    analyze_and_report(thickness=7.0, structural_floor=1.2, activation_pct=60.0, description="De Novo Stricture with Localized SIBO")

    # Post-Op Anastomosis
    analyze_and_report(thickness=4.0, structural_floor=1.5, activation_pct=100.0, description="Post-Op Anastomosis")

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
Patient Profile: Floor=1.5x | Activation=100.0%
Patient Description: (Base Case) Post-Ileocecal Resection with Dense Scarring
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Max potential if acute inflammation is fully resolved.
 - Target Coverage: 90.0%
 - Verdict:  VERY BULLISH (Interception Likely)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Starts with flare (3.0x), heals to baseline (1.5x).
 - Target Coverage: 81.7%
 - Verdict:  VERY BULLISH (Interception Likely)

==========================================
REPORT FOR 7.0mm WALL STRICTURE
Patient Profile: Floor=1.5x | Activation=100.0%
Patient Description: Post-Ileocecal Resection with Dense Scarring
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Max potential if acute inflammation is fully resolved.
 - Target Coverage: 54.5%
 - Verdict:  NEUTRAL (Slowing Progression)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Starts with flare (3.0x), heals to baseline (1.5x).
 - Target Coverage: 49.5%
 - Verdict:  NEUTRAL (Slowing Progression)

==========================================
REPORT FOR 9.0mm WALL STRICTURE
Patient Profile: Floor=1.5x | Activation=100.0%
Patient Description: Post-Ileocecal Resection with Dense Scarring
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Max potential if acute inflammation is fully resolved.
 - Target Coverage: 38.6%
 - Verdict:  NEUTRAL (Slowing Progression)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Starts with flare (3.0x), heals to baseline (1.5x).
 - Target Coverage: 35.0%
 - Verdict:  NEUTRAL (Slowing Progression)

==========================================
REPORT FOR 5.0mm WALL STRICTURE
Patient Profile: Floor=1.2x | Activation=60.0%
Patient Description: De Novo Stricture with Localized SIBO
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Max potential if acute inflammation is fully resolved.
 - Target Coverage: 81.7%
 - Verdict:  VERY BULLISH (Interception Likely)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Starts with flare (3.0x), heals to baseline (1.2x).
 - Target Coverage: 71.7%
 - Verdict:  BULLISH (Effective Treatment)

==========================================
REPORT FOR 7.0mm WALL STRICTURE
Patient Profile: Floor=1.2x | Activation=60.0%
Patient Description: De Novo Stricture with Localized SIBO
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Max potential if acute inflammation is fully resolved.
 - Target Coverage: 50.5%
 - Verdict:  NEUTRAL (Slowing Progression)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Starts with flare (3.0x), heals to baseline (1.2x).
 - Target Coverage: 43.4%
 - Verdict:  NEUTRAL (Slowing Progression)

==========================================
REPORT FOR 4.0mm WALL STRICTURE
Patient Profile: Floor=1.5x | Activation=100.0%
Patient Description: Post-Op Anastomosis
==========================================
THEORETICAL BEST CASE (Structural Baseline)
 - Max potential if acute inflammation is fully resolved.
 - Target Coverage: 100.0%
 - Verdict:  VERY BULLISH (Interception Likely)
----------------------------------------
PREDICTED TRIAL RESULT (Day 14 Dynamic)
 - Starts with flare (3.0x), heals to baseline (1.5x).
 - Target Coverage: 100.0%
 - Verdict:  VERY BULLISH (Interception Likely)
'''