"""
Multi-Agent 3D N-Body Solar-Mass Gravitational Simulation
---------------------------------------------------------
Orchestrates three specialized subagent roles:
1. Subagent 1 (Math): Pairwise 3D Newtonian mechanics, state integration, and Virial equilibrium initialization.
2. Subagent 3 (User UI): Parameter initialization, user prompts, input validation, and argument parsing.
3. Subagent 3 (Visual): Real-time 3D particle vector plotting canvas with orbital trails and velocity quivers.

Author: Antigravity Orchestrator Agent
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Gravitational Constant in Astronomical Units:
# Distance: AU, Mass: M_sun (Solar mass), Time: Year
# G = 4 * pi^2 AU^3 / (M_sun * yr^2)
G_CONST = 4.0 * (np.pi ** 2)


# ==============================================================================
# SUBAGENT 1: MATH AGENT
# ==============================================================================
class MathAgent:
    """
    Subagent 1 (Math):
    Computes pairwise 3D Newtonian mechanics, initializes gravitationally
    bound states using Virial Equilibrium, updates positions & velocities
    via symplectic Velocity-Verlet state integration, and tracks energy conservation.
    """
    def __init__(self, num_stars: int, masses: np.ndarray = None, spatial_radius: float = 5.0, softening: float = 0.05):
        self.N = num_stars
        self.softening = softening  # AU, prevents zero-distance singularities
        self.spatial_radius = spatial_radius

        # Masses in solar mass units (M_sun)
        if masses is None:
            self.masses = np.ones(self.N, dtype=float)  # Default all 1.0 Solar Mass
        else:
            self.masses = np.array(masses, dtype=float)

        self.r = np.zeros((self.N, 3), dtype=float)  # Positions (AU)
        self.v = np.zeros((self.N, 3), dtype=float)  # Velocities (AU/yr)
        self.a = np.zeros((self.N, 3), dtype=float)  # Accelerations (AU/yr^2)

        self.time = 0.0
        self.E0 = 0.0  # Initial total mechanical energy

    def initialize_bound_system(self, seed: int = 42):
        """
        Generates 3D positions and velocities such that bodies are gravitationally bound.
        Uses Virial Equilibrium condition: 2 * Kinetic_Energy + Potential_Energy = 0.
        """
        np.random.seed(seed)
        
        # 1. Random 3D positions sampled from a spherical Gaussian distribution
        self.r = np.random.randn(self.N, 3) * (self.spatial_radius / np.sqrt(3))
        
        # Subtract Center of Mass position
        com_pos = np.average(self.r, axis=0, weights=self.masses)
        self.r -= com_pos

        # 2. Compute initial Gravitational Potential Energy U
        U = self._compute_potential_energy()

        # 3. Generate random velocity vectors with tangential bias around origin
        # Cross product of position vector with random unit vector gives tangential direction
        rand_vecs = np.random.randn(self.N, 3)
        tangential_v = np.cross(self.r, rand_vecs)
        norms = np.linalg.norm(tangential_v, axis=1, keepdims=True)
        # Avoid division by zero for particles near origin
        norms[norms == 0] = 1.0
        unit_tangential = tangential_v / norms

        # Unscaled speeds
        raw_speeds = np.linalg.norm(self.r, axis=1) * (2.0 * np.pi / self.spatial_radius)
        self.v = unit_tangential * raw_speeds[:, np.newaxis]

        # Zero out center-of-mass momentum
        com_vel = np.average(self.v, axis=0, weights=self.masses)
        self.v -= com_vel

        # 4. Enforce Virial Equilibrium: scale velocities so 2 * K = |U| => K = 0.5 * |U|
        K_raw = self._compute_kinetic_energy()
        if K_raw > 0:
            scale_factor = np.sqrt(0.5 * abs(U) / K_raw)
            self.v *= scale_factor

        # 5. Compute initial accelerations & energy baseline
        self.a = self.compute_accelerations(self.r)
        self.E0 = self.get_total_energy()
        print(f"  [MathAgent] System initialized: N={self.N} stars. Initial Virial Energy E0 = {self.E0:.4f} M_sun*(AU/yr)^2")

    def compute_accelerations(self, r_pos: np.ndarray) -> np.ndarray:
        """
        Computes pairwise 3D Newtonian acceleration vectors for all N bodies.
        a_i = G * sum_{j != i} m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
        """
        # Pairwise displacement vectors: dr[i, j, :] = r_pos[j] - r_pos[i]
        dr = r_pos[np.newaxis, :, :] - r_pos[:, np.newaxis, :]  # Shape (N, N, 3)
        
        # Pairwise distance squared + softening factor
        dist_sq = np.sum(dr ** 2, axis=-1) + (self.softening ** 2)  # Shape (N, N)
        inv_dist3 = dist_sq ** (-1.5)
        np.fill_diagonal(inv_dist3, 0.0)  # Avoid self-interaction

        # Multiply by masses of target bodies: shape (N, N)
        weighted_inv_dist3 = inv_dist3 * self.masses[np.newaxis, :]

        # Sum over all j bodies for each dimension
        # accel[i] = G * sum_j (dr[i, j] * weighted_inv_dist3[i, j])
        accel = G_CONST * np.einsum('ijk,ij->ik', dr, weighted_inv_dist3)
        return accel

    def step(self, dt: float):
        """
        Performs one time step using the 2nd-order Symplectic Velocity-Verlet integration:
        r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        a(t + dt) = compute_accelerations(r(t + dt))
        v(t + dt) = v(t) + 0.5*(a(t) + a(t + dt))*dt
        """
        # 1. Update positions
        self.r += self.v * dt + 0.5 * self.a * (dt ** 2)

        # 2. Compute new accelerations
        a_new = self.compute_accelerations(self.r)

        # 3. Update velocities
        self.v += 0.5 * (self.a + a_new) * dt
        self.a = a_new

        # 4. Increment clock
        self.time += dt

    def _compute_kinetic_energy(self) -> float:
        """Computes kinetic energy K = 0.5 * sum(m_i * v_i^2)."""
        v_sq = np.sum(self.v ** 2, axis=1)
        return 0.5 * float(np.sum(self.masses * v_sq))

    def _compute_potential_energy(self) -> float:
        """Computes pairwise gravitational potential energy U."""
        U = 0.0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dr = self.r[j] - self.r[i]
                dist = np.sqrt(np.sum(dr ** 2) + (self.softening ** 2))
                U -= (G_CONST * self.masses[i] * self.masses[j]) / dist
        return U

    def get_total_energy(self) -> float:
        """Returns total mechanical energy E = K + U."""
        return self._compute_kinetic_energy() + self._compute_potential_energy()

    def get_energy_drift(self) -> float:
        """Calculates relative energy drift |E - E0| / |E0|."""
        if abs(self.E0) == 0:
            return 0.0
        return abs(self.get_total_energy() - self.E0) / abs(self.E0)


# ==============================================================================
# SUBAGENT 2: USER UI AGENT
# ==============================================================================
class UserUIAgent:
    """
    Subagent 2 (User UI):
    Handles parameter initialization, user prompts for solar-mass star counts,
    input validation, CLI flag parsing, and simulation settings.
    """
    def __init__(self):
        self.num_stars = 5
        self.dt = 0.005  # Simulation time step in years (~1.8 days)
        self.softening = 0.05  # Softening in AU
        self.spatial_radius = 5.0  # AU
        self.steps_per_frame = 4
        self.random_masses = False
        self.non_interactive = False

    def initialize_parameters(self) -> dict:
        """
        Parses CLI parameters and prompts the user interactively if running in terminal.
        """
        parser = argparse.ArgumentParser(
            description="Multi-Agent 3D N-Body Solar-Mass Gravitational Simulation"
        )
        parser.add_argument("-n", "--num-stars", type=int, help="Number of solar-mass stars")
        parser.add_argument("--dt", type=float, default=0.005, help="Simulation timestep dt in years (default: 0.005)")
        parser.add_argument("--softening", type=float, default=0.05, help="Softening factor in AU (default: 0.05)")
        parser.add_argument("--radius", type=float, default=5.0, help="Initial spatial radius dispersion in AU (default: 5.0)")
        parser.add_argument("--random-masses", action="store_true", help="Assign randomized solar masses between 0.5 and 3.0 M_sun")
        parser.add_argument("--non-interactive", action="store_true", help="Run without stdin user prompts")
        
        args, unknown = parser.parse_known_args()

        self.dt = args.dt
        self.softening = args.softening
        self.spatial_radius = args.radius
        self.random_masses = args.random_masses
        self.non_interactive = args.non_interactive

        if args.num_stars is not None:
            self.num_stars = max(2, min(100, args.num_stars))
            print(f"  [UserUIAgent] Selected N={self.num_stars} stars via command-line arguments.")
        elif not self.non_interactive and sys.stdin.isatty():
            self._prompt_user()
        else:
            print(f"  [UserUIAgent] Non-interactive execution detected. Using default N={self.num_stars} solar-mass stars.")

        # Generate star masses
        if self.random_masses:
            np.random.seed(42)
            masses = np.random.uniform(0.5, 3.0, size=self.num_stars)
        else:
            masses = np.ones(self.num_stars, dtype=float)

        return {
            "num_stars": self.num_stars,
            "masses": masses,
            "dt": self.dt,
            "softening": self.softening,
            "spatial_radius": self.spatial_radius,
            "steps_per_frame": self.steps_per_frame
        }

    def _prompt_user(self):
        """Interactively prompts the user for the number of solar-mass stars."""
        print("\n========================================================")
        print("   MULTI-AGENT 3D N-BODY GRAVITATIONAL SIMULATION")
        print("========================================================")
        while True:
            try:
                user_input = input("Enter the number of solar-mass stars to simulate (2 - 50) [default 5]: ").strip()
                if not user_input:
                    self.num_stars = 5
                    break
                val = int(user_input)
                if 2 <= val <= 50:
                    self.num_stars = val
                    break
                else:
                    print("  --> Please enter an integer between 2 and 50.")
            except (ValueError, KeyboardInterrupt):
                print("\n  --> Invalid input. Defaulting to 5 solar-mass stars.")
                self.num_stars = 5
                break
        print(f"  [UserUIAgent] Initialized simulation for N={self.num_stars} solar-mass stars.\n")


# ==============================================================================
# SUBAGENT 3: VISUAL AGENT
# ==============================================================================
class VisualAgent:
    """
    Subagent 3 (Visual):
    Creates a real-time particle vector plotting canvas in 3D to render star
    orbits, velocity vectors, and fading orbital trajectories.
    """
    def __init__(self, math_agent: MathAgent, config: dict, trail_length: int = 150):
        self.math_agent = math_agent
        self.config = config
        self.trail_length = trail_length
        self.steps_per_frame = config.get("steps_per_frame", 4)
        self.dt = config.get("dt", 0.005)

        # Orbital trajectory history buffer: list of 3D arrays
        self.pos_history = []

        # Color map for stars based on mass or ID
        self.N = math_agent.N
        self.colors = plt.cm.plasma(np.linspace(0.2, 0.9, self.N))
        self.sizes = 40 + (math_agent.masses * 30)  # Visual particle size based on solar mass

    def setup_canvas(self):
        """Sets up the Matplotlib 3D dark-theme plotting canvas."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(10, 8), facecolor='#0b0e14')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='#0b0e14')

        self.ax.set_title(
            f"Multi-Agent 3D N-Body Orbit Visualization (N = {self.N} Solar-Mass Stars)",
            fontsize=12, color='#e0e6ed', pad=15, fontweight='bold'
        )

        bound = self.math_agent.spatial_radius * 1.8
        self.ax.set_xlim(-bound, bound)
        self.ax.set_ylim(-bound, bound)
        self.ax.set_zlim(-bound, bound)

        self.ax.set_xlabel("X (AU)", labelpad=10, color='#8a99a8')
        self.ax.set_ylabel("Y (AU)", labelpad=10, color='#8a99a8')
        self.ax.set_zlabel("Z (AU)", labelpad=10, color='#8a99a8')

        # Clean grid aesthetic
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('#1f2937')
        self.ax.yaxis.pane.set_edgecolor('#1f2937')
        self.ax.zaxis.pane.set_edgecolor('#1f2937')
        self.ax.grid(True, linestyle=':', color='#374151', alpha=0.5)

        # Interactive elements initialization
        self.scatter = self.ax.scatter(
            self.math_agent.r[:, 0],
            self.math_agent.r[:, 1],
            self.math_agent.r[:, 2],
            c=self.colors,
            s=self.sizes,
            depthshade=True,
            edgecolors='white',
            linewidths=0.5
        )

        # Quiver plots for velocity vectors
        self.quiver = self.ax.quiver(
            self.math_agent.r[:, 0], self.math_agent.r[:, 1], self.math_agent.r[:, 2],
            self.math_agent.v[:, 0], self.math_agent.v[:, 1], self.math_agent.v[:, 2],
            color='#00f2fe', length=0.3, normalize=True, alpha=0.7, linewidth=1.2
        )

        # Orbital trail line objects (one per star)
        self.trail_lines = [
            self.ax.plot([], [], [], color=self.colors[i], alpha=0.6, linewidth=1.0)[0]
            for i in range(self.N)
        ]

        # HUD Text Overlay
        self.hud_text = self.ax.text2D(
            0.03, 0.94, "", transform=self.ax.transAxes,
            color='#38ef7d', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#111827', edgecolor='#374151', alpha=0.85)
        )

    def animate_frame(self, frame_idx: int):
        """Callback function for matplotlib.animation.FuncAnimation."""
        # 1. Step the Math Engine multiple sub-steps per frame for smooth mechanics
        for _ in range(self.steps_per_frame):
            self.math_agent.step(self.dt)

        current_r = self.math_agent.r.copy()
        current_v = self.math_agent.v.copy()

        # 2. Record position history for orbital trails
        self.pos_history.append(current_r)
        if len(self.pos_history) > self.trail_length:
            self.pos_history.pop(0)

        # 3. Update scatter positions
        self.scatter._offsets3d = (current_r[:, 0], current_r[:, 1], current_r[:, 2])

        # 4. Update velocity quivers
        self.quiver.remove()
        self.quiver = self.ax.quiver(
            current_r[:, 0], current_r[:, 1], current_r[:, 2],
            current_v[:, 0], current_v[:, 1], current_v[:, 2],
            color='#00f2fe', length=0.4, normalize=True, alpha=0.75, linewidth=1.2
        )

        # 5. Update orbital trail curves
        hist_arr = np.array(self.pos_history)  # Shape (history_len, N, 3)
        for i in range(self.N):
            self.trail_lines[i].set_data(hist_arr[:, i, 0], hist_arr[:, i, 1])
            self.trail_lines[i].set_3d_properties(hist_arr[:, i, 2])

        # 6. Update HUD Stats
        time_yrs = self.math_agent.time
        energy_drift = self.math_agent.get_energy_drift()
        total_E = self.math_agent.get_total_energy()

        hud_info = (
            f"Sim Time : {time_yrs:6.2f} yrs | Frame: {frame_idx:04d}\n"
            f"Total E  : {total_E:8.3f} M_sun*(AU/yr)^2\n"
            f"E-Drift  : {energy_drift*100:6.3f} % | dt = {self.dt} yr"
        )
        self.hud_text.set_text(hud_info)

        return [self.scatter, self.quiver, self.hud_text] + self.trail_lines

    def start_visualization(self, save_path: str = None, total_frames: int = None):
        """Starts real-time animation loop or saves render if requested."""
        self.anim = FuncAnimation(
            self.fig, self.animate_frame, frames=total_frames,
            interval=20, blit=False, cache_frame_data=False
        )

        if save_path:
            print(f"  [VisualAgent] Saving animation render to {save_path}...")
            self.anim.save(save_path, fps=30, dpi=100)
            print("  [VisualAgent] Render saved successfully.")
        else:
            print("  [VisualAgent] Rendering interactive real-time 3D particle canvas...")
            plt.tight_layout()
            plt.show()


# ==============================================================================
# MULTI-AGENT ORCHESTRATOR
# ==============================================================================
class NBodyOrchestrator:
    """
    Multi-Agent System Orchestrator:
    Coordinates communication and workflow across all 3 subagents:
    - UserUIAgent (Parameter discovery)
    - MathAgent (Physics & state integration)
    - VisualAgent (Real-time plotting canvas)
    """
    def __init__(self):
        print("\n========================================================")
        print("    INITIALIZING MULTI-AGENT ORCHESTRATOR SYSTEM        ")
        print("========================================================")
        self.ui_agent = UserUIAgent()

    def run(self, save_path: str = None, max_frames: int = None):
        # Phase 1: Subagent 2 (User UI) gets configuration parameters
        config = self.ui_agent.initialize_parameters()

        # Phase 2: Subagent 1 (Math Engine) initializes 3D physics & state
        print("--> Delegating physics initialization to Subagent 1 (Math)...")
        math_agent = MathAgent(
            num_stars=config["num_stars"],
            masses=config["masses"],
            spatial_radius=config["spatial_radius"],
            softening=config["softening"]
        )
        math_agent.initialize_bound_system(seed=42)

        # Phase 3: Subagent 3 (Visual Canvas) sets up 3D vector plot & starts real-time rendering
        print("--> Delegating live canvas setup to Subagent 3 (Visual)...")
        visual_agent = VisualAgent(math_agent=math_agent, config=config)
        visual_agent.setup_canvas()
        visual_agent.start_visualization(save_path=save_path, total_frames=max_frames)


if __name__ == "__main__":
    orchestrator = NBodyOrchestrator()
    orchestrator.run()
