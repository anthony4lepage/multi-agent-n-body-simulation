\# Multi-Agent 3D N-Body Simulation



A Python-based 3D N-body gravitational simulation developed during a PEARC26 workshop with Google employees using Antigravity IDE.



\## Overview



This project simulates gravitational interactions between multiple solar-mass stars in three-dimensional space. The simulation uses Newtonian mechanics and renders a real-time visualization of particle motion, orbital trails, and velocity vectors.



The application is organized using three specialized agent roles:



\- \*\*Math Agent\*\* — Computes pairwise gravitational interactions, initializes a gravitationally bound system using Virial Equilibrium, and updates particle positions and velocities using Velocity-Verlet integration.

\- \*\*User UI Agent\*\* — Handles command-line arguments, user input, and simulation parameter validation.

\- \*\*Visual Agent\*\* — Creates a real-time 3D visualization with orbital trails, velocity vectors, and energy statistics.



\## Features



\- 3D Newtonian gravitational simulation

\- Multiple solar-mass stars

\- Virial Equilibrium initialization

\- Velocity-Verlet integration

\- Real-time 3D visualization

\- Orbital trajectory trails

\- Velocity vector visualization

\- Energy drift tracking

\- Customizable number of stars

\- Optional randomized star masses

\- Command-line parameter support



\## Technologies



\- Python

\- NumPy

\- Matplotlib

\- Antigravity IDE



\## Installation



Clone the repository:



```bash

git clone https://github.com/YOUR-USERNAME/multi-agent-n-body-simulation.git

cd multi-agent-n-body-simulation

