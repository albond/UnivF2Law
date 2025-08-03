#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Binary Neutron Star Merger Visualization Script - Enhanced Version
Creates an animated GIF showing inspiral, merger, and post-merger phases
Based on the Universal f2 Law model from UnivF2Law paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import imageio
import os
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Configurable parameters for the simulation
MASS_RATIO = 0.8  # q = m2/m1 (m2 <= m1)
M_TOT = 2.7  # Total mass in solar masses
TIDAL_DEFORMABILITY = 400  # Combined tidal deformability ~Λ
F2_FREQUENCY = 2.8  # Post-merger frequency in kHz

# Animation parameters
FPS = 25  # Frames per second
DURATION = 10  # Total duration in seconds
N_FRAMES = FPS * DURATION
DPI = 100
FIG_SIZE = (10, 10)

# Physical parameters (scaled units)
R_INITIAL = 25  # Initial orbital separation
R_MERGER = 3  # Merger radius
R_NS = 1.5  # Neutron star radius
R_REMNANT = 2.5  # Remnant radius
T_MERGER = 0.55 * DURATION  # Time of merger (fraction of total duration)
T_POSTMERGER = 0.45 * DURATION  # Post-merger duration

# Enhanced visual parameters - more realistic palette
BACKGROUND_COLOR = '#000814'
STAR1_COLOR = '#6495ED'  # Cornflower blue (realistic NS color)
STAR2_COLOR = '#87CEEB'  # Sky blue (cooler NS)
REMNANT_COLOR = '#FFA500'  # Orange (hot remnant)
GLOW_COLOR = '#E0FFFF'  # Light cyan glow
TEXT_COLOR = '#E0E1DD'
GRID_COLOR = '#0A0E27'
JET_COLOR = '#ADD8E6'  # Light blue jets (synchrotron radiation)
MATTER_COLOR = '#FF6347'  # Tomato (hot ejecta)

# Create custom colormaps
hot_cmap = LinearSegmentedColormap.from_list('hot_custom', 
    ['#000814', '#001D3D', '#003566', '#00B4D8', '#90E0EF', '#CAF0F8', '#FFFFFF'])
jet_cmap = LinearSegmentedColormap.from_list('jet_custom',
    ['#8338EC', '#C77DFF', '#E0AAFF', '#FFFFFF'])


class BNSMergerSimulation:
    """Enhanced simulation of binary neutron star merger dynamics"""
    
    def __init__(self):
        """Initialize simulation parameters"""
        self.q = MASS_RATIO
        self.m1 = M_TOT / (1 + self.q)
        self.m2 = self.q * self.m1
        self.tidal_def = TIDAL_DEFORMABILITY
        self.f2 = F2_FREQUENCY
        
        # Derived parameters
        self.r1_scale = R_NS * np.power(self.m1 / 1.4, 1/3)
        self.r2_scale = R_NS * np.power(self.m2 / 1.4, 1/3)
        
        # Time arrays
        self.t_inspiral = np.linspace(0, T_MERGER, int(T_MERGER * FPS))
        self.t_postmerger = np.linspace(T_MERGER, DURATION, int(T_POSTMERGER * FPS))
        
        # Orbit history for trails
        self.orbit_history = {'star1': [], 'star2': []}
        self.max_history = 60  # Number of points to keep in trail
        
    def orbital_radius(self, t: float) -> float:
        """Calculate orbital separation as function of time"""
        if t >= T_MERGER:
            return 0
        
        # Accelerating inspiral
        t_norm = t / T_MERGER
        # More realistic inspiral with acceleration
        return R_INITIAL * (1 - t_norm)**2.5 + R_MERGER
    
    def orbital_frequency(self, r: float) -> float:
        """Calculate orbital frequency from Kepler's law"""
        if r <= R_MERGER:
            return 0
        return np.sqrt(M_TOT / r**3) * (1 + 0.1 * (R_INITIAL - r) / R_INITIAL)
    
    def tidal_deformation(self, r: float) -> float:
        """Calculate enhanced tidal deformation factor"""
        if r <= R_MERGER:
            return 0
        
        # Enhanced deformation near merger
        tidal_field = self.tidal_def * M_TOT / r**3
        base_deform = min(0.4, tidal_field / 800)
        
        # Add oscillations for more dynamic look
        if r < R_INITIAL * 0.3:
            oscillation = 0.05 * np.sin(10 * r / R_INITIAL)
            base_deform += oscillation
            
        return base_deform
    
    def get_star_positions(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate positions of both stars during inspiral"""
        r = self.orbital_radius(t)
        
        if r <= R_MERGER:
            return np.array([0, 0]), np.array([0, 0])
        
        # Calculate orbital phase with acceleration
        omega = self.orbital_frequency(r)
        phase = omega * t * 25
        
        # Add precession for visual interest
        precession = 0.1 * np.sin(t * 2)
        phase += precession
        
        # Center of mass frame positions
        r1 = r * self.m2 / M_TOT
        r2 = r * self.m1 / M_TOT
        
        pos1 = np.array([-r1 * np.cos(phase), -r1 * np.sin(phase)])
        pos2 = np.array([r2 * np.cos(phase), r2 * np.sin(phase)])
        
        # Store positions for trails
        if len(self.orbit_history['star1']) > self.max_history:
            self.orbit_history['star1'].pop(0)
            self.orbit_history['star2'].pop(0)
        self.orbit_history['star1'].append(pos1)
        self.orbit_history['star2'].append(pos2)
        
        return pos1, pos2
    
    def get_star_shapes(self, t: float) -> Tuple[dict, dict]:
        """Get enhanced ellipse parameters for deformed stars"""
        r = self.orbital_radius(t)
        deform = self.tidal_deformation(r)
        
        if r <= R_MERGER:
            return None, None
        
        pos1, pos2 = self.get_star_positions(t)
        
        # Calculate deformation direction
        if np.linalg.norm(pos2 - pos1) > 0:
            direction = (pos2 - pos1) / np.linalg.norm(pos2 - pos1)
            angle = np.arctan2(direction[1], direction[0]) * 180 / np.pi
        else:
            angle = 0
        
        # Temperature-based color variation
        temp_factor = 1 - r / R_INITIAL  # Gets hotter as they approach
        
        star1_params = {
            'xy': pos1,
            'width': 2 * self.r1_scale * (1 + deform * 1.2),
            'height': 2 * self.r1_scale * (1 - deform * 0.6),
            'angle': angle,
            'color': self.interpolate_color(STAR1_COLOR, '#FFFFFF', temp_factor * 0.3),
            'alpha': 0.95
        }
        
        star2_params = {
            'xy': pos2,
            'width': 2 * self.r2_scale * (1 + deform * 1.2),
            'height': 2 * self.r2_scale * (1 - deform * 0.6),
            'angle': angle,
            'color': self.interpolate_color(STAR2_COLOR, '#FFFFFF', temp_factor * 0.3),
            'alpha': 0.95
        }
        
        return star1_params, star2_params
    
    def get_remnant_shape(self, t: float) -> dict:
        """Get enhanced remnant parameters during post-merger phase"""
        if t < T_MERGER:
            return None
        
        # Time since merger
        t_post = t - T_MERGER
        
        # Complex oscillation pattern
        osc1 = 0.25 * np.exp(-t_post / (T_POSTMERGER * 0.3))
        osc2 = 0.1 * np.exp(-t_post / (T_POSTMERGER * 0.5))
        deform = osc1 * np.sin(2 * np.pi * self.f2 * t_post) + \
                 osc2 * np.sin(4 * np.pi * self.f2 * t_post)
        
        # Rotating and pulsating pattern
        rotation = 720 * t_post / T_POSTMERGER
        
        # Color evolution
        color_factor = min(1.0, t_post / (T_POSTMERGER * 0.3))
        remnant_color = self.interpolate_color(REMNANT_COLOR, '#FFFFFF', color_factor * 0.5)
        
        remnant_params = {
            'xy': (0, 0),
            'width': 2 * R_REMNANT * (1 + deform),
            'height': 2 * R_REMNANT * (1 - deform * 0.8),
            'angle': rotation % 360,
            'color': remnant_color,
            'alpha': 0.98
        }
        
        return remnant_params
    
    def interpolate_color(self, color1: str, color2: str, factor: float) -> str:
        """Interpolate between two colors"""
        factor = np.clip(factor, 0, 1)  # Ensure factor is between 0 and 1
        c1 = np.array([int(color1[i:i+2], 16) for i in (1, 3, 5)])
        c2 = np.array([int(color2[i:i+2], 16) for i in (1, 3, 5)])
        c = c1 + (c2 - c1) * factor
        # Ensure RGB values are within valid range
        c = np.clip(c, 0, 255)
        return '#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2]))


class MergerVisualizer:
    """Enhanced visualization of BNS merger with stunning effects"""
    
    def __init__(self, simulation: BNSMergerSimulation):
        """Initialize visualizer with simulation"""
        self.sim = simulation
        self.fig, self.ax = self.setup_figure()
        self.patches = []
        self.texts = []
        self.jets = []
        self.matter_particles = []
        
    def setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        """Setup enhanced matplotlib figure and axes"""
        fig = plt.figure(figsize=FIG_SIZE, facecolor=BACKGROUND_COLOR)
        ax = fig.add_subplot(111, facecolor=BACKGROUND_COLOR)
        
        # Set axis properties
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal')
        
        # Enhanced grid with radial lines
        ax.grid(True, alpha=0.05, color=GRID_COLOR, linestyle='-', linewidth=0.5)
        
        # Add circular grid for depth
        for r in [10, 20, 30]:
            circle = Circle((0, 0), r, fill=False, edgecolor=GRID_COLOR, 
                          alpha=0.03, linestyle='--', linewidth=0.5)
            ax.add_patch(circle)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Enhanced title with glow effect
        title = ax.text(0.5, 0.95, 'BINARY NEUTRON STAR MERGER', 
                       transform=ax.transAxes,
                       color=TEXT_COLOR, fontsize=24, 
                       ha='center', weight='bold',
                       fontfamily='monospace')
        
        # Subtitle
        subtitle = ax.text(0.5, 0.92, 'Universal f₂ Law Visualization', 
                          transform=ax.transAxes,
                          color=TEXT_COLOR, fontsize=14, 
                          ha='center', alpha=0.7,
                          fontfamily='monospace')
        
        return fig, ax
    
    def add_background_stars(self, n_stars: int = 500):
        """Add enhanced background with nebula effect"""
        np.random.seed(42)
        
        # Background stars with varying brightness
        x = np.random.uniform(-30, 30, n_stars)
        y = np.random.uniform(-30, 30, n_stars)
        sizes = np.random.exponential(1, n_stars)
        brightness = np.random.uniform(0.1, 0.6, n_stars)
        
        for i in range(n_stars):
            self.ax.scatter(x[i], y[i], s=sizes[i], c='white', 
                          alpha=brightness[i], marker='*')
        
        # Add nebula clouds
        for _ in range(5):
            x_neb = np.random.uniform(-30, 30)
            y_neb = np.random.uniform(-30, 30)
            radius = np.random.uniform(5, 15)
            nebula = Circle((x_neb, y_neb), radius, 
                          color='#4361EE', alpha=0.02)
            self.ax.add_patch(nebula)
    
    def draw_gravitational_waves(self, t: float):
        """Draw enhanced gravitational wave ripples with interference"""
        if t >= T_MERGER - 1 and t <= T_MERGER + 2:
            # Multiple wave sources for interference
            amplitude = np.exp(-1.5 * abs(t - T_MERGER))
            
            for i in range(5):
                radius = 3 + i * 4 + (t - T_MERGER + 1) * 15
                
                # Varying opacity for depth
                alpha = amplitude * (1 - i * 0.15) * 0.7
                
                # Add distortion to circles near merger
                if abs(t - T_MERGER) < 0.2:
                    # Create distorted wave
                    theta = np.linspace(0, 2*np.pi, 100)
                    distortion = 1 + 0.1 * np.sin(8 * theta) * amplitude
                    x = radius * distortion * np.cos(theta)
                    y = radius * distortion * np.sin(theta)
                    self.ax.plot(x, y, color=GLOW_COLOR, alpha=alpha, 
                               linewidth=2.5 - i * 0.3)
                else:
                    circle = Circle((0, 0), radius, 
                                  fill=False, 
                                  edgecolor=GLOW_COLOR,
                                  alpha=alpha,
                                  linewidth=2.5 - i * 0.3)
                    self.ax.add_patch(circle)
    
    def draw_orbital_trails(self):
        """Draw subtle orbital trails behind the stars"""
        for star_key, color in [('star1', STAR1_COLOR), ('star2', STAR2_COLOR)]:
            history = self.sim.orbit_history[star_key]
            if len(history) > 1:
                for i in range(len(history) - 1):
                    alpha = (i / len(history)) * 0.2  # More subtle
                    width = (i / len(history)) * 1.0
                    
                    x = [history[i][0], history[i+1][0]]
                    y = [history[i][1], history[i+1][1]]
                    
                    self.ax.plot(x, y, color='#FFFFFF', alpha=alpha, 
                               linewidth=width, solid_capstyle='round')
    
    def draw_jets_and_matter(self, t: float):
        """Draw realistic helical relativistic jets and ejected matter"""
        if T_MERGER <= t <= T_MERGER + 3:
            t_burst = t - T_MERGER
            
            # Helical relativistic jets with proper structure
            jet_length = 25 * t_burst
            n_segments = 30
            
            for jet_dir in [1, -1]:  # North and south jets
                for i in range(n_segments):
                    z = i * jet_length / n_segments
                    # Helical structure
                    theta = 3 * np.pi * i / n_segments + t_burst * 5
                    width = (1 - i/n_segments) * 2 * np.exp(-t_burst * 0.3)
                    
                    # Jet particle position with helical motion
                    x = width * np.cos(theta)
                    y_pos = jet_dir * z
                    
                    # Fade out with distance and time
                    alpha = 0.3 * (1 - i/n_segments) * np.exp(-t_burst * 0.4)
                    
                    # Draw jet segment as elongated ellipse
                    jet_segment = Ellipse(xy=(x, y_pos),
                                         width=width,
                                         height=jet_length/n_segments * 2,
                                         angle=90,
                                         color=JET_COLOR,
                                         alpha=alpha)
                    self.ax.add_patch(jet_segment)
                    
                    # Add bright core
                    if i < n_segments/2:
                        core = Circle((x * 0.5, y_pos), 
                                    radius=width * 0.3,
                                    color='#FFFFFF',
                                    alpha=alpha * 0.5)
                        self.ax.add_patch(core)
            
            # Torus of ejected matter (kilonova)
            if t_burst < 1.5:
                n_particles = 30
                for i in range(n_particles):
                    angle = 2 * np.pi * i / n_particles
                    # Expanding torus
                    r = 4 + t_burst * 6
                    # Add vertical dispersion
                    z = np.random.normal(0, 1 + t_burst)
                    x = r * np.cos(angle)
                    y = r * np.sin(angle) + z
                    
                    # Particle properties
                    size = 0.3 * np.exp(-t_burst * 0.5)
                    alpha = 0.4 * np.exp(-t_burst * 0.7)
                    
                    particle = Circle((x, y), size, 
                                    color=MATTER_COLOR, 
                                    alpha=alpha)
                    self.ax.add_patch(particle)
    
    def draw_gravitational_lensing(self, t: float):
        """Simulate gravitational lensing effect on background"""
        if t >= T_MERGER - 0.5 and t <= T_MERGER + 1:
            strength = np.exp(-2 * abs(t - T_MERGER))
            
            # Create distortion field
            x = np.linspace(-10, 10, 20)
            y = np.linspace(-10, 10, 20)
            X, Y = np.meshgrid(x, y)
            R = np.sqrt(X**2 + Y**2)
            
            # Lensing displacement
            displacement = strength * 5 / (R + 1)
            
            # Draw distorted grid
            for i in range(len(x)):
                self.ax.plot(X[i, :] + displacement[i, :] * X[i, :]/R[i, :], 
                           Y[i, :] + displacement[i, :] * Y[i, :]/R[i, :],
                           color=GRID_COLOR, alpha=0.1 * strength, linewidth=0.5)
    
    def create_info_text(self, t: float) -> str:
        """Create enhanced information display"""
        phase = "INSPIRAL" if t < T_MERGER else "POST-MERGER"
        
        # Calculate dynamic values
        r = self.sim.orbital_radius(t)
        if r > 0:
            orbital_freq = self.sim.orbital_frequency(r) * 1000  # Convert to Hz
            gw_strain = 1e-21 * (R_INITIAL / r)**2  # Simplified strain
        else:
            orbital_freq = 0
            gw_strain = 1e-20 * np.exp(-(t - T_MERGER) / 2)
        
        info = f"""
  PHASE: {phase:^20}
  TIME: {t:>6.2f} s

  PARAMETERS
  Mass Ratio (q): {self.sim.q:>6.2f}
  Total Mass: {M_TOT:>6.2f} M☉
  Tidal Def (~Λ): {self.sim.tidal_def:>6.0f}
  f₂ Frequency: {self.sim.f2:>6.2f} kHz

  DYNAMICS
  Orbital Sep: {r:>6.1f} km
  Orbital Freq: {orbital_freq:>6.1f} Hz
  GW Strain: {gw_strain:.2e}
        """
        
        return info
    
    def animate_frame(self, frame: int):
        """Animate single frame with all effects"""
        # Clear previous patches
        for patch in self.patches:
            patch.remove()
        self.patches = []
        
        for text in self.texts:
            text.remove()
        self.texts = []
        
        # Calculate time
        t = frame / FPS
        
        # Draw background effects
        self.draw_gravitational_lensing(t)
        self.draw_orbital_trails()
        self.draw_gravitational_waves(t)
        
        # Draw stars or remnant
        if t < T_MERGER:
            # Inspiral phase
            star1_params, star2_params = self.sim.get_star_shapes(t)
            
            if star1_params and star2_params:
                # Subtle glow effect
                for params in [star1_params, star2_params]:
                    # Single subtle glow layer
                    glow = Ellipse(xy=params['xy'], 
                                 width=params['width'] * 1.4,
                                 height=params['height'] * 1.4,
                                 angle=params['angle'],
                                 color=GLOW_COLOR,
                                 alpha=0.15)
                    self.ax.add_patch(glow)
                    self.patches.append(glow)
                
                # Main stars
                star1 = Ellipse(**star1_params)
                star2 = Ellipse(**star2_params)
                
                self.ax.add_patch(star1)
                self.ax.add_patch(star2)
                self.patches.append(star1)
                self.patches.append(star2)
                
        else:
            # Post-merger phase
            remnant_params = self.sim.get_remnant_shape(t)
            
            if remnant_params:
                # Pulsating glow - single layer, more realistic
                glow_scale = 1.5 + 0.2 * np.sin(2 * np.pi * self.sim.f2 * t)
                glow = Ellipse(xy=remnant_params['xy'],
                             width=remnant_params['width'] * glow_scale,
                             height=remnant_params['height'] * glow_scale,
                             angle=remnant_params['angle'],
                             color=GLOW_COLOR,
                             alpha=0.25)
                self.ax.add_patch(glow)
                self.patches.append(glow)
                
                # Main remnant
                remnant = Ellipse(**remnant_params)
                self.ax.add_patch(remnant)
                self.patches.append(remnant)
                
                # Add realistic accretion disk
                t_post = t - T_MERGER
                if t_post > 0.2:  # Disk forms slightly after merger
                    # Single filled disk with proper opacity
                    disk_width = R_REMNANT * 3.5
                    disk_height = R_REMNANT * 0.7
                    disk_alpha = 0.15 * min(1, t_post / 0.5)
                    
                    # Rotating disk
                    rotation = 45 + t_post * 30
                    
                    # Main disk
                    disk = Ellipse(xy=(0, 0),
                                 width=disk_width,
                                 height=disk_height,
                                 angle=rotation % 360,
                                 fill=True,
                                 facecolor=MATTER_COLOR,
                                 alpha=disk_alpha)
                    self.ax.add_patch(disk)
                    self.patches.append(disk)
                    
                    # Inner bright region
                    inner_disk = Ellipse(xy=(0, 0),
                                        width=disk_width * 0.4,
                                        height=disk_height * 0.4,
                                        angle=rotation % 360,
                                        fill=True,
                                        facecolor='#FFA500',
                                        alpha=disk_alpha * 1.5)
                    self.ax.add_patch(inner_disk)
                    self.patches.append(inner_disk)
        
        # Draw jets and matter ejection
        self.draw_jets_and_matter(t)
        
        # Add information display
        info_text = self.create_info_text(t)
        text = self.ax.text(0.02, 0.88, info_text,
                          transform=self.ax.transAxes,
                          fontsize=10,
                          color=TEXT_COLOR,
                          verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round,pad=0.5', 
                                  facecolor=BACKGROUND_COLOR, 
                                  edgecolor=TEXT_COLOR,
                                  alpha=0.8))
        self.texts.append(text)
        
        # Add time bar
        progress = t / DURATION
        bar_bg = Rectangle((0.1, 0.02), 0.8, 0.015,
                          transform=self.ax.transAxes,
                          facecolor=GRID_COLOR,
                          alpha=0.5)
        bar_fill = Rectangle((0.1, 0.02), 0.8 * progress, 0.015,
                            transform=self.ax.transAxes,
                            facecolor=REMNANT_COLOR,
                            alpha=0.8)
        self.ax.add_patch(bar_bg)
        self.ax.add_patch(bar_fill)
        self.patches.append(bar_bg)
        self.patches.append(bar_fill)
        
        return self.patches + self.texts


def create_merger_animation():
    """Main function to create and save the enhanced animation"""
    print("╔════════════════════════════════════════════════╗")
    print("║  BINARY NEUTRON STAR MERGER ANIMATION          ║")
    print("║  Enhanced Visualization Engine v2.0            ║")
    print("╚════════════════════════════════════════════════╝")
    print()
    print("Initializing simulation parameters...")
    
    # Create simulation
    sim = BNSMergerSimulation()
    
    # Create visualizer
    print("Setting up enhanced visualization...")
    viz = MergerVisualizer(sim)
    viz.add_background_stars()
    
    print(f"Creating animation: {N_FRAMES} frames at {FPS} fps")
    print(f"Animation Duration: {DURATION} seconds")
    print()
    
    # Create animation
    anim = animation.FuncAnimation(viz.fig, viz.animate_frame,
                                 frames=N_FRAMES,
                                 interval=1000/FPS,
                                 blit=True)
    
    # Save as GIF
    output_file = 'bns_merger_enhanced.gif'
    print(f"Rendering animation to {output_file}...")
    print("Progress:")
    
    # Save using imageio for better GIF quality
    with imageio.get_writer(output_file, mode='I', fps=FPS, loop=0) as writer:
        for frame in range(N_FRAMES):
            viz.animate_frame(frame)
            viz.fig.canvas.draw()
            
            # Convert to image using buffer_rgba
            buf = viz.fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            # Convert RGBA to RGB
            image = image[:, :, :3]
            writer.append_data(image)
            
            # Progress bar
            if frame % 30 == 0:
                progress = frame / N_FRAMES
                bar_length = 40
                filled = int(bar_length * progress)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"  [{bar}] {progress*100:.1f}% ({frame}/{N_FRAMES})")
    
    plt.close(viz.fig)
    
    print()
    print("╔════════════════════════════════════════════════╗")
    print("║  ANIMATION COMPLETE!                           ║")
    print("╚════════════════════════════════════════════════╝")
    print()
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    print()
    print("Simulation parameters:")
    print(f"  • Mass ratio (q): {MASS_RATIO}")
    print(f"  • Total mass: {M_TOT} M☉")
    print(f"  • Tidal deformability (~Λ): {TIDAL_DEFORMABILITY}")
    print(f"  • Post-merger frequency (f₂): {F2_FREQUENCY} kHz")
    print()
    print("Enhanced features:")
    print("  ✓ Multi-layer glow effects")
    print("  ✓ Orbital trails")
    print("  ✓ Relativistic jets")
    print("  ✓ Matter ejection")
    print("  ✓ Gravitational lensing")
    print("  ✓ Dynamic color temperature")
    print("  ✓ Accretion disk formation")
    print("  ✓ Enhanced information display")
    
    return output_file


if __name__ == "__main__":
    # Run the enhanced animation creation
    output_path = create_merger_animation()