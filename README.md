import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import linear_sum_assignment

# Shape Generation Functions
def generate_filled_square(n):
    """Generate points filling a square area"""
    side = int(np.sqrt(n))
    x = np.linspace(-1, 1, side)
    y = np.linspace(-1, 1, side)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return points[:n]

def generate_circle_border_points(n):
    """Generate points along a circle's circumference"""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

# Grouping Functions
def split_into_quadrants(points):
    """Divide points into 4 quadrants"""
    return [
        points[(points[:, 0] >= 0) & (points[:, 1] >= 0)],  # Q1
        points[(points[:, 0] < 0) & (points[:, 1] >= 0)],   # Q2
        points[(points[:, 0] < 0) & (points[:, 1] < 0)],     # Q3
        points[(points[:, 0] >= 0) & (points[:, 1] < 0)]    # Q4
    ]

def split_into_arcs(points, num_arcs=4):
    """Divide circle points into equal arcs"""
    return np.array_split(points, num_arcs)

# Matching and Path Planning
def match_points(start, end):
    """Optimal point matching using Hungarian algorithm"""
    cost_matrix = np.linalg.norm(start[:, np.newaxis] - end, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return end[col_ind]

def get_bezier_curve(p0, p1, num_points=100, amplitude=0.5):
    """Generate quadratic Bezier curve with control point"""
    mid = (p0 + p1) / 2
    diff = p1 - p0
    norm = np.linalg.norm(diff)
    perp = np.array([-diff[1], diff[0]]) / (norm + 1e-6)  # Avoid division by zero
    control = mid + perp * amplitude
    t = np.linspace(0, 1, num_points)[:, None]
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * control + t**2 * p1

# Animation Setup
def animate(i, bezier_paths, groups_indices, num_groups, frames_per_group, scatter):
    """Update point positions based on precomputed Bezier paths"""
    all_points = np.vstack(initial_groups)  # Start with initial positions
    for group_idx in range(num_groups):
        start_frame = group_idx * frames_per_group
        current_frame = i - start_frame
        
        if 0 <= current_frame < frames_per_group:
            group_curves = bezier_paths[group_idx]
            start_idx, end_idx = groups_indices[group_idx]
            
            for point_idx, curve in enumerate(group_curves):
                all_points[start_idx + point_idx] = curve[current_frame]
    
    scatter.set_offsets(all_points)
    return scatter,

# Main Configuration
n_points = 100
num_groups = 4
frames = 80
frames_per_group = frames // num_groups

# Generate and group points
square_points = generate_filled_square(n_points)
circle_points = generate_circle_border_points(n_points)

initial_groups = split_into_quadrants(square_points)
target_groups = split_into_arcs(circle_points, num_groups)

# Apply visual offsets
square_offset = np.array([1.5, 1.5])
circle_offset = np.array([-1.5, -1.5])
initial_groups = [g + square_offset for g in initial_groups]
target_groups = [g + circle_offset for g in target_groups]

# Match points within groups and compute Bezier paths
bezier_paths = []
for group_idx in range(num_groups):
    matched_target = match_points(initial_groups[group_idx], target_groups[group_idx])
    group_curves = [
        get_bezier_curve(s, e, frames_per_group) 
        for s, e in zip(initial_groups[group_idx], matched_target)
    ]
    bezier_paths.append(group_curves)

# Create group indices mapping
groups_indices = []
current_idx = 0
for group in initial_groups:
    groups_indices.append((current_idx, current_idx + len(group)))
    current_idx += len(group)

# Visualization Setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_title('Formation Transition Visualization')

# Plot leader paths
colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
for group_idx in range(num_groups):
    leader_start = np.mean(initial_groups[group_idx], axis=0)
    leader_end = np.mean(target_groups[group_idx], axis=0)
    curve = get_bezier_curve(leader_start, leader_end, 100)
    ax.plot(curve[:, 0], curve[:, 1], '--', color=colors[group_idx], 
            linewidth=2, label=f'Group {group_idx+1} Leader')

ax.legend()

# Initialize scatter plot
all_initial_points = np.vstack(initial_groups)
scatter = ax.scatter(all_initial_points[:, 0], all_initial_points[:, 1])

# Create and show animation
ani = animation.FuncAnimation(
    fig, animate, frames=frames,
    fargs=(bezier_paths, groups_indices, num_groups, frames_per_group, scatter),
    interval=50, blit=True
)

plt.show()
