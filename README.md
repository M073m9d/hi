import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main(initial_points, target_points, n_groups=4):
    # Group initial points into clusters using KMeans
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    kmeans.fit(initial_points)
    labels = kmeans.labels_
    
    groups = []
    for i in range(n_groups):
        group_indices = np.where(labels == i)[0]
        group_initial = initial_points[group_indices]
        group_target = target_points[group_indices]
        
        # Find leader (closest to the cluster centroid)
        centroid = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(group_initial - centroid, axis=1)
        leader_idx = np.argmin(distances)
        leader_initial = group_initial[leader_idx]
        leader_target = group_target[leader_idx]
        
        groups.append({
            'indices': group_indices,
            'leader_initial': leader_initial,
            'leader_target': leader_target,
            'members_initial': group_initial,
            'members_target': group_target
        })
    
    # Compute smooth paths for leaders using cubic spline
    for group in groups:
        x_i, y_i = group['leader_initial']
        x_t, y_t = group['leader_target']
        t_points = [0, 1]
        x_leader = [x_i, x_t]
        y_leader = [y_i, y_t]
        # Cubic spline with zero initial and final velocity (clamped)
        cs_x = CubicSpline(t_points, x_leader, bc_type=((1, 0), (1, 0)))
        cs_y = CubicSpline(t_points, y_leader, bc_type=((1, 0), (1, 0)))
        group['leader_path_x'] = cs_x
        group['leader_path_y'] = cs_y
    
    # Prepare all member paths
    all_paths_x = []
    all_paths_y = []
    for group in groups:
        leader_x = group['leader_path_x']
        leader_y = group['leader_path_y']
        for i in range(len(group['members_initial'])):
            # Member's initial and target positions
            m_init = group['members_initial'][i]
            m_targ = group['members_target'][i]
            # Leader's initial and target positions
            l_init = group['leader_initial']
            l_targ = group['leader_target']
            # Displacement vectors
            disp_init = m_init - l_init
            disp_targ = m_targ - l_targ
            # Spline interpolation for displacement
            t_disp = [0, 1]
            disp_x_points = [disp_init[0], disp_targ[0]]
            disp_y_points = [disp_init[1], disp_targ[1]]
            cs_disp_x = CubicSpline(t_disp, disp_x_points, bc_type='clamped')
            cs_disp_y = CubicSpline(t_disp, disp_y_points, bc_type='clamped')
            # Member's path functions
            def member_x(t):
                return leader_x(t) + cs_disp_x(t)
            def member_y(t):
                return leader_y(t) + cs_disp_y(t)
            all_paths_x.append(member_x)
            all_paths_y.append(member_y)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(min(initial_points[:,0].min(), target_points[:,0].min()) - 1,
                max(initial_points[:,0].max(), target_points[:,0].max()) + 1)
    ax.set_ylim(min(initial_points[:,1].min(), target_points[:,1].min()) - 1,
                max(initial_points[:,1].max(), target_points[:,1].max()) + 1)
    ax.set_title('Formation Transition Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot initial and target formations
    ax.scatter(initial_points[:,0], initial_points[:,1], c='blue', label='Initial Formation', alpha=0.6)
    ax.scatter(target_points[:,0], target_points[:,1], c='red', marker='x', label='Target Formation', alpha=0.6)
    
    # Highlight group leaders in initial and target positions
    leader_initial_pos = np.array([g['leader_initial'] for g in groups])
    leader_target_pos = np.array([g['leader_target'] for g in groups])
    ax.scatter(leader_initial_pos[:,0], leader_initial_pos[:,1], c='green', s=100, 
               marker='*', label='Initial Leaders', edgecolor='black')
    ax.scatter(leader_target_pos[:,0], leader_target_pos[:,1], c='purple', s=100,
               marker='*', label='Target Leaders', edgecolor='black')
    
    # Create moving points for animation
    moving_dots = ax.scatter([], [], c='black', s=20, label='Moving Points')
    
    # Animation initialization function
    def init():
        moving_dots.set_offsets(np.empty((0, 2)))
        return moving_dots,
    
    # Animation update function
    def update(frame):
        t = frame / 100  # t ranges from 0 to 1 over 100 frames
        x_pos = [fx(t) for fx in all_paths_x]
        y_pos = [fy(t) for fy in all_paths_y]
        moving_dots.set_offsets(np.column_stack((x_pos, y_pos)))
        return moving_dots,
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=50)
    
    # Add legend and controls
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example data: initial points as a filled square and target as a circle
    n_points = 100
    np.random.seed(42)
    initial_points = np.random.rand(n_points, 2) * 10
    
    # Generate target points in a circle
    theta = np.linspace(0, 2 * np.pi, n_points)
    radius = 4
    cx, cy = 5, 5
    target_points = np.column_stack((cx + radius * np.cos(theta), cy + radius * np.sin(theta)))
    
    main(initial_points, target_points, n_groups=4)
