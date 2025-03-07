import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import List, Tuple

class FormationPlanner:
    """
    Plans military parade formations with collision-free movement paths
    Implements square and circle formations with subgroup division and leadership
    """
    
    def __init__(self, shape_type: str, num_people: int, formation_size: float):
        self.shape_type = shape_type
        self.num_people = num_people
        self.formation_size = formation_size  # Radius for circle, side length for square
        self.subgroups = []
        self.leader_indices = []
        self.positions = []
        self.start_positions = []
        
        # Physical constraints
        self.max_speed = 1.4  # m/s (typical marching speed)
        self.min_spacing = 0.9  # meters between individuals
        self.turning_radius = 1.5  # meters

    def generate_formation(self):
        """Generate target positions based on selected shape"""
        if self.shape_type == 'square':
            self._generate_square()
        elif self.shape_type == 'circle':
            self._generate_circle()
        self._divide_subgroups()
        self._assign_leaders()

    def _generate_square(self):
        """Generate positions for square formation"""
        side_length = self.formation_size
        people_per_side = self.num_people // 4
        extra = self.num_people % 4
        
        # Distribute points evenly on all four sides
        positions = []
        for side in range(4):
            n = people_per_side + (1 if side < extra else 0)
            x = np.linspace(-side_length/2, side_length/2, n)
            y = np.linspace(-side_length/2, side_length/2, n)
            
            for i in range(n):
                if side == 0:  # Top
                    positions.append((x[i], side_length/2))
                elif side == 1:  # Right
                    positions.append((side_length/2, y[i]))
                elif side == 2:  # Bottom
                    positions.append((x[i], -side_length/2))
                else:  # Left
                    positions.append((-side_length/2, y[i]))
        
        self.positions = np.array(positions)

    def _generate_circle(self):
        """Generate positions for circular formation"""
        angles = np.linspace(0, 2*np.pi, self.num_people, endpoint=False)
        self.positions = np.array([
            (self.formation_size * np.cos(angle),
             self.formation_size * np.sin(angle))
            for angle in angles
        ])

    def _divide_subgroups(self):
        """Divide formation into logical subgroups"""
        if self.shape_type == 'square':
            # Divide into four sides
            subgroup_size = len(self.positions) // 4
            self.subgroups = np.split(self.positions, 4)
        elif self.shape_type == 'circle':
            # Divide into quadrants
            angles = np.arctan2(self.positions[:,1], self.positions[:,0])
            quadrant_indices = np.digitize(angles, [np.pi/2, np.pi, 3*np.pi/2])
            self.subgroups = [self.positions[quadrant_indices == i] for i in range(4)]

    def _assign_leaders(self):
        """Select subgroup leaders (most central position in subgroup)"""
        self.leader_indices = []
        for subgroup in self.subgroups:
            centroid = np.mean(subgroup, axis=0)
            distances = np.linalg.norm(subgroup - centroid, axis=1)
            leader_idx = np.argmin(distances)
            self.leader_indices.append(leader_idx)

    def plan_movement(self, start_pos: Tuple[float, float] = (0, -10)):
        """Plan movement paths from starting position to formation"""
        self.start_positions = np.tile(start_pos, (self.num_people, 1))
        self.paths = []
        
        for idx, (start, end) in enumerate(zip(self.start_positions, self.positions)):
            if idx in self.leader_indices:
                # Leaders get straight paths with priority
                path = self._straight_path(start, end)
            else:
                path = self._bezier_path(start, end, curvature=0.3)
            self.paths.append(path)

        # Optimize paths to prevent collisions
        self._optimize_paths()

    def _bezier_path(self, start, end, curvature=0.5):
        """Generate Bézier curve path with collision avoidance"""
        direction = end - start
        perp = np.array([-direction[1], direction[0]]) * curvature
        control1 = start + direction * 0.3 + perp
        control2 = end - direction * 0.3 + perp
        return [start, control1, control2, end]

    def _straight_path(self, start, end):
        """Generate straight line path for leaders"""
        return [start, (start + end)/2, (start + end)/2, end]

    def _optimize_paths(self):
        """Optimize paths using gradient-based minimization"""
        # Simplified collision avoidance optimization
        for i in range(len(self.paths)):
            if i in self.leader_indices:
                continue  # Don't optimize leader paths
                
            def collision_cost(params):
                self.paths[i][1] = params[:2]
                self.paths[i][2] = params[2:]
                return self._calculate_collision_risk(i)
                
            initial = np.concatenate([self.paths[i][1], self.paths[i][2]])
            res = minimize(collision_cost, initial, method='L-BFGS-B')
            self.paths[i][1] = res.x[:2]
            self.paths[i][2] = res.x[2:]

    def _calculate_collision_risk(self, path_idx):
        """Calculate collision risk for a single path"""
        risk = 0
        path = self._sample_path(self.paths[path_idx])
        for j, other_path in enumerate(self.paths):
            if j == path_idx:
                continue
            other_samples = self._sample_path(other_path)
            min_dists = np.min(np.linalg.norm(path - other_samples, axis=1))
            if min_dists < self.min_spacing:
                risk += 1/min_dists
        return risk

    def _sample_path(self, path, num_samples=20):
        """Sample points along a Bézier curve"""
        t = np.linspace(0, 1, num_samples)
        return np.array([(1-t)**3 * path[0] + 3*(1-t)**2*t*path[1] 
                       + 3*(1-t)*t**2*path[2] + t**3*path[3] for t in t])

    def visualize(self):
        """Visualize formation and paths"""
        plt.figure(figsize=(10, 10))
        
        # Plot target positions
        plt.scatter(self.positions[:,0], self.positions[:,1], c='blue', label='Target Positions')
        
        # Highlight leaders
        leaders = self.positions[self.leader_indices]
        plt.scatter(leaders[:,0], leaders[:,1], c='red', s=100, label='Leaders')
        
        # Plot paths
        for i, path in enumerate(self.paths):
            samples = self._sample_path(path)
            plt.plot(samples[:,0], samples[:,1], 'grey', alpha=0.3)
            if i in self.leader_indices:
                plt.plot(samples[:,0], samples[:,1], 'red', alpha=0.5)
        
        plt.xlabel('X Position (meters)')
        plt.ylabel('Y Position (meters)')
        plt.title(f'{self.shape_type.capitalize()} Formation Plan')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Square Formation Example
    square_planner = FormationPlanner('square', num_people=40, formation_size=20)
    square_planner.generate_formation()
    square_planner.plan_movement()
    square_planner.visualize()

    # Circle Formation Example
    circle_planner = FormationPlanner('circle', num_people=40, formation_size=15)
    circle_planner.generate_formation()
    circle_planner.plan_movement()
    circle_planner.visualize()
