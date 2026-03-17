import numpy as np
from skbio.stats.composition import clr, clr_inv

import plotly.graph_objects as go

# --- Functions for Pixel-Perfect Circles ---
def ternary_to_cartesian(a, b, c):
    """Converts ternary (a,b,c) to 2D (x,y) for a standard equilateral triangle."""
    x = 0.5 * (2*b + c) / (a + b + c)
    y = (np.sqrt(3)/2) * c / (a + b + c)
    return x, y

def cartesian_to_ternary(x, y):
    """Converts 2D (x,y) back to ternary (a,b,c)."""
    c = y / (np.sqrt(3)/2)
    b = (x - 0.5*c)
    a = 1 - b - c
    return a, b, c

# # --- Functions for CLR Circles ---
def get_clr_ring(center_clr, radius, n_points=100):
    t = np.linspace(0, 2*np.pi, n_points)
    
    v1 = np.array([1, -1, 0]) / np.sqrt(2)
    v2 = np.array([1, 1, -2]) / np.sqrt(6)
    
    ring_points_clr = []
    for angle in t:
        p_clr = center_clr + radius * (np.cos(angle) * v1 + np.sin(angle) * v2)
        ring_points_clr.append(clr_inv(p_clr))
    
    return np.array(ring_points_clr)

def main():

    points = [np.array([0.75, 0.2, 0.05]),
            np.array([0.3, 0.35, 0.35])]

    radii = [0.03, 0.06, 0.09]
    clr_radii = [0.15, 0.3, 0.45]
    colors = ["black", "blue"]
    fig = go.Figure()

    # ---- rings + points ----
    for pi, p in enumerate(points):
        # 1. Add Pixel-Perfect Rings (Green)
        cx, cy = ternary_to_cartesian(*p)
        t = np.linspace(0, 2*np.pi, 100)

        # Euclidean rings
        for ri, r in enumerate(radii):
            x_circ = cx + r * np.cos(t)
            y_circ = cy + r * np.sin(t)
            ta, tb, tc = cartesian_to_ternary(x_circ, y_circ)
            
            fig.add_trace(go.Scatterternary(
                a=ta, b=tb, c=tc,
                mode='lines',
                line=dict(color='green', width=1.5, dash='dot'), # Dotted green for pixel-perfect
                name="Euclidean" if (pi==0 and ri==0) else "",
                showlegend=(pi==0 and ri==0)
            ))
            
        # 2. Add CLR-Transformed Rings (Red)
        point_clr = clr(p)
        for ri, r in enumerate(clr_radii):
            ring_simplex = get_clr_ring(point_clr, r)
            fig.add_trace(go.Scatterternary(
                a=ring_simplex[:, 0], b=ring_simplex[:, 1], c=ring_simplex[:, 2],
                mode='lines',
                line=dict(color='red', width=1.5, dash='solid'), # Solid red for CLR
                name="CLR (Aitchison)" if (pi==0 and ri==0) else "",
                showlegend=(pi==0 and ri==0)
            ))

        # 3. Add the 'X' point
        fig.add_trace(go.Scatterternary(
            a=[p[0]], b=[p[1]], c=[p[2]],
            mode='markers',
            marker=dict(symbol='x', size=10, color=colors[pi], line=dict(width=1)),
            name= f'Point{[float(p[0]), float(p[1]), float(p[2])]}'
        ))

    # 4. Layout and Titles
    fig.update_layout(
        ternary=dict(
            aaxis_title='Component A',
            baxis_title='Component B',
            caxis_title='Component C',
            sum=1 # Ensure it sums to 1
        ),
        title="Comparison of Standard (Green) and CLR-Transformed (Red) Rings",
        showlegend=True,
        width=750, height=650 # Keep aspect ratio for better visual
    )

    fig.show()

if __name__ == "__main__":
    main()