# utils/koch.py
import numpy as np

def rotation_matrix(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])

def initial_equilateral(scale=1.0):
    """Triángulo equilátero base (cerrado)."""
    side = 1.0 * scale
    h = np.sqrt(3) / 2 * side
    p1 = np.array([0.0, 0.0])
    p2 = np.array([side, 0.0])
    p3 = np.array([side / 2.0, h])
    return np.array([p1, p2, p3, p1])

def koch_external_step(points):
    """
    Dado un array Nx2 que define una polilínea cerrada (último igual al primero),
    aplica un paso de Koch que crea el pico hacia afuera en cada segmento.
    """
    new_pts = []
    # iterate segments (points assumed closed: last == first)
    for i in range(len(points) - 1):
        p0 = np.array(points[i])
        p1 = np.array(points[i + 1])
        seg = p1 - p0
        one_third = p0 + seg / 3.0
        two_third = p0 + 2.0 * seg / 3.0

        # vector along middle third
        v = two_third - one_third

        # rotate v by +60 degrees (counter-clockwise) to point outward relative to segment direction
        rot = rotation_matrix(np.pi / 3.0)
        peak = one_third + rot.dot(v)

        # append points, do not duplicate the segment endpoints (we'll close at the end)
        new_pts.append(p0)
        new_pts.append(one_third)
        new_pts.append(peak)
        new_pts.append(two_third)

    # append the last point (closing)
    new_pts.append(points[-1])
    return np.array(new_pts)

def koch_iterations(order, scale=1.0):
    """
    Genera la lista de arrays de puntos para cada nivel desde 0 hasta order.
    Cada elemento es un numpy.ndarray de forma (M,2) con último punto igual al primero (cerrado).
    """
    steps = []
    pts = initial_equilateral(scale=scale)
    steps.append(pts)
    for i in range(1, order + 1):
        pts = koch_external_step(pts)
        steps.append(pts)
    return steps

def get_precise_half(points, axis='x', side='left'):
    """
    Filtra puntos para obtener la "mitad" precisa basada en un eje de simetría.
    - axis: 'x' => horizontal split (superior/inferior)
            'y' => vertical split (left/right)
    - side: for axis=='x' -> 'inferior' or 'superior'
            for axis=='y' -> 'left' or 'right'
    Devuelve los puntos que cumplen la condición (manteniendo orden).
    """
    pts = np.array(points)
    if axis == 'x':
        mid = np.max(pts[:, 1]) / 2.0
        if side == 'inferior':
            mask = pts[:, 1] <= mid + 1e-12
        else:
            mask = pts[:, 1] >= mid - 1e-12
    else:
        mid = np.max(pts[:, 0]) / 2.0
        if side == 'left' or side == 'izquierda':
            mask = pts[:, 0] <= mid + 1e-12
        else:
            mask = pts[:, 0] >= mid - 1e-12

    # Keep only the points that match mask but keep ordering and ensure at least endpoints
    filtered = pts[mask]
    if filtered.shape[0] == 0:
        # fallback: return whole shape
        return pts
    return filtered
