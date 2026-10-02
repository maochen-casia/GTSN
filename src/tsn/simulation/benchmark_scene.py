"""The tsn-1k generator's complete depth geometry, expressed without cache files.

Source: TableSceneNav/tablescenenav/environment.py and assets.py (2026-09-30).
Visual fittings and room geometry are separate from conservative box collisions.
"""

from __future__ import annotations

import numpy as np


BENCHMARK_SETTINGS = {
    "physics_hz": 100, "control_hz": 20,
    "table_center_m": [0.35, 0.0, -0.04], "table_half_size_m": [0.63, 0.62, 0.04],
    "base_position_m": [0.0, 0.0, 0.0], "robot_urdf": "panda_v3.urdf",
    "camera_near_m": 0.01, "camera_far_m": 5.0,
    "solver_position_iterations": 15, "solver_velocity_iterations": 1,
    "enable_ccd": False,
    "balance_passive_force": "disable_link_gravity",
    "arm_stiffness": 1000.0, "arm_damping": 100.0, "arm_force_limit": 100.0,
    "finger_stiffness": 1000.0, "finger_damping": 100.0, "finger_force_limit": 100.0,
}


def collision_half(item):
    half = np.asarray(item["half_size"], dtype=float) + 0.022
    if item["kind"] == "mug":
        half[:2] = np.maximum(half[:2], item["half_size"][0] * 1.98)
    return half


def add_visuals(simulation, description):
    """Reproduce every depth-bearing surface from the benchmark scene generator."""
    trimesh, pyrender = simulation.trimesh, simulation.pyrender

    def mesh(geometry, center, color, yaw=0.0):
        pose = np.eye(4)
        c, s = np.cos(yaw), np.sin(yaw)
        pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        pose[:3, 3] = center
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[*color[:3], 1], roughnessFactor=0.65)
        simulation.render_scene.add(pyrender.Mesh.from_trimesh(geometry, material=material, smooth=False), pose=pose)

    def box(center, half, color, yaw=0.0):
        mesh(trimesh.creation.box(extents=2 * np.asarray(half)), center, color, yaw)

    box([0.35, 0, -0.84], [2, 2, 0.04], [0.30, 0.31, 0.32])
    box([0.35, 0, -0.04], [0.63, 0.62, 0.04], description["table_color"])
    for x in [-0.18, 0.89]:
        for y in [-0.51, 0.51]:
            box([x, y, -0.41], [0.033, 0.033, 0.37], [0.16, 0.17, 0.18])
    for center, half, color in [
        ([0.4, -1.0, 0.5], [2, 0.03, 1.3], [0.76, 0.77, 0.74]),
        ([1.65, 0, 0.5], [0.03, 1.5, 1.3], [0.81, 0.81, 0.78]),
        ([0.4, 1.24, 0.5], [2, 0.03, 1.3], [0.82, 0.81, 0.77]),
        ([-0.85, 0, 0.5], [0.03, 1.5, 1.3], [0.76, 0.78, 0.76]),
        ([0.4, 0, 1.8], [2, 1.5, 0.02], [0.88, 0.88, 0.85]),
        ([0.44, -0.89, 0.48], [0.54, 0.10, 0.016], [0.40, 0.30, 0.21]),
    ]:
        box(center, half, color)
    for k in range(5):
        box([0.15 + k * 0.12, -0.88, 0.57], [0.037, 0.065, 0.073], [0.25 + k * 0.08, 0.34, 0.47 - k * 0.04])

    vessels = {"mug", "bottle", "bowl", "jar"}
    for item in description["objects"]:
        center, half = np.asarray(item["center"]), np.asarray(item["half_size"])
        kind, color, yaw = item["kind"], item["color"], item["yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        def detail(local, size, tint):
            box(center + rotation @ np.asarray(local), size, tint, yaw)

        if kind not in vessels:
            box(center, half, color, yaw)
        if kind in {"storage_box", "toolbox", "box"}:
            detail([0, 0, half[2] * 0.85], half * [1.01, 1.01, 0.09], np.asarray(color) * 0.72)
            detail([-half[0] - 0.0005, 0, 0.01], half * [0, 0.43, 0.24] + [0.001, 0, 0], [0.87, 0.84, 0.73])
            if kind == "toolbox":
                detail([0, 0, half[2] + 0.009], [half[0] * 0.6, 0.035, 0.009], [0.12, 0.13, 0.14])
        elif kind in {"book", "book_stack"}:
            count = 5 if kind == "book_stack" else 1
            for k in range(count):
                z = -half[2] + (2 * k + 1) * half[2] / count
                detail([0, 0, z], [half[0] * 0.97, half[1] * 0.97, half[2] / count * 0.77], [0.83, 0.80, 0.70])
                detail([0, 0, z + half[2] / count * 0.87], [half[0] * 1.01, half[1] * 1.01, half[2] / count * 0.1], np.roll(color, k))
        elif kind in {"carton", "appliance", "canister"}:
            detail([-half[0] - 0.001, 0, 0], [0.001, half[1] * 0.75, half[2] * 0.76], [0.85, 0.83, 0.72] if kind == "carton" else [0.20, 0.23, 0.25])
            for k in range(3):
                detail([-half[0] - 0.002, 0, half[2] * (0.55 - k * 0.36)], [0.001, half[1] * 0.58, 0.009], color)
            detail([0, 0, half[2] - 0.004], [half[0], half[1], 0.004], np.asarray(color) * 0.6)
        elif kind in vessels:
            r, h = half[0], half[2]
            profiles = {
                "mug": [[0, -h], [r, -h], [r, h], [r * .84, h], [r * .84, -h * .77], [0, -h * .77]],
                "bowl": [[0, -h], [r * .38, -h], [r * .72, -h * .35], [r, h], [r * .9, h], [r * .64, -h * .3], [r * .33, -h * .78], [0, -h * .78]],
                "bottle": [[0, -h], [r * .85, -h], [r, -h * .86], [r, h * .42], [r * .45, h * .75], [r * .45, h], [0, h]],
                "jar": [[0, -h], [r, -h], [r, h * .8], [r * .85, h], [0, h]],
            }
            # The generator intentionally builds vessels without the object's yaw.
            mesh(trimesh.creation.revolve(np.asarray(profiles[kind]), sections=48), center, color)
            if kind == "mug":
                a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 40, endpoint=False), np.linspace(0, 2 * np.pi, 12, endpoint=False), indexing="ij")
                tube = min(r * .16, .007)
                vertices = np.stack([r * .56 * np.cos(a) + tube * np.cos(b) * np.cos(a), tube * np.sin(b), h * .66 * np.sin(a) + tube * np.cos(b) * np.sin(a)], axis=-1).reshape(-1, 3)
                faces = []
                for i in range(40):
                    for j in range(12):
                        k, u = i * 12 + j, ((i + 1) % 40) * 12 + j
                        v, w = ((i + 1) % 40) * 12 + (j + 1) % 12, i * 12 + (j + 1) % 12
                        faces.extend([[k, u, v], [k, v, w]])
                mesh(trimesh.Trimesh(vertices=vertices, faces=faces), center + [r * 1.24, 0, 0], color)
            if kind in {"bottle", "jar"}:
                detail([0, 0, h], [r * .6, half[1] * .6, .008], [.19, .19, .20])
