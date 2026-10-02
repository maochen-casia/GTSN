"""Reconstruct the complete tsn-1k depth scene and benchmark controller in OSMesa.

The generator's fixed settings accompany the per-episode scene.json. Textures
affect RGB appearance only; all surfaces affecting geometry maps are retained.
"""

from __future__ import annotations

import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from tsn.data.hdf5_dataset import JOINT_NAMES
from tsn.simulation.benchmark_scene import BENCHMARK_SETTINGS, add_visuals, collision_half


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a scalar-first unit quaternion (4,) into a rotation matrix (3,3)."""
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError("Zero quaternion")
    w, x, y, z = np.asarray(quaternion, dtype=np.float64) / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def orientation_error(rotation: np.ndarray, desired: np.ndarray) -> float:
    """Return the geodesic angle between two 3x3 rotation matrices, in radians."""
    cosine = (np.trace(rotation.T @ desired) - 1) / 2
    return float(np.arccos(np.clip(cosine, -1, 1)))


class EpisodeSimulation:
    """One Panda articulation with static scene geometry and live wrist observations.

    Physics runs at physics_hz; each step executes one 20 Hz position target and
    checks contacts at every physics substep. The two fingers retain initial
    positions. A kinematic consistency guard rejects mismatched TCP/URDF settings.
    """

    def __init__(self, scene_description: dict[str, Any], calibration: np.ndarray,
                 camera_extrinsic: np.ndarray, source_hw: tuple[int, int],
                 initial_qpos: np.ndarray, initial_ee: np.ndarray, joint_names: tuple[str, ...],
                 options: dict[str, Any]) -> None:
        # Set the rendering backend before importing PyOpenGL through pyrender.
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        import sapien
        import sapien.physx as physx
        import pyrender
        import trimesh
        from mani_skill import PACKAGE_ASSET_DIR

        self.sapien, self.pyrender, self.trimesh = sapien, pyrender, trimesh
        options = {**options, **BENCHMARK_SETTINGS}
        self.options = options
        if not np.isfinite(initial_qpos).all() or not np.isfinite(initial_ee).all():
            raise ValueError("Initial robot state must be finite")
        physics_hz, control_hz = int(options["physics_hz"]), int(options["control_hz"])
        if control_hz != 20 or physics_hz < control_hz or physics_hz % control_hz:
            raise ValueError("20 Hz control and an integral physics/control ratio are required")
        self.substeps = physics_hz // control_hz
        self.T_E_C = np.asarray(camera_extrinsic, dtype=np.float64)
        physx.set_scene_config(gravity=np.array([0, 0, -9.81], dtype=np.float32),
                               enable_pcm=True, enable_tgs=True, enable_ccd=options["enable_ccd"])
        physx.set_body_config(solver_position_iterations=options["solver_position_iterations"],
                              solver_velocity_iterations=options["solver_velocity_iterations"])
        self.scene = sapien.Scene([physx.PhysxCpuSystem()])
        self.scene.set_timestep(1 / physics_hz)
        self.render_scene = pyrender.Scene(bg_color=[0.8, 0.8, 0.8, 1], ambient_light=[0.5, 0.5, 0.5])
        self.renderer = None
        self._objects(scene_description)
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf = Path(PACKAGE_ASSET_DIR) / "robots/panda" / options["robot_urdf"]
        if not urdf.is_file():
            raise FileNotFoundError(f"Panda asset missing from Docker image: {urdf}")
        # SAPIEN's URDF parser creates Vulkan materials for inline color tags.
        # OSMesa owns visuals here; remove only materials, retaining all v3
        # geometry, collisions, masses and joints in a disposable container file.
        tree = ET.parse(urdf)
        for parent in tree.iter():
            for child in list(parent):
                if child.tag == "material":
                    parent.remove(child)
        for mesh in tree.iter("mesh"):
            mesh.set("filename", str((urdf.parent / mesh.get("filename")).resolve()))
        with tempfile.TemporaryDirectory(prefix="tsn-urdf-") as temporary:
            physics_urdf = Path(temporary) / "panda.urdf"
            tree.write(physics_urdf)
            builder = loader.load_file_as_articulation_builder(str(physics_urdf), str(urdf.with_suffix(".srdf")))
        if builder is None:
            raise RuntimeError(f"Unable to load Panda asset {urdf}")
        visual_records = [(link.name, list(link.visual_records)) for link in builder.link_builders]
        for link in builder.link_builders:
            link.visual_records = []
        self.robot = builder.build(fix_root_link=True)
        self.robot.set_root_pose(sapien.Pose(options["base_position_m"]))
        self.links = {link.name: link for link in self.robot.get_links()}
        for link in self.links.values():
            link.disable_gravity = True
        self.joints = self.robot.get_active_joints()
        actual_names = tuple(joint.name for joint in self.joints)
        if actual_names != joint_names or actual_names != JOINT_NAMES:
            raise ValueError("Panda URDF and HDF5 joint order differ")
        self.T_W_B = self.links["panda_link0"].entity_pose.to_transformation_matrix()
        self.T_B_W = np.linalg.inv(self.T_W_B)
        self.adjacent = {frozenset((link.name, link.parent.name)) for link in self.links.values()
                         if link.parent is not None}
        self.fingers = np.asarray(initial_qpos[7:], dtype=np.float32).copy()
        self.robot.set_qpos(initial_qpos.astype(np.float32))
        self.robot.set_qvel(np.zeros(9, dtype=np.float32))
        for index, joint in enumerate(self.joints):
            prefix = "arm" if index < 7 else "finger"
            joint.set_drive_properties(float(options[f"{prefix}_stiffness"]),
                                       float(options[f"{prefix}_damping"]),
                                       float(options[f"{prefix}_force_limit"]), "force")
            joint.set_drive_target(float(initial_qpos[index]))
            joint.set_drive_velocity_target(0.0)
        self.robot.set_solver_position_iterations(options["solver_position_iterations"])
        self.robot.set_solver_velocity_iterations(options["solver_velocity_iterations"])
        self.qlimits = np.asarray(self.robot.get_qlimits())[:7]
        snapshot = self.snapshot()
        position_error = np.linalg.norm(snapshot["T_B_E"][:3, 3] - initial_ee[:3])
        rotation_error = orientation_error(snapshot["T_B_E"][:3, :3], quaternion_matrix(initial_ee[3:]))
        if (position_error > options["kinematic_position_tolerance_m"] or
                rotation_error > options["kinematic_rotation_tolerance_rad"]):
            raise ValueError(f"Panda TCP does not match benchmark: position={position_error:.6f} m, "
                             f"orientation={rotation_error:.6f} rad; check ee_link and URDF")
        self.robot_nodes = []
        for link_name, records in visual_records:
            for record in records:
                if record.type != "file":
                    continue
                mesh_scene = trimesh.load(record.filename, force="scene", process=False)
                for geometry in mesh_scene.dump(concatenate=False):
                    geometry.apply_scale(np.asarray(record.scale))
                    mesh = pyrender.Mesh.from_trimesh(geometry, smooth=False)
                    node = self.render_scene.add(mesh)
                    self.robot_nodes.append((link_name, record.pose.to_transformation_matrix(), node))
        height, width = source_hw
        camera = pyrender.IntrinsicsCamera(
            fx=float(calibration[0, 0]), fy=float(calibration[1, 1]),
            cx=float(calibration[0, 2]), cy=float(calibration[1, 2]),
            znear=options["camera_near_m"], zfar=options["camera_far_m"],
        )
        self.camera_node = self.render_scene.add(camera)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0.4, -0.2, 1.5]
        light = pyrender.DirectionalLight(color=scene_description.get("light", [1, 1, 1]), intensity=2.0)
        self.render_scene.add(light, pose=light_pose)
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def _objects(self, description: dict[str, Any]) -> None:
        """Build benchmark collision envelopes separately from detailed visual surfaces."""
        sapien, trimesh, pyrender = self.sapien, self.trimesh, self.pyrender
        table = {
            "name": "table", "kind": "box", "center": self.options["table_center_m"],
            "half_size": self.options["table_half_size_m"],
            "color": description["table_color"], "yaw": 0,
        }
        for item in [table, *description["objects"]]:
            half = np.asarray(item["half_size"], dtype=np.float64)
            yaw = float(item["yaw"])
            pose = sapien.Pose(item["center"], [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=half if item["name"] == "table" else collision_half(item))
            builder.initial_pose = pose
            builder.build_static(item["name"])
        add_visuals(self, description)

    def snapshot(self) -> dict[str, np.ndarray]:
        """Return measured qpos (9,), base-frame TCP and wrist-camera transforms (4,4)."""
        tcp = self.links[self.options["ee_link"]].entity_pose.to_transformation_matrix()
        T_B_E = self.T_B_W @ tcp
        return {"qpos": np.asarray(self.robot.get_qpos(), dtype=np.float32).copy(),
                "T_B_E": T_B_E, "T_B_C": T_B_E @ self.T_E_C}

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Return live RGB (H,W,3) uint8 and camera Z-depth (H,W) float32 in metres."""
        for link_name, local, node in self.robot_nodes:
            world = self.links[link_name].entity_pose.to_transformation_matrix()
            self.render_scene.set_pose(node, world @ local)
        camera_world = self.T_W_B @ self.snapshot()["T_B_C"]
        self.render_scene.set_pose(self.camera_node, camera_world @ np.diag([1, -1, -1, 1]))
        rgb, depth = self.renderer.render(self.render_scene)
        # ManiSkill exports millimetre-quantized depth; preserve that sensor convention.
        depth = (np.floor(np.maximum(depth, 0) * 1000) / 1000).astype(np.float32)
        return rgb[..., :3].astype(np.uint8), depth

    def _contacts(self) -> list[dict[str, Any]]:
        """Filter only adjacent self contacts and the fixed base-table mounting contact."""
        contacts = []
        for contact in self.scene.get_contacts():
            a, b = (body.name for body in contact.bodies)
            pair = frozenset((a, b))
            robot_a, robot_b = a in self.links, b in self.links
            if not robot_a and not robot_b:
                continue
            if pair == frozenset(("panda_link0", "table")) or (robot_a and robot_b and pair in self.adjacent):
                continue
            impulse = sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
            if impulse > self.options["contact_impulse_threshold_ns"]:
                contacts.append({"body_a": a, "body_b": b, "impulse_ns": impulse})
        return contacts

    def step(self, arm_target: np.ndarray) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int]:
        """Execute one absolute arm target (7,), returning state, contacts and clip count."""
        if arm_target.shape != (7,) or not np.isfinite(arm_target).all():
            raise ValueError("Policy produced an invalid arm target")
        bounded = np.clip(arm_target, self.qlimits[:, 0], self.qlimits[:, 1])
        clipped = int(np.count_nonzero(bounded != arm_target))
        for joint, target in zip(self.joints, np.r_[bounded, self.fingers]):
            joint.set_drive_target(float(target))
        events = []
        for substep in range(self.substeps):
            self.scene.step()
            for contact in self._contacts():
                events.append({**contact, "physics_substep": substep})
            if events and self.options["stop_on_collision"]:
                break
        return self.snapshot(), events, clipped

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None
        self.robot_nodes = []
        self.render_scene = None
        self.links = {}
        self.joints = []
        self.robot = None
        self.scene = None

    def __enter__(self) -> "EpisodeSimulation":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
