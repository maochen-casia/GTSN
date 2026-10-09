#!/usr/bin/env python3
"""Reproduce the vendored official FR3 assets (run inside Docker, needs xacro).

The canonical source files are retained verbatim.  The temporary xacro copy
uses local package paths and removes the obsolete ``gazebo`` macro argument
which the 2.9.0 top-level xacro still passes after the macro removed it.
"""
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
import xml.etree.ElementTree as ET

COMMIT = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
REPOSITORY = "https://github.com/frankarobotics/franka_description"
ROOT = Path(__file__).resolve().parent


def download(path):
    destination = ROOT / "upstream" / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/frankarobotics/franka_description/{COMMIT}/{path}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            destination.write_bytes(data)
            return path, hashlib.sha256(data).hexdigest()
        except Exception:
            if attempt == 4:
                raise
            time.sleep(attempt + 1)


def main():
    import xacro
    import trimesh

    with urllib.request.urlopen(f"https://api.github.com/repos/frankarobotics/franka_description/git/trees/{COMMIT}?recursive=1") as response:
        tree = json.load(response)["tree"]
    prefixes = (
        "robots/fr3/", "robots/common/", "end_effectors/franka_hand/",
        "end_effectors/common/", "meshes/robots/fr3/",
        "meshes/robot_ee/franka_hand_white/",
    )
    selected = [item for item in tree if item["type"] == "blob" and
                (item["path"].startswith(prefixes) or item["path"] in ("LICENSE", "NOTICE"))]
    paths = [item["path"] for item in selected]
    def fetch(item):
        path = item["path"]
        destination = ROOT / "upstream" / path
        if destination.exists():
            data = destination.read_bytes()
            blob = f"blob {len(data)}\0".encode() + data
            if hashlib.sha1(blob).hexdigest() == item["sha"]:
                return path, hashlib.sha256(data).hexdigest()
        return download(path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        hashes = dict(executor.map(fetch, selected))
    for filename in ("LICENSE", "NOTICE"):
        shutil.copyfile(ROOT / "upstream" / filename, ROOT / filename)

    # SAPIEN loads each single collision mesh as a convex hull. MPLib's
    # convex=True parser requires that same hull in a ``.convex.stl`` sidecar.
    # Keep the official originals verbatim and record the derived resources.
    cooked_hashes = {}
    for path in paths:
        if "/collision/" not in path or not path.endswith(".stl"):
            continue
        destination = ROOT / "upstream" / (path + ".convex.stl")
        mesh = trimesh.load(str(ROOT / "upstream" / path), force="mesh")
        mesh.convex_hull.export(str(destination))
        cooked_hashes[path + ".convex.stl"] = hashlib.sha256(destination.read_bytes()).hexdigest()

    # Work in the repository cache; leave the upstream evidence unmodified.
    work = ROOT.parents[1] / ".cache" / "franka_xacro"
    shutil.copytree(ROOT / "upstream", work, dirs_exist_ok=True)
    for path in work.rglob("*.xacro"):
        text = path.read_text().replace("$(find franka_description)", str(work))
        if path.name == "fr3.urdf.xacro":
            text = text.replace('gazebo="$(arg gazebo)"', "")
        path.write_text(text)
    mappings = {"with_sc": "false", "hand": "true", "ee_id": "franka_hand", "no_prefix": "false"}
    outputs = {}
    for kind in ("urdf", "srdf"):
        document = xacro.process_file(str(work / "robots" / "fr3" / f"fr3.{kind}.xacro"), mappings=mappings)
        text = document.toprettyxml(indent="  ")
        text = text.replace("package://franka_description/", "upstream/")
        text = text.replace(str(work), "upstream")
        if kind == "urdf":
            # SAPIEN needs strictly positive inertias on collision-free fixed
            # accelerometer links. The official dummy tensor is all zeros.
            robot = ET.fromstring(text)
            for inertial in robot.findall("link/inertial"):
                mass = inertial.find("mass")
                inertia = inertial.find("inertia")
                if float(mass.get("value")) == 0:
                    mass.set("value", "1e-8")
                if all(float(inertia.get(axis)) == 0 for axis in ("ixx", "iyy", "izz")):
                    for axis in ("ixx", "iyy", "izz"):
                        inertia.set(axis, "1e-12")
            ET.indent(robot, space="  ")
            text = '<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n"
        destination = ROOT / f"fr3.{kind}"
        destination.write_text(text)
        outputs[destination.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    provenance = {
        "robot_model": "Franka Research 3", "robot_type": "fr3", "release": "2.9.0",
        "repository": REPOSITORY, "commit": COMMIT, "license": "Apache-2.0",
        "upstream_sha256": hashes, "generated_sha256": outputs,
        "cooked_collision_sha256": cooked_hashes,
        "adaptations": [
            "Generated from official FR3 xacro with the white Franka hand; with_sc=false uses official fine collision meshes.",
            "Replaced ROS package paths with vendored relative mesh paths.",
            "Removed obsolete gazebo argument from the temporary xacro invocation.",
            "Replaced exactly zero dummy fixed-link mass/inertia with negligible positive values for SAPIEN.",
            "Added deterministic convex-hull STL sidecars for MPLib, matching SAPIEN single-convex mesh loading; official source files are unchanged.",
            "Runtime camera housing and optical link are added per episode by tablescenenav.robot.",
        ],
    }
    (ROOT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Vendored {len(paths)} official FR3 files; generated URDF and SRDF.")


if __name__ == "__main__":
    main()
