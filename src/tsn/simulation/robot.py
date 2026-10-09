"""Robot descriptions and episode-calibrated Franka camera housings."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from tsn.common.config import PROJECT_ROOT

# SAPIEN camera (+x forward, +y left, +z up) from OpenCV optical coordinates.
S_FROM_CV = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def robot_spec(kind='panda'):
    if kind == 'fr3':
        path = PROJECT_ROOT/'assets/fr3/fr3.urdf'
    elif kind == 'panda':
        from mani_skill import PACKAGE_ASSET_DIR
        path = Path(PACKAGE_ASSET_DIR)/'robots/panda/panda_v3.urdf'
    else:
        raise ValueError('Unsupported robot: '+kind)
    return dict(kind=kind, urdf=path, base=kind+'_link0', tcp=kind+'_hand_tcp',
                hand=kind+'_hand', wrist=kind+'_link7',
                fingers=(kind+'_leftfinger', kind+'_rightfinger'),
                joints=tuple(f'{kind}_joint{i}' for i in range(1, 8))+
                       (kind+'_finger_joint1', kind+'_finger_joint2'))


def default_camera_extrinsic():
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    transform[:3, 3] = [.0465, -.0200, .0360-.1034]
    return transform


def robot_tree(kind='panda', camera_extrinsic=None):
    """Return canonical geometry with absolute mesh paths and calibrated camera."""
    spec = robot_spec(kind)
    root = ET.parse(spec['urdf']).getroot()
    for mesh in root.iter('mesh'):
        mesh.set('filename', str((spec['urdf'].parent/mesh.get('filename')).resolve()))
    if kind == 'panda' and camera_extrinsic is not None:
        # Keep the canonical camera mesh, collision box and intermediate link.
        # T_TCP_CV locates the optical camera_link; realsense_joint locates its
        # parent camera_base_link, whose remaining fixed offset must be removed.
        hand_from_tcp = np.eye(4); hand_from_tcp[2, 3] = .1034
        cv_from_s = np.eye(4); cv_from_s[:3, :3] = S_FROM_CV.T
        base_from_camera = np.eye(4); base_from_camera[:3, 3] = [0, .02, .0115]
        mount = hand_from_tcp@np.asarray(camera_extrinsic)@cv_from_s@np.linalg.inv(base_from_camera)
        origin = root.find("joint[@name='realsense_joint']/origin")
        origin.set('xyz', ' '.join(map(str, mount[:3, 3])))
        origin.set('rpy', ' '.join(map(str, Rotation.from_matrix(mount[:3, :3]).as_euler('xyz'))))
    if kind == 'fr3':
        optical = default_camera_extrinsic() if camera_extrinsic is None else np.asarray(camera_extrinsic)
        hand_from_tcp = np.eye(4); hand_from_tcp[2, 3] = .1034
        cv_from_s = np.eye(4); cv_from_s[:3, :3] = S_FROM_CV.T
        mount = hand_from_tcp@optical@cv_from_s
        camera = ET.SubElement(root, 'link', name='camera_link')
        inertial = ET.SubElement(camera, 'inertial')
        ET.SubElement(inertial, 'origin', xyz='-0.0125 -0.02 0', rpy='0 0 0')
        ET.SubElement(inertial, 'mass', value='0.08')
        ET.SubElement(inertial, 'inertia', ixx='0.00006888', ixy='0', ixz='0',
                      iyy='0.00000621', iyz='0', izz='0.00006802')
        for tag in ('visual', 'collision'):
            shape = ET.SubElement(camera, tag)
            ET.SubElement(shape, 'origin', xyz='-0.0125 -0.02 0', rpy='0 0 0')
            ET.SubElement(ET.SubElement(shape, 'geometry'), 'box', size='0.02005 0.099 0.023')
        joint = ET.SubElement(root, 'joint', name='camera_joint', type='fixed')
        ET.SubElement(joint, 'parent', link=spec['hand'])
        ET.SubElement(joint, 'child', link='camera_link')
        rpy = [np.pi, -np.pi/2, 0.] if camera_extrinsic is None else Rotation.from_matrix(mount[:3, :3]).as_euler('xyz')
        ET.SubElement(joint, 'origin', xyz=' '.join(map(str, mount[:3, 3])), rpy=' '.join(map(str, rpy)))
    return root
