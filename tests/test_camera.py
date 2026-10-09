"""Calibration covariance, resolution changes, and movable physical cameras."""
import unittest

import torch

from tsn.features.camera import camera_rays, depth_to_base, resize_intrinsics, resize_observation
from tsn.features.maps import GeometryMaps
from tsn.models.c2_embodiment import EmbodimentGeometry
from tsn.simulation.robot import default_camera_extrinsic, S_FROM_CV


class CameraTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.K = torch.tensor([[[110., 0., 160.], [0., 113., 120.], [0., 0., 1.]]])

    def test_half_pixel_resize_preserves_every_output_ray(self):
        resized = resize_intrinsics(self.K, (240, 320), (480, 640))
        torch.testing.assert_close(camera_rays(self.K, (240, 320), (80, 80)),
                                   camera_rays(resized, (480, 640), (80, 80)))
        # Off-center, anisotropic intrinsics change the actual pixel rays.
        changed = self.K.clone(); changed[:, 0, 0] *= 1.5; changed[:, 1, 2] -= 12
        self.assertGreater(float((camera_rays(changed, (240, 320), (80, 80))-
                                  camera_rays(self.K, (240, 320), (80, 80))).abs().max()), .1)

    def test_ray_geometry_is_pose_covariant_and_matches_unclipped_teacher(self):
        depth = torch.full((1, 80, 80), .7, requires_grad=True)
        pose = torch.eye(4)[None]
        base = depth_to_base(depth, self.K, (240, 320), pose)
        other = pose.clone()
        other[0, :3, :3] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        other[0, :3, 3] = torch.tensor([.4, -.1, .2])
        transformed = depth_to_base(depth, self.K, (240, 320), other)
        torch.testing.assert_close(transformed, base@other[0, :3, :3].T+other[0, :3, 3])
        maps = GeometryMaps(dict(clip_points=False))
        target = maps(torch.full((1, 240, 320), .7), self.K, other, torch.zeros(1, 3),
                      torch.zeros(1, 30, 3), torch.ones(1, 30, dtype=torch.bool))
        teacher = target[:, :3].permute(0, 2, 3, 1)*maps.scale+maps.center
        torch.testing.assert_close(transformed, teacher)
        transformed.square().mean().backward()
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(float(depth.grad.abs().sum()), 0)

    def test_mixed_resolution_observations_stack_with_registered_calibration(self):
        rows = []
        for hw in ((192, 256), (240, 424), (360, 480)):
            K = resize_intrinsics(self.K[0], (240, 320), hw)
            row = dict(depth=torch.full(hw, .7), rgb=torch.full((*hw, 3), 128, dtype=torch.uint8), K=K)
            rows.append(resize_observation(row, (192, 256)))
        images = torch.stack([row['rgb'] for row in rows])
        self.assertEqual(images.shape, (3, 192, 256, 3))
        rays = camera_rays(torch.stack([row['K'] for row in rows]), (192, 256), (80, 80))
        torch.testing.assert_close(rays, rays[:1].expand_as(rays))

    def test_fr3_camera_housing_tracks_calibration_without_changing_other_regions(self):
        mount = torch.tensor(default_camera_extrinsic(), dtype=torch.float32)
        rotation = mount[:3, :3]@torch.tensor(S_FROM_CV.T, dtype=torch.float32)
        center = rotation@torch.tensor([-.0125, -.02, 0.])+mount[:3, 3]
        points = center[None]
        for robot in ('panda', 'fr3'):
            with self.subTest(robot=robot):
                body = EmbodimentGeometry(robot)
                before = body.signed_distances(points, torch.full((2,), .02), mount)
                shifted = mount.clone(); shifted[0, 3] += .1
                after = body.signed_distances(points, torch.full((2,), .02), shifted)
                self.assertLess(float(before[0, 5]), 0)
                self.assertGreater(float(after[0, 5]), .05)
                torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)

    def test_panda_physical_camera_link_matches_optical_calibration(self):
        from tsn.simulation.robot import robot_tree
        from tsn.models.c2_embodiment import origin_transform
        mount = torch.tensor(default_camera_extrinsic(), dtype=torch.float32)
        mount[:3, 3] += torch.tensor([-.04, .01, .02])
        tree = robot_tree('panda', mount.numpy())
        actual = (origin_transform(tree.find("joint[@name='realsense_joint']/origin"))@
                  origin_transform(tree.find("joint[@name='camera_link_joint']/origin")))
        expected = mount.numpy().copy()
        expected[:3, :3] = expected[:3, :3]@S_FROM_CV.T
        expected[2, 3] += .1034
        torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected, dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
