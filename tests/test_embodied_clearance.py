import unittest

import torch
from torch.nn import functional as F

from tsn.models.compact_policy import CompactRouteHead
from tsn.models.geometric_energy import RouteTrust
from tsn.models.kinematics import PandaKinematics
from tsn.models.uncertain_clearance import PointUncertainty,UncertainClearancePolicy
from tsn.models.embodied_clearance import (
    PandaToolGeometry,EmbodiedClearancePolicy,BodyFilteredGeometry,
    box_signed_distance,embodied_surface_risk,wrist_pose_candidates,
)


class EmbodiedClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.geometry = PandaToolGeometry()

    def test_box_distance_is_signed_exact_and_rotation_covariant(self):
        points = torch.tensor([[0.,0.,0.],[2.,2.,0.]])
        centers = torch.zeros(1,3); half = torch.ones(1,3)
        identity = torch.eye(3)[None]
        expected = torch.tensor([[-1.],[2.**.5]])
        torch.testing.assert_close(box_signed_distance(points,centers,identity,half),expected)
        r = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        shift = torch.tensor([.4,-.2,.3])
        changed = box_signed_distance(points@r.T+shift,centers+shift,r[None],half)
        torch.testing.assert_close(changed,expected)

    def test_representation_includes_palm_width_and_additional_rigid_tool_bodies(self):
        points = torch.tensor([[.01,.07,-.08],[.01,0.,-.13]])
        hand = self.geometry.distance(points,torch.zeros(2))
        tool = self.geometry.distance(points,torch.zeros(2),tool=True)
        self.assertLess(float(hand[0]),.03)
        self.assertGreater(float(points[0].norm()),.10)
        self.assertTrue((tool<=hand).all())
        self.assertEqual(len(self.geometry.box_centers),9)
        self.assertEqual(int(self.geometry.finger_mask.sum()),8)
        self.assertGreater(len(self.geometry.palm_planes),10)
        self.assertGreater(len(self.geometry.wrist_planes),10)

    def test_finger_geometry_moves_with_measured_aperture(self):
        opening = torch.tensor([.04,0.])
        point = self.geometry.box_centers[3]+torch.einsum('f,fi->i',opening,self.geometry.box_motion[3])
        closed = self.geometry.distance(point[None],torch.zeros(2))
        opened = self.geometry.distance(point[None],opening)
        self.assertGreater(float(closed[0]),.01)
        self.assertEqual(float(opened[0]),0.)

    def test_tcp_only_never_queries_embodiment_and_matches_single_point_equation(self):
        class Forbidden:
            def distance(self,*args,**kwargs):raise AssertionError('Embodiment leaked into TCP control')
        candidates = torch.zeros(1,14,30,3)
        rotation = torch.eye(3).expand(1,30,3,3)
        tcp = torch.eye(4)[None]
        points = torch.tensor([[.1,0.,0.]])
        padding = torch.tensor([.02])
        risk,base,body = embodied_surface_risk(candidates,rotation,tcp,points,padding,Forbidden(),None,'tcp')
        expected = torch.exp(torch.tensor(-.5*(.08/.04)**2))
        torch.testing.assert_close(risk,torch.full_like(risk,expected))
        torch.testing.assert_close(risk,base,rtol=0,atol=0)
        torch.testing.assert_close(risk,body,rtol=0,atol=0)
        zero = embodied_surface_risk(candidates,rotation,tcp,points,padding,Forbidden(),None,'tool',0.)[0]
        torch.testing.assert_close(risk,zero,rtol=0,atol=0)

    def test_embodiment_risk_weight_and_global_pose_invariance(self):
        candidates = torch.zeros(1,14,30,3)
        rotation = torch.eye(3).expand(1,30,3,3)
        tcp = torch.eye(4)[None]
        points = torch.tensor([[.01,.07,-.08]])
        padding = torch.tensor([.015]); fingers = torch.zeros(2)
        risks = [embodied_surface_risk(candidates,rotation,tcp,points,padding,
                    self.geometry,fingers,'tool',weight)[0] for weight in (0.,.35,1.)]
        self.assertTrue((risks[1]>risks[0]).all())
        self.assertTrue((risks[2]>=risks[1]).all())
        r = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        shift = torch.tensor([.4,-.2,.3])
        tcp[:,:3,3] += shift
        changed = embodied_surface_risk(candidates@r.T+shift,r[None,None]@rotation,tcp,
                    points@r.T+shift,padding,self.geometry,fingers,'tool',.35)[0]
        torch.testing.assert_close(changed,risks[1],rtol=1e-5,atol=1e-6)

    def test_empty_invalid_surfaces_have_zero_risk(self):
        candidates = torch.zeros(1,14,30,3)
        rotation = torch.eye(3).expand(1,30,3,3)
        tcp = torch.eye(4)[None]
        for points,padding in ((torch.empty(0,3),torch.empty(0)),
                               (torch.full((1,3),float('nan')),torch.tensor([.03]))):
            for risk in embodied_surface_risk(candidates,rotation,tcp,points,padding,self.geometry,torch.zeros(2),'tool'):
                torch.testing.assert_close(risk,torch.zeros_like(risk))

    def test_signed_cost_retains_penetration_and_uses_same_tcp_equation(self):
        point = torch.tensor([[0.,0.,-.08]])
        unsigned = self.geometry.distance(point,torch.zeros(2),tool=True)
        signed = self.geometry.distance(point,torch.zeros(2),tool=True,signed=True)
        self.assertEqual(float(unsigned[0]),0.)
        self.assertLess(float(signed[0]),-.001)
        candidates = torch.zeros(1,14,30,3)
        rotation = torch.eye(3).expand(1,30,3,3)
        tcp = torch.eye(4)[None]
        risk,_,body = embodied_surface_risk(candidates,rotation,tcp,point,torch.zeros(1),
            self.geometry,torch.zeros(2),'tool',1.,field='signed')
        self.assertTrue((body>1.).all())
        self.assertTrue(torch.isfinite(risk).all())
        control = embodied_surface_risk(candidates,rotation,tcp,torch.tensor([[.1,0.,0.]]),
            torch.tensor([.02]),None,None,'tcp',field='signed')[0]
        expected = F.softplus(torch.tensor(-2.))/torch.log(torch.tensor(2.))
        torch.testing.assert_close(control,torch.full_like(control,expected))

    def test_named_part_features_recover_union_and_retain_contact_distribution(self):
        points = torch.tensor([[0.,0.,-.08],[.1,.1,.1],[.01,.07,-.08]])
        fingers = torch.tensor([.02,.02])
        for tool,count in ((False,4),(True,6)):
            parts = self.geometry.part_distance(points,fingers,tool=tool,signed=True)
            self.assertEqual(parts.shape,(3,count))
            union = self.geometry.distance(points,fingers,tool=tool,signed=True)
            torch.testing.assert_close(parts.amin(-1),union)
        candidates = torch.zeros(1,14,30,3)
        rotation = torch.eye(3).expand(1,30,3,3);tcp = torch.eye(4)[None]
        union = embodied_surface_risk(candidates,rotation,tcp,points,torch.zeros(3),
            self.geometry,fingers,'tool',1.)[0]
        parts = embodied_surface_risk(candidates,rotation,tcp,points,torch.zeros(3),
            self.geometry,fingers,'tool',1.,representation='parts')[0]
        self.assertTrue((parts<union).all())
        control = embodied_surface_risk(candidates,rotation,tcp,points,torch.zeros(3),None,None,
            'tcp',representation='parts')[0]
        original = embodied_surface_risk(candidates,rotation,tcp,points,torch.zeros(3),None,None,'tcp')[0]
        torch.testing.assert_close(control,original,rtol=0,atol=0)

    def test_self_surfaces_are_removed_before_persistent_map_insertion(self):
        memory = BodyFilteredGeometry(self.geometry,True)
        tcp = torch.eye(4)
        tcp[:3,3] = torch.tensor([.55,0.,.30])
        memory.set_pose(tcp,torch.zeros(2))
        points = torch.tensor([[.01,.07,-.08],[.1,.3,0.]])+tcp[:3,3]
        memory.update(points,torch.ones(2,dtype=torch.bool),0,tcp[:3,3],torch.tensor([.8,.3,.3]))
        self.assertTrue(bool(memory.last_self_mask[0]))
        self.assertFalse(bool(memory.last_self_mask[1]))
        torch.testing.assert_close(memory.points,points[1:])

    def test_wrist_candidates_preserve_zero_and_change_lateral_body_risk(self):
        candidates = torch.zeros(1,14,30,3)
        rotation = torch.eye(3).expand(1,30,3,3)
        position,rotations,offsets,angles = wrist_pose_candidates(candidates,rotation,torch.zeros(14,3),torch.ones(1))
        self.assertEqual(position.shape,(1,70,30,3))
        torch.testing.assert_close(position[:,0],candidates[:,0],rtol=0,atol=0)
        torch.testing.assert_close(rotations[:,0],rotation,rtol=0,atol=0)
        orthogonal = rotations.transpose(-1,-2)@rotations
        torch.testing.assert_close(orthogonal,torch.eye(3).expand_as(orthogonal))
        risk,_,_ = embodied_surface_risk(position,rotations,torch.eye(4)[None],
            torch.tensor([[0.,.11,-.075]]),torch.zeros(1),self.geometry,torch.zeros(2),'hand')
        self.assertGreater(float(risk.max()-risk.min()),.001)
        faded = wrist_pose_candidates(candidates,rotation,torch.zeros(14,3),torch.zeros(1))[1]
        torch.testing.assert_close(faded,rotation[:,None].expand_as(faded),rtol=0,atol=0)
        wide = wrist_pose_candidates(candidates,rotation,torch.zeros(14,3),torch.ones(1),wide=True)
        self.assertEqual(wide[0].shape,(1,98,30,3))
        self.assertLessEqual(float(wide[3].abs().max()),torch.pi/2+1e-6)

    def test_complete_tool_volume_and_wide_posture_commands_are_finite(self):
        class Backbone(torch.nn.Module):
            def forward(self,rgb,*args,**kwargs):
                dense = torch.zeros(1,6,20,20)
                dense[:,1] = torch.linspace(-.3,.3,20)[None,None]
                dense[:,2] = .2
                output = (torch.zeros(1,30,7),torch.ones(1,16,768),
                          F.adaptive_avg_pool2d(dense,(4,4)).flatten(2).transpose(1,2))
                return (*output,dense) if kwargs.get('return_maps') else output
        torch.manual_seed(8)
        policy = EmbodiedClearancePolicy(Backbone(),CompactRouteHead(),PandaKinematics(),
            RouteTrust(),PointUncertainty(),'tool',.35,mask_self=True,pose_mode='wide_roll').eval()
        state = torch.zeros(1,16)
        state[:,:7] = torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])/torch.pi
        state[:,7:9] = .5;state[:,9:12] = torch.tensor([[.15,.4,.2]])
        with torch.inference_mode():
            for step in (0,15,30):
                policy.observe_step(step)
                output = policy(torch.zeros(1,1,1,3,dtype=torch.uint8),state,torch.eye(3)[None],torch.eye(4)[None])
                self.assertEqual(output.shape,(1,30,7))
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(policy.diagnostics[-1]['pose_mode'],'wide_roll')
        self.assertIsInstance(policy.geometry_memory,BodyFilteredGeometry)
        policy.reset_episode()
        self.assertIsNone(policy.geometry_memory.last_self_mask)

    def test_axial_mode_matches_C3_complete_commands_and_reset(self):
        class Backbone(torch.nn.Module):
            def forward(self,rgb,*args,**kwargs):
                index = float(rgb[0,0,0,0])
                dense = torch.zeros(1,6,20,20)
                dense[:,0] = index*.008
                dense[:,1] = torch.linspace(-.3,.3,20)[None,None]
                dense[:,2] = .2
                geometry = F.adaptive_avg_pool2d(dense,(4,4)).flatten(2).transpose(1,2)
                output = (torch.zeros(1,30,7),torch.ones(1,16,768)*index,geometry)
                return (*output,dense) if kwargs.get('return_maps') else output
        torch.manual_seed(8)
        modules = (Backbone(),CompactRouteHead(),PandaKinematics(),RouteTrust(),PointUncertainty())
        current = EmbodiedClearancePolicy(*modules,'axial',.7,max_padding=.03).eval()
        reference = UncertainClearancePolicy(*modules,'adaptive',max_padding=.03).eval()
        state = torch.zeros(1,16)
        state[:,:7] = torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])/torch.pi
        state[:,7:9] = .5
        state[:,9:12] = torch.tensor([[.15,.4,.2]])
        with torch.inference_mode():
            for step in range(0,180,15):
                rgb = torch.full((1,1,1,3),step//15,dtype=torch.uint8)
                outputs = []
                for policy in (reference,current):
                    policy.observe_step(step)
                    outputs.append(policy(rgb,state,torch.eye(3)[None],torch.eye(4)[None]))
                torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        torch.testing.assert_close(current.finger_positions,torch.full((2,),.02))
        current.reset_episode()
        self.assertIsNone(current.finger_positions)

    def test_tcp_complete_commands_ignore_self_mask_and_posture_options(self):
        class Backbone(torch.nn.Module):
            def forward(self,rgb,*args,**kwargs):
                dense = torch.zeros(1,6,20,20)
                dense[:,1] = torch.linspace(-.3,.3,20)[None,None]
                dense[:,2] = .2
                output = (torch.zeros(1,30,7),torch.ones(1,16,768),
                          F.adaptive_avg_pool2d(dense,(4,4)).flatten(2).transpose(1,2))
                return (*output,dense) if kwargs.get('return_maps') else output
        torch.manual_seed(8)
        modules = (Backbone(),CompactRouteHead(),PandaKinematics(),RouteTrust(),PointUncertainty())
        original = EmbodiedClearancePolicy(*modules,'tcp',0.).eval()
        control = EmbodiedClearancePolicy(*modules,'tcp',0.,mask_self=True,pose_mode='wide_roll').eval()
        state = torch.zeros(1,16)
        state[:,:7] = torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])/torch.pi
        state[:,7:9] = .5;state[:,9:12] = torch.tensor([[.15,.4,.2]])
        with torch.inference_mode():
            for step in range(0,180,15):
                outputs = []
                for policy in (original,control):
                    policy.observe_step(step)
                    outputs.append(policy(torch.zeros(1,1,1,3,dtype=torch.uint8),state,torch.eye(3)[None],torch.eye(4)[None]))
                torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)

    def test_invalid_body_weight_rejected(self):
        modules = [torch.nn.Identity() for _ in range(5)]
        for weight in (float('nan'),-.01,1.01):
            with self.assertRaises(ValueError):
                EmbodiedClearancePolicy(*modules,body_weight=weight)


if __name__=='__main__':
    unittest.main()
