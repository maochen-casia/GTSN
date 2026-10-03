"""Checks for the revision's causal decoder and kinematic control boundary."""
import unittest
import torch
from tsn.models.cartesian_policy import CartesianHead, CartesianPolicy, GoalServoPolicy, PandaKinematics
from tsn.models.gtsn_policy import geometry_nll


class CartesianTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def sample(self):
        return [torch.randn(2,4,16,768),torch.randn(2,4,16,6),
                torch.eye(4).expand(2,4,4,4).clone(),torch.tensor([[0,0,15,0]]*2),
                torch.tensor([[False,False,True,True]]*2),torch.randn(2,16),
                torch.eye(4).expand(2,4,4).clone()]

    def test_padding_is_inert_and_variance_has_proper_supervision(self):
        model=CartesianHead('memory').eval();args=self.sample()
        original,aux=model(*args)
        args=[x.clone() for x in args]
        args[0][:,:2]*=1000;args[1][:,:2]*=1000
        changed,_=model(*args)
        torch.testing.assert_close(original,changed)
        loss=geometry_nll(aux,torch.zeros(2,16,3),torch.ones(2,16,dtype=torch.bool))
        loss.backward()
        self.assertGreater(model.distribution.weight.grad.abs().sum().item(),0)

    def test_single_frame_ignores_history_and_state_ignores_visuals(self):
        for mode in ('state','visual','uncertainty'):
            model=CartesianHead(mode).eval();args=self.sample()
            original,_=model(*args)
            args[0][:,:-1]*=1000;args[1][:,:-1]*=1000
            if mode=='state':args[0]*=1000;args[1]*=1000
            changed,_=model(*args)
            torch.testing.assert_close(original,changed)

    def test_ik_reaches_nearby_target_with_joint_limits(self):
        kin=PandaKinematics()
        q=torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])
        pose=kin(q)
        target=pose[:,:3,3]+torch.tensor([[.02,-.03,.015]])
        result=kin.inverse(q,target,pose[:,:3,:3])
        self.assertLess((kin(result)[:,:3,3]-target).norm().item(),1e-4)
        self.assertTrue((result>=kin.limits[:,0]).all())
        self.assertTrue((result<=kin.limits[:,1]).all())

    def test_shared_servo_passes_far_actions_and_reaches_near_goal(self):
        class Constant(torch.nn.Module):
            def forward(self,*args):return torch.ones(1,30,7)*.01
        kin=PandaKinematics();q=torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])
        state=torch.zeros(1,16);state[:,:7]=q/torch.pi
        rgb=torch.zeros(1,1,1,3,dtype=torch.uint8);K=torch.eye(3)[None];pose=torch.eye(4)[None]
        policy=GoalServoPolicy(Constant(),kin)
        center=torch.tensor([.65,0,.22]);scale=torch.tensor([.55,.55,.5])
        current=kin(q)[:,:3,3]
        state[:,9:12]=(current+torch.tensor([[.2,0,0]])-center)/scale
        torch.testing.assert_close(policy(rgb,state,K,pose),torch.ones(1,30,7)*.01)
        goal=current+torch.tensor([[.02,-.01,.015]])
        state[:,9:12]=(goal-center)/scale
        chunk=policy(rgb,state,K,pose)
        self.assertLess((kin(q+chunk[:,-1])[:,:3,3]-goal).norm().item(),1e-4)

    def test_cartesian_tracking_preserves_predicted_wrist_orientation(self):
        kin=PandaKinematics();q=torch.tensor([[0.,.4,0.,-1.96,0.,2.35,.78]])
        base=torch.linspace(1/30,1,30)[None,:,None]*torch.tensor([.03,-.02,.01,0,.02,-.01,.04])[None,None]
        desired=kin(q[:,None]+base)
        delta=desired[:,4::5,:3,3]-kin(q)[:,None,:3,3]
        class Perception(torch.nn.Module):
            def forward(self,*args,**kwargs):return base,torch.zeros(1,16,768),torch.zeros(1,16,6)
        class Head(torch.nn.Module):
            def forward(self,*args):return delta,{'risk':torch.zeros(1)}
        policy=CartesianPolicy(Perception(),Head(),kin,servo_radius=0,orientation='baseline')
        state=torch.zeros(1,16);state[:,:7]=q/torch.pi
        chunk=policy(torch.zeros(1,1,1,3,dtype=torch.uint8),state,torch.eye(3)[None],torch.eye(4)[None])
        actual=kin(q[:,None]+chunk)
        self.assertLess((actual[:,:,:3,:3]-desired[:,:,:3,:3]).abs().max().item(),1e-3)
        self.assertLess((actual[:,4::5,:3,3]-desired[:,4::5,:3,3]).abs().max().item(),1e-4)

if __name__=='__main__':unittest.main()
