"""Intervene only when the proposed swept motion has material surface proximity."""
from tsn.models.clearance_policy import ClearancePolicy


class TriggeredClearancePolicy(ClearancePolicy):
    """Apply clearance corrections only above a baseline proximity threshold."""
    def __init__(self, *args, trigger=.15, **kwargs):
        """Configure the risk threshold while retaining the underlying refiner.

        Args:
            *args (tuple): ClearancePolicy positional arguments: backbone,
                head, kinematics, and optional clearance settings.
            trigger (float): Threshold in [0, 1] on the original route's
                dimensionless Gaussian proximity score; not a probability.
            **kwargs (dict[str, object]): Named ClearancePolicy arguments,
                including mode, margin in metres, uncertainty, and penalty.

        Returns:
            None. Initializes inherited memory and stores the trigger threshold.

        Raises:
            ValueError: Trigger is outside [0, 1].
        """
        if not 0 <= trigger <= 1:
            raise ValueError('Trigger must lie in [0, 1]')
        self.trigger = trigger
        super().__init__(*args, **kwargs)
        self.schedule += f'_trigger{trigger}'

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        """Keep the original route unless its proximity score reaches trigger.

        Args:
            waypoints (torch.Tensor): Floating base XYZ targets (1, 30, 3), m.
            rotation (torch.Tensor): Floating TCP-to-base rotations
                (1, 30, 3, 3), used to locate hand proxy samples.
            tcp (torch.Tensor): Floating current TCP-to-base pose (1, 4, 4).
            near (torch.Tensor): Boolean servo mask (1,), passed to the refiner.
            pose (torch.Tensor): Floating camera-to-base pose (1, 4, 4).

        Returns:
            torch.Tensor: Floating selected waypoints (1, 30, 3), metres.
            Returns the original object below threshold; otherwise returns the
            parent's corrected route. Records trigger_active and the proposed
            correction magnitude, then updates the executed choice/correction
            diagnostics when the trigger suppresses an intervention.
        """
        corrected = super().refine_waypoints(waypoints, rotation, tcp, near, pose)
        active = self.last_risk >= self.trigger
        self.diagnostics[-1]['trigger_active'] = active
        self.diagnostics[-1]['proposed_correction_m'] = self.diagnostics[-1]['correction_m']
        if not active:
            self.diagnostics[-1]['choice'] = 0
            self.diagnostics[-1]['correction_m'] = 0.
            return waypoints
        return corrected
