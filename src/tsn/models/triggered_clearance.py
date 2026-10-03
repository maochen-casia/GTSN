"""Intervene only when the proposed swept motion has material surface proximity."""
from tsn.models.clearance_policy import ClearancePolicy


class TriggeredClearancePolicy(ClearancePolicy):
    def __init__(self, *args, trigger=.15, **kwargs):
        if not 0 <= trigger <= 1:
            raise ValueError('Trigger must lie in [0, 1]')
        self.trigger = trigger
        super().__init__(*args, **kwargs)
        self.schedule += f'_trigger{trigger}'

    def refine_waypoints(self, waypoints, rotation, tcp, near, pose):
        corrected = super().refine_waypoints(waypoints, rotation, tcp, near, pose)
        active = self.last_risk >= self.trigger
        self.diagnostics[-1]['trigger_active'] = active
        self.diagnostics[-1]['proposed_correction_m'] = self.diagnostics[-1]['correction_m']
        if not active:
            self.diagnostics[-1]['choice'] = 0
            self.diagnostics[-1]['correction_m'] = 0.
            return waypoints
        return corrected
