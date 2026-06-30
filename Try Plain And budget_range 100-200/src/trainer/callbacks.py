from stable_baselines3.common.callbacks import BaseCallback

class GammaSchedule(BaseCallback):
    def __init__(self, base=0.95, target=0.99, decay_steps=300000, verbose=1):
        super().__init__(verbose)
        self.base = base
        self.target = target
        self.decay_steps = decay_steps

    def _on_rollout_start(self) -> None:
        """Triggers safely at the boundary of a new iteration buffer collection."""
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_gamma = self.base + (self.target - self.base) * progress
        self.model.gamma = new_gamma
        
        if self.verbose > 0:
            print(f"[Step {self.num_timesteps}] Stepped baseline Gamma to: {self.model.gamma:.4f}")

    def _on_step(self) -> bool:
        return True