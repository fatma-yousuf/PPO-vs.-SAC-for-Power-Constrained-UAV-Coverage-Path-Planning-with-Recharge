from stable_baselines3.common.callbacks import BaseCallback

class GammaSchedule(BaseCallback):
    def __init__(self, base=0.95, target=0.999, decay_steps=300_000,decay_rate = 0.1, verbose=1): # decay steps are adjusted based on trainng steps
        super().__init__(verbose)
        self.base = base
        self.target = target
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def _gamma(self, step):
       ## equation for exponential decay of gamma from base to target over decay_steps equ19 in oringal paper
        return 1.0 - (1.0 - self.base) * ( self.decay_rate ** (step / self.decay_steps))

    def _on_rollout_start(self) -> None:
        new_gamma = self._gamma(self.num_timesteps)
        new_gamma = min(new_gamma, self.target)  # Ensure gamma does not exceed the target
        self.model.gamma = new_gamma
        
        if self.verbose > 0:
            print(f"[Step {self.num_timesteps}] Stepped baseline Gamma to: {self.model.gamma:.4f}")

    def _on_step(self) -> bool:
        return True
    
class DualLRCallback(BaseCallback):
    """
    Applies separate, exponentially-decaying learning rates to the actor
    (policy) and critic (value function) parameter groups inside PPO's
    single optimizer.

    SB3's PPO uses one Adam optimizer for the entire policy network.
    This callback splits the parameters by name at the first step and
    then updates each group's lr independently every rollout.
    """

    def __init__(self, actor_lr_config: dict, critic_lr_config: dict, verbose=1):
        """
        Args:
            actor_lr_config:  {"base": 3e-5, "decay_rate": 0.1, "decay_steps": 5_000_000}
            critic_lr_config: {"base": 1e-4, "decay_rate": 0.1, "decay_steps": 5_000_000}
        """
        super().__init__(verbose)
        self.actor_cfg = actor_lr_config
        self.critic_cfg = critic_lr_config
        self._split_done = False

    # Internal helpers
    def _split_param_groups(self):
        """
        Replace the optimizer's single param group with two groups:
        one for actor params, one for critic (value net) params.
        Called once on the very first step.
        """
        optimizer = self.model.policy.optimizer

        # Collect all parameters with their names
        all_params = list(self.model.policy.named_parameters())

        actor_params  = [p for name, p in all_params if "mlp_extractor.policy_net" in name
                                                      or "action_net" in name]
        critic_params = [p for name, p in all_params if "mlp_extractor.value_net" in name
                                                      or "value_net" in name]

        # Params that belong to neither (e.g. shared CNN / feature extractor)
        actor_ids  = {id(p) for p in actor_params}
        critic_ids = {id(p) for p in critic_params}
        shared_params = [p for _, p in all_params
                         if id(p) not in actor_ids and id(p) not in critic_ids]

        # Rebuild optimizer param groups
        # Shared layers get the actor LR (matches original paper convention)
        optimizer.param_groups = []
        optimizer.add_param_group({"params": shared_params + actor_params,
                                   "lr": self.actor_cfg["base"]})
        optimizer.add_param_group({"params": critic_params,
                                   "lr": self.critic_cfg["base"]})

        if self.verbose > 0:
            print(f"[DualLR] Split optimizer: "
                  f"{len(shared_params)+len(actor_params)} actor/shared params, "
                  f"{len(critic_params)} critic params")

        self._split_done = True

    def _decayed_lr(self, cfg: dict, step: int) -> float:
        """Exponential decay: lr = base * decay_rate^(step / decay_steps)"""
        return cfg["base"] * (cfg["decay_rate"] ** (step / cfg["decay_steps"]))

    # SB3 callback hooks
    def _on_training_start(self) -> None:
        self._split_param_groups()

    def _on_rollout_start(self) -> None:
        if not self._split_done:
            self._split_param_groups()

        step = self.num_timesteps
        actor_lr  = self._decayed_lr(self.actor_cfg,  step)
        critic_lr = self._decayed_lr(self.critic_cfg, step)

        optimizer = self.model.policy.optimizer
        optimizer.param_groups[0]["lr"] = actor_lr   # actor + shared
        optimizer.param_groups[1]["lr"] = critic_lr  # critic

        if self.verbose > 0:
            print(f"[Step {step}] actor_lr={actor_lr:.2e}  critic_lr={critic_lr:.2e}")

    def _on_step(self) -> bool:
        return True
