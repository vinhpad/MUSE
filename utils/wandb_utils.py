import os


class WandbLogger:
    def __init__(self, enabled, project, entity, config):
        self.enabled = enabled and self._wandb_available()
        if not self.enabled:
            return
        import wandb

        run_name = f"{config.get('mode', 'unknown')}_{config.get('dataset', 'unknown')}_seed{config.get('rnd_seed', 0)}"
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=self._sanitize_config(config),
            dir=os.path.join(os.getcwd(), "results", "wandb"),
        )

    @staticmethod
    def _wandb_available():
        try:
            import wandb
            return True
        except ImportError:
            return False

    @staticmethod
    def _sanitize_config(config):
        safe = {}
        for k, v in config.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                safe[k] = v
            elif isinstance(v, (list, tuple)):
                safe[k] = str(v)
        return safe

    def log(self, data, step=None, commit=None):
        if not self.enabled:
            return
        import wandb
        wandb.log(data, step=step, commit=commit)

    def finish(self):
        if not self.enabled:
            return
        import wandb
        wandb.finish()
