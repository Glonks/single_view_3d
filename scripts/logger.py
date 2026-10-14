import os
import json
import time
import wandb

from dataclasses import asdict


class Logger:
    def __init__(self, project, config, wandb_mode='online'):
        self.run_dir = os.path.join('runs', config.name)
        if os.path.exists(self.run_dir):
            self.run_dir = f'{self.run_dir}_{time.strftime("%Y%m%d_%H%M%S")}'

        os.makedirs(self.run_dir)

        with open(os.path.join(self.run_dir, 'config.json'), 'w') as file:
            json.dump(asdict(config), file, indent=2)

        wandb.init(
            project=project,
            name=os.path.basename(self.run_dir),
            config=asdict(config),
            dir=self.run_dir,
            mode=wandb_mode
        )

        self.metrics_file = open(os.path.join(self.run_dir, 'metrics.jsonl'), 'a')
        self.last_overwrite = False

    def log(self, metrics: dict, step: int, overwrite=False):
        if self.last_overwrite and not overwrite:
            print()
        self.last_overwrite = overwrite

        parts = [f'Step: {step}']
        for key, value in metrics.items():
            if isinstance(value, float):
                format_spec = '.4f'
            else:
                format_spec = ''

            parts.append(f'{key}: {value:{format_spec}}')

        log_string = ' | '.join(parts)

        if overwrite:
            print('\r' + log_string, end='', flush=True)
        else:
            print(log_string, flush=True)

        wandb.log(metrics, step=step)

        self.metrics_file.write(json.dumps({'step': step, **metrics}) + '\n')
        self.metrics_file.flush()

    def path(self, file_name):
        return os.path.join(self.run_dir, file_name)

    def finish(self):
        self.metrics_file.close()

        wandb.finish()
