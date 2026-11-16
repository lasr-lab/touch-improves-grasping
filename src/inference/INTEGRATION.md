# Integrating BESO Policies into LeRobot

This repo exposes BESO policies from the `policy` package:

- `policy.beso`: standard BESO, registered in LeRobot as `beso`
- `policy.beso_tactile`: tactile BESO, registered in LeRobot as `beso_tactile`

## 1. Install Editable Packages

```bash
cd /path/to/digit260-robotlearning-benchmarking
pip install -e .

cd /path/to/lerobot
pip install -e .
```

For tactile BESO with `tactile_encoder_type="d360"`, install the Sparsh Digit 360 encoder into the same Conda environment:

```bash
git clone https://github.com/facebookresearch/sparsh-multisensory-touch.git
cd sparsh-multisensory-touch
pip install -e .
```

Sanity check:

```bash
python -c "from policy.beso import BESOConfig, BESOPolicy; from policy.beso_tactile import BESOConfig as BESOTactileConfig; print('ok')"
```

## 2. LeRobot Registration

LeRobot needs to import both external policy configs and dispatch them in `lerobot/src/lerobot/policies/factory.py`.

Config exports in `lerobot/src/lerobot/policies/__init__.py`:

```python
from policy.beso.configuration_beso import BESOConfig as BESOConfig
from policy.beso_tactile.configuration_beso import BESOConfig as BESOTactileConfig
```

Factory imports:

```python
from policy.beso.configuration_beso import BESOConfig
from policy.beso_tactile.configuration_beso import BESOConfig as BESOTactileConfig
```

Policy class dispatch:

```python
elif name == "beso":
    from policy.beso.modeling_beso import BESOPolicy
    return BESOPolicy
elif name == "beso_tactile":
    from policy.beso_tactile.modeling_beso import BESOPolicy
    return BESOPolicy
```

Config dispatch:

```python
elif policy_type == "beso":
    return BESOConfig(**kwargs)
elif policy_type == "beso_tactile":
    return BESOTactileConfig(**kwargs)
```

Processor dispatch:

```python
elif isinstance(policy_cfg, BESOConfig):
    from policy.beso.processor_beso import make_beso_pre_post_processors
    processors = make_beso_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
elif isinstance(policy_cfg, BESOTactileConfig):
    from policy.beso_tactile.processor_beso import make_beso_pre_post_processors
    processors = make_beso_pre_post_processors(config=policy_cfg, dataset_stats=kwargs.get("dataset_stats"))
```

## 3. Verify

```python
from lerobot.policies.factory import get_policy_class, make_policy_config

beso_cfg = make_policy_config("beso")
beso_policy = get_policy_class("beso")

tactile_cfg = make_policy_config("beso_tactile")
tactile_policy = get_policy_class("beso_tactile")

assert beso_policy.__name__ == "BESOPolicy"
assert tactile_policy.__name__ == "BESOPolicy"
```

## 4. Evaluation

At inference time, both variants expect:

- an initialized policy in eval mode
- the saved preprocessor and postprocessor from the checkpoint
- an observation dict containing the same keys used during training
- `agent.reset()` at the beginning of every episode or rollout

For standard BESO:

```python
from policy.beso.modeling_beso import BESOPolicy
```

For tactile BESO:

```python
from policy.beso_tactile.modeling_beso import BESOPolicy
```

Minimal checkpoint loading:

```python
import os

import torch
from safetensors.torch import load_file

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from policy.beso.modeling_beso import BESOPolicy

CHECKPOINT_PATH = "/path/to/checkpoint/pretrained_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = PreTrainedConfig.from_pretrained(pretrained_name_or_path=CHECKPOINT_PATH)
agent = BESOPolicy(config)
state_dict = load_file(os.path.join(CHECKPOINT_PATH, "model.safetensors"), device=DEVICE)
agent.load_state_dict(state_dict, strict=False)
agent = agent.to(DEVICE)
agent.eval()

preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=CHECKPOINT_PATH)
agent.reset()
```
