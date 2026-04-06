
## How does LaRA-VLA keep model development modular?

LaRA-VLA keeps a modular code structure for rapid research iteration and clear
boundaries between model, data, training, and evaluation code.

<a id="model"></a>
<details close>
<summary><b>1. Model: Modular and Extensible Framework</b></summary>

LaRA-VLA emphasizes modular model design, following top-down decomposition and a principle of high cohesion and low coupling. We use the following conventions:

1. `laravla.model.framework.yourframework.py` is the main external API of the model and should correspond to the framework figure in your paper.
2. Each `yourframework.py` or `module.py` can run standalone, for example `python yourframework.py` to demo forward and inference.

</details>


<a id="data"></a>
<details close>
<summary><b>2. DataLoader: Model-Agnostic Data Processing</b></summary>

Best practice references: GR00T / LeRobot action data schemas; multimodal data can reuse LLaVA JSON style. Conventions:

1. The dataloader returns raw data such as `PIL.Image`, `str`, normalized actions, and state in a single dict.
2. Model-specific preprocessing should live inside `yourframework.forward()`, not inside the dataloader.
3. The dataloader saves data-processing context such as normalization stats and transforms to the output path.
4. Each `dataset.py` should be runnable standalone to print or validate one legal sample dict, for example `python lerobot_datasets.py`.

</details>


 
<a id="config"></a>
<details close>
<summary><b>3. Config System: Global and Extensible Unified Configuration</b></summary>

LaRA-VLA uses a single global configuration object; all parameter accesses
should follow absolute keys. The configuration is read from `config_yaml` and
converted into an `OmegaConf DictConfig`, which permits redundancy, flexible
grouping, and easy addition of new parameters.

Conventions:
1. Use `OmegaConf.load(args.config_yaml)` as the single configuration entry.
2. Parameters may be intentionally redundant; you can add or override them from the CLI.
3. Save the unified config in the output directory so experiments can be restarted quickly.

</details>


<a id="trainer"></a>
<details close>
<summary><b>4. Trainer: Lightweight and Strategy-Oriented</b></summary>

LaRA-VLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to modify.

Conventions:
1. Store runtime state in dicts where practical.
2. Use multiple dataloaders when adapting to heterogeneous data types or task mixtures.
3. Put each training strategy in its own `trainer_*.py` file instead of large if-else chains.

</details>

<a id="inference"></a>
<details close>
<summary><b>5. Inference: Unified WebSocket Abstraction</b></summary>

LaRA-VLA uses a unified WebSocket layer to decouple training and evaluation environments, providing an environment-agnostic inference interface and simulator-specific adapters.

Conventions:
1. `policy_server.py` should expose only the core inference call: `framework.predict_action()`.
2. Avoid ad-hoc test-time and simulator-specific parameter injection in the evaluation path.
3. Provide per-environment policy clients that handle connection, request packing, retries, and action post-processing.

</details>




---
