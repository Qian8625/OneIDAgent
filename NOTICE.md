# Third-party notices

OneID itself is released under the MIT license (see `LICENSE`). The following
third-party components are either bundled or expected on the user's side.

## Bundled

| Component | Location | License |
|---|---|---|
| Noto Sans CJK Regular | `tool_server/tool_workers/online_workers/fonts/NotoSansCJK-Regular.ttc` | SIL Open Font License 1.1 |

The font is used to render Chinese labels on tool outputs (boxes, legends).
Keep this notice (and the OFL) if you redistribute the bundle.

## Not bundled

Model weights, commercial basemaps and datasets are **not** part of this
repository. If you install or reference any of the following, their own licenses
apply and you are responsible for complying with them:

- detection / segmentation / change-detection checkpoints referenced by
  `tool_server/utils/paths.py` (`OBJECT_DETECTION_*`, `SAM_*`, `TEXT_TO_BBOX_*`,
  `SAR_TO_RGB_MODEL_PATH`, `CHANGE_DETECTION_MODEL_BASE`, `BASE_MODEL_PATH`);
- language / vision-language models used as agents (`PRETRAINED_PATH`);
- geospatial base layers, tile services and any data placed under
  `ONEID_DATA_ROOT`.

## First-party engine

`DS_VI/` (D-S detection fusion, shared-identity evidence inference, cascade
assembly) is part of this project, not a third-party dependency.
