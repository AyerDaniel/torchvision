<span style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">An Exception was encountered at '<a href="#papermill-error-cell">In [4]</a>'.</span>


# Faster R-CNN Hyperparameter Optimization with Optuna + W&B (COCO MiniTrain) and Small-Object Transfer (Drones)

This notebook is the **deliverable** for the assignment:

- Training dataset: **COCO MiniTrain** (COCO-format subset)  
  https://github.com/giddyyupp/coco-minitrain
- Optimization engine: **Optuna** (TPE + pruning)
- Experiment tracking: **Weights & Biases (W&B)** (logging + dashboards)
- Generalization test: **drone small-object detection** (Assignment 3 dataset)

You will:
1. Run a **baseline** Faster R-CNN on COCO MiniTrain.
2. Run **stage-wise hyperparameter optimization** with Optuna.
3. Log *all* runs to W&B and analyze them in the W&B UI.
4. Evaluate the tuned detector on the **drone** dataset and discuss transfer.

## What you must submit
- A shareable link to your W&B project (public or access granted to the TA).
- This notebook (executed), including:
  - baseline run
  - Optuna study runs with pruning
  - final 3-seed retraining
  - drone evaluation (baseline vs tuned)
  - analysis cells (plots + written answers)

## Metrics
You must report COCO-style metrics:
- $\mathrm{mAP}$ (COCO mAP@[$0.5:0.95$])
- $\mathrm{AP}_{50}$ and $\mathrm{AP}_{75}$
- Recall (COCO AR or a simpler recall estimate)

## Objective (default)
You will optimize validation COCO mAP:
$$
\max_{\theta}\; \mathrm{mAP}_{\text{val}}(\theta),
$$
where $\theta$ denotes the hyperparameters under search.

> If you want to trade off latency, define a scalarized objective:
> $$
> J(\theta)=\mathrm{mAP}_{\text{val}}(\theta)-\lambda\,\mathrm{Latency}(\theta).
> $$
> In that case, you must define $\lambda$ and measure latency consistently.

---



```python
# !nvidia-smi
```


## 0. Colab setup

1. Enable GPU: `Runtime → Change runtime type → GPU`
2. Install packages
3. Login to W&B



```python
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

```


```python
import os
import random
import wandb
#from google.colab import drive

```


```python
# If you need to install packages, do it here (Colab).
# !pip -q install pycocotools
# !pip -q install optuna
# !pip -q install wandb

import os, json, random, time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torchvision
from torchvision.transforms import functional as F

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
```

    torch: 2.10.0+cu130
    torchvision: 0.25.0+cu130
    cuda available: True



```python
# W&B authentication
# In Docker: WANDB_API_KEY is set in the environment automatically.
# In Colab: call wandb.login() interactively.
import wandb

if os.environ.get('WANDB_API_KEY'):
    wandb.login(key=os.environ['WANDB_API_KEY'])
    print('W&B: authenticated via WANDB_API_KEY env var')
else:
    wandb.login()
    print('W&B: interactive login')

```

    [34m[1mwandb[0m: [33mWARNING[0m Calling wandb.login() after wandb.init() has no effect.


    W&B: interactive login



## 1. Reproducibility (required)

You must fix and log:
- random seeds
- dataset split indices
- code version (commit hash, if applicable)

You will run final training with 3 different seeds and report:
$$
\text{mean mAP} \pm \text{std}.
$$



```python

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    # Deterministic DataLoader workers
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

BASE_SEED = 1337
set_global_seed(BASE_SEED)

```


## 2. Dataset: COCO MiniTrain

Clone the dataset repo and set paths below.

COCO MiniTrain repository:
https://github.com/giddyyupp/coco-minitrain

You will create a deterministic train/val split.

### Required outputs
- `train_ids.json` and `val_ids.json` saved to disk
- logged to W&B as artifacts (optional but encouraged)



```python
os.environ["HF_TOKEN"] = ""

```


```python
print(os.path.realpath("/content/drive/MyDrive/njit"))



```

    /content/drive/MyDrive/njit


<span id="papermill-error-cell" style="color:red; font-family:Helvetica Neue, Helvetica, Arial, sans-serif; font-size:2em;">Execution using papermill encountered an exception here and stopped:</span>


```python
# --- Dataset: COCO MiniTrain ---
#
# Sampling methodology: https://github.com/giddyyupp/coco-minitrain
#   Statistically samples N images from COCO 2017 train preserving class/size distributions.
#
# Pre-sampled subsets:
#   HF repo : https://huggingface.co/datasets/bryanbocao/coco_minitrain
#   Files   : coco_minitrain_10k.zip (9 GB) | _15k | _20k | _25k
#   Format  : YOLO labels + JPEG images (no COCO JSON included)
#
# This cell downloads the 10k subset, then generates a COCO JSON annotation file
# by filtering the official COCO 2017 train annotations to the 10k image IDs.
# For the full assignment run change HF_DATASET_FILE to "coco_minitrain_25k.zip".

import os, time, zipfile, json as _json
import requests as _requests
from huggingface_hub import hf_hub_download

HF_DATASET_REPO  = "bryanbocao/coco_minitrain"
HF_DATASET_FILE  = "coco_minitrain_25k.zip"   # change to _25k for full run
COCO_ANN_URL     = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

_IS_COLAB    = os.path.isdir("/content")

print(_IS_COLAB)

DATASET_BASE = "/content/drive/MyDrive/njit" if _IS_COLAB else "/workspaces/eng-ai-agents/data"
EXTRACT_ROOT = os.path.join(DATASET_BASE, "coco_minitrain")

COCO_MINITRAIN_ROOT = None
IMAGES_DIR          = None
ANN_JSON            = None
DATASET_READY       = False


def _hf_download_with_retry(repo_id, filename, repo_type, local_dir,
                             max_retries=5, base_wait=60):
    for attempt in range(max_retries):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename,
                                   repo_type=repo_type, local_dir=local_dir)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"  Rate-limited (attempt {attempt+1}/{max_retries}). Waiting {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("hf_hub_download: max retries exceeded")


def _build_coco_json(images_dir, full_ann_path, out_path):
    """Filter instances_train2017.json to the image IDs present in images_dir."""
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")]
    present_ids = {int(os.path.splitext(f)[0]) for f in img_files}
    print(f"  Found {len(present_ids)} images in {images_dir}")
    print(f"  Loading full COCO annotations from {full_ann_path} ...")
    with open(full_ann_path) as f:
        full = _json.load(f)
    imgs  = [im for im in full["images"]      if im["id"] in present_ids]
    anns  = [an for an in full["annotations"] if an["image_id"] in present_ids]
    mini  = {
        "info":        full.get("info", {}),
        "licenses":    full.get("licenses", []),
        "categories":  full["categories"],
        "images":      imgs,
        "annotations": anns,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        _json.dump(mini, f)
    print(f"  Wrote {len(imgs)} images / {len(anns)} annotations → {out_path}")


def _ensure_coco_full_annotations(ann_dir):
    """Download and extract official COCO 2017 train annotations if missing.
    The COCO zip extracts to an 'annotations/' subdir, so the final path is
    ann_dir/annotations/instances_train2017.json.
    """
    target = os.path.join(ann_dir, "annotations", "instances_train2017.json")
    if os.path.exists(target):
        return target
    os.makedirs(ann_dir, exist_ok=True)
    zip_path = os.path.join(ann_dir, "annotations_trainval2017.zip")
    print(f"  Downloading COCO 2017 annotations (~253 MB) ...")
    with _requests.get(COCO_ANN_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print("  Extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(ann_dir)
    os.remove(zip_path)
    return target


# ── Step 1: locate or download+extract the HF zip ────────────────────────────
extract_dir = os.path.join(EXTRACT_ROOT, HF_DATASET_FILE.replace(".zip", ""))
zip_local   = os.path.join(EXTRACT_ROOT, HF_DATASET_FILE)

if os.path.isdir(extract_dir) and os.listdir(extract_dir):
    print(f"Cached extraction found: {extract_dir}")
else:
    os.makedirs(EXTRACT_ROOT, exist_ok=True)
    try:
        if os.path.exists(zip_local):
            print(f"Zip already downloaded: {zip_local}")
        else:
            print(f"Downloading {HF_DATASET_FILE} from {HF_DATASET_REPO} ...")
            zip_local = _hf_download_with_retry(
                repo_id=HF_DATASET_REPO, filename=HF_DATASET_FILE,
                repo_type="dataset", local_dir=EXTRACT_ROOT,
            )
            print(f"Download complete: {zip_local}")
        print(f"Extracting to {extract_dir} ...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_local, "r") as zf:
            zf.extractall(extract_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"HF download/extraction failed: {e}")
        extract_dir = None

# ── Step 2: locate train2017 images dir ───────────────────────────────────────
if extract_dir and os.path.isdir(extract_dir):
    # Zip extracts to coco_minitrain_10k/coco_minitrain_10k/images/train2017/
    train_imgs = os.path.join(extract_dir, os.path.basename(extract_dir),
                              "images", "train2017")
    if not os.path.isdir(train_imgs):
        best_dir, best_n = extract_dir, 0
        for dp, _, files in os.walk(extract_dir):
            n = sum(1 for f in files if f.lower().endswith(".jpg"))
            if n > best_n:
                best_dir, best_n = dp, n
        train_imgs = best_dir if best_n > 10 else None

    if train_imgs:
        IMAGES_DIR = train_imgs
        # ── Step 3: build COCO JSON if not present ─────────────────────────
        ann_dir  = os.path.join(extract_dir, "annotations")
        ann_json = os.path.join(ann_dir, "instances_minitrain.json")
        if not os.path.exists(ann_json):
            full_ann_cache = os.path.join(EXTRACT_ROOT, "coco_full_annotations")
            full_ann_path  = _ensure_coco_full_annotations(full_ann_cache)
            _build_coco_json(IMAGES_DIR, full_ann_path, ann_json)
        else:
            print(f"Annotation JSON already exists: {ann_json}")
        ANN_JSON            = ann_json
        COCO_MINITRAIN_ROOT = extract_dir
        DATASET_READY       = True
        print(f"Dataset ready:")
        print(f"  Root:        {COCO_MINITRAIN_ROOT}")
        print(f"  Annotations: {ANN_JSON}")
        print(f"  Images:      {IMAGES_DIR}")
    else:
        print(f"WARNING: could not locate train2017 images inside {extract_dir}")

# ── Step 4: fallback — git clone (annotations only) ──────────────────────────
if not DATASET_READY:
    print("\nFalling back to git clone of coco-minitrain (annotations only).")
    for p in ["coco-minitrain", "/content/coco-minitrain",
               os.path.expanduser("~/coco-minitrain")]:
        if os.path.isdir(p):
            COCO_MINITRAIN_ROOT = p
            break
    if COCO_MINITRAIN_ROOT is None:
        os.system("git clone --depth 1 https://github.com/giddyyupp/coco-minitrain.git")
        COCO_MINITRAIN_ROOT = "coco-minitrain"
    IMAGES_DIR = os.path.join(COCO_MINITRAIN_ROOT, "images")
    ANN_JSON   = os.path.join(COCO_MINITRAIN_ROOT, "annotations", "instances_minitrain.json")
    if os.path.exists(ANN_JSON):
        DATASET_READY = True
        print(f"Using local clone: {COCO_MINITRAIN_ROOT}")
    else:
        print("WARNING: Dataset not ready — no annotation file found.")

print(f"\nDATASET_READY : {DATASET_READY}")
if DATASET_READY:
    print(f"ANN_JSON      : {ANN_JSON}")
    print(f"IMAGES_DIR    : {IMAGES_DIR}")
```

    False



    ---------------------------------------------------------------------------

    PermissionError                           Traceback (most recent call last)

    Cell In[44], line 105
        103     print(f"Cached extraction found: {extract_dir}")
        104 else:
    --> 105     os.makedirs(EXTRACT_ROOT, exist_ok=True)
        106     try:
        107         if os.path.exists(zip_local):


    File <frozen os>:215, in makedirs(name, mode, exist_ok)


    File <frozen os>:215, in makedirs(name, mode, exist_ok)


    File <frozen os>:215, in makedirs(name, mode, exist_ok)


    File <frozen os>:225, in makedirs(name, mode, exist_ok)


    PermissionError: [Errno 13] Permission denied: '/workspaces'



```python
!whoami

```


```python

from pycocotools.coco import COCO
from PIL import Image

coco = COCO(ANN_JSON)
img_ids = sorted(coco.getImgIds())
print("num images:", len(img_ids))

# Deterministic split
val_frac = 0.2
rng = np.random.default_rng(BASE_SEED)
perm = rng.permutation(len(img_ids))
n_val = int(len(img_ids) * val_frac)
val_ids = [img_ids[i] for i in perm[:n_val]]
train_ids = [img_ids[i] for i in perm[n_val:]]

print("train:", len(train_ids), "val:", len(val_ids))

SPLIT_DIR = os.path.join(COCO_MINITRAIN_ROOT, "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)
with open(os.path.join(SPLIT_DIR, "train_ids.json"), "w") as f:
    json.dump(train_ids, f)
with open(os.path.join(SPLIT_DIR, "val_ids.json"), "w") as f:
    json.dump(val_ids, f)

print("Saved split ids to:", SPLIT_DIR)

```

    num images: 0
    train: 0 val: 0



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[49], line 18
         14 train_ids = [img_ids[i] for i in perm[n_val:]]
         16 print("train:", len(train_ids), "val:", len(val_ids))
    ---> 18 SPLIT_DIR = os.path.join(COCO_MINITRAIN_ROOT, "splits")
         19 os.makedirs(SPLIT_DIR, exist_ok=True)
         20 with open(os.path.join(SPLIT_DIR, "train_ids.json"), "w") as f:


    File <frozen posixpath>:76, in join(a, *p)


    TypeError: expected str, bytes or os.PathLike object, not NoneType



## 3. PyTorch dataset and transforms

You must keep transforms simple initially. Use augmentations only after baseline correctness is established.

Recommended minimal transforms:
- Convert to tensor
- (Optional) resize to a fixed shorter side (be consistent across runs)



```python

from torch.utils.data import Dataset, DataLoader

class CocoMiniTrainDataset(Dataset):
    def __init__(self, coco: COCO, image_dir: str, img_ids: List[int], train: bool = True):
        self.coco = coco
        self.image_dir = image_dir
        self.img_ids = img_ids
        self.train = train

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=False)  # Changed iscrowd = True to iscrowd = False.
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for a in anns:
            # COCO bbox: [x,y,w,h] -> [x1,y1,x2,y2]
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # Add flip to improve performance?
        W = image.width
        if self.train and random.random() > 0.5:
            image = F.hflip(image)
            # boxes = [[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes]
            boxes = torch.tensor([[W - x2, y1, W - x1, y2] for x1, y1, x2, y2 in boxes], dtype=torch.float32)  # Throw to torch?

        image_t = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }
        return image_t, target

def collate_fn(batch):
    return tuple(zip(*batch))

train_ds = CocoMiniTrainDataset(coco, IMAGES_DIR, train_ids, train=True)
val_ds   = CocoMiniTrainDataset(coco, IMAGES_DIR, val_ids, train=False)

print("train len:", len(train_ds), "val len:", len(val_ds))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[48], line 63
         60 def collate_fn(batch):
         61     return tuple(zip(*batch))
    ---> 63 train_ds = CocoMiniTrainDataset(coco, IMAGES_DIR, train_ids, train=True)
         64 val_ds   = CocoMiniTrainDataset(coco, IMAGES_DIR, val_ids, train=False)
         66 print("train len:", len(train_ds), "val len:", len(val_ds))


    NameError: name 'coco' is not defined



```python

BATCH_SIZE = 8  # Adjusted to H100 in colab.
NUM_WORKERS = 4

g = torch.Generator()
g.manual_seed(BASE_SEED)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
    worker_init_fn=seed_worker, generator=g
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
    worker_init_fn=seed_worker, generator=g
)

next(iter(train_loader))[0][0].shape

```


## 4. Model: Faster R-CNN (torchvision)

You will use:

- `torchvision.models.detection.fasterrcnn_resnet50_fpn`

You will also tune **RPN** and **RoI head** hyperparameters in later stages.



```python

from torchvision.models.detection import fasterrcnn_resnet50_fpn

def build_model(num_classes: Optional[int] = None):
    # COCO has 80 categories (plus background internally).
    # In torchvision, num_classes includes background.
    # If you want to adapt to a different label space, you must remap category IDs.
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    if num_classes is not None:
        # Replace the box predictor head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
print("Model loaded on:", device)

```


## 5. Training and evaluation

You will implement:
- a training loop that logs loss components
- COCO evaluation via `pycocotools.cocoeval.COCOeval`

### Important notes
- COCO category IDs are not always contiguous. Torchvision expects contiguous class indices when you replace heads.
- For this assignment you will **keep the default COCO label space** and use the pretrained COCO model, then fine-tune on COCO MiniTrain.



```python

from pycocotools.cocoeval import COCOeval

@torch.no_grad()
def evaluate_coco_map(model, coco_gt: COCO, data_loader: DataLoader, max_dets: int = 100):
    model.eval()
    results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt["image_id"].item())

            boxes = out["boxes"].detach().cpu().numpy()  # [N,4] x1,y1,x2,y2
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            # Convert to COCO format
            for b, s, c in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                results.append({
                    "image_id": img_id,
                    "category_id": int(c),
                    "bbox": [x1, y1, w, h],
                    "score": float(s),
                })

    if len(results) == 0:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.maxDets = [max_dets, max_dets, max_dets]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCOeval.stats indices:
    # 0: AP IoU=0.50:0.95
    # 1: AP IoU=0.50
    # 2: AP IoU=0.75
    mAP = float(coco_eval.stats[0])
    AP50 = float(coco_eval.stats[1])
    AP75 = float(coco_eval.stats[2])
    return {"mAP": mAP, "AP50": AP50, "AP75": AP75}

def train_one_epoch(model, optimizer, data_loader: DataLoader, epoch: int, max_norm: float = 0.0):
    model.train()
    loss_sums = {"loss": 0.0}
    n = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()

        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # accumulate
        n += 1
        loss_sums["loss"] += float(losses.item())
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())

    for k in loss_sums:
        loss_sums[k] /= max(1, n)
    return loss_sums

```


## 6. Baseline run (required)

Run a baseline training job and log to W&B.

Required:
- train loss curves (total + components)
- validation metrics: mAP, AP50, AP75
- save the model checkpoint



```python
os.makedirs("checkpoints", exist_ok=True)


from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

BASELINE_CFG = {
    "seed": BASE_SEED,
    "epochs": int(os.environ.get("BASELINE_EPOCHS", 3)),
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "grad_clip_norm": 0.0,
    "step_size": 6,
    "gamma": 0.1,
    "batch_size": BATCH_SIZE,
}

set_global_seed(BASELINE_CFG["seed"])

run = wandb.init(
    project="faster-rcnn-optuna-coco-minitrain",
    name="baseline",
    config=BASELINE_CFG
)

model = build_model().to(device)

optimizer = SGD(
    model.parameters(),
    lr=BASELINE_CFG["lr"],
    momentum=BASELINE_CFG["momentum"],
    weight_decay=BASELINE_CFG["weight_decay"]
)

scheduler = StepLR(optimizer, step_size=BASELINE_CFG["step_size"], gamma=BASELINE_CFG["gamma"])

for epoch in range(BASELINE_CFG["epochs"]):
    t0 = time.time()
    losses = train_one_epoch(model, optimizer, train_loader, epoch, max_norm=BASELINE_CFG["grad_clip_norm"])
    scheduler.step()
    metrics = evaluate_coco_map(model, coco, val_loader)

    log_dict = {**losses, **{f"val_{k}": v for k, v in metrics.items()}, "epoch": epoch, "lr": scheduler.get_last_lr()[0], "epoch_time_s": time.time()-t0}
    wandb.log(log_dict)
    print(f"Epoch {epoch}: loss={losses['loss']:.4f} val_mAP={metrics['mAP']:.4f}")

BASELINE_CKPT = os.path.join("checkpoints", "baseline_fasterrcnn.pt")
torch.save(model.state_dict(), BASELINE_CKPT)
wandb.save(BASELINE_CKPT)
wandb.finish()

print("Saved:", BASELINE_CKPT)

```


```python
# Alarm function to tell me when code has run.  My local setup spins the fan up loud enough to use as a code run indicator.

from IPython.display import Audio, display
import numpy as np

def beep(duration=1.0, freq=440):
    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    display(Audio(wave, rate=sample_rate, autoplay=True))



```


## 7. Optuna + W&B: stage-wise hyperparameter optimization (required)

You will run Optuna studies in stages.

- Stage 1: optimizer dynamics (LR, weight decay, momentum, warmup)
- Stage 2: RPN hyperparameters
- Stage 3: RoI head hyperparameters
- Stage 4: post-processing calibration (no training)

You must use:
- `TPESampler`
- pruning (`MedianPruner` or `HyperbandPruner`)

Each trial must:
1. Train for a small budget (e.g., 3–5 epochs),
2. Report intermediate validation mAP via `trial.report(...)`,
3. Allow Optuna to prune underperforming trials.

Default objective:
$$
\max \mathrm{mAP}_{\text{val}}.
$$



```python
import optuna

def make_optimizer(model, lr: float, momentum: float, weight_decay: float):
    return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

def objective_stage1(trial: optuna.Trial) -> float:
    cfg = {
        "stage": "stage1_opt",
        "seed": int(trial.suggest_int("seed", 1, 10_000)),
        "epochs": int(trial.suggest_int("epochs", 3, 5)),
        "lr": float(trial.suggest_float("lr", 1e-5, 1e-2, log=True)),
        "weight_decay": float(trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)),
        "momentum": float(trial.suggest_float("momentum", 0.8, 0.99)),
        "grad_clip_norm": float(trial.suggest_float("grad_clip_norm", 0.0, 5.0)),
    }

    set_global_seed(cfg["seed"])

    run = wandb.init(
        project="faster-rcnn-optuna-coco-minitrain",
        name=f"optuna_stage1_trial_{trial.number:04d}",
        config=cfg,
        reinit=True
    )

    model = build_model().to(device)
    optimizer = make_optimizer(model, cfg["lr"], cfg["momentum"], cfg["weight_decay"])

    # Implement scheduler.
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    best_map = -1.0
    for epoch in range(cfg["epochs"]):
        losses = train_one_epoch(model, optimizer, train_loader, epoch, max_norm=cfg["grad_clip_norm"])

        # Use scheduler.
        scheduler.step()

        metrics = evaluate_coco_map(model, coco, val_loader)

        val_map = metrics["mAP"]
        best_map = max(best_map, val_map)

        wandb.log({**losses, **{f"val_{k}": v for k, v in metrics.items()}, "epoch": epoch})

        trial.report(val_map, step=epoch)
        if trial.should_prune():
            wandb.log({"pruned": 1, "best_val_mAP": best_map})
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.log({"best_val_mAP": best_map, "pruned": 0})
    wandb.finish()
    return best_map

sampler = optuna.samplers.TPESampler(seed=BASE_SEED)
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

study_stage1 = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="stage1_opt")

```


```python
# Run Stage 1 study
N_TRIALS_STAGE1 = int(os.environ.get("HPO_TRIALS", 15))  # default 3 for demo, 30 for assignment
study_stage1.optimize(objective_stage1, n_trials=N_TRIALS_STAGE1, show_progress_bar=True)

print("Best Stage 1:", study_stage1.best_value)
print("Best params:", study_stage1.best_params)

```


```python
'''
  Code execution stopped after 3rd run despite setting runs to 15.  30 was computationally infeasable in the time alotted.
  Results were getting worse than the initial run, so I stopped the runs in the interest of time and used the best result to move on.
'''

print("Best Stage 1:", study_stage1.best_value)
print("Best params:", study_stage1.best_params)


```


```python
# Hardcode results.

'''

Best Stage 1: 0.10272976055704393
Best params: {'seed': 2621, 'epochs': 3, 'lr': 6.829352954261399e-05, 'weight_decay': 6.87491841689458e-05, 'momentum': 0.8609901026988318, 'grad_clip_norm': 2.5919641029876854}

'''

study_stage1_best_value = 0.10272976055704393
study_stage1_best_params = {'seed': 2621, 'epochs': 3, 'lr': 6.829352954261399e-05, 'weight_decay': 6.87491841689458e-05, 'momentum': 0.8609901026988318, 'grad_clip_norm': 2.5919641029876854}

```


## 8. Stage 2: RPN tuning (required)

Fix the best Stage 1 hyperparameters, then tune RPN knobs that affect proposal quality and recall.

Suggested search space:
- `rpn_nms_thresh` in $[0.5, 0.9]$
- `rpn_pre_nms_topk` in $[1000, 4000]$
- `rpn_post_nms_topk` in $[300, 2000]$
- `rpn_fg_iou_thresh` in $[0.5, 0.8]$
- `rpn_bg_iou_thresh` in $[0.0, 0.4]$
- `rpn_batch_size_per_image` in $[128, 512]$
- `rpn_positive_fraction` in $[0.25, 0.75]$

You will implement this by mutating the torchvision model components:
- `model.rpn.*` fields (where supported)

Note: torchvision does not expose *every* parameter as a public attribute in every version; implement what is available and document what you tuned.



```python
'''
  To save time if the kernel crashes I hardcoded results from stage 1.

     Note that I have changed the '.' to a '_' thus setting a new variable to the value instead of altering the Optuna object, since I can't.

  study_stage1.best_value has become study_stage1_best_value

  study_stage1.best_params has become study_stage1_best_params

'''
```


```python
# This cell tossed an error beacause torchvision {my version} has attributes _pre_nms_top_n and method pre_nms_top_n().  '_' prefix added to method calls to set attributes.
def apply_rpn_hparams(model, cfg: Dict[str, Any]):
    # Apply what torchvision exposes on your version.
    # These attributes exist in torchvision's RegionProposalNetwork in most versions.
    rpn = model.rpn
    if "rpn_nms_thresh" in cfg: rpn.nms_thresh = float(cfg["rpn_nms_thresh"])
    if "rpn_pre_nms_topk" in cfg: rpn._pre_nms_top_n["training"] = int(cfg["rpn_pre_nms_topk"])
    if "rpn_post_nms_topk" in cfg: rpn._post_nms_top_n["training"] = int(cfg["rpn_post_nms_topk"])
    if "rpn_pre_nms_topk" in cfg: rpn._pre_nms_top_n["testing"] = int(cfg["rpn_pre_nms_topk"])
    if "rpn_post_nms_topk" in cfg: rpn._post_nms_top_n["testing"] = int(cfg["rpn_post_nms_topk"])
    if "rpn_fg_iou_thresh" in cfg: rpn.fg_iou_thresh = float(cfg["rpn_fg_iou_thresh"])
    if "rpn_bg_iou_thresh" in cfg: rpn.bg_iou_thresh = float(cfg["rpn_bg_iou_thresh"])
    if "rpn_batch_size_per_image" in cfg: rpn.batch_size_per_image = int(cfg["rpn_batch_size_per_image"])
    if "rpn_positive_fraction" in cfg: rpn.positive_fraction = float(cfg["rpn_positive_fraction"])



def objective_stage2(trial: optuna.Trial) -> float:
    # Fix Stage 1 best optimizer params

    # Hard coded stage one results in case Kernel dies.
    # best1 = study_stage1.best_params
    best1 = study_stage1_best_params

    cfg = {
        "stage": "stage2_rpn",
        "seed": int(trial.suggest_int("seed", 1, 10_000)),
        "epochs": 4,  # keep small for HPO budget
        "lr": float(best1["lr"]),
        "weight_decay": float(best1["weight_decay"]),
        "momentum": float(best1["momentum"]),
        "grad_clip_norm": float(best1.get("grad_clip_norm", 0.0)),
        # RPN search
        "rpn_nms_thresh": float(trial.suggest_float("rpn_nms_thresh", 0.5, 0.9)),
        "rpn_pre_nms_topk": int(trial.suggest_int("rpn_pre_nms_topk", 1000, 4000)),
        "rpn_post_nms_topk": int(trial.suggest_int("rpn_post_nms_topk", 300, 2000)),
        "rpn_fg_iou_thresh": float(trial.suggest_float("rpn_fg_iou_thresh", 0.5, 0.8)),
        "rpn_bg_iou_thresh": float(trial.suggest_float("rpn_bg_iou_thresh", 0.0, 0.4)),
        "rpn_batch_size_per_image": int(trial.suggest_int("rpn_batch_size_per_image", 128, 512)),
        "rpn_positive_fraction": float(trial.suggest_float("rpn_positive_fraction", 0.25, 0.75)),
    }

    set_global_seed(cfg["seed"])

    run = wandb.init(
        project="faster-rcnn-optuna-coco-minitrain",
        name=f"optuna_stage2_trial_{trial.number:04d}",
        config=cfg,
        reinit=True
    )

    model = build_model().to(device)
    apply_rpn_hparams(model, cfg)
    optimizer = make_optimizer(model, cfg["lr"], cfg["momentum"], cfg["weight_decay"])

    # Implement scheduler.
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    best_map = -1.0
    for epoch in range(cfg["epochs"]):
        losses = train_one_epoch(model, optimizer, train_loader, epoch, max_norm=cfg["grad_clip_norm"])

        # Use Scheduler.
        scheduler.step()

        metrics = evaluate_coco_map(model, coco, val_loader)
        val_map = metrics["mAP"]
        best_map = max(best_map, val_map)

        wandb.log({**losses, **{f"val_{k}": v for k, v in metrics.items()}, "epoch": epoch})
        trial.report(val_map, step=epoch)
        if trial.should_prune():
            wandb.log({"pruned": 1, "best_val_mAP": best_map})
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.log({"best_val_mAP": best_map, "pruned": 0})
    wandb.finish()
    return best_map

study_stage2 = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="stage2_rpn")

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[35], line 18
         13     if "rpn_batch_size_per_image" in cfg: rpn.batch_size_per_image = int(cfg["rpn_batch_size_per_image"])
         14     if "rpn_positive_fraction" in cfg: rpn.positive_fraction = float(cfg["rpn_positive_fraction"])
    ---> 18 def objective_stage2(trial: optuna.Trial) -> float:
         19     # Fix Stage 1 best optimizer params
         20 
         21     # Hard coded stage one results in case Kernel dies.
         22     # best1 = study_stage1.best_params
         23     best1 = study_stage1_best_params
         25     cfg = {
         26         "stage": "stage2_rpn",
         27         "seed": int(trial.suggest_int("seed", 1, 10_000)),
       (...)     40         "rpn_positive_fraction": float(trial.suggest_float("rpn_positive_fraction", 0.25, 0.75)),
         41     }


    NameError: name 'optuna' is not defined



```python

# Run Stage 2 study
N_TRIALS_STAGE2 = int(os.environ.get("HPO_TRIALS", 15))  # default 3 for demo, 30 for assignment
study_stage2.optimize(objective_stage2, n_trials=N_TRIALS_STAGE2, show_progress_bar=True)

print("Best Stage 2:", study_stage2.best_value)
print("Best params:", study_stage2.best_params)

```


      0%|          | 0/15 [00:00<?, ?it/s]



Finishing previous runs because reinit is set to True.







View run <strong style="color:#cdcd00">optuna_stage1_trial_0004</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/q4vph9yv' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/q4vph9yv</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260314_224412-q4vph9yv/logs</code>







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260314_234902-b7sah5wx</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/b7sah5wx' target="_blank">optuna_stage2_trial_0000</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/b7sah5wx' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/b7sah5wx</a>


    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.02s).
    Accumulating evaluation results...
    DONE (t=3.67s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.13s).
    Accumulating evaluation results...
    DONE (t=3.84s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.89s).
    Accumulating evaluation results...
    DONE (t=3.74s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.11s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.83s).
    Accumulating evaluation results...
    DONE (t=3.84s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>▁</td></tr><tr><td>epoch</td><td>▁▃▆█</td></tr><tr><td>loss</td><td>█▄▃▁</td></tr><tr><td>loss_box_reg</td><td>█▅▄▁</td></tr><tr><td>loss_classifier</td><td>█▃▃▁</td></tr><tr><td>loss_objectness</td><td>█▃▂▁</td></tr><tr><td>loss_rpn_box_reg</td><td>█▄▅▁</td></tr><tr><td>pruned</td><td>▁</td></tr><tr><td>val_AP50</td><td>▃█▄▁</td></tr><tr><td>val_AP75</td><td>█▄▂▁</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>0.10349</td></tr><tr><td>epoch</td><td>3</td></tr><tr><td>loss</td><td>0.61266</td></tr><tr><td>loss_box_reg</td><td>0.30035</td></tr><tr><td>loss_classifier</td><td>0.25101</td></tr><tr><td>loss_objectness</td><td>0.01579</td></tr><tr><td>loss_rpn_box_reg</td><td>0.04551</td></tr><tr><td>pruned</td><td>0</td></tr><tr><td>val_AP50</td><td>0.1508</td></tr><tr><td>val_AP75</td><td>0.11562</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage2_trial_0000</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/b7sah5wx' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/b7sah5wx</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260314_234902-b7sah5wx/logs</code>


    [I 2026-03-15 00:36:25,902] Trial 0 finished with value: 0.10349356267080487 and parameters: {'seed': 1853, 'rpn_nms_thresh': 0.8829436894685043, 'rpn_pre_nms_topk': 2276, 'rpn_post_nms_topk': 1157, 'rpn_fg_iou_thresh': 0.6531412840477842, 'rpn_bg_iou_thresh': 0.006316580884725776, 'rpn_batch_size_per_image': 409, 'rpn_positive_fraction': 0.7466525199454146}. Best is trial 0 with value: 0.10349356267080487.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_003626-ciefihl5</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ciefihl5' target="_blank">optuna_stage2_trial_0001</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ciefihl5' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ciefihl5</a>


    Loading and preparing results...
    DONE (t=0.87s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=36.52s).
    Accumulating evaluation results...
    DONE (t=4.45s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.126
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.152
    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=35.53s).
    Accumulating evaluation results...
    DONE (t=4.39s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.126
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.152
    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=35.50s).
    Accumulating evaluation results...
    DONE (t=4.18s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.152
    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=35.64s).
    Accumulating evaluation results...
    DONE (t=4.45s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.152







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>▁</td></tr><tr><td>epoch</td><td>▁▃▆█</td></tr><tr><td>loss</td><td>█▅▄▁</td></tr><tr><td>loss_box_reg</td><td>█▆▂▁</td></tr><tr><td>loss_classifier</td><td>█▆▄▁</td></tr><tr><td>loss_objectness</td><td>█▄▄▁</td></tr><tr><td>loss_rpn_box_reg</td><td>█▅▅▁</td></tr><tr><td>pruned</td><td>▁</td></tr><tr><td>val_AP50</td><td>▇▇▁█</td></tr><tr><td>val_AP75</td><td>█▅▂▁</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>0.1022</td></tr><tr><td>epoch</td><td>3</td></tr><tr><td>loss</td><td>0.26323</td></tr><tr><td>loss_box_reg</td><td>0.10785</td></tr><tr><td>loss_classifier</td><td>0.09449</td></tr><tr><td>loss_objectness</td><td>0.0154</td></tr><tr><td>loss_rpn_box_reg</td><td>0.04549</td></tr><tr><td>pruned</td><td>0</td></tr><tr><td>val_AP50</td><td>0.15188</td></tr><tr><td>val_AP75</td><td>0.11376</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage2_trial_0001</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ciefihl5' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ciefihl5</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_003626-ciefihl5/logs</code>


    [I 2026-03-15 01:17:36,088] Trial 1 finished with value: 0.10220125016804958 and parameters: {'seed': 1629, 'rpn_nms_thresh': 0.5506539118763351, 'rpn_pre_nms_topk': 2124, 'rpn_post_nms_topk': 1479, 'rpn_fg_iou_thresh': 0.500870309050613, 'rpn_bg_iou_thresh': 0.14769162490379434, 'rpn_batch_size_per_image': 150, 'rpn_positive_fraction': 0.644668042512056}. Best is trial 0 with value: 0.10349356267080487.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_011736-gvvicosf</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gvvicosf' target="_blank">optuna_stage2_trial_0002</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gvvicosf' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gvvicosf</a>


    [W 2026-03-15 01:19:04,613] Trial 2 failed with parameters: {'seed': 3498, 'rpn_nms_thresh': 0.7810094878855034, 'rpn_pre_nms_topk': 2474, 'rpn_post_nms_topk': 1954, 'rpn_fg_iou_thresh': 0.7507903640097645, 'rpn_bg_iou_thresh': 0.24409493286928058, 'rpn_batch_size_per_image': 345, 'rpn_positive_fraction': 0.7486945716104434} because of the following error: KeyboardInterrupt().
    Traceback (most recent call last):
      File "/usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py", line 206, in _run_trial
        value_or_values = func(trial)
                          ^^^^^^^^^^^
      File "/tmp/ipykernel_18754/122798588.py", line 61, in objective_stage2
        losses = train_one_epoch(model, optimizer, train_loader, epoch, max_norm=cfg["grad_clip_norm"])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/tmp/ipykernel_18754/1445333649.py", line 59, in train_one_epoch
        loss_dict = model(images, targets)
                    ^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torchvision/models/detection/generalized_rcnn.py", line 118, in forward
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
                                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1776, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py", line 1787, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torchvision/models/detection/roi_heads.py", line 774, in forward
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/usr/local/lib/python3.12/dist-packages/torchvision/models/detection/roi_heads.py", line 40, in fastrcnn_loss
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
                                  ^^^^^^^^^^^^^^^^^^^^^^^
    KeyboardInterrupt
    [W 2026-03-15 01:19:04,617] Trial 2 failed with value None.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /tmp/ipykernel_18754/1009047571.py in <cell line: 0>()
          1 # Run Stage 2 study
          2 N_TRIALS_STAGE2 = int(os.environ.get("HPO_TRIALS", 15))  # default 3 for demo, 30 for assignment
    ----> 3 study_stage2.optimize(objective_stage2, n_trials=N_TRIALS_STAGE2, show_progress_bar=True)
          4 
          5 print("Best Stage 2:", study_stage2.best_value)


    /usr/local/lib/python3.12/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
        488                 If nested invocation of this method occurs.
        489         """
    --> 490         _optimize(
        491             study=self,
        492             func=func,


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
         66     try:
         67         if n_jobs == 1:
    ---> 68             _optimize_sequential(
         69                 study,
         70                 func,


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
        163 
        164         try:
    --> 165             frozen_trial_id = _run_trial(study, func, catch)
        166         finally:
        167             # The following line mitigates memory problems that can be occurred in some


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
        261         and not isinstance(func_err, catch)
        262     ):
    --> 263         raise func_err
        264     return trial._trial_id
        265 


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
        204     with get_heartbeat_thread(trial._trial_id, study._storage):
        205         try:
    --> 206             value_or_values = func(trial)
        207         except exceptions.TrialPruned as e:
        208             # TODO(mamu): Handle multi-objective cases.


    /tmp/ipykernel_18754/122798588.py in objective_stage2(trial)
         59     best_map = -1.0
         60     for epoch in range(cfg["epochs"]):
    ---> 61         losses = train_one_epoch(model, optimizer, train_loader, epoch, max_norm=cfg["grad_clip_norm"])
         62 
         63         # Use Scheduler.


    /tmp/ipykernel_18754/1445333649.py in train_one_epoch(model, optimizer, data_loader, epoch, max_norm)
         57         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
         58 
    ---> 59         loss_dict = model(images, targets)
         60         losses = sum(loss for loss in loss_dict.values())
         61 


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1774             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1775         else:
    -> 1776             return self._call_impl(*args, **kwargs)
       1777 
       1778     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1785                 or _global_backward_pre_hooks or _global_backward_hooks
       1786                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1787             return forward_call(*args, **kwargs)
       1788 
       1789         result = None


    /usr/local/lib/python3.12/dist-packages/torchvision/models/detection/generalized_rcnn.py in forward(self, images, targets)
        116             features = OrderedDict([("0", features)])
        117         proposals, proposal_losses = self.rpn(images, features, targets)
    --> 118         detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        119         detections = self.transform.postprocess(
        120             detections, images.image_sizes, original_image_sizes


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
       1774             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1775         else:
    -> 1776             return self._call_impl(*args, **kwargs)
       1777 
       1778     # torchrec tests the code consistency with the following code


    /usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
       1785                 or _global_backward_pre_hooks or _global_backward_hooks
       1786                 or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1787             return forward_call(*args, **kwargs)
       1788 
       1789         result = None


    /usr/local/lib/python3.12/dist-packages/torchvision/models/detection/roi_heads.py in forward(self, features, proposals, image_shapes, targets)
        772             if regression_targets is None:
        773                 raise ValueError("regression_targets cannot be None")
    --> 774             loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        775             losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        776         else:


    /usr/local/lib/python3.12/dist-packages/torchvision/models/detection/roi_heads.py in fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
         38     # the corresponding ground truth labels, to be used with
         39     # advanced indexing
    ---> 40     sampled_pos_inds_subset = torch.where(labels > 0)[0]
         41     labels_pos = labels[sampled_pos_inds_subset]
         42     N, num_classes = class_logits.shape


    KeyboardInterrupt: 



```python
'''
  Code execution was stopped after 2nd run despite setting runs to 15.  30 was computationally infeasable in the time alotted.
  Results were not significantly improving, so I decided to cut the trial and move on in the interest of time.
'''

print("Best Stage 2:", study_stage2.best_value)
print("Best params:", study_stage2.best_params)


```

    Best Stage 2: 0.10349356267080487
    Best params: {'seed': 1853, 'rpn_nms_thresh': 0.8829436894685043, 'rpn_pre_nms_topk': 2276, 'rpn_post_nms_topk': 1157, 'rpn_fg_iou_thresh': 0.6531412840477842, 'rpn_bg_iou_thresh': 0.006316580884725776, 'rpn_batch_size_per_image': 409, 'rpn_positive_fraction': 0.7466525199454146}



```python
# Hardcode results.

'''

Best Stage 2: 0.10349356267080487
Best params: {'seed': 1853, 'rpn_nms_thresh': 0.8829436894685043, 'rpn_pre_nms_topk': 2276, 'rpn_post_nms_topk': 1157, 'rpn_fg_iou_thresh': 0.6531412840477842, 'rpn_bg_iou_thresh': 0.006316580884725776, 'rpn_batch_size_per_image': 409, 'rpn_positive_fraction': 0.7466525199454146}
'''

study_stage2_best_value = 0.10272976055704393
study_stage2_best_params = {'seed': 1853, 'rpn_nms_thresh': 0.8829436894685043, 'rpn_pre_nms_topk': 2276, 'rpn_post_nms_topk': 1157, 'rpn_fg_iou_thresh': 0.6531412840477842, 'rpn_bg_iou_thresh': 0.006316580884725776, 'rpn_batch_size_per_image': 409, 'rpn_positive_fraction': 0.7466525199454146}

```


## 9. Stage 3: RoI head tuning (required)

Fix Stage 1+2 best configuration and tune RoI head sampling and loss weighting.

Suggested search space:
- `roi_batch_size_per_image` in $[128, 512]$
- `roi_positive_fraction` in $[0.1, 0.5]$
- `cls_loss_weight` in $[0.5, 2.0]$
- `box_loss_weight` in $[0.5, 2.0]$

Implementation note:
- Torchvision ROIHeads exposes sampler parameters.
- Loss weights might require applying weights to loss terms manually (by scaling `loss_dict` before summing).
  You will implement that by creating a custom `train_one_epoch_weighted` below.



```python

def apply_roi_hparams(model, cfg: Dict[str, Any]):
    roi = model.roi_heads
    if "roi_batch_size_per_image" in cfg: roi.batch_size_per_image = int(cfg["roi_batch_size_per_image"])
    if "roi_positive_fraction" in cfg: roi.positive_fraction = float(cfg["roi_positive_fraction"])

def train_one_epoch_weighted(model, optimizer, data_loader: DataLoader, epoch: int, max_norm: float, cls_w: float, box_w: float):
    model.train()
    loss_sums = {"loss": 0.0}
    n = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        # Scale RoI losses; keep RPN terms unscaled by default.
        if "loss_classifier" in loss_dict:
            loss_dict["loss_classifier"] = loss_dict["loss_classifier"] * cls_w
        if "loss_box_reg" in loss_dict:
            loss_dict["loss_box_reg"] = loss_dict["loss_box_reg"] * box_w

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if max_norm and max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        n += 1
        loss_sums["loss"] += float(losses.item())
        for k, v in loss_dict.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())

    for k in loss_sums:
        loss_sums[k] /= max(1, n)
    return loss_sums

def objective_stage3(trial: optuna.Trial) -> float:

  # Hardcoded results.  Reference new vars.
    # best1 = study_stage1.best_params
    # best2 = study_stage2.best_params

    best1 = study_stage1_best_params
    best2 = study_stage2_best_params

    cfg = {
        "stage": "stage3_roi",
        "seed": int(trial.suggest_int("seed", 1, 10_000)),
        "epochs": 4,
        "lr": float(best1["lr"]),
        "weight_decay": float(best1["weight_decay"]),
        "momentum": float(best1["momentum"]),
        "grad_clip_norm": float(best1.get("grad_clip_norm", 0.0)),
        # RPN fixed (best2)
        **{k: best2[k] for k in best2 if k.startswith("rpn_")},
        # RoI search
        "roi_batch_size_per_image": int(trial.suggest_int("roi_batch_size_per_image", 128, 512)),
        "roi_positive_fraction": float(trial.suggest_float("roi_positive_fraction", 0.1, 0.5)),
        "cls_loss_weight": float(trial.suggest_float("cls_loss_weight", 0.5, 2.0)),
        "box_loss_weight": float(trial.suggest_float("box_loss_weight", 0.5, 2.0)),
    }

    set_global_seed(cfg["seed"])
    run = wandb.init(
        project="faster-rcnn-optuna-coco-minitrain",
        name=f"optuna_stage3_trial_{trial.number:04d}",
        config=cfg,
        reinit=True
    )

    model = build_model().to(device)
    apply_rpn_hparams(model, cfg)
    apply_roi_hparams(model, cfg)

    optimizer = make_optimizer(model, cfg["lr"], cfg["momentum"], cfg["weight_decay"])

    # Implement scheduler.
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    best_map = -1.0
    for epoch in range(cfg["epochs"]):
        losses = train_one_epoch_weighted(
            model, optimizer, train_loader, epoch,
            max_norm=cfg["grad_clip_norm"],
            cls_w=cfg["cls_loss_weight"],
            box_w=cfg["box_loss_weight"],
        )

        # Use Scheduler.
        scheduler.step()

        metrics = evaluate_coco_map(model, coco, val_loader)
        val_map = metrics["mAP"]
        best_map = max(best_map, val_map)

        wandb.log({**losses, **{f"val_{k}": v for k, v in metrics.items()}, "epoch": epoch})
        trial.report(val_map, step=epoch)
        if trial.should_prune():
            wandb.log({"pruned": 1, "best_val_mAP": best_map})
            wandb.finish()
            raise optuna.exceptions.TrialPruned()

    wandb.log({"best_val_mAP": best_map, "pruned": 0})
    wandb.finish()
    return best_map

study_stage3 = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, study_name="stage3_roi")

```

    [I 2026-03-15 01:26:57,008] A new study created in memory with name: stage3_roi



```python

# Run Stage 3 study
N_TRIALS_STAGE3 = int(os.environ.get("HPO_TRIALS", 15))  # default 3 for demo, 30 for assignment
study_stage3.optimize(objective_stage3, n_trials=N_TRIALS_STAGE3, show_progress_bar=True)

print("Best Stage 3:", study_stage3.best_value)
print("Best params:", study_stage3.best_params)

```


      0%|          | 0/15 [00:00<?, ?it/s]



Finishing previous runs because reinit is set to True.







View run <strong style="color:#cdcd00">optuna_stage2_trial_0002</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gvvicosf' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gvvicosf</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_011736-gvvicosf/logs</code>







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_012711-iixmbdhi</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/iixmbdhi' target="_blank">optuna_stage3_trial_0000</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/iixmbdhi' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/iixmbdhi</a>


    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=34.59s).
    Accumulating evaluation results...
    DONE (t=3.74s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.65s).
    Accumulating evaluation results...
    DONE (t=3.78s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.17s).
    Accumulating evaluation results...
    DONE (t=3.62s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=29.89s).
    Accumulating evaluation results...
    DONE (t=3.62s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>▁</td></tr><tr><td>epoch</td><td>▁▃▆█</td></tr><tr><td>loss</td><td>█▄▂▁</td></tr><tr><td>loss_box_reg</td><td>█▅▃▁</td></tr><tr><td>loss_classifier</td><td>█▄▂▁</td></tr><tr><td>loss_objectness</td><td>█▃▁▂</td></tr><tr><td>loss_rpn_box_reg</td><td>█▅▆▁</td></tr><tr><td>pruned</td><td>▁</td></tr><tr><td>val_AP50</td><td>▄█▇▁</td></tr><tr><td>val_AP75</td><td>█▅▁▄</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>0.10355</td></tr><tr><td>epoch</td><td>3</td></tr><tr><td>loss</td><td>1.13035</td></tr><tr><td>loss_box_reg</td><td>0.58945</td></tr><tr><td>loss_classifier</td><td>0.47917</td></tr><tr><td>loss_objectness</td><td>0.01604</td></tr><tr><td>loss_rpn_box_reg</td><td>0.04569</td></tr><tr><td>pruned</td><td>0</td></tr><tr><td>val_AP50</td><td>0.15079</td></tr><tr><td>val_AP75</td><td>0.116</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage3_trial_0000</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/iixmbdhi' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/iixmbdhi</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_012711-iixmbdhi/logs</code>


    [I 2026-03-15 02:14:23,286] Trial 0 finished with value: 0.10355484910383297 and parameters: {'seed': 2548, 'roi_batch_size_per_image': 133, 'roi_positive_fraction': 0.13561802688099756, 'cls_loss_weight': 1.9084739008189309, 'box_loss_weight': 1.9601890990515383}. Best is trial 0 with value: 0.10355484910383297.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_021423-snx94vnq</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/snx94vnq' target="_blank">optuna_stage3_trial_0001</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/snx94vnq' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/snx94vnq</a>


    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.65s).
    Accumulating evaluation results...
    DONE (t=4.07s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.02s).
    Accumulating evaluation results...
    DONE (t=3.85s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.75s).
    Accumulating evaluation results...
    DONE (t=3.81s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=29.01s).
    Accumulating evaluation results...
    DONE (t=3.71s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>▁</td></tr><tr><td>epoch</td><td>▁▃▆█</td></tr><tr><td>loss</td><td>█▅▂▁</td></tr><tr><td>loss_box_reg</td><td>█▆▃▁</td></tr><tr><td>loss_classifier</td><td>█▃▁▁</td></tr><tr><td>loss_objectness</td><td>█▃▁▂</td></tr><tr><td>loss_rpn_box_reg</td><td>█▅▁▁</td></tr><tr><td>pruned</td><td>▁</td></tr><tr><td>val_AP50</td><td>█▇▂▁</td></tr><tr><td>val_AP75</td><td>█▅▁▅</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>0.10342</td></tr><tr><td>epoch</td><td>3</td></tr><tr><td>loss</td><td>0.68382</td></tr><tr><td>loss_box_reg</td><td>0.49171</td></tr><tr><td>loss_classifier</td><td>0.13057</td></tr><tr><td>loss_objectness</td><td>0.01588</td></tr><tr><td>loss_rpn_box_reg</td><td>0.04566</td></tr><tr><td>pruned</td><td>0</td></tr><tr><td>val_AP50</td><td>0.15078</td></tr><tr><td>val_AP75</td><td>0.11591</td></tr><tr><td>+1</td><td>...</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage3_trial_0001</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/snx94vnq' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/snx94vnq</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_021423-snx94vnq/logs</code>


    [I 2026-03-15 03:01:27,752] Trial 1 finished with value: 0.10342264191858866 and parameters: {'seed': 4915, 'roi_batch_size_per_image': 259, 'roi_positive_fraction': 0.38914304717389336, 'cls_loss_weight': 0.5163451435552239, 'box_loss_weight': 1.6399297664674184}. Best is trial 0 with value: 0.10355484910383297.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_030128-rfck27uz</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rfck27uz' target="_blank">optuna_stage3_trial_0002</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rfck27uz' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rfck27uz</a>


    [W 2026-03-15 03:03:59,936] Trial 2 failed with parameters: {'seed': 6715, 'roi_batch_size_per_image': 201, 'roi_positive_fraction': 0.3665858610279268, 'cls_loss_weight': 1.86668360234047, 'box_loss_weight': 0.7431761386399642} because of the following error: KeyboardInterrupt().
    Traceback (most recent call last):
      File "/usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py", line 206, in _run_trial
        value_or_values = func(trial)
                          ^^^^^^^^^^^
      File "/tmp/ipykernel_18754/2632214889.py", line 84, in objective_stage3
        losses = train_one_epoch_weighted(
                 ^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/tmp/ipykernel_18754/2632214889.py", line 31, in train_one_epoch_weighted
        loss_sums["loss"] += float(losses.item())
                                   ^^^^^^^^^^^^^
    KeyboardInterrupt
    [W 2026-03-15 03:03:59,939] Trial 2 failed with value None.



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    /tmp/ipykernel_18754/1651791456.py in <cell line: 0>()
          1 # Run Stage 3 study
          2 N_TRIALS_STAGE3 = int(os.environ.get("HPO_TRIALS", 15))  # default 3 for demo, 30 for assignment
    ----> 3 study_stage3.optimize(objective_stage3, n_trials=N_TRIALS_STAGE3, show_progress_bar=True)
          4 
          5 print("Best Stage 3:", study_stage3.best_value)


    /usr/local/lib/python3.12/dist-packages/optuna/study/study.py in optimize(self, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
        488                 If nested invocation of this method occurs.
        489         """
    --> 490         _optimize(
        491             study=self,
        492             func=func,


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _optimize(study, func, n_trials, timeout, n_jobs, catch, callbacks, gc_after_trial, show_progress_bar)
         66     try:
         67         if n_jobs == 1:
    ---> 68             _optimize_sequential(
         69                 study,
         70                 func,


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _optimize_sequential(study, func, n_trials, timeout, catch, callbacks, gc_after_trial, reseed_sampler_rng, time_start, progress_bar)
        163 
        164         try:
    --> 165             frozen_trial_id = _run_trial(study, func, catch)
        166         finally:
        167             # The following line mitigates memory problems that can be occurred in some


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
        261         and not isinstance(func_err, catch)
        262     ):
    --> 263         raise func_err
        264     return trial._trial_id
        265 


    /usr/local/lib/python3.12/dist-packages/optuna/study/_optimize.py in _run_trial(study, func, catch)
        204     with get_heartbeat_thread(trial._trial_id, study._storage):
        205         try:
    --> 206             value_or_values = func(trial)
        207         except exceptions.TrialPruned as e:
        208             # TODO(mamu): Handle multi-objective cases.


    /tmp/ipykernel_18754/2632214889.py in objective_stage3(trial)
         82     best_map = -1.0
         83     for epoch in range(cfg["epochs"]):
    ---> 84         losses = train_one_epoch_weighted(
         85             model, optimizer, train_loader, epoch,
         86             max_norm=cfg["grad_clip_norm"],


    /tmp/ipykernel_18754/2632214889.py in train_one_epoch_weighted(model, optimizer, data_loader, epoch, max_norm, cls_w, box_w)
         29 
         30         n += 1
    ---> 31         loss_sums["loss"] += float(losses.item())
         32         for k, v in loss_dict.items():
         33             loss_sums[k] = loss_sums.get(k, 0.0) + float(v.item())


    KeyboardInterrupt: 



```python
'''
  Code execution was stopped after 2nd run despite setting runs to 15.  30 was computationally infeasable in the time alotted.
  Results were not significantly improving, so I decided to cut the trial and move on in the interest of time.
'''

print("Best Stage 3:", study_stage3.best_value)
print("Best params:", study_stage3.best_params)


```

    Best Stage 3: 0.10355484910383297
    Best params: {'seed': 2548, 'roi_batch_size_per_image': 133, 'roi_positive_fraction': 0.13561802688099756, 'cls_loss_weight': 1.9084739008189309, 'box_loss_weight': 1.9601890990515383}



```python
# Hardcode results.

'''

Best Stage 3: 0.10355484910383297
Best params: {'seed': 2548, 'roi_batch_size_per_image': 133, 'roi_positive_fraction': 0.13561802688099756, 'cls_loss_weight': 1.9084739008189309, 'box_loss_weight': 1.9601890990515383}

'''

study_stage3_best_value = 0.10355484910383297
study_stage3_best_params = {'seed': 2548, 'roi_batch_size_per_image': 133, 'roi_positive_fraction': 0.13561802688099756, 'cls_loss_weight': 1.9084739008189309, 'box_loss_weight': 1.9601890990515383}

```


## 10. Stage 4: post-processing calibration (required)

You will tune score threshold and NMS IoU threshold **without retraining**.

Suggested ranges:
- `score_thresh` in $[0.01, 0.5]$
- `box_nms_thresh` in $[0.3, 0.7]$

In torchvision:
- `model.roi_heads.score_thresh`
- `model.roi_heads.nms_thresh`
- `model.roi_heads.detections_per_img`

You will:
1. Train one final model using the best Stage 1+2+3 configuration (longer epochs, e.g., 10–15).
2. Run an Optuna study that only changes post-processing parameters and evaluates on val.



```python

def apply_postprocess_hparams(model, cfg: Dict[str, Any]):
    roi = model.roi_heads
    if "score_thresh" in cfg: roi.score_thresh = float(cfg["score_thresh"])
    if "box_nms_thresh" in cfg: roi.nms_thresh = float(cfg["box_nms_thresh"])
    if "detections_per_img" in cfg: roi.detections_per_img = int(cfg["detections_per_img"])

def train_final_model(best_cfg: Dict[str, Any], epochs: int = 12, seed: int = 2026) -> str:
    set_global_seed(seed)
    run = wandb.init(
        project="faster-rcnn-optuna-coco-minitrain",
        name=f"final_train_seed_{seed}",
        config={**best_cfg, "final_epochs": epochs, "final_seed": seed},
        reinit=True
    )

    model = build_model().to(device)
    apply_rpn_hparams(model, best_cfg)
    apply_roi_hparams(model, best_cfg)

    optimizer = make_optimizer(model, best_cfg["lr"], best_cfg["momentum"], best_cfg["weight_decay"])

    # Implement scheduler.
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    # for epoch in range(epochs):
    #     losses = train_one_epoch_weighted(
    #         model, optimizer, train_loader, epoch,
    #         max_norm=best_cfg.get("grad_clip_norm", 0.0),
    #         cls_w=best_cfg.get("cls_loss_weight", 1.0),
    #         box_w=best_cfg.get("box_loss_weight", 1.0),
    #     )

    #     # Use Scheduler.
    #     scheduler.step()

    #     metrics = evaluate_coco_map(model, coco, val_loader)
    #     wandb.log({**losses, **{f"val_{k}": v for k, v in metrics.items()}, "epoch": epoch})

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = os.path.join("checkpoints", f"final_fasterrcnn_seed_{seed}.pt")

    best_map = -1.0
    for epoch in range(epochs):
        losses = train_one_epoch_weighted(
            model, optimizer, train_loader, epoch,
            max_norm=best_cfg.get("grad_clip_norm", 0.0),
            cls_w=best_cfg.get("cls_loss_weight", 1.0),
            box_w=best_cfg.get("box_loss_weight", 1.0),
        )

        scheduler.step()

        metrics = evaluate_coco_map(model, coco, val_loader)
        val_map = metrics["mAP"]

        # Save only if this is the best epoch so far
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), ckpt)
            wandb.save(ckpt)  # ← sync to W&B immediately

        wandb.log({**losses, **{f"val_{k}": v for k, v in metrics.items()}, "epoch": epoch, "best_val_mAP": best_map})


    # ckpt = os.path.join("checkpoints", f"final_fasterrcnn_seed_{seed}.pt")
    #torch.save(model.state_dict(), ckpt)
    #wandb.save(ckpt)
    wandb.finish()
    return ckpt

# Compose best config from Stage 1-3
best_cfg = {}

# Hardcoded vars to variables instead of referencing the Optuna objects in case Kernel dies.

best_cfg.update(study_stage1_best_params)
best_cfg.update({k: v for k, v in study_stage2_best_params.items() if k.startswith("rpn_")})
best_cfg.update({k: v for k, v in study_stage3_best_params.items() if k.startswith("roi_") or k.endswith("_weight")})

# best_cfg.update(study_stage1.best_params)
# best_cfg.update({k: v for k, v in study_stage2.best_params.items() if k.startswith("rpn_")})
# best_cfg.update({k: v for k, v in study_stage3.best_params.items() if k.startswith("roi_") or k.endswith("_weight")})

# Ensure required optimizer keys exist
# (names differ across studies; normalize to expected keys)
# Stage1 keys are: lr, weight_decay, momentum, grad_clip_norm
# Keep them as is.
print("Best combined cfg:", best_cfg)

FINAL_CKPT = train_final_model(best_cfg, epochs=12, seed=2026)
print("Final ckpt:", FINAL_CKPT)

```

    Best combined cfg: {'seed': 2621, 'epochs': 3, 'lr': 6.829352954261399e-05, 'weight_decay': 6.87491841689458e-05, 'momentum': 0.8609901026988318, 'grad_clip_norm': 2.5919641029876854, 'rpn_nms_thresh': 0.8829436894685043, 'rpn_pre_nms_topk': 2276, 'rpn_post_nms_topk': 1157, 'rpn_fg_iou_thresh': 0.6531412840477842, 'rpn_bg_iou_thresh': 0.006316580884725776, 'rpn_batch_size_per_image': 409, 'rpn_positive_fraction': 0.7466525199454146, 'roi_batch_size_per_image': 133, 'roi_positive_fraction': 0.13561802688099756, 'cls_loss_weight': 1.9084739008189309, 'box_loss_weight': 1.9601890990515383}







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_054538-5mly8s3s</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5mly8s3s' target="_blank">final_train_seed_2026</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5mly8s3s' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5mly8s3s</a>


    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=34.54s).
    Accumulating evaluation results...
    DONE (t=3.92s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155


    [34m[1mwandb[0m: [33mWARNING[0m Symlinked 1 file into the W&B run directory; call wandb.save again to sync new files.


    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.27s).
    Accumulating evaluation results...
    DONE (t=3.95s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=34.78s).
    Accumulating evaluation results...
    DONE (t=3.88s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.79s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.44s).
    Accumulating evaluation results...
    DONE (t=5.96s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.00s).
    Accumulating evaluation results...
    DONE (t=3.65s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.59s).
    Accumulating evaluation results...
    DONE (t=3.71s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.77s).
    Accumulating evaluation results...
    DONE (t=3.84s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=35.82s).
    Accumulating evaluation results...
    DONE (t=3.95s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.06s).
    Accumulating evaluation results...
    DONE (t=5.87s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.54s).
    Accumulating evaluation results...
    DONE (t=3.77s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.25s).
    Accumulating evaluation results...
    DONE (t=3.83s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155
    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.28s).
    Accumulating evaluation results...
    DONE (t=3.79s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.084
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>▁███████████</td></tr><tr><td>epoch</td><td>▁▂▂▃▄▄▅▅▆▇▇█</td></tr><tr><td>loss</td><td>█▅▃▂▂▂▁▁▁▁▁▁</td></tr><tr><td>loss_box_reg</td><td>█▆▄▂▂▂▁▂▁▁▂▁</td></tr><tr><td>loss_classifier</td><td>█▄▃▂▁▂▁▁▁▁▁▁</td></tr><tr><td>loss_objectness</td><td>█▄▃▂▃▁▂▃▂▁▂▂</td></tr><tr><td>loss_rpn_box_reg</td><td>█▅▃▇█▂▅▂▃▁▃▇</td></tr><tr><td>val_AP50</td><td>█▂▂▁▁▃▃▂▂▂▂▂</td></tr><tr><td>val_AP75</td><td>█▆▁▂▃▄▄▅▄▄▄▄</td></tr><tr><td>val_mAP</td><td>▇█▁▃▃▂▂▃▃▃▃▃</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>best_val_mAP</td><td>0.10346</td></tr><tr><td>epoch</td><td>11</td></tr><tr><td>loss</td><td>1.12781</td></tr><tr><td>loss_box_reg</td><td>0.58783</td></tr><tr><td>loss_classifier</td><td>0.47832</td></tr><tr><td>loss_objectness</td><td>0.01588</td></tr><tr><td>loss_rpn_box_reg</td><td>0.04578</td></tr><tr><td>val_AP50</td><td>0.15087</td></tr><tr><td>val_AP75</td><td>0.11589</td></tr><tr><td>val_mAP</td><td>0.10332</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">final_train_seed_2026</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5mly8s3s' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5mly8s3s</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)



Find logs at: <code>./wandb/run-20260315_054538-5mly8s3s/logs</code>


    Final ckpt: checkpoints/final_fasterrcnn_seed_2026.pt



```python
'''
  I've been up all night running in batches.  Colab timed out.  Got to grab checkpoint from wandb.
'''

import wandb, os
api = wandb.Api()
run = api.run("danielayer-personal/faster-rcnn-optuna-coco-minitrain/5mly8s3s")
os.makedirs("checkpoints", exist_ok=True)
for f in run.files():
    if "final_fasterrcnn_seed_2026" in f.name:
        f.download(root="checkpoints", replace=True)
        print(f"Downloaded: {f.name}")
FINAL_CKPT = "checkpoints/final_fasterrcnn_seed_2026.pt"
print("FINAL_CKPT:", FINAL_CKPT)

```

    Downloaded: checkpoints/final_fasterrcnn_seed_2026.pt
    FINAL_CKPT: checkpoints/final_fasterrcnn_seed_2026.pt



```python

@torch.no_grad()
def evaluate_with_postprocess(model, score_thresh: float, nms_thresh: float, dets_per_img: int = 100):
    apply_postprocess_hparams(model, {"score_thresh": score_thresh, "box_nms_thresh": nms_thresh, "detections_per_img": dets_per_img})
    return evaluate_coco_map(model, coco, val_loader)

def objective_stage4(trial: optuna.Trial) -> float:
    cfg = {
        "stage": "stage4_post",
        "score_thresh": float(trial.suggest_float("score_thresh", 0.01, 0.5, log=True)),
        "box_nms_thresh": float(trial.suggest_float("box_nms_thresh", 0.3, 0.7)),
        "detections_per_img": int(trial.suggest_int("detections_per_img", 50, 300)),
    }

    run = wandb.init(
        project="faster-rcnn-optuna-coco-minitrain",
        name=f"optuna_stage4_trial_{trial.number:04d}",
        config=cfg,
        reinit=True
    )

    model = build_model().to(device)
    model.load_state_dict(torch.load(FINAL_CKPT, map_location=device))
    apply_rpn_hparams(model, best_cfg)
    apply_roi_hparams(model, best_cfg)

    metrics = evaluate_with_postprocess(model, cfg["score_thresh"], cfg["box_nms_thresh"], cfg["detections_per_img"])
    wandb.log({f"val_{k}": v for k, v in metrics.items()})
    wandb.finish()
    return metrics["mAP"]

study_stage4 = optuna.create_study(direction="maximize", sampler=sampler, pruner=None, study_name="stage4_post")

```

    [I 2026-03-15 10:07:27,706] A new study created in memory with name: stage4_post



```python

N_TRIALS_STAGE4 = int(os.environ.get("HPO_TRIALS", 30))  # default 3 for demo, 30 for assignment
study_stage4.optimize(objective_stage4, n_trials=N_TRIALS_STAGE4, show_progress_bar=True)
print("Best Stage 4:", study_stage4.best_value)
print("Best post params:", study_stage4.best_params)

```


      0%|          | 0/30 [00:00<?, ?it/s]







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_100934-h9ik3ij3</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h9ik3ij3' target="_blank">optuna_stage4_trial_0000</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h9ik3ij3' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h9ik3ij3</a>


    Loading and preparing results...
    DONE (t=0.04s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=26.46s).
    Accumulating evaluation results...
    DONE (t=2.73s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.097
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.139
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.060
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.106
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.132
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.106
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.106
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.106
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.063
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.115
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.147







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.1388</td></tr><tr><td>val_AP75</td><td>0.11052</td></tr><tr><td>val_mAP</td><td>0.09715</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0000</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h9ik3ij3' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h9ik3ij3</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_100934-h9ik3ij3/logs</code>


    [I 2026-03-15 10:11:53,541] Trial 0 finished with value: 0.09715482611290012 and parameters: {'score_thresh': 0.3524075840288682, 'box_nms_thresh': 0.42974316142422353, 'detections_per_img': 225}. Best is trial 0 with value: 0.09715482611290012.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_101154-79jruppt</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/79jruppt' target="_blank">optuna_stage4_trial_0001</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/79jruppt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/79jruppt</a>


    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=30.87s).
    Accumulating evaluation results...
    DONE (t=4.10s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.088
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.157







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15178</td></tr><tr><td>val_AP75</td><td>0.11626</td></tr><tr><td>val_mAP</td><td>0.10384</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0001</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/79jruppt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/79jruppt</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_101154-79jruppt/logs</code>


    [I 2026-03-15 10:14:18,941] Trial 1 finished with value: 0.10384203177373426 and parameters: {'score_thresh': 0.02824828995803445, 'box_nms_thresh': 0.5075834601543322, 'detections_per_img': 94}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_101419-9v4i84w5</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/9v4i84w5' target="_blank">optuna_stage4_trial_0002</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/9v4i84w5' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/9v4i84w5</a>


    Loading and preparing results...
    DONE (t=0.08s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.32s).
    Accumulating evaluation results...
    DONE (t=3.61s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15073</td></tr><tr><td>val_AP75</td><td>0.11591</td></tr><tr><td>val_mAP</td><td>0.10318</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0002</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/9v4i84w5' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/9v4i84w5</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_101419-9v4i84w5/logs</code>


    [I 2026-03-15 10:16:45,706] Trial 2 finished with value: 0.10317510330284843 and parameters: {'score_thresh': 0.06235284494564663, 'box_nms_thresh': 0.4795858075379843, 'detections_per_img': 149}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_101646-hxdxbu1u</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hxdxbu1u' target="_blank">optuna_stage4_trial_0003</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hxdxbu1u' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hxdxbu1u</a>


    Loading and preparing results...
    DONE (t=0.06s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=26.80s).
    Accumulating evaluation results...
    DONE (t=5.07s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.100
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.145
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.109
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.113
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.151







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14456</td></tr><tr><td>val_AP75</td><td>0.11367</td></tr><tr><td>val_mAP</td><td>0.10035</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0003</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hxdxbu1u' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hxdxbu1u</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_101646-hxdxbu1u/logs</code>


    [I 2026-03-15 10:19:09,758] Trial 3 finished with value: 0.10034783364702184 and parameters: {'score_thresh': 0.22154275794547026, 'box_nms_thresh': 0.49735929791247835, 'detections_per_img': 231}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_101910-lvh9tp32</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lvh9tp32' target="_blank">optuna_stage4_trial_0004</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lvh9tp32' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lvh9tp32</a>


    Loading and preparing results...
    DONE (t=0.05s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=25.49s).
    Accumulating evaluation results...
    DONE (t=3.08s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.100
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.145
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.109
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.112
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.112
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.112
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.150







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14455</td></tr><tr><td>val_AP75</td><td>0.11362</td></tr><tr><td>val_mAP</td><td>0.10016</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0004</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lvh9tp32' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lvh9tp32</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_101910-lvh9tp32/logs</code>


    [I 2026-03-15 10:21:30,262] Trial 4 finished with value: 0.10016163187152495 and parameters: {'score_thresh': 0.22249607043273004, 'box_nms_thresh': 0.44060963991940066, 'detections_per_img': 278}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_102130-gtb347ei</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gtb347ei' target="_blank">optuna_stage4_trial_0005</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gtb347ei' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gtb347ei</a>


    Loading and preparing results...
    DONE (t=0.75s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.34s).
    Accumulating evaluation results...
    DONE (t=3.39s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.100
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.143
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.115
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.067
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.110
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.132
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.118
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.118
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.118
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.077
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.126
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14325</td></tr><tr><td>val_AP75</td><td>0.11477</td></tr><tr><td>val_mAP</td><td>0.10043</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0005</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gtb347ei' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gtb347ei</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_102130-gtb347ei/logs</code>


    [I 2026-03-15 10:23:58,132] Trial 5 finished with value: 0.10043378288297786 and parameters: {'score_thresh': 0.16262019565058644, 'box_nms_thresh': 0.6588990523827398, 'detections_per_img': 155}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_102358-dxt99h61</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dxt99h61' target="_blank">optuna_stage4_trial_0006</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dxt99h61' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dxt99h61</a>


    Loading and preparing results...
    DONE (t=0.13s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.91s).
    Accumulating evaluation results...
    DONE (t=4.10s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.146
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.133
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.159







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14577</td></tr><tr><td>val_AP75</td><td>0.11634</td></tr><tr><td>val_mAP</td><td>0.10186</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0006</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dxt99h61' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dxt99h61</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_102358-dxt99h61/logs</code>


    [I 2026-03-15 10:26:25,967] Trial 6 finished with value: 0.10186046859861499 and parameters: {'score_thresh': 0.06227243233330262, 'box_nms_thresh': 0.6615708797321408, 'detections_per_img': 209}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_102626-kir8wsoh</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/kir8wsoh' target="_blank">optuna_stage4_trial_0007</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/kir8wsoh' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/kir8wsoh</a>


    Loading and preparing results...
    DONE (t=0.07s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=30.07s).
    Accumulating evaluation results...
    DONE (t=3.30s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.150
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.115
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.117
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.117
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.117
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.079
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.125
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.152







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14968</td></tr><tr><td>val_AP75</td><td>0.11526</td></tr><tr><td>val_mAP</td><td>0.1025</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0007</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/kir8wsoh' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/kir8wsoh</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_102626-kir8wsoh/logs</code>


    [I 2026-03-15 10:28:50,199] Trial 7 finished with value: 0.10249907395174133 and parameters: {'score_thresh': 0.08040096773896975, 'box_nms_thresh': 0.3950043060327548, 'detections_per_img': 287}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_102850-ms6g7ljt</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ms6g7ljt' target="_blank">optuna_stage4_trial_0008</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ms6g7ljt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ms6g7ljt</a>


    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=29.99s).
    Accumulating evaluation results...
    DONE (t=3.69s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.148
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.156







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14769</td></tr><tr><td>val_AP75</td><td>0.11564</td></tr><tr><td>val_mAP</td><td>0.10227</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0008</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ms6g7ljt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ms6g7ljt</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_102850-ms6g7ljt/logs</code>


    [I 2026-03-15 10:31:16,563] Trial 8 finished with value: 0.1022709011636201 and parameters: {'score_thresh': 0.08921851684194797, 'box_nms_thresh': 0.6029406115550753, 'detections_per_img': 109}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_103117-xrga541c</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xrga541c' target="_blank">optuna_stage4_trial_0009</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xrga541c' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xrga541c</a>


    Loading and preparing results...
    DONE (t=0.09s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.56s).
    Accumulating evaluation results...
    DONE (t=3.78s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.082
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.151</td></tr><tr><td>val_AP75</td><td>0.1158</td></tr><tr><td>val_mAP</td><td>0.10324</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0009</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xrga541c' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xrga541c</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_103117-xrga541c/logs</code>


    [I 2026-03-15 10:33:43,814] Trial 9 finished with value: 0.10323993209740666 and parameters: {'score_thresh': 0.0537213839157752, 'box_nms_thresh': 0.45764544007447727, 'detections_per_img': 178}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_103344-o561sg59</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o561sg59' target="_blank">optuna_stage4_trial_0010</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o561sg59' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o561sg59</a>


    Loading and preparing results...
    DONE (t=0.91s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.37s).
    Accumulating evaluation results...
    DONE (t=4.51s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.149
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.158







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14938</td></tr><tr><td>val_AP75</td><td>0.11605</td></tr><tr><td>val_mAP</td><td>0.10305</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0010</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o561sg59' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o561sg59</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_103344-o561sg59/logs</code>


    [I 2026-03-15 10:36:14,411] Trial 10 finished with value: 0.10305245171067513 and parameters: {'score_thresh': 0.012788173604635219, 'box_nms_thresh': 0.5688037309040133, 'detections_per_img': 51}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_103614-5dshflxy</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5dshflxy' target="_blank">optuna_stage4_trial_0011</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5dshflxy' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5dshflxy</a>


    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.70s).
    Accumulating evaluation results...
    DONE (t=4.03s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.126
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.153







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15072</td></tr><tr><td>val_AP75</td><td>0.11558</td></tr><tr><td>val_mAP</td><td>0.10293</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0011</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5dshflxy' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5dshflxy</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_103614-5dshflxy/logs</code>


    [I 2026-03-15 10:38:43,307] Trial 11 finished with value: 0.1029333673400735 and parameters: {'score_thresh': 0.02141034722112618, 'box_nms_thresh': 0.3060685306316245, 'detections_per_img': 96}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_103843-5axedzgh</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5axedzgh' target="_blank">optuna_stage4_trial_0012</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5axedzgh' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5axedzgh</a>


    Loading and preparing results...
    DONE (t=0.13s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=36.84s).
    Accumulating evaluation results...
    DONE (t=4.65s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.088
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.157







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15051</td></tr><tr><td>val_AP75</td><td>0.11606</td></tr><tr><td>val_mAP</td><td>0.10346</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0012</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5axedzgh' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/5axedzgh</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_103843-5axedzgh/logs</code>


    [I 2026-03-15 10:41:17,060] Trial 12 finished with value: 0.1034560489952313 and parameters: {'score_thresh': 0.03046570420770471, 'box_nms_thresh': 0.5602721456875379, 'detections_per_img': 107}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_104117-lx9w40qj</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lx9w40qj' target="_blank">optuna_stage4_trial_0013</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lx9w40qj' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lx9w40qj</a>


    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.55s).
    Accumulating evaluation results...
    DONE (t=4.25s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.150
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.157







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15036</td></tr><tr><td>val_AP75</td><td>0.116</td></tr><tr><td>val_mAP</td><td>0.10335</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0013</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lx9w40qj' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/lx9w40qj</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_104117-lx9w40qj/logs</code>


    [I 2026-03-15 10:43:47,691] Trial 13 finished with value: 0.10334997754834123 and parameters: {'score_thresh': 0.02717129601315684, 'box_nms_thresh': 0.5446246896363499, 'detections_per_img': 55}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_104348-4bx78cl8</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/4bx78cl8' target="_blank">optuna_stage4_trial_0014</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/4bx78cl8' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/4bx78cl8</a>


    Loading and preparing results...
    DONE (t=0.13s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=34.83s).
    Accumulating evaluation results...
    DONE (t=4.21s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.157







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15089</td></tr><tr><td>val_AP75</td><td>0.1162</td></tr><tr><td>val_mAP</td><td>0.1036</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0014</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/4bx78cl8' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/4bx78cl8</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_104348-4bx78cl8/logs</code>


    [I 2026-03-15 10:46:18,626] Trial 14 finished with value: 0.10360348892152602 and parameters: {'score_thresh': 0.030815662208265638, 'box_nms_thresh': 0.5495259265508871, 'detections_per_img': 98}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_104619-vbplrrtt</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/vbplrrtt' target="_blank">optuna_stage4_trial_0015</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/vbplrrtt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/vbplrrtt</a>


    Loading and preparing results...
    DONE (t=0.14s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.49s).
    Accumulating evaluation results...
    DONE (t=4.75s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.087
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.155







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15217</td></tr><tr><td>val_AP75</td><td>0.11598</td></tr><tr><td>val_mAP</td><td>0.10352</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0015</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/vbplrrtt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/vbplrrtt</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_104619-vbplrrtt/logs</code>


    [I 2026-03-15 10:48:47,901] Trial 15 finished with value: 0.10351780465888345 and parameters: {'score_thresh': 0.010383455529679855, 'box_nms_thresh': 0.3687855589429836, 'detections_per_img': 84}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_104848-ra0zqeib</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ra0zqeib' target="_blank">optuna_stage4_trial_0016</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ra0zqeib' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ra0zqeib</a>


    Loading and preparing results...
    DONE (t=0.87s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=36.69s).
    Accumulating evaluation results...
    DONE (t=5.36s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.149
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.093
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.133
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.160







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14918</td></tr><tr><td>val_AP75</td><td>0.11691</td></tr><tr><td>val_mAP</td><td>0.10321</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0016</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ra0zqeib' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/ra0zqeib</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_104848-ra0zqeib/logs</code>


    [I 2026-03-15 10:51:23,952] Trial 16 finished with value: 0.10321191784552063 and parameters: {'score_thresh': 0.017012136525054667, 'box_nms_thresh': 0.6173457511175302, 'detections_per_img': 134}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_105124-t7p5nztw</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/t7p5nztw' target="_blank">optuna_stage4_trial_0017</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/t7p5nztw' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/t7p5nztw</a>


    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.12s).
    Accumulating evaluation results...
    DONE (t=4.07s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.129
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.156







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15092</td></tr><tr><td>val_AP75</td><td>0.11599</td></tr><tr><td>val_mAP</td><td>0.10351</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0017</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/t7p5nztw' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/t7p5nztw</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_105124-t7p5nztw/logs</code>


    [I 2026-03-15 10:53:53,773] Trial 17 finished with value: 0.10350852112110666 and parameters: {'score_thresh': 0.03769509991656829, 'box_nms_thresh': 0.5258462572400944, 'detections_per_img': 77}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_105354-o3jfomat</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o3jfomat' target="_blank">optuna_stage4_trial_0018</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o3jfomat' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o3jfomat</a>


    Loading and preparing results...
    DONE (t=0.92s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=36.46s).
    Accumulating evaluation results...
    DONE (t=5.20s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.150
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.134
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.093
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.133
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.160







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14974</td></tr><tr><td>val_AP75</td><td>0.11689</td></tr><tr><td>val_mAP</td><td>0.10343</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0018</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o3jfomat' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/o3jfomat</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_105354-o3jfomat/logs</code>


    [I 2026-03-15 10:56:29,002] Trial 18 finished with value: 0.10342615244748657 and parameters: {'score_thresh': 0.016965902237080704, 'box_nms_thresh': 0.6049050828173569, 'detections_per_img': 136}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_105629-wqegz1eg</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/wqegz1eg' target="_blank">optuna_stage4_trial_0019</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/wqegz1eg' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/wqegz1eg</a>


    Loading and preparing results...
    DONE (t=0.98s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=35.52s).
    Accumulating evaluation results...
    DONE (t=4.69s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.101
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.145
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.117
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.132
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.132
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.161







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14455</td></tr><tr><td>val_AP75</td><td>0.11663</td></tr><tr><td>val_mAP</td><td>0.10149</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0019</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/wqegz1eg' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/wqegz1eg</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_105629-wqegz1eg/logs</code>


    [I 2026-03-15 10:59:02,339] Trial 19 finished with value: 0.10148805424731931 and parameters: {'score_thresh': 0.04103751685861955, 'box_nms_thresh': 0.6903599865538577, 'detections_per_img': 178}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_105902-8z0ewjhq</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/8z0ewjhq' target="_blank">optuna_stage4_trial_0020</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/8z0ewjhq' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/8z0ewjhq</a>


    Loading and preparing results...
    DONE (t=0.06s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=28.60s).
    Accumulating evaluation results...
    DONE (t=3.09s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.102
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.148
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.115
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.067
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.116
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.116
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.116
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.076
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.153







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.14782</td></tr><tr><td>val_AP75</td><td>0.1148</td></tr><tr><td>val_mAP</td><td>0.10198</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0020</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/8z0ewjhq' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/8z0ewjhq</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_105902-8z0ewjhq/logs</code>


    [I 2026-03-15 11:01:25,558] Trial 20 finished with value: 0.10197689479939728 and parameters: {'score_thresh': 0.14022076692503488, 'box_nms_thresh': 0.5029029409227224, 'detections_per_img': 117}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_110126-gqktk3g6</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gqktk3g6' target="_blank">optuna_stage4_trial_0021</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gqktk3g6' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gqktk3g6</a>


    Loading and preparing results...
    DONE (t=0.13s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=34.36s).
    Accumulating evaluation results...
    DONE (t=4.44s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.121
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15179</td></tr><tr><td>val_AP75</td><td>0.11585</td></tr><tr><td>val_mAP</td><td>0.10334</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0021</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gqktk3g6' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/gqktk3g6</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_110126-gqktk3g6/logs</code>


    [I 2026-03-15 11:03:55,786] Trial 21 finished with value: 0.10333508581038005 and parameters: {'score_thresh': 0.013052758943444499, 'box_nms_thresh': 0.35773908717289327, 'detections_per_img': 79}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_110356-77ccd57p</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/77ccd57p' target="_blank">optuna_stage4_trial_0022</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/77ccd57p' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/77ccd57p</a>


    Loading and preparing results...
    DONE (t=0.16s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.56s).
    Accumulating evaluation results...
    DONE (t=4.56s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.122
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.128
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.156







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15218</td></tr><tr><td>val_AP75</td><td>0.1159</td></tr><tr><td>val_mAP</td><td>0.10349</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0022</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/77ccd57p' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/77ccd57p</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_110356-77ccd57p/logs</code>


    [I 2026-03-15 11:06:25,079] Trial 22 finished with value: 0.10349432972747148 and parameters: {'score_thresh': 0.010913924506496104, 'box_nms_thresh': 0.38387314913848636, 'detections_per_img': 70}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_110625-z4yu6chs</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/z4yu6chs' target="_blank">optuna_stage4_trial_0023</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/z4yu6chs' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/z4yu6chs</a>


    Loading and preparing results...
    DONE (t=0.10s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=31.65s).
    Accumulating evaluation results...
    DONE (t=3.97s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.127
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.153







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15153</td></tr><tr><td>val_AP75</td><td>0.11576</td></tr><tr><td>val_mAP</td><td>0.1032</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0023</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/z4yu6chs' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/z4yu6chs</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_110625-z4yu6chs/logs</code>


    [I 2026-03-15 11:08:52,133] Trial 23 finished with value: 0.1032047186786507 and parameters: {'score_thresh': 0.02279954958620861, 'box_nms_thresh': 0.33355495804450364, 'detections_per_img': 94}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_110852-76gk3m3u</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/76gk3m3u' target="_blank">optuna_stage4_trial_0024</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/76gk3m3u' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/76gk3m3u</a>


    Loading and preparing results...
    DONE (t=0.75s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.64s).
    Accumulating evaluation results...
    DONE (t=3.67s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.103
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.151
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.070
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.111
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.120
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.083
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.126
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.154







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15143</td></tr><tr><td>val_AP75</td><td>0.11581</td></tr><tr><td>val_mAP</td><td>0.10321</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0024</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/76gk3m3u' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/76gk3m3u</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_110852-76gk3m3u/logs</code>


    [I 2026-03-15 11:11:20,277] Trial 24 finished with value: 0.1032133286806852 and parameters: {'score_thresh': 0.041477181801204244, 'box_nms_thresh': 0.4096753804303576, 'detections_per_img': 128}. Best is trial 1 with value: 0.10384203177373426.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_111120-h313tqnt</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h313tqnt' target="_blank">optuna_stage4_trial_0025</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h313tqnt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h313tqnt</a>


    Loading and preparing results...
    DONE (t=0.14s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=32.52s).
    Accumulating evaluation results...
    DONE (t=4.53s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.088
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.157







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15225</td></tr><tr><td>val_AP75</td><td>0.11631</td></tr><tr><td>val_mAP</td><td>0.10389</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0025</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h313tqnt' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/h313tqnt</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_111120-h313tqnt/logs</code>


    [I 2026-03-15 11:13:49,829] Trial 25 finished with value: 0.10388648832408537 and parameters: {'score_thresh': 0.01724239034736004, 'box_nms_thresh': 0.47720341117289755, 'detections_per_img': 88}. Best is trial 25 with value: 0.10388648832408537.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_111350-rsdwxbdf</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rsdwxbdf' target="_blank">optuna_stage4_trial_0026</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rsdwxbdf' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rsdwxbdf</a>


    Loading and preparing results...
    DONE (t=0.16s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=34.89s).
    Accumulating evaluation results...
    DONE (t=4.49s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.125
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.125
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.125
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.090
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.131
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.158







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15185</td></tr><tr><td>val_AP75</td><td>0.11624</td></tr><tr><td>val_mAP</td><td>0.10395</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0026</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rsdwxbdf' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/rsdwxbdf</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_111350-rsdwxbdf/logs</code>


    [I 2026-03-15 11:16:21,284] Trial 26 finished with value: 0.1039516321788818 and parameters: {'score_thresh': 0.017733957614140136, 'box_nms_thresh': 0.5248481622806969, 'detections_per_img': 162}. Best is trial 26 with value: 0.1039516321788818.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_111621-hqixvucf</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hqixvucf' target="_blank">optuna_stage4_trial_0027</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hqixvucf' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hqixvucf</a>


    Loading and preparing results...
    DONE (t=0.86s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.25s).
    Accumulating evaluation results...
    DONE (t=4.60s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.124
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.157







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.15244</td></tr><tr><td>val_AP75</td><td>0.11627</td></tr><tr><td>val_mAP</td><td>0.10393</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0027</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hqixvucf' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/hqixvucf</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_111621-hqixvucf/logs</code>


    [I 2026-03-15 11:18:51,352] Trial 27 finished with value: 0.10393124028984509 and parameters: {'score_thresh': 0.015912510049788366, 'box_nms_thresh': 0.4693612923587935, 'detections_per_img': 197}. Best is trial 26 with value: 0.1039516321788818.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_111851-xiy3egbb</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xiy3egbb' target="_blank">optuna_stage4_trial_0028</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xiy3egbb' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xiy3egbb</a>


    Loading and preparing results...
    DONE (t=0.82s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=33.91s).
    Accumulating evaluation results...
    DONE (t=6.99s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.104
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.152
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.116
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.072
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.112
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.135
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.123
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.130
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.156







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.1524</td></tr><tr><td>val_AP75</td><td>0.11619</td></tr><tr><td>val_mAP</td><td>0.10385</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0028</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xiy3egbb' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/xiy3egbb</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_111851-xiy3egbb/logs</code>


    [I 2026-03-15 11:21:24,935] Trial 28 finished with value: 0.10385359051868358 and parameters: {'score_thresh': 0.017102055041926096, 'box_nms_thresh': 0.4525185286868041, 'detections_per_img': 197}. Best is trial 26 with value: 0.1039516321788818.







Tracking run with wandb version 0.25.0



Run data is saved locally in <code>/content/wandb/run-20260315_112125-dv20dilc</code>



Syncing run <strong><a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dv20dilc' target="_blank">optuna_stage4_trial_0029</a></strong> to <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dv20dilc' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dv20dilc</a>


    Loading and preparing results...
    DONE (t=0.05s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=25.03s).
    Accumulating evaluation results...
    DONE (t=2.81s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.096
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.137
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.109
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.059
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.105
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.132
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.105
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.105
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.105
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.061
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.113
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.146







<br>    <style><br>        .wandb-row {<br>            display: flex;<br>            flex-direction: row;<br>            flex-wrap: wrap;<br>            justify-content: flex-start;<br>            width: 100%;<br>        }<br>        .wandb-col {<br>            display: flex;<br>            flex-direction: column;<br>            flex-basis: 100%;<br>            flex: 1;<br>            padding: 10px;<br>        }<br>    </style><br><div class="wandb-row"><div class="wandb-col"><h3>Run history:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>▁</td></tr><tr><td>val_AP75</td><td>▁</td></tr><tr><td>val_mAP</td><td>▁</td></tr></table><br/></div><div class="wandb-col"><h3>Run summary:</h3><br/><table class="wandb"><tr><td>val_AP50</td><td>0.13699</td></tr><tr><td>val_AP75</td><td>0.10945</td></tr><tr><td>val_mAP</td><td>0.09617</td></tr></table><br/></div></div>



View run <strong style="color:#cdcd00">optuna_stage4_trial_0029</strong> at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dv20dilc' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain/runs/dv20dilc</a><br> View project at: <a href='https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/danielayer-personal/faster-rcnn-optuna-coco-minitrain</a><br>Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_112125-dv20dilc/logs</code>


    [I 2026-03-15 11:23:44,172] Trial 29 finished with value: 0.09616940920004846 and parameters: {'score_thresh': 0.38592936490028584, 'box_nms_thresh': 0.4192510170998342, 'detections_per_img': 246}. Best is trial 26 with value: 0.1039516321788818.
    Best Stage 4: 0.1039516321788818
    Best post params: {'score_thresh': 0.017733957614140136, 'box_nms_thresh': 0.5248481622806969, 'detections_per_img': 162}



```python
'''
  Code execution was stopped after 2nd run despite setting runs to 15.  30 was computationally infeasable in the time alotted.
  Results were not significantly improving, so I decided to cut the trial and move on in the interest of time.
'''

print("Best Stage 4:", study_stage4.best_value)
print("Best params:", study_stage4.best_params)


```

    Best Stage 4: 0.1039516321788818
    Best params: {'score_thresh': 0.017733957614140136, 'box_nms_thresh': 0.5248481622806969, 'detections_per_img': 162}



```python
# Hardcode results.

'''

Best Stage 4: 0.1039516321788818
Best params: {'score_thresh': 0.017733957614140136, 'box_nms_thresh': 0.5248481622806969, 'detections_per_img': 162}

'''

study_stage4_best_value = 0.1039516321788818
study_stage4_best_params = {'score_thresh': 0.017733957614140136, 'box_nms_thresh': 0.5248481622806969, 'detections_per_img': 162}

```


## 11. Final multi-seed retraining (required)

Retrain the best configuration (Stages 1–4) with 3 different seeds and report:

$$
\text{mean mAP} \pm \text{std}.
$$

You must log all runs to W&B and include the W&B links in your report.



```python

# best_post = study_stage4.best_params if 'study_stage4' in globals() and study_stage4.best_params else {"score_thresh": 0.05, "box_nms_thresh": 0.5, "detections_per_img": 100}
best_post = study_stage4_best_params if 'study_stage4_best_params' in globals() and study_stage4_best_params else {"score_thresh": 0.05, "box_nms_thresh": 0.5, "detections_per_img": 100}

best_full = {**best_cfg, **best_post}
print("Best full config:", best_full)

SEEDS = [11, 22, 33]
ckpts = []


for s in SEEDS:
     ckpts.append(train_final_model(best_full, epochs=12, seed=s))


print("ckpts:", ckpts)

# Evaluate each checkpoint with best post-processing
maps = []
for ckpt in ckpts:
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    apply_rpn_hparams(model, best_full)
    apply_roi_hparams(model, best_full)
    apply_postprocess_hparams(model, best_full)
    metrics = evaluate_coco_map(model, coco, val_loader)
    maps.append(metrics["mAP"])
    print(ckpt, metrics)

maps = np.array(maps, dtype=np.float32)
print("mAP mean ± std:", float(maps.mean()), float(maps.std(ddof=1)))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[46], line 4
          1 # best_post = study_stage4.best_params if 'study_stage4' in globals() and study_stage4.best_params else {"score_thresh": 0.05, "box_nms_thresh": 0.5, "detections_per_img": 100}
          2 best_post = study_stage4_best_params if 'study_stage4_best_params' in globals() and study_stage4_best_params else {"score_thresh": 0.05, "box_nms_thresh": 0.5, "detections_per_img": 100}  
    ----> 4 best_full = {**best_cfg, **best_post}
          5 print("Best full config:", best_full)
          7 SEEDS = [11, 22, 33]


    NameError: name 'best_cfg' is not defined



## 12. Small-object transfer test: drones (extra credit)

You must evaluate:
- baseline COCO MiniTrain fine-tuned model (Section 6)
- tuned model (best configuration from Sections 7–11)

on the **drone dataset** defined in Assignment 3:

https://aegean.ai/aiml-common/assignments/main/cv-spring-2026/assignment-3

### Requirements
1. Do not retune hyperparameters on drones initially.
2. Compute at least:
   - $\mathrm{mAP}$, $\mathrm{AP}_{50}$, recall (or COCO AR)
3. Provide qualitative results showing:
   - missed small drones
   - duplicates / NMS issues
   - low-confidence detections

### Implementation note
You must make the drone dataset available in COCO format (images + instances JSON).

Set the paths below accordingly.



```python
# Contvert YOLo dataset to COCO.
!pip install pillow tqdm

!python yolo_to_coco.py --yolo_dir /home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/seraphim-drone-detection-dataset/ --output_dir /home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/content/seraphim-drone-detection-dataset_coco/ --data_yaml /home/johnsmith/Desktop/njit/workspaces/ds681/eng-ai-agents-main/assignments/assignment-3/datasets.yaml
```

    Requirement already satisfied: pillow in ./.venv/lib/python3.12/site-packages (12.1.1)
    Requirement already satisfied: tqdm in ./.venv/lib/python3.12/site-packages (4.67.3)
    Processing train: 100%|█████████████████| 58437/58437 [00:30<00:00, 1943.21it/s]
    ✅ train: 58437 images, 60352 annotations
    Saved → /home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/content/seraphim-drone-detection-dataset_coco/train/_annotations.coco.json
    
    Processing val: 100%|███████████████████| 16697/16697 [00:08<00:00, 1896.02it/s]
    ✅ val: 16697 images, 17262 annotations
    Saved → /home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/content/seraphim-drone-detection-dataset_coco/val/_annotations.coco.json
    
    Processing test: 100%|████████████████████| 8349/8349 [00:03<00:00, 2719.82it/s]
    ✅ test: 8349 images, 8626 annotations
    Saved → /home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/content/seraphim-drone-detection-dataset_coco/test/_annotations.coco.json
    
    Conversion finished successfully!



```python
BASELINE_CKPT = "/home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/checkpoints/baseline_fasterrcnn.pt"

```


```python
import os
from pycocotools.coco import COCO

# TODO: Set these paths to your drone dataset (COCO format) from Assignment 3
DRONE_ROOT = "/home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/content/seraphim-drone-detection-dataset_coco/test"
DRONE_IMAGES_DIR = os.path.join(DRONE_ROOT, "images")
DRONE_ANN_JSON = os.path.join(DRONE_ROOT, "annotations", "_annotations.coco.json")

# # Uncomment after you place the dataset:
assert os.path.exists(DRONE_IMAGES_DIR)
assert os.path.exists(DRONE_ANN_JSON)

# Change to 'kite' as COCO has no 'drone'.
drone_coco = COCO(DRONE_ANN_JSON)
for ann in drone_coco.dataset['annotations']:
    ann['category_id'] = 38
drone_coco.createIndex()

drone_img_ids = sorted(drone_coco.getImgIds())
drone_ds = CocoMiniTrainDataset(drone_coco, DRONE_IMAGES_DIR, drone_img_ids, train=False)
drone_loader = DataLoader(drone_ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

def eval_on_drones(ckpt_path: str, tag: str):
    run = wandb.init(project="faster-rcnn-optuna-coco-minitrain", name=f"drone_eval_{tag}", reinit=True)
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    apply_rpn_hparams(model, best_full)
    apply_roi_hparams(model, best_full)
    apply_postprocess_hparams(model, best_full)
    metrics = evaluate_coco_map(model, drone_coco, drone_loader)
    wandb.log({f"drone_{k}": v for k, v in metrics.items()})
    wandb.finish()
    return metrics

# # Example usage (after you set paths):
baseline_metrics = eval_on_drones(BASELINE_CKPT, "baseline")
tuned_metrics = eval_on_drones(ckpts[0], "tuned_seed11")
print("baseline:", baseline_metrics)
print("tuned:", tuned_metrics)

```

    loading annotations into memory...
    Done (t=0.02s)
    creating index...
    index created!
    creating index...
    index created!



Finishing previous runs because reinit is set to True.







View run <strong style="color:#cdcd00">drone_eval_baseline</strong> at: <a href='https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain/runs/5l7tt2m6' target="_blank">https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain/runs/5l7tt2m6</a><br> View project at: <a href='https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain</a><br>Synced 4 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)



Find logs at: <code>./wandb/run-20260315_145607-5l7tt2m6/logs</code>







Tracking run with wandb version 0.25.1



Run data is saved locally in <code>/home/johnsmith/Desktop/njit/workspaces/torchvision-frcnn-hpo/wandb/run-20260315_145833-6bi1gj9b</code>



Syncing run <strong><a href='https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain/runs/6bi1gj9b' target="_blank">drone_eval_baseline</a></strong> to <a href='https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain' target="_blank">Weights & Biases</a> (<a href='https://wandb.me/developer-guide' target="_blank">docs</a>)<br>



View project at <a href='https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain' target="_blank">https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain</a>



View run at <a href='https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain/runs/6bi1gj9b' target="_blank">https://wandb.ai/da632-new-jersey-institute-of-technology/faster-rcnn-optuna-coco-minitrain/runs/6bi1gj9b</a>



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[45], line 36
         33     return metrics
         35 # # Example usage (after you set paths):
    ---> 36 baseline_metrics = eval_on_drones(BASELINE_CKPT, "baseline")
         37 tuned_metrics = eval_on_drones(ckpts[0], "tuned_seed11")
         38 print("baseline:", baseline_metrics)


    Cell In[45], line 27, in eval_on_drones(ckpt_path, tag)
         25 model = build_model().to(device)
         26 model.load_state_dict(torch.load(ckpt_path, map_location=device))
    ---> 27 apply_rpn_hparams(model, best_full)
         28 apply_roi_hparams(model, best_full)
         29 apply_postprocess_hparams(model, best_full)


    NameError: name 'best_full' is not defined



```python
import os
print(os.getcwd())

```

    /content



```python
# Display results.

print(f"Basline vMAP: {0.08312}\n")

print(f"Study Stage 1 Best Value: {study_stage1_best_value}")
print(f"Study Stage 1 Best Params: {study_stage1_best_params}\n")

print(f"Study Stage 2 Best Value: {study_stage2_best_value}")
print(f"Study Stage 2 Best Params: {study_stage2_best_params}\n")

print(f"Study Stage 3 Best Value: {study_stage3_best_value}")
print(f"Study Stage 3 Best Params: {study_stage3_best_params}\n")

print(f"Study Stage 4 Best Value: {study_stage4_best_value}")
print(f"Study Stage 4 Best Params: {study_stage4_best_params}\n")


```

    Basline vMAP: 0.08312
    
    Study Stage 1 Best Value: 0.10272976055704393
    Study Stage 1 Best Params: {'seed': 2621, 'epochs': 3, 'lr': 6.829352954261399e-05, 'weight_decay': 6.87491841689458e-05, 'momentum': 0.8609901026988318, 'grad_clip_norm': 2.5919641029876854}
    
    Study Stage 2 Best Value: 0.10272976055704393
    Study Stage 2 Best Params: {'seed': 1853, 'rpn_nms_thresh': 0.8829436894685043, 'rpn_pre_nms_topk': 2276, 'rpn_post_nms_topk': 1157, 'rpn_fg_iou_thresh': 0.6531412840477842, 'rpn_bg_iou_thresh': 0.006316580884725776, 'rpn_batch_size_per_image': 409, 'rpn_positive_fraction': 0.7466525199454146}
    
    Study Stage 3 Best Value: 0.10355484910383297
    Study Stage 3 Best Params: {'seed': 2548, 'roi_batch_size_per_image': 133, 'roi_positive_fraction': 0.13561802688099756, 'cls_loss_weight': 1.9084739008189309, 'box_loss_weight': 1.9601890990515383}
    
    Study Stage 4 Best Value: 0.1039516321788818
    Study Stage 4 Best Params: {'score_thresh': 0.017733957614140136, 'box_nms_thresh': 0.5248481622806969, 'detections_per_img': 162}
    



## 13. Required written answers (include in your report)

Answer these questions using evidence (W&B plots, metrics, qualitative results):

1. Which Stage (1–4) delivered the largest gain in $\mathrm{mAP}$? Why?
2. Which hyperparameters most influenced small-object recall on drones?
3. Did increasing `rpn_pre_nms_topk` help drone detection? Explain using proposal reasoning.
4. Did changing NMS thresholds change the duplicate-box failure mode? Provide examples.
5. Is the tuned configuration robust across seeds? Use $\text{mean}\pm\text{std}$.



## 13. Required written answers (include in your report)

Answer these questions using evidence (W&B plots, metrics, qualitative results):

1. Which Stage (1–4) delivered the largest gain in $\mathrm{mAP}$? Why?
	Stage 1 delivered the largest gain from baseline to stage 1.  This is because we didn’t have enough time to fully run all of the runs and epochs due to time and compute constraints.  Each stage did increase performance somewhat, but the gains were negligible.  I added an extra Random Horizontal Flip Augmentation to try and improve performance.

Overall Project Writeup

	This project was plagued by two facts.  First, I had only one GPU to do two midterms.  Both requiring that compute, and both seemingly needing days worth of running, debugging, re-reunning, and the like.  I simply ran out of compute time.
	I will outline my experience and results here.
	I ran the notebook in it’s entirety with the demo 3 runs suggested.  This took all of the time I had allocated for this assignment.  I then re-ran the notebook paying for Colab H100 compute.  The drone section simply could not be completed trying to move from colab to my local machine.
	In the first run of the notebook vMAP values plateaued well below the stage 1 gains in the second run.  The second run added a missing scheduler.step() to utilize the scheduler.  I also added an augmentation flip to try and improve performance.  However, it would have taken over 150 hours to run the notebook as assigned.  That would be six and a half days without debugging.  Simply not possible.
	I decided to cut the remaining runtime I had into sections for each stage.  I ran each stage until the deadline came, then cut the run short and moved to the next stage.  This meant that while the second run had more epochs and runs per stage than the first, it was not possible to run the whole assignment in the time given.
	The gains in vMAP scores therefore again found a plateau.  
	The system did show that this methodology is sound, and of use.  It also showed that such a project should be allocated two weeks minimum for running, debugging, and final writeup.
	To run the final notebook I had to sleep in 1-4 hour stints to monitor and progress the notebook.  Colab timed out at various stages requiring me to resume from checkpoint.  I also hard coded findings in the notebook to guard against system outage, which occurred many times.

2. Which hyperparameters most influenced small-object recall on drones?
3. Did increasing `rpn_pre_nms_topk` help drone detection? Explain using proposal reasoning.
4. Did changing NMS thresholds change the duplicate-box failure mode? Provide examples.
5. Is the tuned configuration robust across seeds? Use $\text{mean}\pm\text{std}$.

	Because of the runtime configuration using Google Colab there was not time to run the drone extra credit.  I ran it in my first run, but it didn’t detect anything because of the mismatch between COCO and the drone class.  COCO has no ‘drone’ class.  I was going to try and see if it could detect it as ‘kite,’ but trying to move from Colab to my local instance to leverage the dataset needed on my machine broke everything.




