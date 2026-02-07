import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from flcore.trainmodel.models import FedAvgCNN


class EarlyExitWrapperBase(nn.Module):
    def __init__(self, backbone, num_classes, model_family, init_head=None, target_dim=None):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.model_family = model_family
        self.head_by_exit = nn.ModuleDict()
        self.adapter_by_exit = nn.ModuleDict()
        self.exit_dims = {}
        self.final_dim = None
        self.target_dim = target_dim
        self._init_head = init_head
        self._init_head_applied = False

    def extract_exit_features(self, x):
        raise NotImplementedError

    def _adapt_exit_features(self, feats_by_exit):
        if not feats_by_exit:
            return feats_by_exit
        final_exit = max(feats_by_exit.keys())
        final_feats = feats_by_exit[final_exit]
        if final_feats.ndim != 2:
            raise ValueError(f"Final exit features must be 2D, got {final_feats.shape}")
        self.final_dim = final_feats.shape[1]
        target_dim = self.target_dim or self.final_dim

        adapted = {}
        for exit_id, feats in feats_by_exit.items():
            if feats.ndim != 2:
                raise ValueError(f"Exit {exit_id} features must be 2D, got {feats.shape}")
            if feats.shape[1] != target_dim:
                key = str(exit_id)
                if key not in self.adapter_by_exit:
                    self.adapter_by_exit[key] = nn.Linear(feats.shape[1], target_dim).to(feats.device)
                feats = self.adapter_by_exit[key](feats)
            adapted[exit_id] = feats
        return adapted

    def _ensure_heads_initialized(self, feats_by_exit):
        for exit_id, feats in feats_by_exit.items():
            if feats.ndim != 2:
                raise ValueError(f"Exit {exit_id} features must be 2D, got {feats.shape}")
            key = str(exit_id)
            if key not in self.head_by_exit:
                self.head_by_exit[key] = nn.Linear(feats.shape[1], self.num_classes).to(feats.device)
            self.exit_dims[exit_id] = feats.shape[1]

        # initialize the final exit head from existing head if provided
        if self._init_head is not None and not self._init_head_applied:
            final_exit = max(self.exit_dims.keys())
            head = self.head_by_exit[str(final_exit)]
            if isinstance(self._init_head, nn.Linear) and head.in_features == self._init_head.in_features:
                head.load_state_dict(self._init_head.state_dict())
            self._init_head_applied = True

    def get_exit_features(self, x):
        feats_by_exit = self.extract_exit_features(x)
        return self._adapt_exit_features(feats_by_exit)

    def forward(self, x, exit_id=None, return_features=False):
        feats_by_exit = self.get_exit_features(x)
        self._ensure_heads_initialized(feats_by_exit)
        if exit_id is None:
            exit_id = max(feats_by_exit.keys())
        feats = feats_by_exit[exit_id]
        logits = self.head_by_exit[str(exit_id)](feats)
        if return_features:
            return logits, feats
        return logits

    def forward_all_exits(self, x):
        feats_by_exit = self.get_exit_features(x)
        self._ensure_heads_initialized(feats_by_exit)
        logits_by_exit = {}
        for exit_id, feats in feats_by_exit.items():
            logits_by_exit[exit_id] = self.head_by_exit[str(exit_id)](feats)
        return logits_by_exit

    def get_exit_ids(self):
        if not self.head_by_exit:
            return []
        return sorted([int(k) for k in self.head_by_exit.keys()])

    def load_exit_heads(self, state_for_family):
        if not state_for_family:
            return
        device = next(self.parameters()).device
        for exit_id, payload in state_for_family.items():
            try:
                exit_key = int(exit_id)
            except (TypeError, ValueError):
                exit_key = exit_id
            in_dim = payload.get("in_dim", None)
            state_dict = payload.get("state_dict", None)
            if in_dim is None or state_dict is None:
                continue
            key = str(exit_key)
            if key not in self.head_by_exit or self.head_by_exit[key].in_features != in_dim:
                self.head_by_exit[key] = nn.Linear(in_dim, self.num_classes).to(device)
            self.head_by_exit[key].load_state_dict(state_dict)
            self.exit_dims[exit_key] = in_dim

    def export_exit_heads(self):
        export = {}
        for exit_id, dim in self.exit_dims.items():
            head = self.head_by_exit[str(exit_id)]
            export[exit_id] = {
                "in_dim": dim,
                "state_dict": head.state_dict(),
            }
        return export


def _pool_flatten(x):
    x = F.adaptive_avg_pool2d(x, 1)
    return torch.flatten(x, 1)


class ResNet18EarlyExit(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        exit0 = _pool_flatten(x)
        x = b.layer3(x)
        x = b.layer4(x)
        exit1 = _pool_flatten(x)
        return {0: exit0, 1: exit1}


class ResNet18EarlyExit3(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        x = b.layer2(x)
        exit0 = _pool_flatten(x)
        x = b.layer3(x)
        exit1 = _pool_flatten(x)
        x = b.layer4(x)
        exit2 = _pool_flatten(x)
        return {0: exit0, 1: exit1, 2: exit2}


class MobileNetV2EarlyExit(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        feats = self.backbone.features
        x = feats[:7](x)
        exit0 = _pool_flatten(x)
        x = feats[7:](x)
        exit1 = _pool_flatten(x)
        return {0: exit0, 1: exit1}


class MobileNetV2EarlyExit3(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        feats = self.backbone.features
        n = len(feats)
        idx1 = 7 if n >= 7 else max(1, n // 3)
        idx2 = 14 if n >= 14 else max(idx1 + 1, (2 * n) // 3)
        idx1 = min(idx1, n)
        idx2 = min(idx2, n)
        if idx2 <= idx1:
            idx2 = min(idx1 + 1, n)

        x = feats[:idx1](x)
        exit0 = _pool_flatten(x)
        x = feats[idx1:idx2](x)
        exit1 = _pool_flatten(x)
        x = feats[idx2:](x)
        exit2 = _pool_flatten(x)
        return {0: exit0, 1: exit1, 2: exit2}


class GoogLeNetEarlyExit(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.maxpool1(x)
        x = b.conv2(x)
        x = b.conv3(x)
        x = b.maxpool2(x)
        x = b.inception3a(x)
        x = b.inception3b(x)
        x = b.maxpool3(x)
        x = b.inception4a(x)
        x = b.inception4b(x)
        x = b.inception4c(x)
        x = b.inception4d(x)
        exit0 = _pool_flatten(x)
        x = b.inception4e(x)
        x = b.maxpool4(x)
        x = b.inception5a(x)
        x = b.inception5b(x)
        x = b.avgpool(x)
        exit1 = torch.flatten(x, 1)
        return {0: exit0, 1: exit1}


class GoogLeNetEarlyExit3(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.maxpool1(x)
        x = b.conv2(x)
        x = b.conv3(x)
        x = b.maxpool2(x)
        x = b.inception3a(x)
        x = b.inception3b(x)
        x = b.maxpool3(x)
        x = b.inception4a(x)
        x = b.inception4b(x)
        exit0 = _pool_flatten(x)
        x = b.inception4c(x)
        x = b.inception4d(x)
        exit1 = _pool_flatten(x)
        x = b.inception4e(x)
        x = b.maxpool4(x)
        x = b.inception5a(x)
        x = b.inception5b(x)
        x = b.avgpool(x)
        exit2 = torch.flatten(x, 1)
        return {0: exit0, 1: exit1, 2: exit2}


class FedAvgCNNEarlyExit(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        x = b.conv2(x)
        exit0 = torch.flatten(x, 1)
        exit1 = b.fc1(exit0)
        return {0: exit0, 1: exit1}


class FedAvgCNNEarlyExit3(EarlyExitWrapperBase):
    def extract_exit_features(self, x):
        b = self.backbone
        x = b.conv1(x)
        exit0 = torch.flatten(x, 1)
        x = b.conv2(x)
        exit1 = torch.flatten(x, 1)
        exit2 = b.fc1(exit1)
        return {0: exit0, 1: exit1, 2: exit2}


def infer_model_family(model):
    if isinstance(model, EarlyExitWrapperBase):
        return model.model_family
    base = model.base if hasattr(model, "base") else model
    if isinstance(base, FedAvgCNN):
        return "fedavgcnn"
    if isinstance(base, torchvision.models.ResNet):
        return "resnet"
    if isinstance(base, torchvision.models.MobileNetV2):
        return "mobilenet_v2"
    if isinstance(base, torchvision.models.GoogLeNet):
        return "googlenet"
    name = base.__class__.__name__.lower()
    if "resnet" in name:
        return "resnet"
    if "mobilenet" in name:
        return "mobilenet_v2"
    if "googlenet" in name:
        return "googlenet"
    if "fedavgcnn" in name:
        return "fedavgcnn"
    return "unknown"


def wrap_with_early_exit(model, model_family, num_classes, feature_dim=None, num_exits=None):
    if isinstance(model, EarlyExitWrapperBase):
        return model

    base = model.base if hasattr(model, "base") else model
    family = model_family or infer_model_family(model)

    init_head = None
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        init_head = model.head
    elif hasattr(base, "fc") and isinstance(base.fc, nn.Linear):
        init_head = base.fc
    elif hasattr(base, "classifier") and isinstance(base.classifier, nn.Linear):
        init_head = base.classifier

    use_three_exits = num_exits == 3
    if family in {"resnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"}:
        if use_three_exits:
            return ResNet18EarlyExit3(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
        return ResNet18EarlyExit(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
    if family == "mobilenet_v2":
        if use_three_exits:
            return MobileNetV2EarlyExit3(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
        return MobileNetV2EarlyExit(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
    if family == "googlenet":
        if use_three_exits:
            return GoogLeNetEarlyExit3(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
        return GoogLeNetEarlyExit(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
    if family == "fedavgcnn":
        if use_three_exits:
            return FedAvgCNNEarlyExit3(base, num_classes, family, init_head=init_head, target_dim=feature_dim)
        return FedAvgCNNEarlyExit(base, num_classes, family, init_head=init_head, target_dim=feature_dim)

    raise NotImplementedError(f"Early-exit wrapper not implemented for model family: {family}")
