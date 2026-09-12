"""Self-contained HyperMolTab core for the exact paper-405 benchmark."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class HyperMolTabConfig:
    hidden_dim: int = 128
    dropout: float = 0.20
    tabm_k: int = 16
    learning_rate: float = 4e-4
    weight_decay: float = 3e-4
    batch_size: int = 64
    epochs: int = 180
    patience: int = 25
    lr_step_size: int = 0
    lr_scheduler_gamma: float = 1.0
    pretrain_epochs: int = 25
    focal_alpha: float = 1.0
    focal_gamma: float = 1.0
    rank_weight: float = 0.0
    distill_weight: float = 0.0
    use_graph: bool = True
    use_tabm: bool = True
    use_kan: bool = True
    use_tree: bool = True


@dataclass
class HyperMolTabArtifact:
    config: dict
    atom_dim: int
    vector_dim: int
    teacher_dim: int
    state_dict: dict
    selected_epoch: int
    seed: int


def variant_config(name: str, locked: dict) -> HyperMolTabConfig:
    variants = {
        "hyper_moltab": {},
        "hyper_moltab_no_graph": {"use_graph": False},
        "hyper_moltab_no_tabm": {"use_tabm": False},
        "hyper_moltab_no_kan": {"use_kan": False},
        "hyper_moltab_no_tree": {"use_tree": False},
        "hyper_moltab_no_cl": {"pretrain_epochs": 0},
        "hyper_moltab_no_graph_no_cl": {"use_graph": False, "pretrain_epochs": 0},
        "hyper_moltab_distill_rank": {
            "rank_weight": float(locked["rank_weight"]),
            "distill_weight": float(locked["distill_weight"]),
        },
    }
    if name not in variants:
        raise ValueError(f"Unknown HyperMolTab variant: {name}")
    base = {
        "hidden_dim": int(locked["hidden_dim"]),
        "dropout": float(locked["dropout"]),
        "tabm_k": int(locked["tabm_k"]),
        "learning_rate": float(locked["learning_rate"]),
        "weight_decay": float(locked["weight_decay"]),
        "batch_size": int(locked["batch_size"]),
        "epochs": int(locked["epochs"]),
        "patience": int(locked["patience"]),
        "lr_step_size": int(locked.get("lr_step_size", 0)),
        "lr_scheduler_gamma": float(locked.get("lr_scheduler_gamma", 1.0)),
        "pretrain_epochs": int(locked["pretrain_epochs"]),
        "focal_alpha": float(locked["focal_alpha"]),
        "focal_gamma": float(locked["focal_gamma"]),
    }
    return HyperMolTabConfig(**{**base, **variants[name]})


def focal_loss(logits, labels, *, alpha: float, gamma: float):
    """Binary focal loss; alpha weights positives while negative weight remains one."""
    log_probability = F.log_softmax(logits, dim=1)
    probability = log_probability.exp()
    row = torch.arange(labels.shape[0], device=labels.device)
    log_pt = log_probability[row, labels]
    pt = probability[row, labels]
    alpha_t = torch.where(labels == 1, float(alpha), 1.0).to(logits.dtype)
    return (-alpha_t * (1.0 - pt).pow(float(gamma)) * log_pt).mean()


_ATOM_NUMBERS = (1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53)


def _one_hot(value, choices) -> list[float]:
    return [float(value == item) for item in choices] + [float(value not in choices)]


def _atom_features(atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), _ATOM_NUMBERS)
        + _one_hot(atom.GetDegree(), (0, 1, 2, 3, 4, 5))
        + _one_hot(atom.GetTotalNumHs(), (0, 1, 2, 3, 4))
        + _one_hot(str(atom.GetHybridization()), ("SP", "SP2", "SP3", "SP3D", "SP3D2"))
        + [
            atom.GetFormalCharge() / 3.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() / 200.0,
        ]
    )


def smiles_to_graph(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"RDKit cannot parse curated paper SMILES: {smiles!r}")
    nodes = np.asarray([_atom_features(atom) for atom in molecule.GetAtoms()], dtype=np.float32)
    adjacency = np.eye(len(nodes), dtype=np.float32)
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[left, right] = adjacency[right, left] = 1.0
    degree = np.maximum(adjacency.sum(axis=1), 1.0)
    inverse = degree**-0.5
    return nodes, (inverse[:, None] * adjacency * inverse[None, :]).astype(np.float32)


class KANLinear(nn.Module):
    def __init__(self, inputs: int, outputs: int, grid_size: int = 5, order: int = 3):
        super().__init__()
        self.inputs, self.outputs, self.grid_size, self.order = (
            inputs,
            outputs,
            grid_size,
            order,
        )
        spacing = 2.0 / grid_size
        grid = torch.arange(-order, grid_size + order + 1) * spacing - 1.0
        self.register_buffer("grid", grid.expand(inputs, -1).contiguous())
        self.base = nn.Parameter(torch.empty(outputs, inputs))
        self.spline = nn.Parameter(torch.empty(outputs, inputs, grid_size + order))
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
        nn.init.normal_(self.spline, std=0.02)

    def _bases(self, values):
        expanded = values.unsqueeze(-1)
        bases = ((expanded >= self.grid[:, :-1]) & (expanded < self.grid[:, 1:])).to(
            values.dtype
        )
        for level in range(1, self.order + 1):
            left_den = self.grid[:, level:-1] - self.grid[:, : -(level + 1)]
            right_den = self.grid[:, level + 1 :] - self.grid[:, 1:-level]
            bases = (expanded - self.grid[:, : -(level + 1)]) / (left_den + 1e-8) * bases[
                :, :, :-1
            ] + (self.grid[:, level + 1 :] - expanded) / (right_den + 1e-8) * bases[:, :, 1:]
        return bases

    def forward(self, values):
        base = F.linear(F.silu(values), self.base)
        spline = F.linear(self._bases(values).flatten(1), self.spline.flatten(1))
        return base + spline

    def regularization_loss(self):
        return self.spline.abs().mean()


class KANHead(nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer = KANLinear(hidden, 2)
        self.shortcut = nn.Linear(hidden, 2)

    def forward(self, values):
        return self.layer(self.dropout(values)) + self.shortcut(values)

    def regularization_loss(self):
        return self.layer.regularization_loss()


class HyperMolTab(nn.Module):
    def __init__(
        self, atom_dim: int, vector_dim: int, teacher_dim: int, cfg: HyperMolTabConfig
    ):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.hidden_dim
        branches = 1
        if cfg.use_graph:
            self.graph_first = nn.Linear(atom_dim, hidden)
            self.graph_second = nn.Linear(hidden, hidden)
            self.graph_attention = nn.Linear(hidden, 1)
            self.graph_project = nn.Linear(2 * hidden, hidden)
            branches += 1
        if cfg.use_tabm:
            self.vector_norm = nn.LayerNorm(vector_dim)
            self.experts = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(vector_dim, hidden),
                        nn.SiLU(),
                        nn.Dropout(cfg.dropout),
                        nn.Linear(hidden, hidden),
                    )
                    for _ in range(cfg.tabm_k)
                ]
            )
        else:
            self.vector = nn.Sequential(
                nn.LayerNorm(vector_dim),
                nn.Linear(vector_dim, hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
            )
        if cfg.use_tree:
            self.teacher = nn.Sequential(
                nn.LayerNorm(teacher_dim),
                nn.Linear(teacher_dim, hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
            )
            branches += 1
        self.gate = nn.Sequential(
            nn.Linear(branches * hidden, hidden), nn.SiLU(), nn.Linear(hidden, branches)
        )
        self.head = (
            KANHead(hidden, cfg.dropout)
            if cfg.use_kan
            else nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden, 2),
            )
        )
        self.reconstruction = nn.Linear(hidden, vector_dim)

    def encode(self, node, adjacency, mask, vector, teacher):
        pieces = []
        if self.cfg.use_graph:
            mask_float = mask.unsqueeze(-1).to(node.dtype)
            hidden = F.silu(self.graph_first(torch.bmm(adjacency, node))) * mask_float
            hidden = F.silu(self.graph_second(torch.bmm(adjacency, hidden))) * mask_float
            scores = self.graph_attention(hidden).squeeze(-1).masked_fill(~mask, -1e9)
            attention = torch.softmax(scores, dim=1)
            attended = (hidden * attention.unsqueeze(-1)).sum(dim=1)
            mean = hidden.sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)
            pieces.append(F.silu(self.graph_project(torch.cat([attended, mean], dim=1))))
        if self.cfg.use_tabm:
            normal = self.vector_norm(vector)
            pieces.append(
                torch.stack([expert(normal) for expert in self.experts], dim=1).mean(dim=1)
            )
        else:
            pieces.append(self.vector(vector))
        if self.cfg.use_tree:
            pieces.append(self.teacher(teacher))
        weights = torch.softmax(self.gate(torch.cat(pieces, dim=1)), dim=1)
        return (torch.stack(pieces, dim=1) * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, node, adjacency, mask, vector, teacher):
        return self.head(self.encode(node, adjacency, mask, vector, teacher))

    def reconstruct(self, node, adjacency, mask, vector, teacher):
        return self.reconstruction(self.encode(node, adjacency, mask, vector, teacher))


class Rows(Dataset):
    def __init__(self, graphs, vectors, teachers, labels=None):
        self.graphs = graphs
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.teachers = np.asarray(teachers, dtype=np.float32)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, index):
        item = (self.graphs[index], self.vectors[index], self.teachers[index])
        return item if self.labels is None else (*item, self.labels[index])


def collate(batch):
    labelled = len(batch[0]) == 4
    graphs = [row[0] for row in batch]
    max_nodes = max(graph[0].shape[0] for graph in graphs)
    atom_dim = graphs[0][0].shape[1]
    node = torch.zeros((len(batch), max_nodes, atom_dim), dtype=torch.float32)
    adjacency = torch.zeros((len(batch), max_nodes, max_nodes), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_nodes), dtype=torch.bool)
    for index, (features, adj) in enumerate(graphs):
        count = len(features)
        node[index, :count] = torch.from_numpy(features)
        adjacency[index, :count, :count] = torch.from_numpy(adj)
        mask[index, :count] = True
    vectors = torch.from_numpy(np.asarray([row[1] for row in batch], dtype=np.float32))
    teachers = torch.from_numpy(np.asarray([row[2] for row in batch], dtype=np.float32))
    if not labelled:
        return node, adjacency, mask, vectors, teachers
    return (
        node,
        adjacency,
        mask,
        vectors,
        teachers,
        torch.tensor([row[3] for row in batch], dtype=torch.long),
    )


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _loader(graphs, vectors, teachers, labels, cfg, *, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Rows(graphs, vectors, teachers, labels),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=collate,
    )


def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for node, adjacency, mask, vector, teacher in loader:
            logits = model(
                node.to(device),
                adjacency.to(device),
                mask.to(device),
                vector.to(device),
                teacher.to(device),
            )
            output.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    probability = np.concatenate(output)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise FloatingPointError("HyperMolTab returned invalid probabilities")
    return np.clip(probability, 1e-7, 1 - 1e-7)


def _rank_loss(logits, labels):
    scores = logits[:, 1] - logits[:, 0]
    positive, negative = scores[labels == 1], scores[labels == 0]
    if not positive.numel() or not negative.numel():
        return logits.new_zeros(())
    return F.softplus(-(positive[:, None] - negative[None, :])).mean()


def fit_model(
    X_train,
    teacher_train,
    smiles_train: Sequence[str],
    y_train,
    *,
    X_validation=None,
    teacher_validation=None,
    smiles_validation=None,
    y_validation=None,
    config: HyperMolTabConfig,
    seed: int,
    requested_device: str = "auto",
    fixed_epochs: int | None = None,
):
    from sklearn.metrics import average_precision_score

    _seed(seed)
    device = _device(requested_device)
    train_graphs = [smiles_to_graph(value) for value in smiles_train]
    valid_graphs = (
        None
        if smiles_validation is None
        else [smiles_to_graph(value) for value in smiles_validation]
    )
    y_train = np.asarray(y_train, dtype=np.int64)
    teacher_train = np.asarray(teacher_train, dtype=np.float32)
    if not config.use_tree:
        teacher_train = np.zeros((len(y_train), 1), dtype=np.float32)
        if teacher_validation is not None:
            teacher_validation = np.zeros((len(teacher_validation), 1), dtype=np.float32)
    model = HyperMolTab(
        train_graphs[0][0].shape[1], X_train.shape[1], teacher_train.shape[1], config
    ).to(device)
    train_loader = _loader(
        train_graphs, X_train, teacher_train, y_train, config, shuffle=True, seed=seed
    )
    if config.pretrain_epochs:
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        pretrain_scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimiser,
                step_size=config.lr_step_size,
                gamma=config.lr_scheduler_gamma,
            )
            if config.lr_step_size > 0
            else None
        )
        for _ in range(config.pretrain_epochs):
            model.train()
            for node, adjacency, mask, vector, teacher, _labels in train_loader:
                node, adjacency, mask, vector, teacher = (
                    node.to(device),
                    adjacency.to(device),
                    mask.to(device),
                    vector.to(device),
                    teacher.to(device),
                )
                optimiser.zero_grad(set_to_none=True)
                loss = F.smooth_l1_loss(
                    model.reconstruct(node, adjacency, mask, vector, teacher), vector
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            if pretrain_scheduler is not None:
                pretrain_scheduler.step()
    valid_loader = None
    if fixed_epochs is None:
        if any(
            value is None
            for value in (X_validation, teacher_validation, smiles_validation, y_validation)
        ):
            raise ValueError("Epoch selection requires a complete validation partition")
        valid_loader = _loader(
            valid_graphs,
            X_validation,
            teacher_validation,
            None,
            config,
            shuffle=False,
            seed=seed,
        )
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size=config.lr_step_size,
            gamma=config.lr_scheduler_gamma,
        )
        if config.lr_step_size > 0
        else None
    )
    total_epochs = int(config.epochs if fixed_epochs is None else fixed_epochs)
    best_state, best_epoch, best_score, stale = None, total_epochs, -np.inf, 0
    for epoch in range(1, total_epochs + 1):
        model.train()
        for node, adjacency, mask, vector, teacher, labels in train_loader:
            node, adjacency, mask, vector, teacher, labels = (
                node.to(device),
                adjacency.to(device),
                mask.to(device),
                vector.to(device),
                teacher.to(device),
                labels.to(device),
            )
            optimiser.zero_grad(set_to_none=True)
            logits = model(node, adjacency, mask, vector, teacher)
            loss = focal_loss(
                logits, labels, alpha=config.focal_alpha, gamma=config.focal_gamma
            )
            if config.rank_weight:
                loss = loss + config.rank_weight * _rank_loss(logits, labels)
            if config.distill_weight:
                target = ((teacher[:, 0] + teacher[:, 2]) / 2.0).clamp(1e-6, 1 - 1e-6)
                loss = loss + config.distill_weight * F.binary_cross_entropy(
                    torch.softmax(logits, dim=1)[:, 1].clamp(1e-6, 1 - 1e-6), target
                )
            if hasattr(model.head, "regularization_loss"):
                loss = loss + 1e-6 * model.head.regularization_loss()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite HyperMolTab focal loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        if scheduler is not None:
            scheduler.step()
        if valid_loader is not None:
            probability = _predict(model, valid_loader, device)
            score = float(average_precision_score(np.asarray(y_validation), probability))
            if score > best_score + 1e-8:
                best_score, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                )
            else:
                stale += 1
                if stale >= config.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    valid_probability = None if valid_loader is None else _predict(model, valid_loader, device)
    artifact = HyperMolTabArtifact(
        asdict(config),
        train_graphs[0][0].shape[1],
        X_train.shape[1],
        teacher_train.shape[1],
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        best_epoch,
        seed,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return artifact, valid_probability


def predict_model(
    artifact: HyperMolTabArtifact,
    X,
    teachers,
    smiles: Sequence[str],
    *,
    requested_device: str = "auto",
) -> np.ndarray:
    cfg = HyperMolTabConfig(**artifact.config)
    if not cfg.use_tree:
        teachers = np.zeros((len(X), 1), dtype=np.float32)
    if (
        X.shape[1] != artifact.vector_dim
        or np.asarray(teachers).shape[1] != artifact.teacher_dim
    ):
        raise ValueError("HyperMolTab prediction feature dimensions differ from the fit")
    device = _device(requested_device)
    model = HyperMolTab(artifact.atom_dim, artifact.vector_dim, artifact.teacher_dim, cfg).to(
        device
    )
    model.load_state_dict(artifact.state_dict)
    graphs = [smiles_to_graph(value) for value in smiles]
    loader = _loader(graphs, X, teachers, None, cfg, shuffle=False, seed=artifact.seed)
    probability = _predict(model, loader, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probability
