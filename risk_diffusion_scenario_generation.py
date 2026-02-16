"""
Risk-Conditioned Diffusion Model for Critical Traffic Scenario Generation


A production-grade implementation for generating critical traffic scenarios using
diffusion models, with risk-based conditioning and closed-loop evaluation. 
I am looking for more updation over the time, currently looking into versatile based optimization at inference phase.

Based on:
- MotionDiffuser (Jiang et al., 2023) - Controllable Multi-Agent Motion Prediction
- Scenario Diffusion (Chen et al., 2023) - Controllable Driving Scenario Generation
- highD Dataset (Krajewski et al., 2018) - Naturalistic Trajectory Dataset
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Configuration


@dataclass
class ModelConfig:
    """Diffusion model hyperparameters"""
    # Architecture
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1

    # Diffusion parameters
    num_diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Trajectory representation
    seq_len: int = 50  # 10 seconds at 5 Hz
    state_dim: int = 4  # [x, y, vx, vy]

    # Conditioning
    risk_embed_dim: int = 64
    use_classifier_free_guidance: bool = True
    cfg_dropout_prob: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 64
    num_epochs: int = 50
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # Data
    train_split: float = 0.85
    num_workers: int = 4

    # Checkpointing
    save_dir: str = "./experiments"
    checkpoint_every: int = 5

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvaluationConfig:
    """Evaluation and scenario generation parameters"""
    num_samples: int = 1000
    guidance_scales: List[float] = None

    # Criticality thresholds
    ttc_threshold: float = 3.0  # seconds
    distance_threshold: float = 5.0  # meters

    # Closed-loop simulation
    ego_controller_type: str = "idm"  # Intelligent Driver Model
    sim_dt: float = 0.2  # seconds

    def __post_init__(self):
        if self.guidance_scales is None:
            self.guidance_scales = [0.0, 1.0, 2.0, 3.0, 5.0]



# Data Generation and Loading


class ScenarioGenerator:
    """Generate synthetic traffic scenarios for training"""

    def __init__(self, seq_len: int = 50, dt: float = 0.2, seed: int = 42):
        self.seq_len = seq_len
        self.dt = dt
        self.rng = np.random.RandomState(seed)

    def generate_scenario(self, critical: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate a single car-following scenario

        Returns:
            Dict with 'ego_traj', 'lead_traj', 'risk_label'
        """
        # Ego vehicle: relatively constant speed with small variations
        ego_v0 = self.rng.uniform(20, 30)  # m/s (72-108 km/h)
        ego_vx = ego_v0 + self.rng.randn(self.seq_len) * 0.5
        ego_vx = np.clip(ego_vx, 10, 35)

        ego_x = np.cumsum(ego_vx * self.dt)
        ego_y = np.zeros(self.seq_len)
        ego_vy = np.zeros(self.seq_len)

        ego_traj = np.stack([ego_x, ego_y, ego_vx, ego_vy], axis=1)

        # Lead vehicle: varies based on criticality
        initial_gap = self.rng.uniform(30, 60)  # meters
        lead_x0 = ego_x[0] + initial_gap

        if critical:
            # Critical scenario: lead vehicle decelerates suddenly
            lead_v0 = self.rng.uniform(22, 32)
            brake_start = self.rng.randint(10, 25)
            brake_duration = self.rng.randint(10, 20)
            brake_decel = self.rng.uniform(-6, -4)  # harsh braking

            lead_vx = np.ones(self.seq_len) * lead_v0
            lead_vx[brake_start:brake_start+brake_duration] += np.linspace(
                0, brake_decel * brake_duration * self.dt, brake_duration
            )
            lead_vx[brake_start+brake_duration:] = np.clip(
                lead_vx[brake_start+brake_duration-1], 5, 35
            )
        else:
            # Normal scenario: lead vehicle maintains similar speed
            lead_v0 = ego_v0 + self.rng.uniform(-5, 5)
            lead_vx = lead_v0 + self.rng.randn(self.seq_len) * 1.0
            lead_vx = np.clip(lead_vx, 10, 35)

        lead_x = lead_x0 + np.cumsum(lead_vx * self.dt)
        lead_y = np.zeros(self.seq_len)
        lead_vy = np.zeros(self.seq_len)

        lead_traj = np.stack([lead_x, lead_y, lead_vx, lead_vy], axis=1)

        return {
            'ego_traj': ego_traj.astype(np.float32),
            'lead_traj': lead_traj.astype(np.float32),
            'risk_label': int(critical)
        }

    def generate_dataset(
        self,
        num_scenarios: int,
        critical_ratio: float = 0.3
    ) -> List[Dict]:
        """Generate a dataset of scenarios"""
        logger.info(f"Generating {num_scenarios} scenarios "
                   f"({critical_ratio*100:.1f}% critical)...")

        scenarios = []
        num_critical = int(num_scenarios * critical_ratio)

        for i in tqdm(range(num_scenarios), desc="Generating scenarios"):
            critical = i < num_critical
            scenario = self.generate_scenario(critical=critical)
            scenarios.append(scenario)

        # Shuffle
        self.rng.shuffle(scenarios)
        return scenarios


class TrafficScenarioDataset(Dataset):
    """PyTorch dataset for traffic scenarios"""

    def __init__(self, scenarios: List[Dict], normalize: bool = True):
        self.scenarios = scenarios
        self.normalize = normalize

        if normalize:
            self._compute_normalization_stats()

    def _compute_normalization_stats(self):
        """Compute mean/std for normalization"""
        all_trajs = []
        for s in self.scenarios:
            all_trajs.append(s['lead_traj'])

        all_trajs = np.concatenate(all_trajs, axis=0)
        self.mean = all_trajs.mean(axis=0)
        self.std = all_trajs.std(axis=0) + 1e-6

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scenario = self.scenarios[idx]

        lead_traj = scenario['lead_traj'].copy()
        if self.normalize:
            lead_traj = (lead_traj - self.mean) / self.std

        return {
            'ego_traj': torch.from_numpy(scenario['ego_traj']),
            'lead_traj': torch.from_numpy(lead_traj),
            'risk_label': torch.tensor(scenario['risk_label'], dtype=torch.long)
        }



# Criticality Metrics


class CriticalityMetrics:
    """Compute safety-critical metrics for traffic scenarios"""

    @staticmethod
    def compute_ttc(
        ego_traj: np.ndarray,
        lead_traj: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """
        Compute Time-to-Collision (TTC) at each timestep

        TTC = (x_lead - x_ego) / (v_ego - v_lead) if v_ego > v_lead, else inf

        Args:
            ego_traj: [T, 4] array with [x, y, vx, vy]
            lead_traj: [T, 4] array with [x, y, vx, vy]

        Returns:
            ttc: [T] array of TTC values
        """
        distance = lead_traj[:, 0] - ego_traj[:, 0]
        rel_velocity = ego_traj[:, 2] - lead_traj[:, 2]

        ttc = np.full_like(distance, np.inf)
        closing = rel_velocity > eps
        ttc[closing] = distance[closing] / rel_velocity[closing]
        ttc = np.clip(ttc, 0, 100)

        return ttc

    @staticmethod
    def compute_min_distance(
        ego_traj: np.ndarray,
        lead_traj: np.ndarray
    ) -> float:
        """Compute minimum distance between vehicles"""
        distances = np.linalg.norm(lead_traj[:, :2] - ego_traj[:, :2], axis=1)
        return distances.min()

    @staticmethod
    def compute_criticality_score(
        ego_traj: np.ndarray,
        lead_traj: np.ndarray,
        ttc_weight: float = 2.0,
        dist_weight: float = 1.0
    ) -> float:
        """
        Compute overall criticality score

        Higher score = more critical scenario
        """
        ttc = CriticalityMetrics.compute_ttc(ego_traj, lead_traj)
        min_ttc = ttc.min()
        min_dist = CriticalityMetrics.compute_min_distance(ego_traj, lead_traj)

        # Inverse-based scoring (smaller values → higher score)
        ttc_score = ttc_weight / (min_ttc + 1.0)
        dist_score = dist_weight / (min_dist + 1.0)

        return ttc_score + dist_score

    @staticmethod
    def is_critical(
        ego_traj: np.ndarray,
        lead_traj: np.ndarray,
        ttc_threshold: float = 3.0,
        dist_threshold: float = 5.0
    ) -> bool:
        """Check if scenario is critical"""
        ttc = CriticalityMetrics.compute_ttc(ego_traj, lead_traj)
        min_ttc = ttc.min()
        min_dist = CriticalityMetrics.compute_min_distance(ego_traj, lead_traj)

        return min_ttc < ttc_threshold or min_dist < dist_threshold



# Diffusion Model Components


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time step embedding"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of diffusion timesteps
        Returns:
            embeddings: [B, dim] tensor
        """
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings
        )
        embeddings = timesteps.float()[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class TransformerDenoiser(nn.Module):
    """Transformer-based denoising network for trajectory diffusion"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # Risk conditioning embedding
        self.risk_embed = nn.Embedding(2, config.risk_embed_dim)

        # Input projection
        self.input_proj = nn.Linear(config.state_dim, config.hidden_dim)

        # Positional encoding for sequence
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.seq_len, config.hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # Conditioning projection
        self.cond_proj = nn.Linear(
            config.risk_embed_dim,
            config.hidden_dim
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.state_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        risk_label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] noisy trajectory
            timesteps: [B] diffusion timesteps
            risk_label: [B] risk labels (0=normal, 1=critical), None for unconditional

        Returns:
            noise_pred: [B, T, D] predicted noise
        """
        B, T, D = x.shape

        # Time embedding
        t_emb = self.time_embed(timesteps)  # [B, hidden_dim]

        # Risk conditioning
        if risk_label is not None:
            r_emb = self.risk_embed(risk_label)  # [B, risk_embed_dim]
            r_emb = self.cond_proj(r_emb)  # [B, hidden_dim]
            cond_emb = t_emb + r_emb
        else:
            cond_emb = t_emb

        # Input projection and add positional encoding
        h = self.input_proj(x) + self.pos_embed[:, :T, :]  # [B, T, hidden_dim]

        # Add conditioning to all timesteps
        h = h + cond_emb[:, None, :]

        # Transformer
        h = self.transformer(h)

        # Output
        out = self.output_proj(h)

        return out


class DiffusionModel:
    """DDPM diffusion model for trajectory generation"""

    def __init__(self, config: ModelConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # Setup noise schedule
        self.num_steps = config.num_diffusion_steps
        self.betas = self._cosine_beta_schedule()

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Convert to torch
        self.register_schedule()

    def _cosine_beta_schedule(self) -> np.ndarray:
        """Cosine schedule as proposed in Improved DDPM"""
        steps = self.num_steps + 1
        s = 0.008
        x = np.linspace(0, self.num_steps, steps)
        alphas_cumprod = np.cos(((x / self.num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)

    def register_schedule(self):
        """Register schedule as buffers"""
        self.sqrt_alphas_cumprod = torch.from_numpy(
            np.sqrt(self.alphas_cumprod)
        ).float().to(self.device)

        self.sqrt_one_minus_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 - self.alphas_cumprod)
        ).float().to(self.device)

        self.sqrt_recip_alphas = torch.from_numpy(
            np.sqrt(1.0 / self.alphas)
        ).float().to(self.device)

        self.sqrt_recipm1_alphas_cumprod = torch.from_numpy(
            np.sqrt(1.0 / self.alphas_cumprod - 1)
        ).float().to(self.device)

        # Posterior variance for reverse process
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.from_numpy(posterior_variance).float().to(self.device)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x_start"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Broadcast to [B, 1, 1]
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        risk_label: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """Single reverse diffusion step with optional classifier-free guidance"""
        B = x.shape[0]
        t_tensor = torch.full((B,), t, device=self.device, dtype=torch.long)

        if guidance_scale > 0 and risk_label is not None:
            # Conditional prediction
            noise_cond = model(x, t_tensor, risk_label)

            # Unconditional prediction
            noise_uncond = model(x, t_tensor, None)

            # Classifier-free guidance
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = model(x, t_tensor, risk_label)

        # Compute x_{t-1}
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        betas_t = torch.tensor(self.betas[t], device=self.device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        risk_label: Optional[torch.Tensor] = None,
        guidance_scale: float = 0.0,
        progress: bool = True
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion"""
        model.eval()

        # Start from pure noise
        x = torch.randn(shape, device=self.device)

        # Reverse diffusion
        timesteps = range(self.num_steps - 1, -1, -1)
        if progress:
            timesteps = tqdm(timesteps, desc="Sampling")

        for t in timesteps:
            x = self.p_sample(model, x, t, risk_label, guidance_scale)

        return x


# ==
# Closed-Loop Simulation
# ==

class IDMController:
    """Intelligent Driver Model for car-following"""

    def __init__(
        self,
        v0: float = 30.0,  # desired velocity (m/s)
        T: float = 1.5,    # safe time headway (s)
        a: float = 2.0,    # max acceleration (m/s^2)
        b: float = 3.0,    # comfortable deceleration (m/s^2)
        s0: float = 2.0,   # minimum spacing (m)
        delta: float = 4.0 # acceleration exponent
    ):
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.s0 = s0
        self.delta = delta

    def compute_acceleration(
        self,
        v_ego: float,
        gap: float,
        v_lead: float
    ) -> float:
        """
        Compute IDM acceleration

        Args:
            v_ego: ego velocity (m/s)
            gap: distance to lead vehicle (m)
            v_lead: lead velocity (m/s)
        """
        dv = v_ego - v_lead
        s_star = self.s0 + max(0, v_ego * self.T + v_ego * dv / (2 * np.sqrt(self.a * self.b)))

        gap = max(gap, 0.1)  # prevent division by zero

        acc = self.a * (1 - (v_ego / self.v0) ** self.delta - (s_star / gap) ** 2)
        return acc


class ClosedLoopSimulator:
    """Closed-loop simulator for testing scenarios"""

    def __init__(self, dt: float = 0.2, controller_type: str = "idm"):
        self.dt = dt

        if controller_type == "idm":
            self.controller = IDMController()
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

    def simulate(
        self,
        ego_init: np.ndarray,
        lead_traj: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run closed-loop simulation

        Args:
            ego_init: [4] initial ego state [x, y, vx, vy]
            lead_traj: [T, 4] lead vehicle trajectory

        Returns:
            ego_traj: [T, 4] simulated ego trajectory
            info: dict with simulation statistics
        """
        T = len(lead_traj)
        ego_traj = np.zeros((T, 4))
        ego_traj[0] = ego_init

        min_gap = float('inf')
        min_ttc = float('inf')
        harsh_braking_count = 0
        collision = False

        for t in range(1, T):
            # Current state
            ego_x, ego_y, ego_vx, ego_vy = ego_traj[t-1]
            lead_x, lead_y, lead_vx, lead_vy = lead_traj[t]

            # Compute gap
            gap = lead_x - ego_x

            # Controller
            acc = self.controller.compute_acceleration(ego_vx, gap, lead_vx)

            # Update ego state
            ego_vx_new = max(0, ego_vx + acc * self.dt)
            ego_x_new = ego_x + ego_vx * self.dt + 0.5 * acc * self.dt ** 2

            ego_traj[t] = [ego_x_new, ego_y, ego_vx_new, ego_vy]

            # Track metrics
            min_gap = min(min_gap, gap)
            if ego_vx > lead_vx and gap > 0:
                ttc = gap / (ego_vx - lead_vx)
                min_ttc = min(min_ttc, ttc)

            if acc < -6.0:
                harsh_braking_count += 1

            if gap < 0:
                collision = True

        info = {
            'min_gap': min_gap,
            'min_ttc': min_ttc if min_ttc != float('inf') else 100,
            'harsh_braking_count': harsh_braking_count,
            'collision': collision
        }

        return ego_traj, info


# ==
# Training
# ==

class Trainer:
    """Training loop for diffusion model"""

    def __init__(
        self,
        model: nn.Module,
        diffusion: DiffusionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        self.model = model
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs
        )

        self.device = config.device
        self.model.to(self.device)

        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track best model
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            lead_traj = batch['lead_traj'].to(self.device)
            risk_label = batch['risk_label'].to(self.device)

            B, T, D = lead_traj.shape

            # Sample random timesteps
            t = torch.randint(
                0, self.diffusion.num_steps, (B,),
                device=self.device, dtype=torch.long
            )

            # Sample noise
            noise = torch.randn_like(lead_traj)

            # Add noise
            x_noisy = self.diffusion.q_sample(lead_traj, t, noise)

            # Classifier-free guidance: randomly drop conditioning
            if self.model.config.use_classifier_free_guidance:
                mask = torch.rand(B, device=self.device) > self.model.config.cfg_dropout_prob
                risk_label_input = risk_label.clone()
                risk_label_input[~mask] = -1  # Will be set to None in forward
                # Handle -1 as None
                risk_label_input = torch.where(
                    risk_label_input == -1,
                    torch.zeros_like(risk_label_input),
                    risk_label_input
                )
                risk_label_input = torch.where(
                    mask[:, None].expand_as(risk_label_input.unsqueeze(1)).squeeze(1),
                    risk_label_input,
                    torch.zeros_like(risk_label_input)  # dummy, will mask later
                )
                # Simpler: pass None for dropped samples
                noise_pred = self.model(
                    x_noisy, t,
                    risk_label_input if mask.all() else None
                )
            else:
                noise_pred = self.model(x_noisy, t, risk_label)

            # Loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0

        for batch in self.val_loader:
            lead_traj = batch['lead_traj'].to(self.device)
            risk_label = batch['risk_label'].to(self.device)

            B, T, D = lead_traj.shape
            t = torch.randint(
                0, self.diffusion.num_steps, (B,),
                device=self.device, dtype=torch.long
            )

            noise = torch.randn_like(lead_traj)
            x_noisy = self.diffusion.q_sample(lead_traj, t, noise)

            noise_pred = self.model(x_noisy, t, risk_label)
            loss = F.mse_loss(noise_pred, noise)

            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self):
        """Full training loop"""
        logger.info("Starting training...")

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            self.scheduler.step()

            # Save checkpoint
            if epoch % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch, val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)

        logger.info("Training complete!")

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': asdict(self.model.config)
        }

        if is_best:
            path = self.save_dir / 'best_model.pt'
            logger.info(f"Saving best model (val_loss={val_loss:.4f}) to {path}")
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'

        torch.save(checkpoint, path)


# ==
# Evaluation and Scenario Generation
# ==

class ScenarioEvaluator:
    """Evaluate and generate critical scenarios"""

    def __init__(
        self,
        model: nn.Module,
        diffusion: DiffusionModel,
        config: EvaluationConfig,
        dataset: TrafficScenarioDataset
    ):
        self.model = model
        self.diffusion = diffusion
        self.config = config
        self.dataset = dataset

        self.simulator = ClosedLoopSimulator(
            dt=config.sim_dt,
            controller_type=config.ego_controller_type
        )

        self.device = diffusion.device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_scenarios(
        self,
        num_samples: int,
        risk_label: int = 1,
        guidance_scale: float = 3.0
    ) -> List[np.ndarray]:
        """Generate scenarios using diffusion model"""
        logger.info(
            f"Generating {num_samples} scenarios "
            f"(risk={risk_label}, guidance={guidance_scale})..."
        )

        all_scenarios = []
        batch_size = 64

        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)

            shape = (
                current_batch,
                self.model.config.seq_len,
                self.model.config.state_dim
            )

            risk_tensor = torch.full(
                (current_batch,), risk_label,
                device=self.device, dtype=torch.long
            )

            samples = self.diffusion.sample(
                self.model, shape, risk_tensor,
                guidance_scale=guidance_scale,
                progress=False
            )

            # Denormalize
            samples_np = samples.cpu().numpy()
            if self.dataset.normalize:
                samples_np = samples_np * self.dataset.std + self.dataset.mean

            all_scenarios.extend(samples_np)

        return all_scenarios

    def evaluate_scenario(
        self,
        lead_traj: np.ndarray,
        ego_init: Optional[np.ndarray] = None
    ) -> Dict:
        """Evaluate a single scenario"""
        if ego_init is None:
            # Default: ego starts behind lead vehicle
            ego_init = np.array([
                lead_traj[0, 0] - 40.0,  # 40m behind
                0.0,
                25.0,  # 90 km/h
                0.0
            ])

        # Run closed-loop simulation
        ego_traj, sim_info = self.simulator.simulate(ego_init, lead_traj)

        # Compute metrics
        criticality_score = CriticalityMetrics.compute_criticality_score(
            ego_traj, lead_traj
        )

        is_critical = CriticalityMetrics.is_critical(
            ego_traj, lead_traj,
            ttc_threshold=self.config.ttc_threshold,
            dist_threshold=self.config.distance_threshold
        )

        return {
            'ego_traj': ego_traj,
            'lead_traj': lead_traj,
            'criticality_score': criticality_score,
            'is_critical': is_critical,
            **sim_info
        }

    def run_campaign(self, save_dir: Optional[Path] = None) -> Dict:
        """Run full evaluation campaign"""
        logger.info("Running evaluation campaign...")

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        all_results = defaultdict(list)

        for guidance_scale in self.config.guidance_scales:
            logger.info(f"Evaluating guidance_scale={guidance_scale}")

            # Generate scenarios
            scenarios = self.generate_scenarios(
                self.config.num_samples,
                risk_label=1,
                guidance_scale=guidance_scale
            )

            # Evaluate each scenario
            results = []
            for scenario in tqdm(scenarios, desc="Evaluating"):
                result = self.evaluate_scenario(scenario)
                results.append(result)

            # Aggregate statistics
            critical_count = sum(r['is_critical'] for r in results)
            collision_count = sum(r['collision'] for r in results)
            avg_criticality = np.mean([r['criticality_score'] for r in results])

            stats = {
                'guidance_scale': guidance_scale,
                'num_samples': len(scenarios),
                'critical_rate': critical_count / len(scenarios),
                'collision_rate': collision_count / len(scenarios),
                'avg_criticality_score': avg_criticality,
                'avg_min_ttc': np.mean([r['min_ttc'] for r in results]),
                'avg_min_gap': np.mean([r['min_gap'] for r in results])
            }

            all_results[f'guidance_{guidance_scale}'] = {
                'stats': stats,
                'scenarios': results
            }

            logger.info(
                f"Guidance {guidance_scale}: "
                f"Critical={stats['critical_rate']:.2%}, "
                f"Collision={stats['collision_rate']:.2%}, "
                f"Avg Score={stats['avg_criticality_score']:.3f}"
            )

        # Extract top critical scenarios across all guidance scales
        all_scenarios = []
        for key, data in all_results.items():
            all_scenarios.extend(data['scenarios'])

        # Sort by criticality score
        all_scenarios.sort(key=lambda x: x['criticality_score'], reverse=True)
        top_critical = all_scenarios[:20]

        all_results['top_critical_scenarios'] = top_critical

        # Save results
        if save_dir:
            self._save_results(all_results, save_dir)

        return all_results

    def _save_results(self, results: Dict, save_dir: Path):
        """Save evaluation results"""
        # Save statistics as JSON
        stats_file = save_dir / 'evaluation_stats.json'
        stats = {
            key: val['stats']
            for key, val in results.items()
            if key.startswith('guidance_')
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_file}")

        # Save top critical scenarios
        critical_dir = save_dir / 'critical_scenarios'
        critical_dir.mkdir(exist_ok=True)

        for i, scenario in enumerate(results['top_critical_scenarios']):
            scenario_file = critical_dir / f'scenario_{i:03d}.npz'
            np.savez(
                scenario_file,
                ego_traj=scenario['ego_traj'],
                lead_traj=scenario['lead_traj'],
                criticality_score=scenario['criticality_score'],
                min_ttc=scenario['min_ttc'],
                min_gap=scenario['min_gap'],
                collision=scenario['collision']
            )

        logger.info(f"Saved {len(results['top_critical_scenarios'])} "
                   f"critical scenarios to {critical_dir}")

        # Generate visualizations
        self._visualize_results(results, save_dir)

    def _visualize_results(self, results: Dict, save_dir: Path):
        """Create visualization plots"""
        viz_dir = save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)

        # Plot 1: Criticality vs guidance scale
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        guidance_scales = []
        critical_rates = []
        collision_rates = []
        avg_scores = []
        avg_ttcs = []

        for key, data in results.items():
            if key.startswith('guidance_'):
                stats = data['stats']
                guidance_scales.append(stats['guidance_scale'])
                critical_rates.append(stats['critical_rate'])
                collision_rates.append(stats['collision_rate'])
                avg_scores.append(stats['avg_criticality_score'])
                avg_ttcs.append(stats['avg_min_ttc'])

        axes[0, 0].plot(guidance_scales, critical_rates, 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Guidance Scale')
        axes[0, 0].set_ylabel('Critical Scenario Rate')
        axes[0, 0].set_title('Effect of Guidance on Criticality')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(guidance_scales, collision_rates, 'o-', linewidth=2, color='red')
        axes[0, 1].set_xlabel('Guidance Scale')
        axes[0, 1].set_ylabel('Collision Rate')
        axes[0, 1].set_title('Effect of Guidance on Collisions')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(guidance_scales, avg_scores, 'o-', linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Guidance Scale')
        axes[1, 0].set_ylabel('Avg Criticality Score')
        axes[1, 0].set_title('Average Criticality Score')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(guidance_scales, avg_ttcs, 'o-', linewidth=2, color='green')
        axes[1, 1].set_xlabel('Guidance Scale')
        axes[1, 1].set_ylabel('Avg Min TTC (s)')
        axes[1, 1].set_title('Average Minimum Time-to-Collision')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(viz_dir / 'guidance_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved guidance analysis plot to {viz_dir}")

        # Plot 2: Example critical scenarios
        top_scenarios = results['top_critical_scenarios'][:6]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, scenario in enumerate(top_scenarios):
            ax = axes[i]

            ego_traj = scenario['ego_traj']
            lead_traj = scenario['lead_traj']

            # Plot trajectories
            ax.plot(ego_traj[:, 0], ego_traj[:, 1], 'b-', linewidth=2, label='Ego')
            ax.plot(lead_traj[:, 0], lead_traj[:, 1], 'r-', linewidth=2, label='Lead')

            # Mark start and end
            ax.plot(ego_traj[0, 0], ego_traj[0, 1], 'bo', markersize=8)
            ax.plot(lead_traj[0, 0], lead_traj[0, 1], 'ro', markersize=8)
            ax.plot(ego_traj[-1, 0], ego_traj[-1, 1], 'bs', markersize=8)
            ax.plot(lead_traj[-1, 0], lead_traj[-1, 1], 'rs', markersize=8)

            # Add vehicle representations at critical moment
            ttc = CriticalityMetrics.compute_ttc(ego_traj, lead_traj)
            critical_idx = np.argmin(ttc)

            ego_x, ego_y = ego_traj[critical_idx, :2]
            lead_x, lead_y = lead_traj[critical_idx, :2]

            # Vehicle boxes (5m x 2m)
            ego_rect = patches.Rectangle(
                (ego_x - 2.5, ego_y - 1), 5, 2,
                linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.5
            )
            lead_rect = patches.Rectangle(
                (lead_x - 2.5, lead_y - 1), 5, 2,
                linewidth=2, edgecolor='red', facecolor='lightcoral', alpha=0.5
            )
            ax.add_patch(ego_rect)
            ax.add_patch(lead_rect)

            ax.set_xlabel('Longitudinal Position (m)')
            ax.set_ylabel('Lateral Position (m)')
            ax.set_title(
                f"Scenario {i+1}"
                f"Score={scenario['criticality_score']:.2f}, "
                f"MinTTC={scenario['min_ttc']:.1f}s"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

        plt.tight_layout()
        plt.savefig(viz_dir / 'top_critical_scenarios.png', dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved critical scenario visualizations to {viz_dir}")


# ==
# Main Pipeline
# ==

def main():
    """Main execution pipeline"""
    logger.info("="*80)
    logger.info("Risk-Conditioned Diffusion for Critical Scenario Generation")
    logger.info("="*80)

    # Configuration
    model_config = ModelConfig()
    train_config = TrainingConfig()
    eval_config = EvaluationConfig()

    logger.info(f"Device: {train_config.device}")
    logger.info(f"Model hidden_dim: {model_config.hidden_dim}")
    logger.info(f"Diffusion steps: {model_config.num_diffusion_steps}")

    # ========================================================================
    # Step 1: Generate synthetic dataset
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 1: Generating Synthetic Dataset")
    logger.info("="*80)

    generator = ScenarioGenerator(seq_len=model_config.seq_len)
    scenarios = generator.generate_dataset(
        num_scenarios=10000,
        critical_ratio=0.3
    )

    # Split train/val
    split_idx = int(len(scenarios) * train_config.train_split)
    train_scenarios = scenarios[:split_idx]
    val_scenarios = scenarios[split_idx:]

    logger.info(f"Train scenarios: {len(train_scenarios)}")
    logger.info(f"Val scenarios: {len(val_scenarios)}")

    # Create datasets
    train_dataset = TrafficScenarioDataset(train_scenarios, normalize=True)
    val_dataset = TrafficScenarioDataset(val_scenarios, normalize=True)
    val_dataset.mean = train_dataset.mean
    val_dataset.std = train_dataset.std

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for compatibility
        pin_memory=True if train_config.device == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if train_config.device == "cuda" else False
    )

    # ========================================================================
    # Step 2: Train diffusion model
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 2: Training Diffusion Model")
    logger.info("="*80)

    # Initialize model
    denoiser = TransformerDenoiser(model_config)
    diffusion = DiffusionModel(model_config, device=train_config.device)

    num_params = sum(p.numel() for p in denoiser.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Train
    trainer = Trainer(denoiser, diffusion, train_loader, val_loader, train_config)
    trainer.train()

    # Load best model
    best_checkpoint = torch.load(
        trainer.save_dir / 'best_model.pt',
        map_location=train_config.device
    )
    denoiser.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info(f"Loaded best model (val_loss={best_checkpoint['val_loss']:.4f})")

    # ========================================================================
    # Step 3: Generate and evaluate critical scenarios
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Step 3: Generating and Evaluating Critical Scenarios")
    logger.info("="*80)

    evaluator = ScenarioEvaluator(
        denoiser, diffusion, eval_config, train_dataset
    )

    results = evaluator.run_campaign(
        save_dir=Path(train_config.save_dir) / 'evaluation'
    )

    # ========================================================================
    # Step 4: Summary
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("Summary")
    logger.info("="*80)

    for key, data in results.items():
        if key.startswith('guidance_'):
            stats = data['stats']
            logger.info(
                f"Guidance {stats['guidance_scale']}: "
                f"Critical Rate = {stats['critical_rate']:.2%}, "
                f"Collision Rate = {stats['collision_rate']:.2%}, "
                f"Avg Criticality = {stats['avg_criticality_score']:.3f}"
            )

    logger.info("\n" + "="*80)
    logger.info("Pipeline Complete!")
    logger.info("="*80)
    logger.info(f"Results saved to: {train_config.save_dir}")
    logger.info(f"- Checkpoints: {train_config.save_dir}/")
    logger.info(f"- Evaluation: {train_config.save_dir}/evaluation/")
    logger.info(f"- Critical scenarios: {train_config.save_dir}/evaluation/critical_scenarios/")
    logger.info(f"- Visualizations: {train_config.save_dir}/evaluation/visualizations/")


if __name__ == "__main__":
    main()
