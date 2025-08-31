import argparse
import os
import sys
import math
import random
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

# Ensure project root import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from src.model import tiny_model
from src.contrastive import SupConLoss
from src.synthetic import EEGConfig, generate_synthetic_eeg
from src.blocks import TwoStageCLSPool
from src.revin import RevIN


class EEGWindows(Dataset):
	def __init__(self, X: np.ndarray, subj_ids: np.ndarray, task_ids: np.ndarray, L: int, stride: int):
		self.L = L
		self.stride = stride
		self.X = X  # (Nseq, T, C)
		self.subj = subj_ids
		self.task = task_ids
		self.windows = []
		for i in range(X.shape[0]):
			T = X[i].shape[0]
			for t0 in range(0, max(1, T - L + 1), stride):
				self.windows.append((i, t0))

	def __len__(self):
		return len(self.windows)

	def __getitem__(self, idx: int):
		i, t0 = self.windows[idx]
		x = torch.from_numpy(self.X[i, t0:t0 + self.L]).float()  # (L, C)
		return x, int(self.subj[i]), int(self.task[i])



class DualHeadEncoder(nn.Module):
	def __init__(self, num_channels: int, r: int, proj_dim: int = 64, sink_time: int = 2, sink_channel: int = 2):
		super().__init__()
		self.backbone = tiny_model(num_channels=num_channels, r=r)
		self.cls = TwoStageCLSPool(
			num_channels=num_channels,
			r=r,
			r_v=max(4, r//2),
			d_qk=max(8, r),
			z=max(8, r),
			num_sink_q_time=max(0, sink_time),
			num_sink_q_channel=max(0, sink_channel),
		)
		self.proj = nn.Sequential(
			nn.Linear(r, r), nn.GELU(), nn.Linear(r, 2 * proj_dim)
		)

	def forward(self, x: Tensor):
		# x: (B, L, C)
		h = self.backbone(x)  # (B, L, C, r)
		p = self.cls(h)  # (B, r)
		z = self.proj(p)
		d = z.shape[-1] // 2
		return z[:, :d], z[:, d:]


def main():
	parser = argparse.ArgumentParser(description='Contrastive training on synthetic EEG (subject/task)')
	parser.add_argument('--subjects', type=int, default=8)
	parser.add_argument('--tasks', type=int, default=3)
	parser.add_argument('--T', type=int, default=2000)
	parser.add_argument('--C', type=int, default=16)
	parser.add_argument('--L', type=int, default=256)
	parser.add_argument('--stride', type=int, default=128)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--r', type=int, default=8)
	parser.add_argument('--proj_dim', type=int, default=64)
	parser.add_argument('--sink_time', type=int, default=2, help='Number of sink queries in temporal attention')
	parser.add_argument('--sink_channel', type=int, default=2, help='Number of sink queries in channel attention')
	# augmentation config
	parser.add_argument('--no_aug', action='store_true', help='Disable all augmentations and use 1 view')
	parser.add_argument('--num_views', type=int, default=2, help='Number of jittered views per sample')
	parser.add_argument('--aug_amp', type=float, default=0.2, help='Amplitude scaling range ±value')
	parser.add_argument('--aug_noise', type=float, default=0.01, help='Gaussian noise std')
	parser.add_argument('--aug_shift', type=float, default=0.1, help='Max time shift fraction of L')
	parser.add_argument('--aug_resample', type=float, default=0.1, help='Max global resample factor delta')
	# loss selection
	parser.add_argument('--loss', type=str, default='supcon', choices=['supcon', 'infonce_instance', 'infonce_masked'], help='Loss type for contrastive training')
	parser.add_argument('--lr', type=float, default=3e-3)
	parser.add_argument('--wd', type=float, default=1e-4)
	parser.add_argument('--temp', type=float, default=0.1)
	parser.add_argument('--cpu', action='store_true')
	args = parser.parse_args()

	device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
	print('Device:', device)

	cfg = EEGConfig(T=args.T, C=args.C, num_subjects=args.subjects, num_tasks=args.tasks)
	X, subj_ids, task_ids = generate_synthetic_eeg(cfg)
	ds = EEGWindows(X, subj_ids, task_ids, L=args.L, stride=args.stride)
	pin = device.type == 'cuda'
	loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=pin)

	# Optional: apply RevIN normalization to inputs before encoding
	revin = RevIN(num_channels=args.C, affine=True).to(device)

	model = DualHeadEncoder(
		num_channels=args.C,
		r=args.r,
		proj_dim=args.proj_dim,
		sink_time=args.sink_time,
		sink_channel=args.sink_channel,
	).to(device)
	crit = SupConLoss(temperature=args.temp)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
	def info_nce(q: Tensor, k: Tensor, temperature: float) -> Tensor:
		# q, k: (B, D)
		q = torch.nn.functional.normalize(q, dim=-1)
		k = torch.nn.functional.normalize(k, dim=-1)
		logits = (q @ k.t()) / max(1e-8, temperature)  # (B,B)
		target = torch.arange(q.size(0), device=q.device)
		return torch.nn.functional.cross_entropy(logits, target)

	def masked_info_nce(q: Tensor, k: Tensor, labels: Tensor, temperature: float) -> Tensor:
		# q, k: (B, D); labels: (B,) ints. Positive is diag. Mask out entries where labels[i]==labels[j] and i!=j.
		q = torch.nn.functional.normalize(q, dim=-1).float()
		k = torch.nn.functional.normalize(k, dim=-1).float()
		logits = (q @ k.t()) / max(1e-8, temperature)  # (B,B) float32 for stability
		B = labels.size(0)
		same = labels.view(B, 1).eq(labels.view(1, B))
		eye = torch.eye(B, device=labels.device, dtype=torch.bool)
		mask_bad = same & (~eye)
		# Use a large negative sentinel safe for float32 and float16 contexts
		logits = logits.masked_fill(mask_bad, -1e4)
		target = torch.arange(B, device=logits.device)
		return torch.nn.functional.cross_entropy(logits, target)


	scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

	# Effective augmentation knobs (respect --no_aug)
	eff_num_views = 1 if args.no_aug else max(1, args.num_views)
	eff_aug_amp = 0.0 if args.no_aug else max(0.0, args.aug_amp)
	eff_aug_noise = 0.0 if args.no_aug else max(0.0, args.aug_noise)
	eff_aug_shift = 0.0 if args.no_aug else max(0.0, args.aug_shift)
	eff_aug_resample = 0.0 if args.no_aug else max(0.0, args.aug_resample)

	def amp_scale(x: Tensor, strength: float) -> Tensor:
		if strength <= 0: return x
		scale = 1.0 + (2 * torch.rand(x.size(0), 1, 1, device=x.device) - 1.0) * strength
		return x * scale

	def add_noise(x: Tensor, sigma: float) -> Tensor:
		if sigma <= 0: return x
		return x + torch.randn_like(x) * sigma

	def time_shift(x: Tensor, max_frac: float) -> Tensor:
		if max_frac <= 0: return x
		B, L, C = x.shape
		shift_max = max(1, int(L * max_frac))
		shifts = torch.randint(-shift_max, shift_max + 1, (B,), device=x.device)
		idx = (torch.arange(L, device=x.device).unsqueeze(0) - shifts.unsqueeze(1)) % L
		gather_idx = idx.unsqueeze(-1).expand(B, L, C)
		return x.gather(dim=1, index=gather_idx)

	def global_resample(x: Tensor, max_delta: float) -> Tensor:
		# Resample with factor f in [1-delta, 1+delta] via linear interpolation
		if max_delta <= 0: return x
		B, L, C = x.shape
		factors = 1.0 + (2 * torch.rand(B, device=x.device) - 1.0) * max_delta
		t = torch.arange(L, device=x.device).float()
		out = torch.empty_like(x)
		for i in range(B):
			f = factors[i].item()
			# map output times to input positions
			src = t / f
			src_clamped = src.clamp(0, L - 1)
			lo = src_clamped.floor().long()
			hi = (lo + 1).clamp(max=L - 1)
			alpha = (src_clamped - lo.float()).view(L, 1)
			out[i] = x[i, lo] * (1 - alpha) + x[i, hi] * alpha
		return out

	def make_views(x: Tensor) -> Tensor:
		# Compose augmentations; order: resample -> shift -> amp -> noise
		v = x
		v = global_resample(v, eff_aug_resample)
		v = time_shift(v, eff_aug_shift)
		v = amp_scale(v, eff_aug_amp)
		v = add_noise(v, eff_aug_noise)
		return v

	for epoch in range(1, args.epochs + 1):
		model.train()
		total = 0.0
		count = 0
		for xb, s_lb, t_lb in loader:
			xb = xb.to(device, non_blocking=True)
			xb = revin(xb)  # instance normalize per sample over time
			s_lb = s_lb.to(device)
			t_lb = t_lb.to(device)
			opt.zero_grad(set_to_none=True)
			ctx = torch.amp.autocast('cuda') if scaler is not None else nullcontext()
			with ctx:
				# Build K views per sample
				views_s = []
				views_t = []
				for _ in range(eff_num_views):
					xv = make_views(xb)
					zs, zt = model(xv)
					views_s.append(zs)
					views_t.append(zt)
				if args.loss == 'supcon':
					# Concatenate features and repeat labels
					feat_s = torch.cat(views_s, dim=0)
					feat_t = torch.cat(views_t, dim=0)
					s_lab = s_lb.repeat_interleave(eff_num_views)
					t_lab = t_lb.repeat_interleave(eff_num_views)
					loss = crit(feat_s, s_lab) + crit(feat_t, t_lab)
				else:
					# InfoNCE (instance) requires at least 2 views
					if eff_num_views < 2:
						# fall back to two identical views to avoid crash
						views_s.append(views_s[0])
						views_t.append(views_t[0])
					q_s, k_s = views_s[0], views_s[1]
					q_t, k_t = views_t[0], views_t[1]
					if args.loss == 'infonce_instance':
						loss = info_nce(q_s, k_s, args.temp) + info_nce(q_t, k_t, args.temp)
					else:
						# Masked variant: exclude same-subject (for subject head) or same-task (for task head) from negatives
						loss = masked_info_nce(q_s, k_s, s_lb, args.temp) + masked_info_nce(q_t, k_t, t_lb, args.temp)
			if scaler is not None:
				scaler.scale(loss).backward()
				scaler.step(opt)
				scaler.update()
			else:
				loss.backward()
				opt.step()
			total += loss.item() * xb.size(0)
			count += xb.size(0)
		print(f"Epoch {epoch:02d}/{args.epochs} | loss {total/max(1,count):.4f}")

if __name__ == '__main__':
	from contextlib import nullcontext
	main()
