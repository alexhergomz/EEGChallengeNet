import argparse
import os
import sys
import math
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

# Ensure project root import
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
	sys.path.insert(0, ROOT)

from src.synthetic import EEGConfig, generate_synthetic_eeg
from src.blocks import TwoStageCLSPool
from src.model import tiny_model
from src.revin import RevIN
from src.datasets import S3EEGIterableDataset


class EEGWindows(Dataset):
	def __init__(self, X: np.ndarray, subj_ids: np.ndarray, task_ids: np.ndarray, L: int, stride: int):
		self.L = L
		self.stride = stride
		self.X = X
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
		x = torch.from_numpy(self.X[i, t0:t0 + self.L]).float()
		return x, int(self.subj[i]), int(self.task[i])



class DualHeadTeacherStudent(nn.Module):
	def __init__(self, num_channels: int, r: int, proj_dim: int, sink_time: int, sink_channel: int,
	             r_v: Optional[int] = None, d_qk: Optional[int] = None, z: Optional[int] = None,
	             num_q_heads_time: int = 1, num_q_heads_channel: int = 1):
		super().__init__()
		# Shared student backbone
		self.student_backbone = tiny_model(num_channels=num_channels, r=r)
		self.student_pool = TwoStageCLSPool(num_channels=num_channels, r=r,
			r_v=(r_v if r_v is not None else max(4, r//2)), d_qk=(d_qk if d_qk is not None else max(8, r)), z=(z if z is not None else max(8, r)),
			num_sink_q_time=sink_time, num_sink_q_channel=sink_channel,
			num_q_heads_time=num_q_heads_time, num_q_heads_channel=num_q_heads_channel)
		self.student_proj = nn.Sequential(nn.Linear(r, r), nn.GELU(), nn.Linear(r, 2 * proj_dim))
		# Teacher EMA copies (frozen forward under no grad)
		self.teacher_backbone = tiny_model(num_channels=num_channels, r=r)
		self.teacher_pool = TwoStageCLSPool(num_channels=num_channels, r=r,
			r_v=(r_v if r_v is not None else max(4, r//2)), d_qk=(d_qk if d_qk is not None else max(8, r)), z=(z if z is not None else max(8, r)),
			num_sink_q_time=sink_time, num_sink_q_channel=sink_channel,
			num_q_heads_time=num_q_heads_time, num_q_heads_channel=num_q_heads_channel)
		self.teacher_proj = nn.Sequential(nn.Linear(r, r), nn.GELU(), nn.Linear(r, 2 * proj_dim))
		# init teacher = student
		self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
		self.teacher_pool.load_state_dict(self.student_pool.state_dict())
		self.teacher_proj.load_state_dict(self.student_proj.state_dict())
		for p in self.teacher_backbone.parameters(): p.requires_grad = False
		for p in self.teacher_pool.parameters(): p.requires_grad = False
		for p in self.teacher_proj.parameters(): p.requires_grad = False

	@torch.no_grad()
	def _ema_update(self, tau: float):
		for m_s, m_t in [
			(self.student_backbone, self.teacher_backbone),
			(self.student_pool, self.teacher_pool),
			(self.student_proj, self.teacher_proj),
		]:
			for ps, pt in zip(m_s.parameters(), m_t.parameters()):
				pt.data.mul_(tau).add_(ps.data, alpha=1 - tau)

	def forward(self, x: Tensor):
		# Student forward
		hs = self.student_backbone(x)
		ps = self.student_pool(hs)
		zs = self.student_proj(ps)
		d = zs.shape[-1] // 2
		zs_subj, zs_task = zs[:, :d], zs[:, d:]
		return zs_subj, zs_task

	@torch.no_grad()
	def teacher_embed(self, x: Tensor):
		ht = self.teacher_backbone(x)
		pt = self.teacher_pool(ht)
		zt = self.teacher_proj(pt)
		d = zt.shape[-1] // 2
		return zt[:, :d], zt[:, d:]


def cosine_loss(student: Tensor, teacher: Tensor, temp: float = 0.04) -> Tensor:
	student = torch.nn.functional.normalize(student, dim=-1)
	teacher = torch.nn.functional.normalize(teacher, dim=-1)
	# maximize cosine similarity => minimize 1 - cos
	return (1.0 - (student * teacher).sum(dim=-1)).mean()


def main():
	parser = argparse.ArgumentParser(description='Self-distillation (EMA teacher) on synthetic EEG with dual invariances')
	parser.add_argument('--subjects', type=int, default=8)
	parser.add_argument('--tasks', type=int, default=3)
	parser.add_argument('--T', type=int, default=2000)
	parser.add_argument('--C', type=int, default=16)
	parser.add_argument('--L', type=int, default=256)
	parser.add_argument('--stride', type=int, default=128)
	parser.add_argument('--s3', type=str, default='', help='S3 prefix to stream EEG windows (BIDS paths). If set, overrides synthetic generator')
	parser.add_argument('--s3_max_files', type=int, default=0, help='Max files to scan from S3 (0 = no limit)')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--r', type=int, default=8)
	parser.add_argument('--proj_dim', type=int, default=64)
	parser.add_argument('--lr', type=float, default=3e-3)
	parser.add_argument('--wd', type=float, default=1e-4)
	parser.add_argument('--ema_tau', type=float, default=0.99)
	parser.add_argument('--sink_time', type=int, default=2)
	parser.add_argument('--sink_channel', type=int, default=2)
	parser.add_argument('--no_aug', action='store_true', help='Disable data augmentations (use identical views)')
	parser.add_argument('--cpu', action='store_true')
	args = parser.parse_args()

	device = torch.device('cpu' if (args.cpu or not torch.cuda.is_available()) else 'cuda')
	print('Device:', device)

	if args.s3:
		max_files = args.s3_max_files if args.s3_max_files > 0 else None
		ds = S3EEGIterableDataset(s3_uri=args.s3, window_length=args.L, stride=args.stride, max_files=max_files, channels=args.C)
		# IterableDataset: shuffle must be False
		loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=(device.type=='cuda'))
	else:
		cfg = EEGConfig(T=args.T, C=args.C, num_subjects=args.subjects, num_tasks=args.tasks)
		X, subj_ids, task_ids = generate_synthetic_eeg(cfg)
		ds = EEGWindows(X, subj_ids, task_ids, L=args.L, stride=args.stride)
		loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=(device.type=='cuda'))

	revin = RevIN(num_channels=args.C, affine=True).to(device)
	model = DualHeadTeacherStudent(num_channels=args.C, r=args.r, proj_dim=args.proj_dim, sink_time=args.sink_time, sink_channel=args.sink_channel).to(device)
	opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
	scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

	for epoch in range(1, args.epochs + 1):
		model.train()
		total = 0.0
		count = 0
		for xb, s_lb, t_lb in loader:
			xb = xb.to(device, non_blocking=True)
			xb = revin(xb)
			# Build two views: if --no_aug, use identical inputs; else small Gaussian jitter
			if args.no_aug:
				x1 = xb
				x2 = xb
			else:
				x1 = xb + torch.randn_like(xb) * 0.01
				x2 = xb + torch.randn_like(xb) * 0.01

			opt.zero_grad(set_to_none=True)
			ctx = torch.amp.autocast('cuda') if scaler is not None else nullcontext()
			with ctx:
				zs1_subj, zs1_task = model(x1)
				zs2_subj, zs2_task = model(x2)
				with torch.no_grad():
					zt1_subj, zt1_task = model.teacher_embed(x1)
					zt2_subj, zt2_task = model.teacher_embed(x2)
					# teacher averages per sample for stability
					zt_subj = 0.5 * (zt1_subj + zt2_subj)
					zt_task = 0.5 * (zt1_task + zt2_task)
					# build batch prototypes by subject and task
					s_lb_t = torch.tensor(s_lb, device=zt_subj.device, dtype=torch.long)
					t_lb_t = torch.tensor(t_lb, device=zt_task.device, dtype=torch.long)
					# subject prototypes
					uniq_s, inv_s = torch.unique(s_lb_t, return_inverse=True)
					K_s = uniq_s.size(0)
					proto_sum_s = torch.zeros(K_s, zt_subj.size(1), device=zt_subj.device, dtype=zt_subj.dtype)
					counts_s = torch.zeros(K_s, device=zt_subj.device, dtype=zt_subj.dtype)
					proto_sum_s.index_add_(0, inv_s, zt_subj)
					counts_s.index_add_(0, inv_s, torch.ones_like(inv_s, dtype=zt_subj.dtype))
					protos_s = proto_sum_s / counts_s.clamp(min=1e-6).unsqueeze(1)
					aligned_proto_s = protos_s[inv_s]  # (B, d)
					# task prototypes
					uniq_t, inv_t = torch.unique(t_lb_t, return_inverse=True)
					K_t = uniq_t.size(0)
					proto_sum_t = torch.zeros(K_t, zt_task.size(1), device=zt_task.device, dtype=zt_task.dtype)
					counts_t = torch.zeros(K_t, device=zt_task.device, dtype=zt_task.dtype)
					proto_sum_t.index_add_(0, inv_t, zt_task)
					counts_t.index_add_(0, inv_t, torch.ones_like(inv_t, dtype=zt_task.dtype))
					protos_t = proto_sum_t / counts_t.clamp(min=1e-6).unsqueeze(1)
					aligned_proto_t = protos_t[inv_t]
				# student matches its group prototypes from both views
				loss = (
					cosine_loss(zs1_subj, aligned_proto_s) + cosine_loss(zs2_subj, aligned_proto_s)
					+ cosine_loss(zs1_task, aligned_proto_t) + cosine_loss(zs2_task, aligned_proto_t)
				)
			if scaler is not None:
				scaler.scale(loss).backward()
				scaler.step(opt)
				scaler.update()
			else:
				loss.backward()
				opt.step()
			total += loss.item() * xb.size(0)
			count += xb.size(0)
			# EMA update teacher
			model._ema_update(args.ema_tau)
		print(f"Epoch {epoch:02d}/{args.epochs} | loss {total/max(1,count):.4f}")


if __name__ == '__main__':
	from contextlib import nullcontext
	main()


