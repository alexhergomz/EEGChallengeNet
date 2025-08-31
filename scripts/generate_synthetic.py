import argparse
import json
import os
import sys
import numpy as np

# Ensure project root is importable when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.synthetic import SyntheticConfig, generate_synthetic_mts


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate synthetic multivariate time series (T x C)')
    parser.add_argument('--out', type=str, default='synthetic.csv')
    parser.add_argument('--meta', type=str, default='synthetic_meta.json')
    parser.add_argument('--T', type=int, default=4000)
    parser.add_argument('--C', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base_correlation', type=float, default=0.3)
    parser.add_argument('--noise_scale', type=float, default=1.0)
    parser.add_argument('--ar_min', type=float, default=0.4)
    parser.add_argument('--ar_max', type=float, default=0.95)
    parser.add_argument('--num_seasonal', type=int, default=2)
    parser.add_argument('--num_segments', type=int, default=4)
    parser.add_argument('--num_jumps', type=int, default=8)
    parser.add_argument('--jump_scale', type=float, default=2.0)
    parser.add_argument('--spike_prob', type=float, default=0.001)
    parser.add_argument('--spike_scale', type=float, default=6.0)
    args = parser.parse_args()

    cfg = SyntheticConfig(
        T=args.T,
        C=args.C,
        seed=args.seed,
        base_correlation=args.base_correlation,
        noise_scale=args.noise_scale,
        ar_min=args.ar_min,
        ar_max=args.ar_max,
        num_seasonal=args.num_seasonal,
        num_segments=args.num_segments,
        num_jumps=args.num_jumps,
        jump_scale=args.jump_scale,
        spike_prob=args.spike_prob,
        spike_scale=args.spike_scale,
    )

    X, meta = generate_synthetic_mts(cfg)
    np.savetxt(args.out, X, delimiter=',')
    with open(args.meta, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print(f'Wrote {X.shape} to {args.out}')
    print(f'Meta -> {args.meta}')


if __name__ == '__main__':
    main()


