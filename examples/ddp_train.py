"""Two-process CPU DDP lab. Run directly; see notebook 18 for the explanation."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def worker(rank, world_size, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=90))
    try:
        # All ranks construct identical data; the sampler selects disjoint shards.
        generator = torch.Generator().manual_seed(42)
        x = torch.randn(256, 3, generator=generator)
        y = x @ torch.tensor([[2.], [-1.], [0.5]]) + 0.3
        dataset = TensorDataset(x, y)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=42)
        loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=0)
        torch.manual_seed(42)
        model = DDP(nn.Linear(3, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
        for epoch in range(25):
            sampler.set_epoch(epoch)  # Each epoch uses a new, coordinated shuffle.
            model.train()
            local_sum, local_count = 0., 0
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward()  # DDP synchronizes / averages gradients here.
                optimizer.step()
                local_sum += loss.item() * len(xb)
                local_count += len(xb)
            totals = torch.tensor([local_sum, local_count], dtype=torch.float64)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if rank == 0 and epoch in (0, 24):
                print(f'Epoch {epoch + 1}: global training MSE={totals[0] / totals[1]:.6f}', flush=True)

        # Check actual replica parameters, not just matching loss values.
        flat = torch.cat([p.detach().flatten() for p in model.module.parameters()])
        gathered = [torch.zeros_like(flat) for _ in range(world_size)]
        dist.all_gather(gathered, flat)
        replica_difference = max((flat - other).abs().max().item() for other in gathered)
        if rank == 0:
            model.eval()
            with torch.inference_mode():
                final_mse = nn.functional.mse_loss(model.module(x), y).item()
            report = {'world_size': world_size, 'backend': 'gloo', 'epochs': 25,
                      'local_batch_size': 32, 'global_batch_size': 32 * world_size,
                      'replica_max_difference': replica_difference, 'final_mse': final_mse}
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(model.module.state_dict(), folder / 'model.pt')
            (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report), flush=True)
        assert replica_difference < 1e-6
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/18_ddp'))
    args = parser.parse_args()
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError('This PyTorch installation needs the Gloo distributed backend.')
    # Use a new, nonexistent rendezvous file for every run; avoids port collisions.
    with tempfile.TemporaryDirectory(prefix='pytorch-course-ddp-') as temporary:
        rendezvous = (Path(temporary) / 'rendezvous').resolve().as_uri()
        mp.spawn(worker, args=(2, rendezvous, str(args.output_dir.resolve())), nprocs=2, join=True)


if __name__ == '__main__':
    main()
