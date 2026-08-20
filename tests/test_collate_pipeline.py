"""Equivalence tests for the dataloader rebuild (commit "gpu optimized changes").

The old pipeline's GPUCollate moved every tensor to the GPU inside the
DataLoader worker. The rebuild (detection_collate + move_batch_to_device)
stacks on CPU and transfers once in the training loop. These tests pin the
contract that made that change safe: the new path must deliver bitwise-
identical batches to what GPUCollate produced.

Run: .venv/bin/python -m pytest tests/test_collate_pipeline.py -v
(GPU tests skip automatically on machines without CUDA.)

"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils.transforms import detection_collate, move_batch_to_device  # noqa: E402


def reference_gpu_collate(batch, device):
    """The pre-rebuild GPUCollate behavior, kept here as the reference."""
    images, targets = [], []
    for image, target in batch:
        images.append(image.to(device))
        targets.append({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in target.items()
        })
    return torch.stack(images, dim=0), targets


def make_batch(n, h=416, w=832, boxes_per_image=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = []
    for i in range(n):
        image = torch.rand(3, h, w, generator=g)
        nb = boxes_per_image if i % 3 else 0  # every 3rd image has no boxes
        xy = torch.rand(nb, 2, generator=g) * torch.tensor([w - 40.0, h - 40.0])
        wh = torch.rand(nb, 2, generator=g) * 30 + 10
        target = {
            "boxes": torch.cat([xy, xy + wh], dim=1),
            "labels": torch.ones(nb, dtype=torch.int64),
            "image_id": torch.tensor([i]),
            "orig_size": (h, w),  # non-tensor value must pass through
        }
        batch.append((image, target))
    return batch


def assert_batches_equal(ref, new):
    ref_images, ref_targets = ref
    new_images, new_targets = new
    assert ref_images.shape == new_images.shape
    assert ref_images.dtype == new_images.dtype
    assert ref_images.device == new_images.device
    assert torch.equal(ref_images, new_images)
    assert len(ref_targets) == len(new_targets)
    for rt, nt in zip(ref_targets, new_targets):
        assert rt.keys() == nt.keys()
        for k in rt:
            if isinstance(rt[k], torch.Tensor):
                assert nt[k].device == rt[k].device, k
                assert torch.equal(rt[k], nt[k]), k
            else:
                assert rt[k] == nt[k], k


def test_cpu_collate_stacks_and_preserves_targets():
    batch = make_batch(16)
    images, targets = detection_collate(batch)
    assert images.shape == (16, 3, 416, 832)
    assert images.device.type == "cpu"
    assert len(targets) == 16
    assert targets[0]["boxes"].device.type == "cpu"
    # order and identity preserved
    for i, (image, target) in enumerate(batch):
        assert torch.equal(images[i], image)
        assert targets[i]["image_id"].item() == target["image_id"].item()


@pytest.mark.parametrize("n", [16, 3, 1])  # full, partial, single-sample batch
def test_matches_old_gpu_collate_bitwise(n):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    batch = make_batch(n, seed=n)

    ref = reference_gpu_collate(batch, device)

    images, targets = detection_collate(batch)
    # exercise the pinned path the DataLoader pin_memory thread would use
    images = images.pin_memory()
    targets = [
        {k: v.pin_memory() if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]
    new = move_batch_to_device(images, targets, device)
    torch.cuda.synchronize()  # flush non_blocking copies before comparing

    assert_batches_equal(ref, new)


# Module-level so it is picklable: spawn-context workers receive the dataset
# by pickling it, and classes defined inside a function cannot be pickled.
class SyntheticDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed(idx)
        nb = 2 if idx % 3 else 0  # every 3rd item has no boxes
        return (
            torch.rand(3, 64, 64, generator=g),
            {"boxes": torch.rand(nb, 4, generator=g),
             "labels": torch.ones(nb, dtype=torch.int64),
             "image_id": torch.tensor([idx])},
        )


def test_dataloader_integration_spawn_workers_pinned():
    """The production DataLoader configuration, end to end.

    Uses the spawn context (as train.py does) so the collate function and
    dataset really are pickled to workers, enables the real pin_memory
    thread, iterates twice to exercise persistent-worker reuse, and checks
    batch CONTENT against the deterministic dataset, not just counts.
    """
    pin = torch.cuda.is_available()
    dataset = SyntheticDataset()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, num_workers=2, shuffle=False,
        collate_fn=detection_collate, persistent_workers=True,
        pin_memory=pin, multiprocessing_context="spawn",
    )
    for epoch in range(2):  # second pass reuses the persistent workers
        seen = 0
        for batch_idx, (images, targets) in enumerate(loader):
            assert images.device.type == "cpu"
            if pin:
                assert images.is_pinned()
            for j in range(images.shape[0]):
                idx = batch_idx * 4 + j
                ref_image, ref_target = dataset[idx]
                assert torch.equal(images[j], ref_image)
                assert torch.equal(targets[j]["boxes"], ref_target["boxes"])
                assert targets[j]["image_id"].item() == idx
                if pin and targets[j]["boxes"].numel() > 0:
                    assert targets[j]["boxes"].is_pinned()
            seen += images.shape[0]
        assert seen == 12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
