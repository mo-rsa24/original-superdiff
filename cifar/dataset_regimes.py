import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np

# ---------- helpers: ink masks + dilation ----------
def binarize_ink(glyph, thr=0.20, relative=True):
    g = glyph[0]
    t = (float(g.max()) * thr) if relative else thr
    return (g > t).to(torch.float32)[None, ...]

def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask[None, ...], kernel_size=k, stride=1, padding=radius)[0]

def can_place(occupied, mask_dilated, x0, y0):
    h, w = mask_dilated.shape[-2:]
    patch = occupied[:, y0:y0+h, x0:x0+w]
    return (patch * mask_dilated).sum().item() == 0.0

def paste_max(canvas, glyph, x0, y0):
    h, w = glyph.shape[-2:]
    patch = canvas[:, y0:y0+h, x0:x0+w]
    canvas[:, y0:y0+h, x0:x0+w] = torch.maximum(patch, glyph)
    return canvas

def paste_mask(occupied, mask, x0, y0):
    h, w = mask.shape[-2:]
    patch = occupied[:, y0:y0+h, x0:x0+w]
    occupied[:, y0:y0+h, x0:x0+w] = torch.maximum(patch, mask)
    return occupied

class TwoDigitMNISTCanvasClean(Dataset):
    def __init__(self, mnist_base, length=60000, canvas_size=48, mode="mixed",
                 p_both=0.8, digit_size_range=(18, 22), min_margin=4, min_gap=10,
                 ink_thr=0.20, ink_thr_relative=True, enforce_side_by_side=False,
                 corridor_gap=4, max_tries=1500, max_restarts=50, antialias_resize=True, seed=0):
        super().__init__()
        self.mnist = mnist_base
        self.length = int(length)
        self.H = self.W = int(canvas_size)
        self.mode = mode
        self.p_both = float(p_both)
        self.sz_lo, self.sz_hi = digit_size_range
        self.min_margin = int(min_margin)
        self.min_gap = int(min_gap)
        self.ink_thr = float(ink_thr)
        self.ink_thr_relative = bool(ink_thr_relative)
        self.enforce_side_by_side = bool(enforce_side_by_side)
        self.corridor_gap = int(corridor_gap)
        self.max_tries = int(max_tries)
        self.max_restarts = int(max_restarts)
        self.antialias_resize = bool(antialias_resize)
        self.base_seed = int(seed)
        self.rng = np.random.RandomState(self.base_seed)

        self.by_digit = {d: [] for d in range(10)}
        for i in range(len(self.mnist)):
            _, y = self.mnist[i]
            self.by_digit[int(y)].append(i)

    def __len__(self):
        return self.length

    def _sample_glyph(self, digit):
        idx = self.rng.choice(self.by_digit[int(digit)])
        x, _ = self.mnist[idx]
        return x.clone()

    def _resize(self, glyph, size):
        return TF.resize(glyph, [size, size], antialias=self.antialias_resize)

    def _sample_top_left(self, size, region=None):
        if region is None:
            x_min = self.min_margin
            y_min = self.min_margin
            x_max = self.W - self.min_margin - size
            y_max = self.H - self.min_margin - size
        else:
            x_min, x_max, y_min, y_max = region
            x_min = max(int(x_min), self.min_margin)
            y_min = max(int(y_min), self.min_margin)
            x_max = min(int(x_max), self.W - self.min_margin - size)
            y_max = min(int(y_max), self.H - self.min_margin - size)
        if x_max < x_min or y_max < y_min:
            return None
        x0 = int(self.rng.randint(x_min, x_max + 1))
        y0 = int(self.rng.randint(y_min, y_max + 1))
        return x0, y0

    def _choose_digits_and_labels(self):
        y = {"has4": torch.tensor(0.0), "has7": torch.tensor(0.0)}
        if self.mode == "exists4":
            digits = [4]; y["has4"] = torch.tensor(1.0)
        elif self.mode == "exists7":
            digits = [7]; y["has7"] = torch.tensor(1.0)
        elif self.mode == "both47":
            digits = [4, 7]; y["has4"] = torch.tensor(1.0); y["has7"] = torch.tensor(1.0)
        elif self.mode == "mixed":
            if self.rng.rand() < self.p_both:
                digits = [4, 7]; y["has4"] = torch.tensor(1.0); y["has7"] = torch.tensor(1.0)
            else:
                if self.rng.rand() < 0.5:
                    digits = [4]; y["has4"] = torch.tensor(1.0)
                else:
                    digits = [7]; y["has7"] = torch.tensor(1.0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return digits, y

    def _side_by_side_regions_dynamic(self, size_left, size_right):
        usable_w = self.W - 2 * self.min_margin
        if size_left + size_right + self.corridor_gap > usable_w:
            return None, None
        left_block_min = size_left
        left_block_max = usable_w - self.corridor_gap - size_right
        left_block = int(self.rng.randint(left_block_min, left_block_max + 1))

        xL_min = self.min_margin
        xL_max = self.min_margin + left_block - 1
        xR_min = self.min_margin + left_block + self.corridor_gap
        xR_max = self.W - self.min_margin - 1

        region_left  = (xL_min, xL_max, self.min_margin, self.H - self.min_margin - 1)
        region_right = (xR_min, xR_max, self.min_margin, self.H - self.min_margin - 1)
        return region_left, region_right

    def __getitem__(self, idx):
        for _restart in range(self.max_restarts):
            canvas = torch.zeros(1, self.H, self.W)
            occupied = torch.zeros(1, self.H, self.W)

            digits, y = self._choose_digits_and_labels()
            sizes = [int(self.rng.randint(self.sz_lo, self.sz_hi + 1)) for _ in digits]

            regions = [None] * len(digits)
            if len(digits) == 2 and self.enforce_side_by_side:
                r0, r1 = self._side_by_side_regions_dynamic(sizes[0], sizes[1])
                if r0 is not None:
                    regions[0], regions[1] = r0, r1

            ok_all = True
            for i, d in enumerate(digits):
                glyph = self._sample_glyph(d)
                glyph = self._resize(glyph, sizes[i])

                mask = binarize_ink(glyph, thr=self.ink_thr, relative=self.ink_thr_relative)
                mask_gap = dilate_mask(mask, self.min_gap)

                placed = False
                for _ in range(self.max_tries):
                    pos = self._sample_top_left(sizes[i], region=regions[i])
                    if pos is None: continue
                    x0, y0 = pos
                    if not can_place(occupied, mask_gap, x0, y0): continue
                    canvas = paste_max(canvas, glyph, x0, y0)
                    occupied = paste_mask(occupied, mask, x0, y0)
                    placed = True
                    break

                if not placed:
                    ok_all = False
                    break

            if ok_all:
                x = canvas * 2.0 - 1.0
                return x, y

        raise RuntimeError("Could not generate a valid sample; relax constraints.")

class TwoDigitMNISTCanvasCleanPlus(TwoDigitMNISTCanvasClean):
    def __init__(self, *args, p_extra=0.3, max_extra=1, forbid_digit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_extra = float(p_extra)
        self.max_extra = int(max_extra)
        self.forbid_digit = forbid_digit

    def _choose_digits_and_labels(self):
        digits, y = super()._choose_digits_and_labels()
        if self.p_extra > 0 and self.rng.rand() < self.p_extra:
            n_extra = int(self.rng.randint(1, self.max_extra + 1))
            for _ in range(n_extra):
                d = int(self.rng.randint(0, 10))
                if self.forbid_digit is not None and d == int(self.forbid_digit):
                    d = (d + 1) % 10
                digits.append(d)
        return digits, y

class PadTo48(Dataset):
    def __init__(self, base_subset):
        self.base = base_subset
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]            # [1,28,28] in [0,1]
        x = TF.pad(x, [10,10,10,10])   # -> 48x48 centered
        x = x*2 - 1                    # [-1,1]
        return x, y

def filter_digit_subset(mnist_ds, digit: int):
    idx = [i for i, (_, y) in enumerate(mnist_ds) if int(y) == int(digit)]
    return torch.utils.data.Subset(mnist_ds, idx)

def worker_init_fn(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.RandomState(ds.base_seed + 1000 * worker_id)