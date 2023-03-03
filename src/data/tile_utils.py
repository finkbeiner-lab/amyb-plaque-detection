from typing import List, Mapping, Tuple
from collections import OrderedDict

def intdiv(
    a: int,
    b: int,
) -> int:
    return (a // b) if a >= 0 else -((b - 1 - a) // b)

def coords_to_tiles(
    a: int,
    b: int,
    size: int,
    stride: int,
    offset: int,
) -> List[int]:
    return range(intdiv(a - size - offset, stride) + 1, intdiv(b - 1 - offset, stride) + 1)

def bbox_to_tiles(
    a: Tuple[int, int],
    b: Tuple[int, int],
    size: Tuple[int, int],
    stride: Tuple[int, int],
    offset: Tuple[int, int],
) -> List[Tuple[int, int]]:
    return [(xi, yi) for xi in coords_to_tiles(a[0], b[0], size[0], stride[0], offset[0]) for yi in coords_to_tiles(a[1], b[1], size[1], stride[1], offset[1])]

def bboxes_to_tiles_map(
    bboxes: List[Tuple[int, int, int, int]],
    size: Tuple[int, int],
    stride: Tuple[int, int],
    offset: Tuple[int, int],
) -> Mapping[Tuple[int, int], int]:
    tiles_map = OrderedDict()
    for idx, bbox in enumerate(bboxes):
        for tile in bbox_to_tiles(bbox[:2], bbox[2:], size, stride, offset):
            tiles_map.setdefault(tile, list()).append(idx)
    return tiles_map
