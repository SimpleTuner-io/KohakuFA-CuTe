"""Scalar spatial traversal shared by CPU qualification and CuTe JIT.

These functions contain only integer arithmetic and a static direction choice. Keep
them free of tensor operations: production allocates no spatial mask or index
table. Bounds may conservatively include unused tiles, but never omit an edge.
"""


def spatial_bounds(start, rows, height, width, window_h, window_w, inverse, minimum=min, maximum=max):
    last = minimum(start + rows - 1, height * width - 1)
    y0, y1 = start // width, last // width
    x0 = (start % width) * (y0 == y1)
    x1 = last % width + (width - 1 - last % width) * (y0 != y1)
    # inverse is a Python bool selected by the owner at compilation. Runtime
    # bounds use integer min/max and predicate arithmetic, so this same helper
    # also traces when applied through CuTe's programmatic JIT decorator.
    if inverse:
        # Invert start(q) <= key < start(q) + window. Boundary shifts
        # expand the inverse interval; a forward window about the key is wrong.
        y0 = maximum(y0 - (window_h - 1 - window_h // 2), 0) * (y0 >= window_h)
        y1 = minimum(height, y1 + window_h // 2 + 1 + window_h * (y1 >= height - window_h))
        x0 = maximum(x0 - (window_w - 1 - window_w // 2), 0) * (x0 >= window_w)
        x1 = minimum(width, x1 + window_w // 2 + 1 + window_w * (x1 >= width - window_w))
    else:
        y0 = minimum(maximum(y0 - window_h // 2, 0), height - window_h)
        y1 = minimum(maximum(y1 - window_h // 2, 0), height - window_h) + window_h
        x0 = minimum(maximum(x0 - window_w // 2, 0), width - window_w)
        x1 = minimum(maximum(x1 - window_w // 2, 0), width - window_w) + window_w
    return y0, y1, x0, x1


def spatial_tile(iteration, y0, x0, x1, width, columns, maximum=max):
    # Row alignment varies when width is not divisible by columns. Reserve
    # enough slots for any alignment and predicate unused slots before loading.
    slots = (x1 - x0 + 2 * columns - 2) // columns
    y = y0 + iteration // slots
    first = (y * width + x0) // columns
    previous_stop = ((y - 1) * width + x1 + columns - 1) // columns
    first = maximum(first, previous_stop * (y > y0))
    block = first + iteration % slots
    stop = (y * width + x1 + columns - 1) // columns
    return block, block < stop


def full_row_tile_bounds(y0, y1, width, columns):
    """Contiguous tile interval for a union of complete spatial rows."""
    return y0 * width // columns, (y1 * width + columns - 1) // columns


def normalize_exact_contiguous(selection):
    """Normalize a serialized experiment's forward/query/KV traversal flags."""
    if type(selection) is bool:
        return (selection,) * 3
    try:
        selected = tuple(selection)
    except TypeError as error:
        raise ValueError("Expected a bool or three bool traversal flags.") from error
    if len(selected) != 3 or any(type(value) is not bool for value in selected):
        raise ValueError("Expected a bool or three bool traversal flags.")
    return selected


def volume_bounds(start, rows, depth, height, width, window_d, window_h, window_w, inverse, minimum=min, maximum=max):
    """Conservative 3D box for flattened query owners or inverse KV owners."""
    last = minimum(start + rows - 1, depth * height * width - 1)
    z0, z1 = start // (height * width), last // (height * width)
    y0 = (start // width % height) * (z0 == z1)
    y1 = last // width % height + (height - 1 - last // width % height) * (z0 != z1)
    same_row = start // width == last // width
    x0 = start % width * same_row
    x1 = last % width + (width - 1 - last % width) * (same_row == False)
    if inverse:
        z0 = maximum(z0 - (window_d - 1 - window_d // 2), 0) * (z0 >= window_d)
        z1 = minimum(depth, z1 + window_d // 2 + 1 + window_d * (z1 >= depth - window_d))
        y0 = maximum(y0 - (window_h - 1 - window_h // 2), 0) * (y0 >= window_h)
        y1 = minimum(height, y1 + window_h // 2 + 1 + window_h * (y1 >= height - window_h))
        x0 = maximum(x0 - (window_w - 1 - window_w // 2), 0) * (x0 >= window_w)
        x1 = minimum(width, x1 + window_w // 2 + 1 + window_w * (x1 >= width - window_w))
    else:
        z0 = minimum(maximum(z0 - window_d // 2, 0), depth - window_d)
        z1 = minimum(maximum(z1 - window_d // 2, 0), depth - window_d) + window_d
        y0 = minimum(maximum(y0 - window_h // 2, 0), height - window_h)
        y1 = minimum(maximum(y1 - window_h // 2, 0), height - window_h) + window_h
        x0 = minimum(maximum(x0 - window_w // 2, 0), width - window_w)
        x1 = minimum(maximum(x1 - window_w // 2, 0), width - window_w) + window_w
    return z0, z1, y0, y1, x0, x1


def volume_tile(iteration, z0, y0, y1, x0, x1, height, width, columns, maximum=max):
    """Visit each flattened tile intersecting the box once, even across planes."""
    slots = (x1 - x0 + 2 * columns - 2) // columns
    row_index = iteration // slots
    z = z0 + row_index // (y1 - y0)
    y = y0 + row_index % (y1 - y0)
    row = z * height + y
    # The previous selected row can be in the preceding plane. Excluding its
    # last tile prevents double-counted probabilities and gradients at seams.
    previous = row - 1 - (height - (y1 - y0)) * (y == y0)
    first = maximum((row * width + x0) // columns, ((previous * width + x1 + columns - 1) // columns) * (row_index > 0))
    block = first + iteration % slots
    return block, block < (row * width + x1 + columns - 1) // columns


def compact_volume_row(row, height, width, tile_d, tile_h, tile_w):
    """Map a compact tile index to the original width-fastest sequence index."""
    size = tile_d * tile_h * tile_w
    blocks_w, blocks_h = width // tile_w, height // tile_h
    block, inner = row // size, row % size
    x = block % blocks_w * tile_w + inner % tile_w
    y = block // blocks_w % blocks_h * tile_h + inner // tile_w % tile_h
    z = block // (blocks_w * blocks_h) * tile_d + inner // (tile_w * tile_h)
    return (z * height + y) * width + x


def compact_volume_bounds(
    start,
    depth,
    height,
    width,
    window_d,
    window_h,
    window_w,
    tile_d,
    tile_h,
    tile_w,
    minimum=min,
    maximum=max,
    key_tile_d=0,
    key_tile_h=0,
    key_tile_w=0,
):
    size = tile_d * tile_h * tile_w
    block = start // size
    x = block % (width // tile_w) * tile_w
    y = block // (width // tile_w) % (height // tile_h) * tile_h
    z = block // ((width // tile_w) * (height // tile_h)) * tile_d
    z0 = minimum(maximum(z - window_d // 2, 0), depth - window_d)
    z1 = minimum(maximum(z + tile_d - 1 - window_d // 2, 0), depth - window_d) + window_d
    y0 = minimum(maximum(y - window_h // 2, 0), height - window_h)
    y1 = minimum(maximum(y + tile_h - 1 - window_h // 2, 0), height - window_h) + window_h
    x0 = minimum(maximum(x - window_w // 2, 0), width - window_w)
    x1 = minimum(maximum(x + tile_w - 1 - window_w // 2, 0), width - window_w) + window_w
    tile_d = key_tile_d or tile_d
    tile_h = key_tile_h or tile_h
    tile_w = key_tile_w or tile_w
    return (
        z0 // tile_d,
        (z1 + tile_d - 1) // tile_d,
        y0 // tile_h,
        (y1 + tile_h - 1) // tile_h,
        x0 // tile_w,
        (x1 + tile_w - 1) // tile_w,
    )


def compact_volume_tile(iteration, z0, z1, y0, y1, x0, x1, blocks_h, blocks_w):
    x = x0 + iteration % (x1 - x0)
    y = y0 + iteration // (x1 - x0) % (y1 - y0)
    z = z0 + iteration // ((x1 - x0) * (y1 - y0))
    return (z * blocks_h + y) * blocks_w + x
