"""Out-of-core vertex-clustering simplification of a large binary PLY mesh.

Vertices falling in the same voxel are merged (mean position and color) and degenerate triangles
are dropped, as in Open3D simplify_vertex_clustering, but the mesh is read through np.memmap in
chunks, so meshes larger than the available RAM can be simplified. Only fully triangulated binary
PLY files with per-vertex colors are supported. The input file is never modified.

    python scripts/prepare/simplify_mesh.py mesh.ply --voxel 0.03
    python scripts/prepare/simplify_mesh.py mesh.ply --target-faces 25000000
"""
import argparse
import time
from pathlib import Path

import numpy as np

PLY_SCALAR = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
CHUNK = 8_000_000
VOXEL_LADDER = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.20]


def parse_ply_header(path):
    """-> (data offset, endianness, [(element, count, properties)])."""
    with open(path, "rb") as f:
        raw = b""
        while b"end_header" not in raw:
            block = f.read(4096)
            if not block:
                raise SystemExit(f"{path}: 'end_header' not found")
            raw += block
    end = raw.find(b"end_header") + len(b"end_header")
    end += 2 if raw[end:end + 2] == b"\r\n" else 1
    lines = raw[:end].decode("ascii", errors="replace").splitlines()

    fmt = next((line.split()[1] for line in lines if line.startswith("format")), None)
    endian = {"binary_little_endian": "<", "binary_big_endian": ">"}.get(fmt)
    if endian is None:
        raise SystemExit(f"{path}: format '{fmt}' is not supported (binary PLY required)")

    elements = []
    for line in lines:
        tok = line.split()
        if tok and tok[0] == "element":
            elements.append((tok[1], int(tok[2]), []))
        elif tok and tok[0] == "property" and elements:
            # list properties: ("list", count type, item type, name); scalars: (name, type)
            elements[-1][2].append(("list", tok[2], tok[3], tok[4]) if tok[1] == "list"
                                   else (tok[2], tok[1]))
    return end, endian, elements


def open_mesh(path):
    """Memory-mapped vertex and face arrays, and the names of the color properties."""
    data_off, endian, elements = parse_ply_header(path)
    by_name = {name: (n, props) for name, n, props in elements}
    nv, vprops = by_name["vertex"]
    nf, fprops = by_name["face"]
    if any(p[0] == "list" for p in vprops):
        raise SystemExit("list properties on vertices are not supported")
    vdt = np.dtype([(name, endian + PLY_SCALAR[t]) for name, t in vprops])
    if not {"x", "y", "z", "red", "green", "blue"} <= set(vdt.names):
        raise SystemExit(f"{path}: vertices need x, y, z and red, green, blue")

    _, count_type, index_type, _ = fprops[0]
    face_stride = np.dtype(PLY_SCALAR[count_type]).itemsize + 3 * np.dtype(PLY_SCALAR[index_type]).itemsize
    face_off = data_off + nv * vdt.itemsize
    # Fixed-size faces fill the rest of the file exactly only if every face is a triangle.
    if Path(path).stat().st_size - face_off != nf * face_stride:
        raise SystemExit(f"{path}: only fully triangulated meshes are supported")
    fdt = np.dtype([("n", endian + PLY_SCALAR[count_type]), ("v", endian + PLY_SCALAR[index_type], 3)])
    verts = np.memmap(path, dtype=vdt, mode="r", offset=data_off, shape=(nv,))
    faces = np.memmap(path, dtype=fdt, mode="r", offset=face_off, shape=(nf,))
    return verts, faces, ["red", "green", "blue"]


def chunks(n, size=CHUNK):
    for a in range(0, n, size):
        yield a, min(a + size, n)


def xyz_of(block):
    return np.stack([block["x"], block["y"], block["z"]], axis=1).astype(np.float64)


def bbox(verts):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for a, b in chunks(len(verts)):
        xyz = xyz_of(verts[a:b])
        lo, hi = np.minimum(lo, xyz.min(axis=0)), np.maximum(hi, xyz.max(axis=0))
    return lo, hi


def voxel_keys(verts, lo, hi, voxel):
    """Linear voxel index of every vertex."""
    dims = np.maximum(np.ceil((hi - lo) / voxel).astype(np.int64) + 1, 1)
    keys = np.empty(len(verts), dtype=np.int64)
    for a, b in chunks(len(verts)):
        ijk = np.floor((xyz_of(verts[a:b]) - lo) / voxel).astype(np.int64)
        np.clip(ijk, 0, dims - 1, out=ijk)
        keys[a:b] = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    return keys, dims


def cluster(verts, lo, hi, voxel, colors):
    """-> (vertex -> cluster index, cluster mean of [x, y, z, r, g, b], grid dimensions)."""
    keys, dims = voxel_keys(verts, lo, hi, voxel)
    uniq, inverse = np.unique(keys, return_inverse=True)
    n_clusters = len(uniq)
    del keys, uniq
    inverse = inverse.ravel().astype(np.int64, copy=False)

    sums = np.zeros((n_clusters, 6), dtype=np.float64)
    counts = np.zeros(n_clusters, dtype=np.int64)
    for a, b in chunks(len(verts)):
        block, inv = verts[a:b], inverse[a:b]
        counts += np.bincount(inv, minlength=n_clusters)
        for k, name in enumerate(["x", "y", "z", *colors]):
            sums[:, k] += np.bincount(inv, weights=block[name].astype(np.float64), minlength=n_clusters)
    return inverse.astype(np.int32), sums / np.maximum(counts, 1)[:, None], dims


def remap_faces(faces, vert2cluster, cluster_xyz, max_edge):
    """Triangles on the clusters, without degenerate, duplicate or overly long triangles.

    Clustering ignores the triangle geometry, so on irregular photogrammetry meshes a small
    triangle can end up joining two distant clusters. Triangles with an edge above max_edge are
    dropped. Duplicates are detected on the sorted vertex triplet, keeping the original winding.
    """
    kept, dropped_long = [], 0
    for a, b in chunks(len(faces)):
        c = vert2cluster[np.asarray(faces[a:b]["v"], dtype=np.int64)]
        c = c[(c[:, 0] != c[:, 1]) & (c[:, 1] != c[:, 2]) & (c[:, 0] != c[:, 2])]
        if max_edge is not None and len(c):
            p0, p1, p2 = cluster_xyz[c[:, 0]], cluster_xyz[c[:, 1]], cluster_xyz[c[:, 2]]
            longest = np.maximum(np.maximum(np.linalg.norm(p0 - p1, axis=1),
                                            np.linalg.norm(p1 - p2, axis=1)),
                                 np.linalg.norm(p2 - p0, axis=1))
            short = longest <= max_edge
            dropped_long += int((~short).sum())
            c = c[short]
        if len(c):
            kept.append(c.astype(np.int32))
    if dropped_long:
        print(f"  dropped {dropped_long:,} bridging triangles (edge > {max_edge:.2f} m)")
    if not kept:
        raise SystemExit("no triangle left: the voxel is too large")
    tris = np.concatenate(kept)

    srt = np.sort(tris, axis=1)
    order = np.lexsort((srt[:, 2], srt[:, 1], srt[:, 0]))   # stable: keeps the first occurrence
    ss = srt[order]
    is_new = np.empty(len(ss), dtype=bool)
    is_new[0] = True
    np.any(ss[1:] != ss[:-1], axis=1, out=is_new[1:])
    return tris[np.sort(order[is_new])]


def write_ply(path, xyz, rgb, tris):
    vdt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    varr = np.empty(len(xyz), dtype=vdt)
    varr["x"], varr["y"], varr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    varr["red"], varr["green"], varr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    farr = np.empty(len(tris), dtype=np.dtype([("n", "u1"), ("v", "<i4", 3)]))
    farr["n"], farr["v"] = 3, tris
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(tris)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        varr.tofile(f)
        farr.tofile(f)


def count_clusters(verts, lo, hi, candidates):
    """Exact number of clusters for each voxel size with a single pass over the vertices.

    Cells are computed at the finest size; coarser sizes (integer multiples) are obtained by
    dividing the cell indices instead of reading the vertices again.
    """
    base = min(candidates)
    keys, dims = voxel_keys(verts, lo, hi, base)
    uniq = np.unique(keys)
    del keys
    k = uniq // dims[2]
    cells = np.stack([k // dims[1], k % dims[1], uniq % dims[2]], axis=1)
    out = {}
    for v in sorted(candidates):
        f = int(round(v / base))
        if f <= 1:
            out[v] = len(cells)
        else:
            c, d = cells // f, dims // f + 2
            out[v] = len(np.unique((c[:, 0] * d[1] + c[:, 1]) * d[2] + c[:, 2]))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mesh", type=Path)
    p.add_argument("--out", type=Path, default=None, help="default: <name>_simplified.ply")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--voxel", type=float, help="voxel size in meters")
    group.add_argument("--target-faces", type=int,
                       help="pick the voxel size whose estimated face count is closest")
    p.add_argument("--max-edge-factor", type=float, default=20.0,
                   help="drop triangles with an edge longer than voxel * factor (0 disables)")
    args = p.parse_args()

    out = args.out or args.mesh.with_name(f"{args.mesh.stem}_simplified.ply")
    if out.resolve() == args.mesh.resolve():
        raise SystemExit("the output would overwrite the input mesh")

    t0 = time.time()
    verts, faces, colors = open_mesh(args.mesh)
    faces_per_vertex = len(faces) / len(verts)
    print(f"{args.mesh.name}: {len(verts):,} vertices, {len(faces):,} faces "
          f"({args.mesh.stat().st_size / 1e9:.1f} GB), faces/vertices = {faces_per_vertex:.2f}")
    lo, hi = bbox(verts)
    print(f"  bbox {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()} "
          f"(extent {np.round(hi - lo, 1).tolist()} m) [{time.time() - t0:.0f}s]")
    voxel = args.voxel
    if voxel is None:
        estimate = count_clusters(verts, lo, hi, VOXEL_LADDER)
        print("  voxel (m)      vertices   estimated faces")
        for v in VOXEL_LADDER:
            print(f"  {v:9.2f}  {estimate[v]:12,}   {int(estimate[v] * faces_per_vertex):15,}")
        voxel = min(VOXEL_LADDER, key=lambda v: abs(estimate[v] * faces_per_vertex - args.target_faces))
        print(f"  --target-faces {args.target_faces:,} -> voxel {voxel} m "
              f"(~{int(estimate[voxel] * faces_per_vertex):,} faces)")

    print(f"clustering at voxel {voxel} m...", flush=True)
    vert2cluster, centroids, dims = cluster(verts, lo, hi, voxel, colors)
    print(f"  grid {dims.tolist()} -> {len(centroids):,} vertices "
          f"({100 * len(centroids) / len(verts):.1f}% of the original) [{time.time() - t0:.0f}s]",
          flush=True)
    max_edge = voxel * args.max_edge_factor if args.max_edge_factor > 0 else None
    tris = remap_faces(faces, vert2cluster, centroids[:, :3], max_edge)
    print(f"  {len(tris):,} faces ({100 * len(tris) / len(faces):.1f}% of the original) "
          f"[{time.time() - t0:.0f}s]", flush=True)

    rgb = centroids[:, 3:6]
    # Colors may be floats in [0, 1] or already in [0, 255]. The mean is used to decide because a
    # few overshooting values (e.g. 1.000001) would fool the maximum.
    scale = 255.0 if rgb.mean() <= 2.0 else 1.0
    write_ply(out, centroids[:, :3].astype(np.float32),
              np.clip(rgb * scale, 0, 255).round().astype(np.uint8), tris)
    print(f"{len(verts):,} -> {len(centroids):,} vertices, {len(faces):,} -> {len(tris):,} faces: {out}")
    print(f"  {out.stat().st_size / 1e9:.2f} GB, {args.mesh.stat().st_size / out.stat().st_size:.1f}x "
          f"smaller, {time.time() - t0:.0f}s in total")


if __name__ == "__main__":
    main()
