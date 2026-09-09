#!/usr/bin/env python3
"""Rewrite a GGUF with a larger tensor-data alignment.

The Vulkan mmap-import path needs every tensor offset to be a multiple of
minStorageBufferOffsetAlignment (64 on Intel Arrow Lake), but gguf writes
tensor data padded to general.alignment, which defaults to 32.

Metadata is copied verbatim; only general.alignment and the tensor offsets
change. Tensor payloads are byte-identical, just moved to aligned boundaries.

usage: gguf_realign.py SRC.gguf DST.gguf [alignment]
"""
import os
import struct
import sys

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

FIXED = {
    UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
    FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8,
}

ALIGN_KEY = b"general.alignment"
COPY_CHUNK = 8 * 1024 * 1024


def pad_up(n, a):
    return (n + a - 1) // a * a


class Reader:
    def __init__(self, f):
        self.f = f

    def tell(self):
        return self.f.tell()

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated gguf")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def bstr(self):
        return self.raw(self.u64())

    def skip_value(self, vtype):
        if vtype in FIXED:
            self.raw(FIXED[vtype])
        elif vtype == STRING:
            self.bstr()
        elif vtype == ARRAY:
            etype = self.u32()
            count = self.u64()
            if etype in FIXED:
                self.raw(FIXED[etype] * count)
            elif etype == STRING:
                for _ in range(count):
                    self.bstr()
            else:
                raise ValueError("unsupported array element type %d" % etype)
        else:
            raise ValueError("unsupported value type %d" % vtype)


def write_bstr(b):
    return struct.pack("<Q", len(b)) + b


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    src, dst = sys.argv[1], sys.argv[2]
    align = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    if align <= 0 or align & (align - 1):
        print("alignment must be a power of two")
        return 1

    src_size = os.path.getsize(src)

    with open(src, "rb") as f:
        r = Reader(f)

        if r.raw(4) != b"GGUF":
            print("not a gguf file")
            return 1

        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        kv_spans = []
        old_align = 32
        for _ in range(n_kv):
            start = r.tell()
            key = r.bstr()
            vtype = r.u32()
            if key == ALIGN_KEY and vtype == UINT32:
                old_align = r.u32()
            else:
                r.skip_value(vtype)
            kv_spans.append((key, start, r.tell()))

        names, ndims, dims, types, old_offs = [], [], [], [], []
        for _ in range(n_tensors):
            names.append(r.bstr())
            nd = r.u32()
            ndims.append(nd)
            dims.append([r.u64() for _ in range(nd)])
            types.append(r.u32())
            old_offs.append(r.u64())

        header_end = r.tell()
        src_data_start = pad_up(header_end, old_align)
        src_data_len = src_size - src_data_start

        # tensor payload sizes come from the gaps between consecutive offsets, so no
        # ggml type table is needed here. Trailing pad bytes get copied too, harmlessly.
        order = sorted(range(n_tensors), key=lambda i: old_offs[i])
        sizes = [0] * n_tensors
        for k, i in enumerate(order):
            nxt = old_offs[order[k + 1]] if k + 1 < len(order) else src_data_len
            sizes[i] = nxt - old_offs[i]
            if sizes[i] < 0:
                print("overlapping tensor offsets, refusing to rewrite")
                return 1

        f.seek(0)
        header = f.read(header_end)

        new_kv = [header[a:b] for key, a, b in kv_spans if key != ALIGN_KEY]
        new_kv.append(write_bstr(ALIGN_KEY) + struct.pack("<II", UINT32, align))

        new_offs = [0] * n_tensors
        cursor = 0
        for i in order:
            new_offs[i] = cursor
            cursor += pad_up(sizes[i], align)

        with open(dst, "wb") as g:
            g.write(b"GGUF")
            g.write(struct.pack("<I", version))
            g.write(struct.pack("<Q", n_tensors))
            g.write(struct.pack("<Q", len(new_kv)))
            for blob in new_kv:
                g.write(blob)

            for i in range(n_tensors):
                g.write(write_bstr(names[i]))
                g.write(struct.pack("<I", ndims[i]))
                for d in dims[i]:
                    g.write(struct.pack("<Q", d))
                g.write(struct.pack("<I", types[i]))
                g.write(struct.pack("<Q", new_offs[i]))

            g.write(b"\0" * (pad_up(g.tell(), align) - g.tell()))
            dst_data_start = g.tell()

            for i in order:
                g.seek(dst_data_start + new_offs[i])
                f.seek(src_data_start + old_offs[i])
                left = sizes[i]
                while left:
                    chunk = f.read(min(COPY_CHUNK, left))
                    if not chunk:
                        raise EOFError("truncated tensor data")
                    g.write(chunk)
                    left -= len(chunk)

            g.truncate(pad_up(g.tell(), align))

    misaligned = sum(1 for o in new_offs if o % align)
    print("wrote %s" % dst)
    print("  alignment  : %d -> %d" % (old_align, align))
    print("  data start : %d -> %d" % (src_data_start, dst_data_start))
    print("  tensors    : %d (%d misaligned)" % (n_tensors, misaligned))
    print("  size       : %.1f MB -> %.1f MB" % (src_size / 1e6, os.path.getsize(dst) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
