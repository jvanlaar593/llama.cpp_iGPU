#!/usr/bin/env python3
"""Report whether a GGUF's tensor data offsets meet a backend's alignment.

The Vulkan mmap-import path can only be used when every tensor offset is a
multiple of minStorageBufferOffsetAlignment (64 on Intel Arrow Lake). GGUF pads
tensor data to general.alignment, which defaults to 32.

usage: gguf_align_check.py MODEL.gguf [alignment]
"""
import struct
import sys

# gguf metadata value types
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

FIXED = {
    UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
    FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8,
}


class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated gguf")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def skip_value(self, vtype):
        if vtype in FIXED:
            self.raw(FIXED[vtype])
        elif vtype == STRING:
            self.string()
        elif vtype == ARRAY:
            etype = self.u32()
            count = self.u64()
            if etype in FIXED:
                self.raw(FIXED[etype] * count)
            elif etype == STRING:
                for _ in range(count):
                    self.string()
            else:
                raise ValueError("bad array element type %d" % etype)
        else:
            raise ValueError("bad value type %d" % vtype)

    def read_value(self, vtype):
        if vtype == UINT32:
            return self.u32()
        if vtype == UINT64:
            return self.u64()
        self.skip_value(vtype)
        return None


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    path = sys.argv[1]
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    with open(path, "rb") as f:
        r = Reader(f)

        if r.raw(4) != b"GGUF":
            print("not a gguf file")
            return 1

        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        alignment = 32
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            val = r.read_value(vtype)
            if key == "general.alignment" and val is not None:
                alignment = val

        offsets = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            r.raw(8 * n_dims)
            r.u32()  # ggml type
            offsets.append((name, r.u64()))

    bad = [(n, o) for n, o in offsets if o % want]

    print("file            : %s" % path)
    print("gguf version    : %d" % version)
    print("tensors         : %d" % n_tensors)
    print("general.alignment: %d" % alignment)
    print("required        : %d" % want)
    print("misaligned      : %d of %d" % (len(bad), n_tensors))

    if bad:
        print("\nzero-copy import will NOT engage. First offenders:")
        for n, o in bad[:10]:
            print("  %-48s offset %d (mod %d = %d)" % (n, o, want, o % want))
        print("\nRewrite the file with general.alignment >= %d to enable it." % want)
    else:
        print("\nall tensor offsets are %d-byte aligned: zero-copy import can engage" % want)

    return 0


if __name__ == "__main__":
    sys.exit(main())
