"""Enter a fresh, private-data-free selector jail before loading candidate code."""
import ctypes
import os
from pathlib import Path
import resource
import runpy
import sys

jail = Path(sys.argv[1]).resolve(strict=True)
result_descriptor = int(sys.argv[2])
cpu_capability = sys.argv[3]
if cpu_capability not in ("default", "avx2", "avx512"):
    raise ValueError("Invalid trusted CPU dispatch mode")
allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, allowed[-2:])
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_FSIZE, (32768, 32768))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (128, 128))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
os.chdir(jail)
os.chroot(jail)
os.chdir("/")
os.setgroups([])
os.setgid(65533)
os.setuid(65533)
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(38, 1, 0, 0, 0):
    raise OSError(ctypes.get_errno(), "Cannot enforce no-new-privileges")
os.environ.clear()
os.environ.update({"PATH": "/usr/local/bin", "PYTHONDONTWRITEBYTECODE": "1",
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "TMPDIR": "/tmp",
                   "ATEN_CPU_CAPABILITY": cpu_capability,
                   "ARM_SELECTOR_RESULT_FD": str(result_descriptor)})


class Filter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte),
                ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class Program(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filters", ctypes.POINTER(Filter))]


# x86-64: allow thread creation, deny processes, namespaces and networking.
# clone3 returns ENOSYS so the C runtime uses the checked clone fallback.
instructions = [(0x20, 0, 0, 4), (0x15, 1, 0, 0xc000003e), (0x06, 0, 0, 0x80000000),
                (0x20, 0, 0, 0), (0x35, 0, 1, 0x40000000), (0x06, 0, 0, 0x80000000),
                (0x15, 0, 1, 435), (0x06, 0, 0, 0x00050026),
                (0x15, 0, 3, 56), (0x20, 0, 0, 16), (0x45, 1, 0, 0x10000),
                (0x06, 0, 0, 0x00050001), (0x20, 0, 0, 0)]
for syscall, flags_offset in ((2, 24), (257, 32)):
    instructions.extend([(0x15, 0, 3, syscall), (0x20, 0, 0, flags_offset),
                         (0x45, 0, 1, 0x643), (0x06, 0, 0, 0x00050001),
                         (0x20, 0, 0, 0)])
for syscall in (29, 30, 31, 64, 65, 66, 67, 68, 69, 70, 71, 220, 142,
                41, 42, 53, 57, 58, 59, 101, 109, 112, 155, 165, 166,
                272, 308, 310, 311, 321, 322, 323, 437,
                76, 77, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 133,
                258, 259, 260, 263, 264, 265, 266, 268, 316):
    instructions.extend([(0x15, 0, 1, syscall), (0x06, 0, 0, 0x00050001)])
instructions.append((0x06, 0, 0, 0x7fff0000))
filters = (Filter * len(instructions))(*(Filter(*row) for row in instructions))
program = Program(len(instructions), filters)
if libc.prctl(22, 2, ctypes.byref(program), 0, 0):
    raise OSError(ctypes.get_errno(), "Cannot install the selector syscall policy")
runpy.run_path("/entry.py", run_name="__main__")
