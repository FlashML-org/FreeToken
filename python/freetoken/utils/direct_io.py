"""Positional reads that bypass the page cache: ``O_DIRECT`` + ``preadv`` on POSIX, ``FILE_FLAG_NO_BUFFERING`` on Windows.

Either way the file offset, the length and the destination address of a read are multiples of :data:`DIRECT_ALIGN`,
and a read that reaches the end of the file returns the bytes up to it. On Windows the handle is overlapped, so the
threads of one reader read the same handle concurrently; a synchronous handle would serialize them. Read through
the file cache, a process at very low I/O priority moves a small fraction of what the same reads move unbuffered.
"""

from __future__ import annotations

import ctypes
import os
import sys

DIRECT_ALIGN = 4096

DIRECT_READS = sys.platform == "win32" or (hasattr(os, "O_DIRECT") and hasattr(os, "preadv"))

if sys.platform == "win32":
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _GENERIC_READ = 0x80000000
    _FILE_SHARE_ALL = 0x00000007
    _OPEN_EXISTING = 3
    _FILE_FLAG_NO_BUFFERING = 0x20000000
    _FILE_FLAG_OVERLAPPED = 0x40000000
    _INVALID_HANDLE = ctypes.c_void_p(-1).value
    _ERROR_HANDLE_EOF = 38
    _ERROR_IO_PENDING = 997
    _FILE_STORAGE_INFO_CLASS = 16
    # ReadFile takes a DWORD length; a chunk past this is read in pieces.
    _MAX_READ = 1 << 30

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    class _FileStorageInfo(ctypes.Structure):
        _fields_ = [
            ("LogicalBytesPerSector", wintypes.ULONG),
            ("PhysicalBytesPerSectorForAtomicity", wintypes.ULONG),
            ("PhysicalBytesPerSectorForPerformance", wintypes.ULONG),
            ("FileSystemEffectivePhysicalBytesPerSectorForAtomicity", wintypes.ULONG),
            ("Flags", wintypes.ULONG),
            ("ByteOffsetForSectorAlignment", wintypes.ULONG),
            ("ByteOffsetForPartitionAlignment", wintypes.ULONG),
        ]

    _kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    _kernel32.CreateFileW.restype = wintypes.HANDLE
    _kernel32.ReadFile.argtypes = (
        wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(_Overlapped),
    )
    _kernel32.ReadFile.restype = wintypes.BOOL
    _kernel32.GetOverlappedResult.argtypes = (
        wintypes.HANDLE, ctypes.POINTER(_Overlapped), ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
    )
    _kernel32.GetOverlappedResult.restype = wintypes.BOOL
    _kernel32.CreateEventW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR)
    _kernel32.CreateEventW.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _kernel32.GetFileInformationByHandleEx.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    _kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL

    def _windows_error(what: str, path: str) -> OSError:
        code = ctypes.get_last_error()
        return OSError(code, f"{what} {path}: {ctypes.FormatError(code).strip()}")


def _address_of(view: memoryview) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(view))


class DirectFile:
    """One file opened for reads that bypass the page cache; ``OSError`` where the file system refuses that."""

    def __init__(self, path: str):
        self.path = path
        if sys.platform == "win32":
            handle = _kernel32.CreateFileW(
                path, _GENERIC_READ, _FILE_SHARE_ALL, None, _OPEN_EXISTING,
                _FILE_FLAG_NO_BUFFERING | _FILE_FLAG_OVERLAPPED, None,
            )
            if handle in (None, _INVALID_HANDLE):
                raise _windows_error("cannot open unbuffered", path)
            self._handle = handle
            try:
                self._check_sector_size()
            except BaseException:
                self.close()
                raise
        else:
            self._fd = os.open(path, os.O_RDONLY | os.O_DIRECT)

    def _check_sector_size(self) -> None:
        info = _FileStorageInfo()
        if not _kernel32.GetFileInformationByHandleEx(
            self._handle, _FILE_STORAGE_INFO_CLASS, ctypes.byref(info), ctypes.sizeof(info),
        ):
            raise _windows_error("cannot read the sector size of", self.path)
        sector = info.LogicalBytesPerSector
        if sector == 0 or DIRECT_ALIGN % sector:
            raise OSError(f"{self.path}: {sector}-byte sectors do not divide the {DIRECT_ALIGN}-byte alignment")

    def pread_into(self, view: memoryview, offset: int) -> int:
        """Read into ``view`` from ``offset``; returns the bytes read, fewer than ``len(view)`` only at the end of the file."""
        if len(view) == 0:
            return 0
        if sys.platform != "win32":
            return os.preadv(self._fd, [view], offset)
        return self._read_windows(view, offset)

    def _read_windows(self, view: memoryview, offset: int) -> int:
        address = _address_of(view)
        done = 0
        while done < len(view):
            want = min(len(view) - done, _MAX_READ)
            got = self._read_once(address + done, want, offset + done)
            done += got
            if got < want:
                break
        return done

    def _read_once(self, address: int, length: int, offset: int) -> int:
        event = _kernel32.CreateEventW(None, True, False, None)
        if not event:
            raise _windows_error("cannot create a read event for", self.path)
        try:
            overlapped = _Overlapped(Offset=offset & 0xFFFFFFFF, OffsetHigh=offset >> 32, hEvent=event)
            if not _kernel32.ReadFile(self._handle, address, length, None, ctypes.byref(overlapped)):
                code = ctypes.get_last_error()
                if code == _ERROR_HANDLE_EOF:
                    return 0
                if code != _ERROR_IO_PENDING:
                    raise OSError(code, f"read {length} bytes at {offset} of {self.path}: {ctypes.FormatError(code).strip()}")
            transferred = wintypes.DWORD(0)
            if not _kernel32.GetOverlappedResult(self._handle, ctypes.byref(overlapped), ctypes.byref(transferred), True):
                code = ctypes.get_last_error()
                if code == _ERROR_HANDLE_EOF:
                    return 0
                raise OSError(code, f"read {length} bytes at {offset} of {self.path}: {ctypes.FormatError(code).strip()}")
            return transferred.value
        finally:
            _kernel32.CloseHandle(event)

    def close(self) -> None:
        if sys.platform == "win32":
            handle, self._handle = getattr(self, "_handle", None), None
            if handle:
                _kernel32.CloseHandle(handle)
        else:
            fd, self._fd = getattr(self, "_fd", None), None
            if fd is not None:
                os.close(fd)

    def __enter__(self) -> "DirectFile":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def drop_page_cache(path: str, offset: int = 0, nbytes: int = 0) -> None:
    """Drop ``path``'s cached pages from ``offset`` on, ``nbytes`` of them or to the end: banks + full checkpoint cache don't both fit in host RAM (OOM)."""
    # Windows has no posix_fadvise; its file cache is standby memory the memory manager repurposes on demand.
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


__all__ = ["DIRECT_ALIGN", "DIRECT_READS", "DirectFile", "drop_page_cache"]
