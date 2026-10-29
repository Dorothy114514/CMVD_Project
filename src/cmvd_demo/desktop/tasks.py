"""Serialized background work with one replaceable pending request.

All callbacks land on the owning Qt thread. Cancellation is cooperative; a
revision check discards even a result that finished just before cancellation.
"""

from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from cmvd_demo.io import DicomReadError, ReadCancelled
from .diagnostics import error_code, record_error


class _Signals(QObject):
    finished = Signal(int, object, str)
    progress = Signal(int, str, int, int)


class _Work(QRunnable):
    def __init__(self, revision, function, stage):
        super().__init__()
        self.revision, self.function = revision, function
        self.cancelled = Event()
        self.signals = _Signals()
        self.stage = stage

    def run(self):
        payload, error = None, ""
        try:
            if self.cancelled.is_set():
                raise ReadCancelled()
            payload = self.function(
                lambda stage, done, total: self.signals.progress.emit(self.revision, stage, done, total),
                self.cancelled.is_set,
            )
            if self.cancelled.is_set():
                raise ReadCancelled()
        except Exception as exception:
            payload = None
            error = error_code(exception)
            if not isinstance(exception, ReadCancelled):
                record_error(error, self.stage)
        self.signals.finished.emit(self.revision, payload, error)


class TaskRunner(QObject):
    result = Signal(int, object)
    failed = Signal(int, str)
    progress = Signal(int, str, int, int)
    idle = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self.revision = 0
        self._active = None
        self._pending = None
        self.stage = type(parent).__name__ if parent is not None else 'background'

    @property
    def busy(self):
        return self._active is not None or self._pending is not None

    def submit(self, function):
        self.invalidate()
        self._pending = (self.revision, function)
        self._start_pending()
        return self.revision

    def invalidate(self):
        self.revision += 1
        self._pending = None
        if self._active is not None:
            self._active.cancelled.set()

    def _start_pending(self):
        if self._active is not None or self._pending is None:
            return
        self._active = work = _Work(*self._pending, self.stage)
        self._pending = None
        work.signals.finished.connect(self._finished)
        work.signals.progress.connect(self._progress)
        self.pool.start(work)

    @Slot(int, str, int, int)
    def _progress(self, revision, stage, done, total):
        if revision == self.revision:
            self.progress.emit(revision, stage, done, total)

    @Slot(int, object, str)
    def _finished(self, revision, payload, error):
        self._active = None
        if revision == self.revision:
            if error:
                self.failed.emit(revision, error)
            else:
                self.result.emit(revision, payload)
        self._start_pending()
        if not self.busy:
            self.idle.emit()
