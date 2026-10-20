"""Bounded, series-isolated geometry history. Never restores reviews or curves."""

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class _Edit:
    serial: int
    before: str
    after: str
    context: tuple

    @property
    def size(self):
        return len(self.before.encode("utf-8"))+len(self.after.encode("utf-8"))


class RoiHistory:
    """At most 50 operations / 16 MiB in the current case, across all series."""

    def __init__(self, max_steps=50, max_bytes=16*1024**2):
        self.max_steps, self.max_bytes = max_steps, max_bytes
        self.clear()

    def clear(self, key=None):
        if key is None:
            self._stacks = {}
            self._serial = 0
        else:
            self._stacks.pop(key, None)

    @staticmethod
    def _encode(tracks):
        return json.dumps(tracks, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def bytes_used(self):
        return sum(e.size for pair in self._stacks.values() for stack in pair for e in stack)

    @property
    def steps(self):
        return sum(len(stack) for pair in self._stacks.values() for stack in pair)

    def can_undo(self, key):
        return bool(self._stacks.get(key, ([], []))[0])

    def can_redo(self, key):
        return bool(self._stacks.get(key, ([], []))[1])

    def record(self, key, before, after, context):
        before, after = self._encode(before), self._encode(after)
        if before == after:
            return
        self._serial += 1
        undo, redo = self._stacks.setdefault(key, ([], []))
        redo.clear()
        edit = _Edit(self._serial, before, after, tuple(context))
        if edit.size > self.max_bytes or self.max_steps < 1:
            # Never undo across a new, unrecorded edit.
            self.clear(key)
            return
        undo.append(edit)
        while self.steps > self.max_steps or self.bytes_used > self.max_bytes:
            candidates = [(stack[0].serial, k, i) for k, pair in self._stacks.items()
                          for i, stack in enumerate(pair) if stack]
            _, k, i = min(candidates)
            if i == 0:
                self._stacks[k][0].pop(0)
            else:
                # Dropping part of a redo chain could skip a prerequisite state.
                self._stacks[k][1].clear()

    def restore(self, key, tracks, *, redo=False):
        pair = self._stacks.get(key)
        source = pair[1 if redo else 0] if pair else []
        if not source:
            return None
        edit = source[-1]
        expected = edit.before if redo else edit.after
        if self._encode(tracks) != expected:
            self.clear(key)
            return None
        source.pop()
        pair[0 if redo else 1].append(edit)
        return json.loads(edit.after if redo else edit.before), edit.context
