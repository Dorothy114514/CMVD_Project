"""Geometry history isolation and bounded retention, independent of Qt."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cmvd_demo.roi.history import RoiHistory


class HistoryTests(unittest.TestCase):
    def test_order_noop_redo_and_context(self):
        h=RoiHistory()
        h.record("a",[],[1],(0,0,"blood_pool",0))
        h.record("a",[1],[1],(5,0,"blood_pool",0))
        h.record("a",[1],[2],(2,0,"blood_pool",1))
        self.assertEqual(h.steps,2)
        self.assertEqual(h.restore("a",[2]),([1],(2,0,"blood_pool",1)))
        self.assertEqual(h.restore("a",[1]),([],(0,0,"blood_pool",0)))
        self.assertEqual(h.restore("a",[],redo=True),([1],(0,0,"blood_pool",0)))

    def test_series_isolated_and_new_edit_clears_redo(self):
        h=RoiHistory()
        h.record("a",[],[1],())
        h.record("b",[],[2],())
        self.assertIsNone(h.restore("c",[]))
        h.restore("a",[1])
        h.record("a",[],[3],())
        self.assertFalse(h.can_redo("a"))
        self.assertTrue(h.can_undo("b"))
        h.clear()
        self.assertEqual(h.steps,0)

    def test_changed_document_cannot_replay_stale_history(self):
        h=RoiHistory()
        h.record("a",[],[1],())
        self.assertIsNone(h.restore("a",[99]))
        self.assertFalse(h.can_undo("a"))

    def test_global_operation_and_byte_limits(self):
        h=RoiHistory(max_steps=3,max_bytes=160)
        for i in range(25):
            h.record(str(i%2),[i],[i+1],())
            self.assertLessEqual(h.steps,3)
            self.assertLessEqual(h.bytes_used,160)
        h.record("large",[],["x"*200],())
        self.assertFalse(h.can_undo("large"))
        self.assertLessEqual(h.bytes_used,160)

    def test_eviction_keeps_retained_undo_chain_consistent(self):
        h=RoiHistory(max_steps=2)
        for i in range(4):
            h.record("a",[i],[i+1],(i,))
        self.assertEqual(h.restore("a",[4]),([3],(3,)))
        self.assertEqual(h.restore("a",[3]),([2],(2,)))
        self.assertIsNone(h.restore("a",[2]))
        self.assertEqual(h.restore("a",[2],redo=True),([3],(2,)))


if __name__=="__main__":
    unittest.main()
