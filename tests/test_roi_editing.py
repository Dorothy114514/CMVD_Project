"""Mouse-driven native Qt regression for point capture, navigation and undo."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PySide6.QtCore import Qt,QPoint,QPointF,QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from cmvd_demo.desktop.window import BrowserWindow
from test_desktop import APP,until,dispose_window


def drag(widget,start,end,button=Qt.LeftButton):
    """Route real Qt mouse events through the viewport and GraphicsScene."""
    QTest.mouseMove(widget,start)
    APP.processEvents()
    QTest.mousePress(widget,button,Qt.NoModifier,start)
    for fraction in (.3,.6,1.):
        # GraphicsScene throttles moves; use an actual interactive cadence so a
        # sub-threshold first move cannot cause all later moves to be dropped.
        QTest.qWait(20)
        p=QPointF(start)+(QPointF(end)-QPointF(start))*fraction
        event=QMouseEvent(QEvent.MouseMove,p,QPointF(widget.mapToGlobal(p.toPoint())),
                          Qt.NoButton,button,Qt.NoModifier)
        QApplication.sendEvent(widget,event)
        APP.processEvents()
    QTest.mouseRelease(widget,button,Qt.NoModifier,end)
    APP.processEvents()


class RoiEditingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.w=BrowserWindow()
        self.p=self.w.roi_panel
        self.p.output_root=Path(self.temp.name)
        self.w.show()
        self.w.open_synthetic()
        until(lambda:self.w.current_pixels is not None and not self.w.prefetch.busy)
        for role in (0,1):
            self.p.role.setCurrentIndex(role)
            self.p.add_reference()
        APP.processEvents()

    def tearDown(self):
        dispose_window(self.w)
        self.temp.cleanup()

    def tracks(self):
        return copy.deepcopy(self.p.document["tracks"])

    def control(self):
        item=self.p.overlays["blood_pool"][0]
        handle=item.getHandles()[0]
        point=self.w.graphics.mapFromScene(handle.scenePos())
        self.assertTrue(self.w.graphics.viewport().rect().contains(point))
        return item,point

    def test_point_drag_moves_only_one_vertex_and_keeps_image_fixed(self):
        before=self.tracks()
        view=copy.deepcopy(self.w.view.viewRange())
        steps=self.p.history.steps
        item,point=self.control()
        self.assertEqual(item.getHandles()[0].radius,8)
        # Offset is within the enlarged handle, outside the old small square.
        point+=QPoint(4,0)
        drag(self.w.graphics.viewport(),point,point+QPoint(17,9))
        after=self.tracks()
        np.testing.assert_allclose(self.w.view.viewRange(),view,atol=1e-9)
        self.assertEqual(before[0],after[0])
        a,b=np.array(before[1]["contour"]["outer_rc"]),np.array(after[1]["contour"]["outer_rc"])
        self.assertEqual(np.count_nonzero(np.any(np.abs(a-b)>1e-6,axis=1)),1)
        self.assertEqual(self.p.history.steps,steps+1)
        QTest.mouseClick(self.w.roi_undo_button,Qt.LeftButton)
        self.assertEqual(self.tracks(),before)
        QTest.mouseClick(self.w.roi_redo_button,Qt.LeftButton)
        self.assertEqual(self.tracks(),after)

    def test_missed_point_does_not_pan_and_explicit_pan_preserves_roi(self):
        widget=self.w.graphics.viewport()
        before=self.tracks()
        view=copy.deepcopy(self.w.view.viewRange())
        drag(widget,QPoint(20,20),QPoint(65,40))
        np.testing.assert_allclose(self.w.view.viewRange(),view)
        self.assertEqual(self.tracks(),before)
        self.w.set_roi_tool("pan")
        drag(widget,QPoint(20,20),QPoint(65,40))
        self.assertFalse(np.allclose(self.w.view.viewRange(),view))
        self.assertEqual(self.tracks(),before)

    def test_middle_pan_works_in_point_mode_without_changing_roi(self):
        before=self.tracks()
        view=copy.deepcopy(self.w.view.viewRange())
        drag(self.w.graphics.viewport(),QPoint(20,20),QPoint(65,40),Qt.MiddleButton)
        self.assertFalse(np.allclose(self.w.view.viewRange(),view))
        self.assertEqual(self.tracks(),before)

    def test_contour_mode_moves_whole_roi_then_undo_restores(self):
        before=self.tracks()
        self.w.set_roi_tool("contour")
        item=self.p.overlays["blood_pool"][0]
        center=item.mapToScene(item.shape().boundingRect().center())
        point=self.w.graphics.mapFromScene(center)
        view=copy.deepcopy(self.w.view.viewRange())
        drag(self.w.graphics.viewport(),point,point+QPoint(12,8))
        after=self.tracks()
        shifts=np.asarray(after[1]["contour"]["outer_rc"])-np.asarray(before[1]["contour"]["outer_rc"])
        self.assertGreater(float(np.max(np.abs(shifts))),0)
        np.testing.assert_allclose(shifts,np.tile(shifts[0],(len(shifts),1)),atol=1e-8)
        np.testing.assert_allclose(self.w.view.viewRange(),view)
        self.p.undo()
        self.assertEqual(self.tracks(),before)

    def test_contour_can_be_dragged_from_segment_without_adding_vertex(self):
        before=self.tracks()
        self.w.set_roi_tool("contour")
        item,_=self.control()
        a,b=[h["item"].scenePos() for h in item.segments[0].handles]
        point=self.w.graphics.mapFromScene((a+b)*.5)
        view=copy.deepcopy(self.w.view.viewRange())
        drag(self.w.graphics.viewport(),point,point+QPoint(17,9))
        after=self.tracks()
        self.assertNotEqual(before[1],after[1])
        self.assertEqual(len(after[1]["contour"]["outer_rc"]),8)
        np.testing.assert_allclose(self.w.view.viewRange(),view)

    def test_whole_myocardium_move_keeps_inner_and_outer_boundaries_together(self):
        self.p.role.setCurrentIndex(0)
        self.w.set_roi_tool("contour")
        before=self.tracks()
        # Starting at the inner boundary also moves the outer boundary.
        inner=self.p.overlays["myocardium"][1]
        a,b=[h["item"].scenePos() for h in inner.segments[0].handles]
        point=self.w.graphics.mapFromScene((a+b)*.5)
        drag(self.w.graphics.viewport(),point,point+QPoint(17,9))
        after=self.tracks()
        outer_shift=np.asarray(after[0]["contour"]["outer_rc"])-np.asarray(before[0]["contour"]["outer_rc"])
        inner_shift=np.asarray(after[0]["contour"]["inner_rc"])-np.asarray(before[0]["contour"]["inner_rc"])
        self.assertGreater(float(np.max(np.abs(outer_shift))),0)
        np.testing.assert_allclose(outer_shift,inner_shift,atol=1e-8)
        self.assertEqual(after[1],before[1])
        self.p.undo()
        self.assertEqual(self.tracks(),before)
        self.p.redo()
        self.assertEqual(self.tracks(),after)

    def test_point_tool_body_drag_and_locked_reference_cannot_change_roi(self):
        before=self.tracks()
        item,_=self.control()
        center=self.w.graphics.mapFromScene(item.mapToScene(item.shape().boundingRect().center()))
        view=copy.deepcopy(self.w.view.viewRange())
        drag(self.w.graphics.viewport(),center,center+QPoint(17,9))
        self.assertEqual(self.tracks(),before)
        np.testing.assert_allclose(self.w.view.viewRange(),view)
        self.p.scope.setCurrentIndex(1)
        item.setPos(QPointF(1,0))
        self.p.scope.setCurrentIndex(0)
        self.assertFalse(self.p.overlays["blood_pool"][0].isEnabled())
        self.assertFalse(any(h.isVisible() for h in item.getHandles()))
        before=self.tracks()
        point=self.w.graphics.mapFromScene(item.getHandles()[0].scenePos())
        drag(self.w.graphics.viewport(),point,point+QPoint(17,9))
        self.assertEqual(self.tracks(),before)
        np.testing.assert_allclose(self.w.view.viewRange(),view)

    def test_added_point_is_one_undo_and_topology_reuses_editor(self):
        before=self.tracks()
        item,_=self.control()
        identity=id(item)
        segment=item.segments[0]
        a,b=[handle["item"].scenePos() for handle in segment.handles]
        point=self.w.graphics.mapFromScene((a+b)*.5)
        steps=self.p.history.steps
        QTest.mouseClick(self.w.graphics.viewport(),Qt.LeftButton,Qt.NoModifier,point)
        self.assertEqual(len(self.tracks()[1]["contour"]["outer_rc"]),9)
        self.assertEqual(self.p.history.steps,steps+1)
        after=self.tracks()
        for i in range(12):
            self.p.undo()
            self.assertEqual(self.tracks(),before)
            self.p.redo()
            self.assertEqual(self.tracks(),after)
            self.assertEqual(id(self.p.overlays["blood_pool"][0]),identity)

    def test_invalid_drag_undo_recovers_valid_contour_and_clears_measurements(self):
        before=self.tracks()
        self.p.orientation_review.setChecked(True)
        self.p.motion_review.setChecked(True)
        self.p.calculate()
        until(lambda:self.p.result is not None)
        item,_=self.control()
        item.setPos(item.pos()+QPointF(200,0))
        self.assertTrue(self.p.invalid_edit)
        self.assertTrue(self.p.undo_button.isEnabled())
        self.p.undo()
        self.assertFalse(self.p.invalid_edit)
        self.assertEqual(self.tracks(),before)
        self.assertIsNone(self.p.result)
        self.assertFalse(any(self.p.document["reviews"].values()))
        self.assertTrue(self.p.calculate_button.isEnabled())

    def test_frame_override_undo_returns_to_edited_frame_and_redo_is_local(self):
        base=self.tracks()
        self.w.time_slider.setValue(3)
        until(lambda:self.w.displayed_indices==(3,0))
        self.p.scope.setCurrentIndex(1)
        item,_=self.control()
        item.setPos(item.pos()+QPointF(1,0))
        after=self.tracks()
        self.w.time_slider.setValue(4)
        until(lambda:self.w.displayed_indices==(4,0))
        self.p.undo()
        until(lambda:self.w.displayed_indices==(3,0))
        self.assertEqual(self.tracks(),base)
        self.p.redo()
        self.assertEqual(self.tracks(),after)
        self.assertEqual(after[1]["overrides"][0]["frame_index"],3)

    def test_delete_hole_and_reset_override_can_be_undone(self):
        before=self.tracks()
        self.p.remove_track()
        self.p.undo()
        self.assertEqual(self.tracks(),before)
        self.p.role.setCurrentIndex(0)
        self.p.hole.setChecked(False)
        self.assertIsNone(self.tracks()[0]["contour"]["inner_rc"])
        self.p.undo()
        self.assertEqual(self.tracks(),before)
        self.p.scope.setCurrentIndex(1)
        self.p.overlays["myocardium"][0].setPos(QPointF(1,0))
        after=self.tracks()
        self.p.reset_frame()
        self.assertFalse(self.tracks()[0]["overrides"])
        self.p.undo()
        self.assertEqual(self.tracks(),after)

    def test_keyboard_undo_redo_and_new_case_reset(self):
        before=self.tracks()
        item,_=self.control()
        item.setPos(item.pos()+QPointF(1,0))
        after=self.tracks()
        self.w.graphics.setFocus()
        self.w.activateWindow()
        APP.processEvents()
        QTest.keyClick(self.w.graphics,Qt.Key_Z,Qt.ControlModifier)
        self.assertEqual(self.tracks(),before)
        QTest.keyClick(self.w.graphics,Qt.Key_Y,Qt.ControlModifier)
        self.assertEqual(self.tracks(),after)
        self.w.open_synthetic()
        until(lambda:self.w.current_pixels is not None)
        self.assertEqual(self.p.history.steps,0)
        self.assertFalse(self.p.undo_button.isEnabled())

    def test_text_field_keeps_its_own_undo(self):
        before=self.tracks()
        self.w.workspace_tabs.setCurrentIndex(1)
        field=self.w.feature_panel.protocol
        field.setText("")
        self.w.activateWindow()
        field.setFocus()
        APP.processEvents()
        QTest.keyClicks(field,"protocol-typed")
        QTest.keyClick(field,Qt.Key_Z,Qt.ControlModifier)
        self.assertEqual(self.tracks(),before)
        self.assertEqual(field.text(),"")

    def test_new_edit_after_undo_clears_redo(self):
        item,_=self.control()
        item.setPos(item.pos()+QPointF(1,0))
        self.p.undo()
        self.assertTrue(self.p.history.can_redo(self.p.key))
        self.p.overlays["blood_pool"][0].setPos(QPointF(-1,0))
        self.assertFalse(self.p.history.can_redo(self.p.key))

    def test_removed_vertex_can_be_undone_and_redone(self):
        before=self.tracks()
        item,_=self.control()
        steps=self.p.history.steps
        item.getHandles()[0].removeAction.trigger()
        self.assertFalse(self.p.invalid_edit)
        self.assertEqual(len(self.tracks()[1]["contour"]["outer_rc"]),7)
        self.assertEqual(self.p.history.steps,steps+1)
        after=self.tracks()
        self.p.undo()
        self.assertEqual(self.tracks(),before)
        self.p.redo()
        self.assertEqual(self.tracks(),after)

    def test_history_isolated_per_series_and_retained_on_switch_back(self):
        self.w.open_synthetic_perfusion()
        until(lambda:self.w.current_pixels is not None)
        rest="synthetic-perfusion-rest"
        stress="synthetic-perfusion-stress"
        self.w.select_choice(rest)
        until(lambda:self.w.current_pixels is not None)
        self.p.add_reference()
        self.assertTrue(self.p.undo_button.isEnabled())
        self.w.select_choice(stress)
        until(lambda:self.w.current_pixels is not None)
        self.assertFalse(self.p.undo_button.isEnabled())
        self.p.undo()
        self.assertFalse(self.p.document["tracks"])
        self.w.select_choice(rest)
        until(lambda:self.w.current_pixels is not None)
        self.assertTrue(self.p.undo_button.isEnabled())
        self.p.undo()
        self.assertFalse(self.p.document["tracks"])

    def test_roi_reload_is_one_undoable_operation(self):
        before=self.tracks()
        self.p.save()
        until(lambda:self.p.saved_output is not None)
        path=self.p.saved_output/"roi.json"
        self.p.remove_track()
        removed=self.tracks()
        self.p.load_roi(path)
        until(lambda:not self.p.runner.busy)
        self.assertEqual(self.tracks(),before)
        self.p.undo()
        self.assertEqual(self.tracks(),removed)
        self.p.redo()
        self.assertEqual(self.tracks(),before)


if __name__=="__main__":
    unittest.main()
