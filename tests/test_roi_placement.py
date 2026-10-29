from test_desktop import discard_unsaved
"""Known placement geometry and real Windows Qt drawing gestures."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cmvd_demo.roi import new_document, set_track, document_hash, RoiError
from cmvd_demo.roi.model import contour_at
from cmvd_demo.roi.placement import ellipse_vertices, placed_contour
from cmvd_demo.desktop.data import PhantomGrid
from cmvd_demo.desktop.window import BrowserWindow
from test_desktop import APP, until, dispose_window
from test_roi_editing import drag


class PlacementGeometryTests(unittest.TestCase):
    def test_ellipse_uses_requested_pixel_center_bounds_in_both_drag_directions(self):
        for a,b in (([10,20],[30,60]),([30,60],[10,20])):
            points=np.asarray(ellipse_vertices(a,b))
            np.testing.assert_allclose(points.mean(axis=0),[20,40])
            np.testing.assert_allclose(points.min(axis=0),[10,20])
            np.testing.assert_allclose(points.max(axis=0),[30,60])
            self.assertEqual(len(points),16)

    def test_invalid_or_tiny_bounds_are_rejected(self):
        for a,b in (([1,2],[1,8]),([0,0],[1,1]),([0,0],[float('nan'),3])):
            with self.assertRaises(RoiError):ellipse_vertices(a,b)

    def test_inner_redraw_preserves_outer_and_rejects_crossing(self):
        doc=new_document(PhantomGrid())
        outer=ellipse_vertices([10,10],[70,100])
        base=placed_contour(doc,'myocardium',0,0,outer,'both',True)
        doc=set_track(doc,'myocardium',0,0,base)
        before=copy.deepcopy(doc)
        inner=ellipse_vertices([30,30],[50,65])
        got=placed_contour(doc,'myocardium',0,0,inner,'inner',True)
        self.assertEqual(got['outer_rc'],outer)
        self.assertEqual(got['inner_rc'],inner)
        self.assertEqual(doc,before)
        with self.assertRaises(RoiError):
            placed_contour(doc,'myocardium',0,0,ellipse_vertices([0,0],[90,120]),'inner',True)

    def test_inner_cannot_exist_without_outer(self):
        with self.assertRaisesRegex(RoiError,'roi_draw_outer_first'):
            placed_contour(new_document(PhantomGrid()),'myocardium',0,0,ellipse_vertices([10,10],[20,20]),'inner',True)


class PlacementDesktopTests(unittest.TestCase):
    def setUp(self):
        self.w=BrowserWindow(unsaved_prompt=discard_unsaved)
        self.p=self.w.roi_panel
        self.w.show()
        self.w.activateWindow()
        self.w.open_synthetic()
        until(lambda:self.w.current_pixels is not None and not self.w.runner.busy)
        self.widget=self.w.graphics.viewport()

    def tearDown(self):
        from shiboken6 import isValid
        if isValid(self.w):
            dispose_window(self.w)

    def position(self,r,c):
        result=self.w.graphics.mapFromScene(self.w.image.mapToScene(QPointF(c+.5,r+.5)))
        self.assertTrue(self.widget.rect().contains(result))
        return result

    def start(self,mode='polygon',role=1):
        self.p.role.setCurrentIndex(role)
        self.p.draw_shape.setCurrentIndex(self.p.draw_shape.findData(mode))
        self.p.draw_button.click()
        self.assertTrue(self.p.drawing.active)

    def click(self,r,c):
        QTest.mouseClick(self.widget,Qt.LeftButton,Qt.NoModifier,self.position(r,c))
        APP.processEvents()

    def polygon(self,points):
        for r,c in points:self.click(r,c)
        self.w.graphics.setFocus()
        QTest.keyClick(self.w.graphics,Qt.Key_Return)
        APP.processEvents()

    def test_native_ellipse_placed_off_center_is_one_undo_and_keeps_view(self):
        view=copy.deepcopy(self.w.view.viewRange())
        self.start('ellipse')
        drag(self.widget,self.position(20,25),self.position(42,62))
        self.assertFalse(self.p.drawing.active)
        points=np.array(self.p.document['tracks'][0]['contour']['outer_rc'])
        np.testing.assert_allclose(points.mean(axis=0),[31,43.5],atol=.8)
        self.assertEqual(len(points),16)
        np.testing.assert_allclose(self.w.view.viewRange(),view)
        self.assertEqual(self.p.history.steps,1)
        self.p.undo()
        self.assertFalse(self.p.document['tracks'])
        self.p.redo()
        np.testing.assert_allclose(self.p.document['tracks'][0]['contour']['outer_rc'],points)

    def test_native_polygon_backspace_enter_and_no_background_pan(self):
        view=copy.deepcopy(self.w.view.viewRange())
        self.start()
        self.click(20,20);self.click(20,60);self.click(40,60)
        QTest.keyClick(self.w.graphics,Qt.Key_Backspace)
        self.assertEqual(len(self.p.drawing.points),2)
        self.click(50,60);self.click(50,20)
        QTest.keyClick(self.w.graphics,Qt.Key_Return)
        self.assertFalse(self.p.drawing.active)
        self.assertEqual(len(self.p.document['tracks'][0]['contour']['outer_rc']),4)
        np.testing.assert_allclose(self.w.view.viewRange(),view)

    def test_cancel_preserves_committed_roi_curves_and_reviews(self):
        for role in (0,1):
            self.p.role.setCurrentIndex(role);self.p.add_reference()
        self.p.orientation_review.setChecked(True)
        self.p.calculate()
        until(lambda:self.p.result is not None)
        doc,curve=copy.deepcopy(self.p.document),copy.deepcopy(self.p.result)
        self.start()
        self.click(20,20)
        self.assertTrue(self.p.drawing.preview.isVisible())
        self.assertFalse(self.w.save_work_button.isEnabled())
        QTest.keyClick(self.w.graphics,Qt.Key_Escape)
        self.assertFalse(self.p.drawing.active)
        self.assertEqual(self.p.document,doc)
        self.assertEqual(self.p.result,curve)
        self.assertTrue(self.w.save_work_button.isEnabled())

    def test_invalid_polygon_stays_draft_and_can_be_corrected(self):
        self.start()
        self.polygon([(20,20),(50,60),(20,60),(50,20)])
        self.assertTrue(self.p.drawing.active)
        self.assertTrue(self.p.drawing.error)
        self.assertFalse(self.p.document['tracks'])
        for _ in range(4):QTest.keyClick(self.w.graphics,Qt.Key_Backspace)
        self.polygon([(20,20),(20,60),(50,60),(50,20)])
        self.assertFalse(self.p.drawing.active)
        self.assertEqual(len(self.p.document['tracks']),1)

    def test_frame_and_role_switch_discard_unfinished_drafts(self):
        self.start();self.click(20,20);self.click(20,60)
        self.w.time_slider.setValue(2)
        until(lambda:self.w.displayed_indices==(2,0))
        self.assertFalse(self.p.drawing.active)
        self.assertFalse(self.p.drawing.preview.isVisible())
        self.assertFalse(self.p.document['tracks'])
        self.start();self.click(20,20)
        self.p.role.setCurrentIndex(0)
        self.assertFalse(self.p.drawing.active)

    def test_inner_only_redraw_and_locked_outer_native_drag(self):
        self.p.role.setCurrentIndex(0);self.p.add_reference()
        original=copy.deepcopy(self.p.document['tracks'][0]['contour']['outer_rc'])
        self.p.boundary.setCurrentIndex(self.p.boundary.findData('inner'))
        self.start('ellipse',0)
        drag(self.widget,self.position(40,55),self.position(55,73))
        self.assertFalse(self.p.drawing.active,self.p.drawing.error)
        self.assertEqual(self.p.document['tracks'][0]['contour']['outer_rc'],original)
        outer=self.p.overlays['myocardium'][0]
        inner=self.p.overlays['myocardium'][1]
        self.assertFalse(outer.isEnabled())
        self.assertTrue(inner.isEnabled())
        point=self.w.graphics.mapFromScene(outer.getHandles()[0].scenePos())
        before=document_hash(self.p.document)
        drag(self.widget,point,point+QPointF(10,5).toPoint())
        self.assertEqual(document_hash(self.p.document),before)

    def test_drawing_current_frame_creates_only_local_override(self):
        self.p.role.setCurrentIndex(1);self.p.add_reference()
        original=copy.deepcopy(self.p.document['tracks'][0]['contour'])
        self.w.time_slider.setValue(3)
        until(lambda:self.w.displayed_indices==(3,0))
        self.p.scope.setCurrentIndex(1)
        self.start('ellipse')
        drag(self.widget,self.position(40,55),self.position(56,74))
        track=self.p.document['tracks'][0]
        self.assertEqual(track['contour'],original)
        self.assertEqual([o['frame_index'] for o in track['overrides']],[3])
        self.assertEqual(contour_at(track,4),original)
        self.p.scope.setCurrentIndex(0)
        self.p.draw_button.click()
        self.assertFalse(self.p.drawing.active)

    def test_repeated_drawing_reuses_editor_and_case_change_hides_preview(self):
        identity=None
        for i in range(10):
            self.start('ellipse' if i%2 else 'polygon')
            if i%2:drag(self.widget,self.position(35,50),self.position(58,76))
            else:self.polygon([(35,50),(35,76),(58,64)])
            self.assertFalse(self.p.drawing.active)
            item=self.p.overlays['blood_pool'][0]
            if identity is None:identity=id(item)
            self.assertEqual(id(item),identity)
        self.start();self.click(20,20)
        self.w.open_synthetic_perfusion()
        until(lambda:self.w.current_pixels is not None)
        self.assertFalse(self.p.drawing.active)
        self.assertFalse(self.p.drawing.preview.isVisible())
        self.assertFalse(self.p.document['tracks'])

    def test_right_click_cancels_and_ctrl_z_removes_draft_point_only(self):
        self.start();self.click(20,20);self.click(20,60)
        QTest.keyClick(self.w.graphics,Qt.Key_Z,Qt.ControlModifier)
        self.assertEqual(len(self.p.drawing.points),1)
        QTest.mouseClick(self.widget,Qt.RightButton,Qt.NoModifier,self.position(25,25))
        self.assertFalse(self.p.drawing.active)
        self.assertEqual(self.p.history.steps,0)

    def test_click_back_to_first_point_closes_polygon(self):
        self.start()
        for point in ((20,20),(20,60),(50,40),(20,20)):self.click(*point)
        self.assertFalse(self.p.drawing.active)
        self.assertEqual(len(self.p.document['tracks'][0]['contour']['outer_rc']),3)

    def test_double_click_finishes_without_duplicate_last_vertex(self):
        self.start()
        self.click(20,20);self.click(20,60);self.click(50,60)
        QTest.mouseDClick(self.widget,Qt.LeftButton,Qt.NoModifier,self.position(50,20))
        QTest.mouseRelease(self.widget,Qt.LeftButton,Qt.NoModifier,self.position(50,20))
        APP.processEvents()
        self.assertFalse(self.p.drawing.active,self.p.drawing.error)
        self.assertEqual(len(self.p.document['tracks'][0]['contour']['outer_rc']),4)

    def test_outer_redraw_preserves_inner_and_edits_clear_curves_and_reviews(self):
        for role in (0,1):
            self.p.role.setCurrentIndex(role);self.p.add_reference()
        self.p.orientation_review.setChecked(True)
        self.p.calculate()
        until(lambda:self.p.result is not None)
        self.p.role.setCurrentIndex(0)
        self.p.boundary.setCurrentIndex(self.p.boundary.findData('outer'))
        inner=copy.deepcopy(self.p.document['tracks'][0]['contour']['inner_rc'])
        self.start('ellipse',0)
        drag(self.widget,self.position(18,20),self.position(78,110))
        self.assertFalse(self.p.drawing.active,self.p.drawing.error)
        self.assertEqual(self.p.document['tracks'][0]['contour']['inner_rc'],inner)
        self.assertIsNone(self.p.result)
        self.assertFalse(any(self.p.document['reviews'].values()))

    def test_analysis_tab_and_dock_hide_cancel_draft(self):
        self.start();self.click(20,20)
        self.w.workspace_tabs.setCurrentIndex(1)
        self.assertFalse(self.p.drawing.active)
        self.w.workspace_tabs.setCurrentIndex(0)
        self.start();self.click(20,20)
        self.w.roi_dock.hide()
        self.assertFalse(self.p.drawing.active)

    def test_removed_native_handles_and_menus_are_deleted_on_gui_thread(self):
        from shiboken6 import isValid
        from PySide6.QtCore import QCoreApplication, QEvent
        self.p.role.setCurrentIndex(1);self.p.add_reference()
        item=self.p.overlays['blood_pool'][0]
        handle=item.getHandles()[0]
        menu=handle.menu
        self.assertIs(menu.parent(),self.w)
        self.assertIs(self.p.plot.getPlotItem().ctrlMenu.parent(),self.p.plot)
        self.assertIs(self.p.plot.getViewBox().menu.parent(),self.p.plot)
        for owned in (menu,self.p.plot.getPlotItem().ctrlMenu,self.p.plot.getViewBox().menu):
            self.assertTrue(owned.windowFlags() & Qt.Popup)
            self.assertFalse(owned.isVisible())
        handle.removeClicked()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete)
        self.assertFalse(isValid(handle))
        self.assertFalse(isValid(menu))

    def test_window_deferred_delete_releases_owned_menus_before_worker_gc(self):
        import gc
        from threading import Thread
        from shiboken6 import isValid
        self.p.add_reference()
        item=self.p.overlays['myocardium'][0]
        menus=[h.menu for pair in self.p.overlays.values() for roi in pair if roi is not None for h in roi.getHandles()]
        menus += [item.getMenu(),self.p.plot.getPlotItem().ctrlMenu,self.p.plot.getViewBox().menu]
        dispose_window(self.w)
        self.assertTrue(all(not isValid(menu) for menu in menus))
        worker=Thread(target=gc.collect,daemon=True)
        worker.start()
        worker.join(3)
        self.assertFalse(worker.is_alive(),'worker_gc_stalled')


if __name__=='__main__':unittest.main()
