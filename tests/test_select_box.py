"""Native vertical dropdown selection, dismissal and restored layout semantics."""
import unittest
from PySide6.QtCore import Qt,QCoreApplication,QEvent,QEventLoop,QTimer
from PySide6.QtTest import QTest,QSignalSpy
from PySide6.QtWidgets import QComboBox
from test_desktop import APP,until,dispose_window,discard_unsaved
from cmvd_demo.desktop.select_box import SelectBox
from cmvd_demo.desktop.window import BrowserWindow


def popup_ready(box):
    QTest.mouseClick(box,Qt.LeftButton)
    until(lambda:box.view().isVisible())
    # The native popup also ignores synthetic release events immediately after
    # opening (plain QComboBox comparison retained in QA). Let Qt settle first.
    loop=QEventLoop();QTimer.singleShot(APP.doubleClickInterval()+120,loop.quit);loop.exec()
    return box.view()


def click_row(box,index):
    view=box.view();point=view.visualRect(box.model().index(index,0)).center()
    QTest.mouseMove(view.viewport(),point);APP.processEvents()
    QTest.mouseClick(view.viewport(),Qt.LeftButton,pos=point)


class SelectBoxTests(unittest.TestCase):
    def setUp(self):
        self.box=SelectBox()
        for label,value in [('请选择已核对状态',None),('静息','rest'),('负荷','stress')]:
            self.box.addItem(label,value)
        self.box.resize(280,38);self.box.move(350,220);self.box.show();APP.processEvents()

    def tearDown(self):
        self.box.hidePopup();self.box.close();self.box.deleteLater()
        QCoreApplication.sendPostedEvents(None,QEvent.DeferredDelete);APP.processEvents()

    def open(self):
        return popup_ready(self.box)

    def test_popup_click_commits_once_and_blocked_restore_keeps_data(self):
        spy=QSignalSpy(self.box.currentIndexChanged)
        view=self.open()
        click_row(self.box,2)
        until(lambda:not view.isVisible())
        self.assertEqual(self.box.currentData(),'stress');self.assertEqual(spy.count(),1)
        self.box.blockSignals(True);self.box.setCurrentIndex(self.box.findData(None));self.box.blockSignals(False)
        self.assertIsNone(self.box.currentData());self.assertEqual(spy.count(),1)

    def test_hover_and_escape_do_not_commit_a_different_acquisition(self):
        self.box.setCurrentIndex(1);spy=QSignalSpy(self.box.currentIndexChanged)
        view=self.open()
        QTest.mouseMove(view.viewport(),view.visualRect(self.box.model().index(2,0)).center())
        self.assertEqual(self.box.currentData(),'rest')
        QTest.keyClick(view,Qt.Key_Escape);until(lambda:not view.isVisible())
        self.assertEqual(self.box.currentData(),'rest');self.assertEqual(spy.count(),0)

    def test_disabled_control_and_long_scrollable_list(self):
        self.box.setEnabled(False);QTest.mouseClick(self.box,Qt.LeftButton)
        self.assertFalse(self.box.view().isVisible());self.assertIsNone(self.box.currentData())
        self.box.setEnabled(True)
        for i in range(40):self.box.addItem('较长的影像序列候选 '+str(i),i)
        view=self.open();last=self.box.model().index(self.box.count()-1,0)
        view.scrollTo(last);APP.processEvents()
        self.assertGreater(view.verticalScrollBar().maximum(),0)
        click_row(self.box,self.box.count()-1)
        until(lambda:not view.isVisible());self.assertEqual(self.box.currentData(),39)


class OriginalLayoutTests(unittest.TestCase):
    def test_original_side_by_side_roi_dropdowns_and_mouse_choice(self):
        w=BrowserWindow(unsaved_prompt=discard_unsaved)
        try:
            w.show();w.open_synthetic_perfusion();until(lambda:w.current_pixels is not None)
            p=w.roi_panel
            self.assertIsInstance(p.role,QComboBox)
            self.assertEqual(p.role.geometry().y(),p.scope.geometry().y())
            self.assertEqual(p.boundary.itemText(0),'内外膜一起编辑')
            self.assertEqual(w.roi_tool.itemText(0),'编辑轮廓点')
            self.assertEqual(w.roi_tool.currentData(),'pan')
            self.assertIsNone(w.feature_panel.condition.currentData())
            view=popup_ready(p.role)
            click_row(p.role,1)
            until(lambda:not view.isVisible())
            self.assertEqual(p.role.currentData(),'blood_pool');self.assertFalse(p.boundary.isEnabled())
        finally:dispose_window(w)
