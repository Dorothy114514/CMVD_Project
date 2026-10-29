"""Native-Qt frozen artifact checks. Only synthetic images may be captured."""

import hashlib
import json
import re
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QCoreApplication, QEvent
from PySide6.QtWidgets import (QPushButton, QLabel, QAbstractButton, QPlainTextEdit,
                              QLineEdit, QComboBox, QTabWidget, QTreeWidget)
from PySide6.QtGui import QAction

from cmvd_demo.io import DicomReadError
from cmvd_demo.paths import workspace_root, frozen
from cmvd_demo.perfusion import load_json
from cmvd_demo.perfusion.storage import write_json
from cmvd_demo.roi import document_hash
from .presentation import PRODUCT_NAME, PRODUCT_VERSION, WINDOW_TITLE


def check_product_language(window):
    """Inspect actual widget content, including hidden tabs and cohort results."""
    texts = [window.windowTitle(), window.cohort_dialog.windowTitle()]
    for cls in (QLabel, QAbstractButton, QAction):
        texts.extend(item.text() for item in window.findChildren(cls))
    texts.extend(item.toPlainText() for item in window.findChildren(QPlainTextEdit))
    texts.extend(item.placeholderText() for item in window.findChildren(QLineEdit))
    for combo in window.findChildren(QComboBox):
        texts.extend(combo.itemText(i) for i in range(combo.count()))
    for tabs in window.findChildren(QTabWidget):
        texts.extend(tabs.tabText(i) for i in range(tabs.count()))
    for tree in window.findChildren(QTreeWidget):
        nodes = [tree.headerItem()]+[tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while nodes:
            node = nodes.pop()
            texts.extend(node.text(c) for c in range(tree.columnCount()))
            nodes.extend(node.child(i) for i in range(node.childCount()))
    check(not any(re.search(r"(?<![A-Za-z0-9])D[0-8](?![0-9])", text) for text in texts),
          "release_development_code_in_product_ui")
    check(window.windowTitle() == WINDOW_TITLE, "release_product_title_wrong")


def wait_for(condition, timeout=60000):
    if condition():
        return
    loop, timer, deadline = QEventLoop(), QTimer(), QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: loop.quit() if condition() else None)
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    timer.start()
    deadline.start(timeout)
    loop.exec()
    timer.stop()
    deadline.stop()
    if not condition():
        raise DicomReadError("release_check_timeout")


def check(ok, code):
    if not ok:
        raise DicomReadError(code)


def curves(window):
    p = window.roi_panel
    for role in (0, 1):
        p.role.setCurrentIndex(role)
        p.add_reference()
    p.document["purpose"] = "engineering_fixture"
    p.calculate()
    wait_for(lambda: p.result is not None or bool(p.last_error))
    check(not p.last_error, "release_roi_curve_failed")
    return p


def check_manual_drawing(app, window, output):
    """Drive actual viewport events, including the release's gesture routing."""
    from PySide6.QtCore import Qt, QPointF
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    p, widget = window.roi_panel, window.graphics.viewport()
    def position(r,c):
        return window.graphics.mapFromScene(window.image.mapToScene(QPointF(c+.5,r+.5)))
    wait_for(lambda:not window.runner.busy)
    p.role.setCurrentIndex(0)
    p.draw_shape.setCurrentIndex(0)
    p.draw_button.click()
    check(p.drawing.active, 'release_manual_draw_start_failed')
    start,end = position(20,33),position(76,95)
    QTest.mouseMove(widget,start)
    app.processEvents()
    QTest.mousePress(widget,Qt.LeftButton,Qt.NoModifier,start)
    for fraction in (.3,.6,1.):
        QTest.qWait(20)
        point=QPointF(start)+(QPointF(end)-QPointF(start))*fraction
        event=QMouseEvent(QEvent.MouseMove,point,QPointF(widget.mapToGlobal(point.toPoint())),
                          Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
        QApplication.sendEvent(widget,event)
        app.processEvents()
    QTest.mouseRelease(widget,Qt.LeftButton,Qt.NoModifier,end)
    check(not p.drawing.active and len(p.document['tracks'])==1,'release_manual_ellipse_failed')
    expected=document_hash(p.document)
    p.undo()
    check(not p.document['tracks'],'release_manual_draw_undo_failed')
    p.redo()
    check(document_hash(p.document)==expected,'release_manual_draw_redo_failed')
    p.boundary.setCurrentIndex(2)
    check(not p.overlays['myocardium'][0].isEnabled() and p.overlays['myocardium'][1].isEnabled(),
          'release_manual_boundary_lock_failed')
    p.role.setCurrentIndex(1)
    p.draw_shape.setCurrentIndex(1)
    p.draw_button.click()
    for r,c in ((42,58),(42,70),(54,70),(54,58)):
        QTest.mouseClick(widget,Qt.LeftButton,Qt.NoModifier,position(r,c))
    p.finish_draw_button.click()
    check(not p.drawing.active and len(p.document['tracks'])==2,'release_manual_polygon_failed')
    expected=document_hash(p.document)
    p.draw_button.click()
    QTest.mouseClick(widget,Qt.LeftButton,Qt.NoModifier,position(30,45))
    p.cancel_draw_button.click()
    check(not p.drawing.active and document_hash(p.document)==expected,'release_manual_draft_cancel_failed')
    p.boundary.setCurrentIndex(0)
    p.calculate()
    wait_for(lambda:p.result is not None or bool(p.last_error))
    check(not p.last_error and p.result is not None,'release_manually_drawn_curve_failed')
    check(window.grab().save(str(output/'synthetic-roi-drawing.png')),'release_screenshot_failed')


def check_curve_link(app, window, output):
    from PySide6.QtCore import Qt,QPointF
    from PySide6.QtTest import QTest
    p,f,link=window.roi_panel,window.feature_panel,window.curve_link
    curves(window)
    f.condition.setCurrentIndex(1)
    f.baseline.setValue(2);f.start.setValue(3);f.end.setValue(7)
    window.workspace_tabs.setCurrentIndex(1)
    window.workspace_tabs.widget(1).ensureWidgetVisible(f.curve_plot)
    app.processEvents()
    def click(time):
        vb=f.curve_plot.getViewBox()
        lo,hi=vb.viewRange()[1]
        point=f.curve_plot.mapFromScene(vb.mapViewToScene(QPointF(time,lo+(hi-lo)*.7)))
        QTest.mouseClick(f.curve_plot.viewport(),Qt.LeftButton,Qt.NoModifier,point)
        app.processEvents()
    click(7.9)
    wait_for(lambda:window.displayed_indices==(5,0))
    check(link.cursor.value()==8. and p._cursor.value()==8.,'release_curve_seek_failed')
    f.curve_tool.setCurrentIndex(2)
    click(3.9);click(10.)
    check(f.start.value()==4 and f.end.value()==7,'release_curve_window_click_failed')
    check(all(pair[1].getRegion()==(4.,10.) for pair in link.regions),'release_curve_regions_not_synced')
    f.calculate_mr()
    wait_for(lambda:f.candidate is not None or bool(f.last_error))
    check(f.candidate is not None and not f.window_review.isChecked(),'release_curve_recalculation_failed')
    check(window.grab().save(str(output/'synthetic-curve-link.png')),'release_screenshot_failed')
    f.start.setValue(3)
    f.curve_tool.setCurrentIndex(0)
    check(f.candidate is None,'release_changed_window_candidate_not_cleared')


def check_sequence_review(app,window,output):
    """Synthetic capability/header/grid/frame failures; never capture real data."""
    from .data import Choice,PhantomGrid
    w=window;p=w.sequences;key='synthetic-review'
    w._reset_case()
    w._populate([Choice(key,'合成方向与时间体模','synthetic','unknown',60,PhantomGrid()),
        Choice('synthetic-index-only','合成截图示例','secondary_capture','unknown',1,None)],
        {'indexed_objects':61,'series_count':2,'issues':[{'stage':'header_read','code':'read_failed_unclassified','file_token':'c'*16}]})
    wait_for(lambda:w.current_pixels is not None)
    check(p.model.rows[key]['state']=='previewed','release_sequence_preview_state_wrong')
    w.sequence_filter.setCurrentIndex(w.sequence_filter.findData('index_only'))
    w.select_choice('synthetic-index-only')
    check(w.current_pixels is None and w.grid is None and p.model.rows['synthetic-index-only']['state']=='index_only',
          'release_index_only_state_wrong')
    w.sequence_filter.setCurrentIndex(0)
    original=w.backend.grid
    def failed(*args):raise DicomReadError('missing_time_axis')
    try:
        w.backend.grid=failed;w.select_choice(key)
        wait_for(lambda:bool(w.last_error))
    finally:w.backend.grid=original
    check(p.model.rows[key]['state']=='partial_failure','release_sequence_failure_lost')
    p.open()
    def locate(code):
        index=next(i for i,row in enumerate(p._visible) if row['code']==code)
        p.table.selectRow(index);p.locate_button.click()
    locate('missing_time_axis');wait_for(lambda:w.current_pixels is not None)
    p.model.failure(key,'pixels','pixel_decode_failed',[3,2]);p.refresh()
    locate('pixel_decode_failed');wait_for(lambda:w.displayed_indices==(3,2))
    check(p.model.rows[key]['state']=='previewed' and not any(row['active'] for row in p.model.issues if row['stage'] in ('grid','pixels')),
          'release_sequence_retry_not_recorded')
    index=next(i for i,row in enumerate(p._visible) if row['file_token']=='c'*16)
    p.table.selectRow(index);p.disposition.setCurrentIndex(1);p.mark_button.click()
    p.export_button.click();wait_for(lambda:not p.runner.busy)
    check(not p.last_error and p.saved_output is not None,'release_sequence_export_failed')
    record=p.saved_output
    check((record/'bundle.json').is_file() and not load_json(record/'review.json')['all_pixels_checked'],
          'release_sequence_export_claim_wrong')
    p.import_history(record/'review.json');wait_for(lambda:not p.runner.busy)
    check(not p.last_error and p.model.rows[key]['state']=='previewed' and any(row['origin']=='history' for row in p.model.issues),
          'release_sequence_history_association_failed')
    p.visibility.setCurrentIndex(1);app.processEvents()
    check_product_language(w)
    check(p.dialog.grab().save(str(output/'synthetic-sequence-issues.png')),'release_screenshot_failed')
    p.dialog.hide();w.open_synthetic();wait_for(lambda:w.current_pixels is not None)
    return record.relative_to(workspace_root()).as_posix()


def check_comparison_research(app,window,output):
    from cmvd_demo.research_record import load_research
    from cmvd_demo.roi import document_hash
    w=window;c=w.comparison;r=w.research
    expected=document_hash(w.roi_panel.document);current=w.current_choice.key
    w.compare_button.click();wait_for(lambda:all(p.displayed is not None for p in c.panes.values()))
    left,right=(c.panes[s] for s in ('rest','stress'))
    check(not c.link.isChecked(),'release_comparison_should_start_independent')
    left.time.setValue(4);wait_for(lambda:left.displayed==(4,0))
    check(right.displayed==(0,0),'release_comparison_independent_seek_failed')
    c.link.setChecked(True);left.time.setValue(5)
    wait_for(lambda:left.displayed==(5,0) and right.displayed==(5,0))
    check(w.current_choice.key==current and document_hash(w.roi_panel.document)==expected,'release_comparison_changed_main_work')
    check(c.table.rowCount()==17,'release_comparison_retained_measurements_missing')
    check(c.dialog.grab().save(str(output/'synthetic-rest-stress-comparison.png')),'release_screenshot_failed')
    w.research_button.click();r.export_button.click();wait_for(lambda:not r.busy)
    check(not r.last_error and r.last_output is not None,'release_research_export_failed')
    root=r.last_output;data=load_research(root)
    check(data['clinical_validated'] is False and data['synthetic_data'] is True,'release_research_flags_wrong')
    check('MR灌注储备指数' in (root/'summary.html').read_text(encoding='utf-8'),'release_research_summary_missing_values')
    r.verify(root/'research.json');wait_for(lambda:not r.busy)
    check(not r.last_error,'release_research_integrity_check_failed')
    r.restore_button.click();wait_for(lambda:not w.work.busy)
    check(not w.work.last_error and document_hash(w.roi_panel.document)==expected,'release_research_work_restore_failed')
    check(r.dialog.grab().save(str(output/'synthetic-research-record.png')),'release_screenshot_failed')
    r.dialog.hide();c.dialog.close()
    return root.relative_to(workspace_root()).as_posix()


def check_safety_fixes(app,window,output):
    """Exercise QA fixes in source and actual frozen packages; synthetic only."""
    from threading import Event
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox,QScrollArea
    from cmvd_demo.io import ReadCancelled
    from . import research_controller,sequence_panel
    w=window
    w.open_synthetic_perfusion();wait_for(lambda:w.current_pixels is not None)
    w.roi_panel.add_reference()
    before=document_hash(w.roi_panel.document);epoch=w.case_revision
    response=[]
    def dismiss_prompt():
        dialog=app.activeModalWidget()
        if isinstance(dialog,QMessageBox):
            response.append(dialog.button(QMessageBox.Cancel).text())
            dialog.grab().save(str(output/'synthetic-unsaved-prompt.png'))
            dialog.button(QMessageBox.Cancel).click()
    w.unsaved.prompt=None
    QTimer.singleShot(0,dismiss_prompt)
    w.open_synthetic()
    check(response==['取消操作'] and w.case_revision==epoch and document_hash(w.roi_panel.document)==before,
          'release_unsaved_cancel_failed')
    w.unsaved.prompt=lambda action:QMessageBox.Save
    w.open_synthetic();wait_for(lambda:w.case_revision>epoch)
    check(w.work.saved_output is not None and not w.unsaved.dirty(),'release_save_before_switch_failed')
    w.unsaved.prompt=lambda action:QMessageBox.Discard
    w.open_synthetic_perfusion();wait_for(lambda:w.current_pixels is not None)
    c=w.comparison;c.open();wait_for(lambda:all(p.displayed is not None for p in c.panes.values()))
    pane=c.panes['rest'];original=w.backend.plane;started,release=Event(),Event()
    def delayed_plane(*args):started.set();release.wait(5);return original(*args)
    w.backend.plane=delayed_plane
    try:
        pane.time.setValue(3);wait_for(started.is_set)
        QTest.keyClick(c.dialog,Qt.Key_Escape)
        check(pane._job is None and not c.dialog.isVisible(),'release_comparison_escape_failed')
        release.set();wait_for(lambda:not pane.runner.busy)
        check(pane.displayed is None,'release_comparison_late_frame')
    finally:release.set();w.backend.plane=original
    for controller,module,name in ((w.research,research_controller,'save_research'),
                                    (w.sequences,sequence_panel,'save_review')):
        original=getattr(module,name);started,release=Event(),Event()
        def delayed(*args,**kwargs):started.set();release.wait(5);raise ReadCancelled()
        setattr(module,name,delayed)
        try:
            controller.open();controller.export();wait_for(started.is_set)
            QTest.keyClick(controller.dialog,Qt.Key_Escape)
            check(controller._job is None and w.centralWidget().isEnabled(),'release_report_escape_failed')
            release.set();wait_for(lambda:not controller.runner.busy)
        finally:release.set();setattr(module,name,original)
    r=w.research;oldroot=r.output_root;blocked=output/'synthetic-output-file'
    blocked.write_text('Synthetic output-path failure fixture.',encoding='utf-8');r.output_root=blocked
    try:
        r.open();r.export();wait_for(lambda:not r.busy)
        check(r.last_error=='output_path_unavailable' and r.last_output is None,'release_output_error_not_specific')
        events=list((workspace_root()/'outputs/runtime_logs').glob('*-events.jsonl'))
        check(any('output_path_unavailable' in p.read_text(encoding='utf-8') for p in events),'release_business_error_log_missing')
        r.dialog.grab().save(str(output/'synthetic-output-error.png'))
    finally:r.output_root=oldroot;r.dialog.close()
    check(isinstance(w.centralWidget(),QScrollArea),'release_main_scroll_missing')
    check(bool(c.dialog.findChildren(QScrollArea)),'release_comparison_scroll_missing')


def run_checks(app, window, output, *, case_dir=None, resume_record=None):
    from PySide6.QtWidgets import QMessageBox
    window.unsaved.prompt=lambda action: QMessageBox.Discard
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"completed": False, "passed": False, "frozen": frozen(), "qt_platform": app.platformName(),
              "product_name": PRODUCT_NAME, "product_version": PRODUCT_VERSION,
              "checks": [], "failures": [], "clinical_validation": False, "human_acceptance": False,
              "independent_viewer_comparison": False, "real_case_directories_checked": 0}
    write_json(output/"attempt.json", report)
    try:
        check(window.current_pixels is None, "release_expected_empty_start")
        check(window.viewer_stack.currentWidget() is window.empty_view, "release_empty_state_missing")
        # Observe beyond QProgressDialog's delayed-show threshold. No case
        # should start indexing or show a reading dialog during idle startup.
        idle_progress=[]
        idle_loop,idle_timer=QEventLoop(),QTimer()
        idle_timer.setInterval(25)
        idle_timer.timeout.connect(lambda:idle_progress.append(window._index_dialog.isVisible() or window._pending_index))
        idle_timer.start();QTimer.singleShot(6000,idle_loop.quit);idle_loop.exec();idle_timer.stop()
        check(not any(idle_progress) and not window.choices and not window.runner.busy,'release_idle_case_reading_unexpected')
        report['checks'].append('idle_start_no_case_progress_after_delayed_show_threshold')
        check_product_language(window)
        check(window.grab().save(str(output/"empty-workstation.png")), "release_screenshot_failed")
        report["checks"].append("empty_start")
        if resume_record is not None:
            window.open_synthetic_perfusion()
            wait_for(lambda:window.current_pixels is not None)
            window.work.restore(resume_record)
            wait_for(lambda:not window.work.busy)
            check(not window.work.last_error and window.work.restored_output is not None,"release_cross_process_work_restore_failed")
            check(np.isclose(window.feature_panel._mr_pair()["measurements"]["mr_mpri"]["value"],2.),"release_restored_pair_wrong")
            check(not any(window.roi_panel.document["reviews"].values()),"release_restored_review_upgraded")
            report["checks"].append("work_restore_from_previous_executable_process")
        check_safety_fixes(app,window,output)
        report['checks'].append('unsaved_save_cancel_dismissal_layout_safe_business_logging')
        check(not any("合成" in b.text() for b in window.centralWidget().findChildren(QPushButton)),
              "release_demo_button_in_main_workspace")
        check(window.synthetic_demo_action in window.demo_menu.actions()
              and window.perfusion_demo_action in window.demo_menu.actions(), "release_demo_menu_missing")
        window.synthetic_demo_action.trigger()
        wait_for(lambda: window.current_pixels is not None)
        report["checks"].append("menu_only_synthetic_entry")
        for t in (3, 8, 1, 11):
            window.time_slider.setValue(t)
        wait_for(lambda: window.displayed_indices == (11, 0))
        np.testing.assert_array_equal(window.current_pixels, window.grid.read_plane(11, 0))
        report["checks"].append("synthetic_render_and_final_seek")
        report['sequence_review_record']=check_sequence_review(app,window,output)
        report['checks'].append('sequence_states_index_only_failure_retry_disposition_export_advisory_import')
        check_manual_drawing(app,window,output)
        report['checks'].append('manual_ellipse_polygon_boundary_lock_and_draft_cancel')
        p = curves(window)
        expected_doc = document_hash(p.document)
        p.save()
        wait_for(lambda: p.saved_output is not None or bool(p.last_error))
        check(not p.last_error and (p.saved_output/"bundle.json").is_file(), "release_roi_save_failed")
        p.load_roi(p.saved_output/"roi.json")
        wait_for(lambda: p._job is None)
        check(not p.last_error and document_hash(p.document) == expected_doc, "release_roi_reload_failed")
        report["checks"].append("roi_curve_save_reload")
        check(window.grab().save(str(output/"synthetic-workstation.png")), "release_screenshot_failed")
        window.perfusion_demo_action.trigger()
        wait_for(lambda: window.current_pixels is not None)
        f = window.feature_panel
        f.protocol.setText("SYNTH-RELEASE")
        f.baseline.setValue(2)
        f.start.setValue(3)
        f.end.setValue(7)
        check_curve_link(app,window,output)
        report['checks'].append('curve_click_seek_manual_windows_and_invalidation')
        for condition, index in (("rest", 1), ("stress", 2)):
            window.select_choice("synthetic-perfusion-"+condition)
            wait_for(lambda: window.current_pixels is not None and window.current_choice.key == "synthetic-perfusion-"+condition)
            curves(window)
            f.condition.setCurrentIndex(index)
            f.calculate_mr()
            wait_for(lambda: f.candidate is not None or bool(f.last_error))
            check(not f.last_error, "release_mr_calculation_failed")
            f.capture_mr()
        pair = f._mr_pair()
        check(np.isclose(pair["measurements"]["mr_mpri"]["value"], 2.), "release_synthetic_mpri_wrong")
        f.save()
        wait_for(lambda: f.saved_output is not None or bool(f.last_error))
        check(not f.last_error and (f.saved_output/"bundle.json").is_file(), "release_feature_save_failed")
        check(not load_json(f.saved_output/"features.json")["qc"]["model_input_ready"], "release_synthetic_ready_incorrect")
        report["checks"].append("d4_synthetic_mpri_and_save")
        expected_doc = document_hash(window.roi_panel.document)
        expected_curves = document_hash(window.roi_panel.result)
        expected_mr = document_hash(f.candidate)
        window.work.save(single=True)
        wait_for(lambda:not window.work.busy)
        check(not window.work.last_error and window.work.saved_output is not None,"release_single_mr_save_failed")
        single_marker = load_json(window.work.saved_output/"bundle.json")
        check(single_marker["scope"] == "single_mr" and not single_marker["model_input_ready"],"release_single_mr_scope_wrong")
        window.work.save()
        wait_for(lambda:not window.work.busy)
        check(not window.work.last_error and window.work.saved_output is not None,"release_work_save_failed")
        work_record = window.work.saved_output
        report["saved_work_record"] = work_record.relative_to(workspace_root()).as_posix()
        window.open_synthetic_perfusion()
        wait_for(lambda:window.current_pixels is not None)
        window.work.restore(work_record)
        wait_for(lambda:not window.work.busy)
        check(not window.work.last_error and window.work.restored_output is not None,"release_work_restore_failed")
        check(document_hash(window.roi_panel.document) == expected_doc
              and document_hash(window.roi_panel.result) == expected_curves
              and document_hash(window.feature_panel.candidate) == expected_mr,"release_work_recomputed_values_differ")
        check(not any(window.roi_panel.document["reviews"].values()),"release_restored_review_upgraded")
        report["checks"].append("single_mr_save_case_work_restore_recompute")
        report['research_record']=check_comparison_research(app,window,output)
        report['checks'].append('rest_stress_independent_linked_view_research_export_verify_restore')
        window.workspace_tabs.setCurrentIndex(1)
        app.processEvents()
        check_product_language(window)
        check(window.grab().save(str(output/"synthetic-perfusion.png")), "release_screenshot_failed")
        d = window.cohort_dialog
        d.run_action("check")
        wait_for(lambda: not d.runner.busy)
        check(not d.last_error and not d.audit["report"]["training_inputs_ready"], "release_missing_cohort_gate_failed")
        d.run_action("templates")
        wait_for(lambda: not d.runner.busy)
        check(not d.last_error and (d.last_output/"clinical_features.csv").is_file(), "release_template_export_failed")
        window.cohort_demo_action.trigger()
        wait_for(lambda: not d.runner.busy)
        check(not d.last_error and d.audit["report"]["development_included_count"] == 6, "release_synthetic_cohort_failed")
        check_product_language(window)
        check(d.grab().save(str(output/"synthetic-cohort.png")), "release_screenshot_failed")
        d.hide()
        report["checks"].append("product_language_empty_state_and_demo_navigation")
        report["checks"].append("d5_missing_gate_templates_synthetic")
        if case_dir is not None:
            window.open_case(case_dir)
            wait_for(lambda: window.current_pixels is not None or bool(window.last_error), timeout=180000)
            check(not window.last_error, "release_case_read_failed")
            check(not window.index_summary["issues"], "release_case_header_issues")
            seen, planes = set(), 0
            for choice in window.choices:
                if choice.kind not in ("pet_static", "pet_dynamic", "pet_gated", "mr_perfusion", "mr_cine", "cine_stack") or choice.kind in seen:
                    continue
                window.select_choice(choice.key)
                wait_for(lambda: (window.current_pixels is not None and window.current_choice.key == choice.key) or bool(window.last_error))
                check(not window.last_error, "release_case_grid_failed")
                grid = window.grid
                for t, z in ((0, 0), (grid.shape[0]-1, grid.shape[1]-1)):
                    window.time_slider.setValue(t)
                    window.slice_slider.setValue(z)
                    wait_for(lambda: window.displayed_indices == (t, z))
                    np.testing.assert_array_equal(window.current_pixels, grid.read_plane(t, z))
                    planes += 1
                seen.add(choice.kind)
            check(bool(seen), "release_no_supported_case_branch")
            report["real_case_directories_checked"] = 1
            report["case_branches_checked"] = sorted(seen)
            report["case_pixel_planes_checked"] = planes
            report["checks"].append("one_case_first_last_supported_branches")
    except Exception as error:
        report["failures"].append(error.code if isinstance(error, DicomReadError) else type(error).__name__)
    finally:
        window.close()
        try:
            wait_for(lambda: not any(r.busy for r in window.background_runners()))
            app.processEvents()
            window.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            report["checks"].append("background_drain_and_native_cleanup")
        except Exception as error:
            report["failures"].append(type(error).__name__)
    report.update(completed=True, passed=not report["failures"])
    write_json(output/"report.json", report)
    if report["passed"]:
        write_json(output/"bundle.json", {"completed": True, "files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir()) if p.is_file()}})
    return 0 if report["passed"] else 1
