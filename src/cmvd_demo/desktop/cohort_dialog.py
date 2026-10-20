"""Local, modeless D5 cohort manager; controlled file work stays off Qt."""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTreeWidget, QTreeWidgetItem, QHeaderView, QPlainTextEdit)

from cmvd_demo.cohort import assemble_cohort, save_cohort, export_templates
from cmvd_demo.cohort.synthetic import create_synthetic_project
from cmvd_demo.perfusion import load_json
from .tasks import TaskRunner


ISSUES = {
    "manifest": "旧研究编号及处理清单", "associations": "患者与检查关联表", "clinical": "临床表",
    "labels": "逐例医生诊断表", "perfusion": "D4特征登记与审核表", "partitions": "患者用途划分表",
    "missingness": "临床缺失原因表",
    "reported_manifest_count_mismatch": "清单例数与申报例数未对齐",
    "policy_pending:reconciliation_reviewed": "病例对账尚未确认",
    "policy_pending:clinical_schema_reviewed": "候选临床字段及单位尚未确认",
    "policy_pending:exclusions_reviewed": "纳入排除尚未复核",
    "policy_pending:reviewer_code": "未填写审核人代号",
    "policy_pending:label_definition": "诊断定义及依据尚未确认",
    "partition_patient_coverage_mismatch": "划分表未覆盖全部已关联患者",
    "independent_patient_used_or_unknown_development": "独立患者已参与开发或开发用途未核对",
    "index_exam_not_unique": "同一患者需明确选定一次检查",
    "reserved_independent_not_loaded": "独立保留；未纳入开发数据",
    "non_index_exam_or_ambiguous_selection": "非选定检查，或选定检查尚不唯一",
    "explicit_patient_exclusion": "已明确排除该患者",
    "patient_association_missing": "缺少已核对的患者关联",
    "patient_partition_unassigned": "患者用途未分配",
    "cohort_identity_review_pending": "队列患者关联尚未完成核对",
    "manifest_processing_failed_or_empty": "清单记录处理失败或无有效影像",
    "clinical_schema_not_confirmed": "临床字段及单位尚未确认",
    "clinical_provenance_missing": "临床资料时点或来源缺失",
    "physician_label_missing_invalid_or_unreviewed": "医生诊断缺失、定义不符或尚未审核",
    "perfusion_measurement_review_missing": "D4特征缺失或测量尚未审核",
    "cohort_perfusion_branch_missing": "PET或MR分支缺失",
    "cohort_perfusion_qc_pending": "灌注特征QC未通过",
}


def issue_label(code):
    if code.startswith("input_missing:"):
        return "缺少："+ISSUES.get(code.split(":", 1)[1], code)
    return ISSUES.get(code, code)


class CohortDialog(QDialog):
    def __init__(self, owner, workspace=None):
        super().__init__(owner)
        self.workspace = Path(workspace) if workspace else Path(__file__).resolve().parents[3]
        self.project_path = self.workspace/"configs/cohort_project.template.json"
        self.audit = None
        self.last_output = None
        self.last_error = ""
        self.runner = TaskRunner(self)
        self.runner.result.connect(self._result)
        self.runner.failed.connect(self._failed)
        self.runner.progress.connect(lambda rev, stage, done, total: self.status.setText(f"处理中 {done}/{total} · 可取消"))
        self.runner.idle.connect(self._idle)
        self.setWindowTitle("D5 · 队列整理与患者划分")
        self.resize(1100, 720)
        layout = QVBoxLayout(self)
        title = QLabel("D5  /  开发队列与独立患者保留")
        title.setObjectName("brand")
        layout.addWidget(title)
        self.mode_label = QLabel("真实队列 · 尚未校验")
        self.mode_label.setObjectName("pill")
        layout.addWidget(self.mode_label)
        note = QLabel("复用旧编号，不从阳性/阴性目录推导诊断。独立患者全部检查排除在开发特征与交叉验证之外。")
        note.setWordWrap(True)
        layout.addWidget(note)
        controls = QHBoxLayout()
        self.choose_button = QPushButton("打开队列配置")
        self.choose_button.clicked.connect(self._choose)
        self.check_button = QPushButton("校验队列")
        self.check_button.clicked.connect(lambda: self.run_action("check"))
        self.template_button = QPushButton("导出空白补录表")
        self.template_button.clicked.connect(lambda: self.run_action("templates"))
        self.synthetic_button = QPushButton("完全合成队列演示")
        self.synthetic_button.clicked.connect(lambda: self.run_action("synthetic"))
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel)
        self.cancel_button.setEnabled(False)
        for button in (self.choose_button, self.check_button, self.template_button, self.synthetic_button, self.cancel_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        self.summary = QLabel("尚未读取清单，不创建真实病例。可先导出空白表或运行完全合成演示。")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.issues = QPlainTextEdit()
        self.issues.setReadOnly(True)
        self.issues.setMaximumHeight(170)
        self.issues.setPlaceholderText("待补资料、关联冲突与排除原因将在校验后显示")
        layout.addWidget(self.issues)
        self.table = QTreeWidget()
        self.table.setHeaderLabels(["原研究编号", "患者代号", "用途", "状态 / 原因"])
        self.table.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.header().setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)
        bottom = QHBoxLayout()
        self.export_button = QPushButton("重新校验并导出审计")
        self.export_button.clicked.connect(lambda: self.run_action("export"))
        self.freeze_button = QPushButton("重新校验并冻结患者划分")
        self.freeze_button.setEnabled(False)
        self.freeze_button.clicked.connect(lambda: self.run_action("freeze"))
        bottom.addWidget(self.export_button)
        bottom.addWidget(self.freeze_button)
        self.output_button = QPushButton("打开本次输出目录")
        self.output_button.setEnabled(False)
        self.output_button.clicked.connect(self.open_output)
        bottom.addWidget(self.output_button)
        bottom.addStretch()
        layout.addLayout(bottom)
        self.status = QLabel("未训练模型 · 未输出诊断")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _choose(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开受控队列配置", str(self.workspace/"outputs"), "JSON (*.json)")
        if path:
            self.select_project(path)

    def select_project(self, path):
        self.cancel()
        self.project_path = Path(path)
        self._clear()
        self.mode_label.setText("已选择队列配置 · 请重新校验")

    def _clear(self):
        self.audit, self.last_output, self.last_error = None, None, ""
        self.table.clear()
        self.issues.clear()
        self.freeze_button.setEnabled(False)
        self.output_button.setEnabled(False)
        self.summary.setText("未校验 · 旧结果已清除")

    def run_action(self, action):
        self._clear()
        self.status.setText("正在处理本地受控表与快照…")
        self.cancel_button.setEnabled(True)
        path, workspace = self.project_path, self.workspace
        def work(progress, cancel):
            if action == "templates":
                run = export_templates(workspace, workspace/"outputs/cohort_requests", cancel=cancel)
                return {"action": action, "project_path": run/"cohort_project.json", "output": run}
            selected = create_synthetic_project(workspace, workspace/"outputs/cohort_fixtures", progress=progress, cancel=cancel) if action == "synthetic" else path
            audit = assemble_cohort(workspace, load_json(selected), progress=progress, cancel=cancel)
            output = save_cohort(workspace/"outputs/cohort", audit, freeze=action == "freeze", cancel=cancel) if action in ("synthetic", "export", "freeze") else None
            return {"action": action, "audit": audit, "project_path": selected, "output": output}
        self.runner.submit(work)

    def _result(self, revision, payload):
        self.project_path, self.last_output = payload["project_path"], payload["output"]
        self.output_button.setEnabled(self.last_output is not None)
        if payload["action"] == "templates":
            self.mode_label.setText("真实队列 · 已导出空白补录表")
            self.summary.setText("没有创建旧清单或真实编号。补录表与队列配置已写入 outputs/cohort_requests/ 的新目录。")
            self.status.setText("已选中新配置；填写后点击“校验队列”。旧研究清单仍需找回。")
            return
        self.audit = audit = payload["audit"]
        report = audit["report"]
        synthetic = report["mode"] == "synthetic_fixture"
        self.mode_label.setText("完全合成队列 · 仅验证工程流程 · 不可用于真实训练" if synthetic else "真实队列 · 本地受控资料")
        self.summary.setText(f'清单检查 {report["manifest_case_count"]} 次 / 关联患者 {report["associated_patient_count"]} 人；'
                             f'开发完整行 {report["development_included_count"]}；保留独立患者 {report["reserved_independent_patients"]} 人。'
                             f'患者交叉验证：{"可构建" if report["folds"]["ready"] else "条件不足"}。')
        messages = [issue_label(code) for code in report["issues"]]
        if synthetic:
            messages.insert(0, "所有病例、临床值、标签与测量均为合成；D4仍保留工程体模标记，真实特征就绪状态为否。")
        if not report["freeze_ready"]:
            messages.append("患者划分暂不可冻结：需完整关联与对账、明确开发/独立用途及人工复核。")
        messages.append("没有执行填补、标准化、训练或临床验证。")
        self.issues.setPlainText("\n".join(messages))
        for row in audit["cases"]:
            partition = {"development": "开发", "independent_test": "独立保留", "excluded": "排除"}.get(row["partition"], "未分配")
            status = "合成工程行" if synthetic and row["included"] else "候选完整行" if row["included"] else " | ".join(issue_label(c) for c in row["reasons"])
            self.table.addTopLevelItem(QTreeWidgetItem([row["research_id"], row["patient_key"], partition, status]))
        frozen = payload["action"] == "freeze"
        self.status.setText("已冻结患者划分，输出位于 outputs/cohort/ 新目录。" if frozen else
                            "已导出审计，输出位于 outputs/cohort/ 新目录。" if self.last_output else "校验结束；可导出审计与逐例缺失原因。")
        if synthetic:
            self.status.setText(self.status.text()+" 完全合成示例不能进入真实训练。")

    def _failed(self, revision, code):
        self.last_error = code
        self.status.setText("未完成 · "+code+" · 未发布有效队列")

    def open_output(self):
        if self.last_output is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_output)))

    def _idle(self):
        self.cancel_button.setEnabled(False)
        self.freeze_button.setEnabled(bool(self.audit and self.audit["report"]["freeze_ready"]))

    def cancel(self):
        self.runner.invalidate()
        self._clear()
        self.cancel_button.setEnabled(False)
        self.status.setText("已取消 · 旧结果不会发布")

    def reject(self):
        self.cancel()
        super().reject()

    def closeEvent(self, event):
        self.cancel()
        event.accept()
