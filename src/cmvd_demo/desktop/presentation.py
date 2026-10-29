"""Product identity and safe, patient-independent presentation language."""

PRODUCT_NAME = "心灌影像工作站"
PRODUCT_SUBTITLE = "PET/MR 心肌灌注研究"
PRODUCT_VERSION = "0.7.4"
WINDOW_TITLE = PRODUCT_NAME + " · " + PRODUCT_SUBTITLE


from cmvd_demo.perfusion.labels import unit_label

ERROR_MESSAGES = {
    "output_disk_full": "保存位置空间不足，未完成记录不会记为成功；请释放空间或改用可写工作目录。",
    "filesystem_permission_denied": "没有访问所选文件或保存位置的权限；请核对访问权限或改用可写工作目录。",
    "output_path_unavailable": "保存目录被同名文件占用或目录类型不正确；请保留原有记录，改用可用的工作目录后重试。",
    "file_unavailable": "所需文件已不存在或暂不可访问，请核对原资料和完整记录目录后重试。",
    "compare_time_outside": "另一侧采样时间范围外，未跳转；保留原位置，不补帧或插值。",
    "compare_invalid_position": "对照位置无效，请重新选择序列或帧。",
    "research_wait_comparison": "请先等待两侧影像到达目标位置，并完成或取消勾画，再导出研究记录。",
    "research_invalid_record": "研究记录格式或来源校验未通过，请保留完整导出目录。",
    "research_case_mismatch": "序列记录与当前病例不匹配，已停止导出。",
    "research_stale_result": "病例、轮廓、分析或对照位置已变化，未发布旧记录，请重新导出。",
    "research_file_changed": "导出文件被改写或派生内容不一致，完整性核对未通过。",
    "research_incomplete": "研究记录缺失文件或完成标记；中断目录不作为完整记录。",
    "research_file_too_large": "研究记录超出单文件容量，已停止导出；请保留未完成目录排查。",
    "sequence_report_incomplete": "历史报告缺少完成标记或文件哈希不符，请选择完整的逐包报告目录。",
    "sequence_report_case_mismatch": "报告与当前病例/序列清单不匹配，请选对应逐包报告，不能用总体汇总报告替代。",
    "sequence_report_changed": "报告在读取过程中发生变化，已停止关联。",
    "sequence_report_too_large": "报告超出大小或记录数量限制，未关联。",
    "sequence_invalid_report": "报告格式或字段不符合要求，当前序列状态未改变。",
    "feature_window_two_frames_required": "区间至少需要两个不同采样帧。",
    "roi_drawing_too_small": "轮廓太小，请拖出至少两个像素宽和高的区域。",
    "roi_draw_outer_first": "请先勾画心外膜，再单独勾画心内膜。",
    "case_directory_missing": "病例目录不存在，请重新选择已解压的病例目录。",
    "case_no_indexed_objects": "目录中没有可索引的影像序列，请选择单个已解压病例目录。",
    "duplicate_sop_instance": "存在重复影像标识，请核对原始导出版本。",
    "duplicate_spatial_time_cell": "同一层面和时点存在多个影像，需核对导出版本。",
    "missing_spatial_time_cell": "影像的层面或时间点不完整，请核对采集与导出。",
    "declared_frame_count_mismatch": "实际时间点数与影像声明不一致，请核对完整导出。",
    "declared_slice_count_mismatch": "实际层面数与影像声明不一致，请核对完整导出。",
    "zero_rescale_slope": "影像缩放参数为零，暂不能用于定量，请联系医院核对导出。",
    "missing_time_axis": "缺少可靠时间信息，暂不能组装动态序列。",
    "missing_geometry": "缺少空间定位信息，暂不能组装此序列。",
    "inconsistent_geometry": "序列内空间信息不一致，请核对采集与导出。",
    "source_changed_since_index": "影像文件已变化，请重新打开病例后再操作。",
    "source_changed_during_read": "读取时影像文件发生变化，请重新打开病例。",
    "series_not_supported_for_assembly": "此类影像暂不支持预览，可保留供独立查看器核对。",
    "unsupported_pixel_layout": "此影像像素格式暂不支持预览。",
    "desktop_plane_memory_budget_exceeded": "单幅影像超过当前预览容量，请选择其他序列。",
    "roi_myocardium_blood_overlap": "心肌与血池轮廓重叠，请调整后重新计算。",
    "roi_no_tracks_on_plane": "当前层面尚无轮廓，请先勾画心肌与血池。",
    "roi_time_series_required": "当前影像没有可计算的时间序列，可先保存轮廓。",
    "roi_stale_curve_result": "轮廓已变化，请重新计算曲线后再保存。",
    "roi_source_binding_mismatch": "轮廓或工作记录与当前影像的来源、几何、时间或单位不一致，已停止载入。",
    "roi_invalid_json": "轮廓文件内容无效，请选择完整保存的轮廓文件。",
    "feature_current_curves_required": "请先在“轮廓与曲线”中计算当前层面的心肌、血池曲线。",
    "feature_condition_required": "请先确认并选择静息或负荷状态。",
    "feature_wrong_branch": "所选序列不适用于当前分析，请切换对应灌注序列。",
    "feature_mr_pair_required": "请分别保留静息和负荷MR结果后再配对。",
    "feature_invalid_json": "结果文件内容无效，请选择完整的分析或工作站结果文件。",
    "work_open_case_first": "请先打开原病例目录或对应合成演示，再恢复工作记录。",
    "work_wait_current_tasks": "请先等待影像或曲线计算完成，再保存工作。",
    "work_case_mismatch": "工作记录与当前病例不匹配，请打开原病例后再恢复。",
    "work_series_missing": "工作记录所需序列未找到，请核对是否打开原病例目录。",
    "work_completion_required": "工作记录未完整保存，不能恢复；请保留该目录供排查。",
    "work_file_changed": "工作记录文件发生变化，已停止恢复，请使用完整的原保存目录。",
    "work_recalculation_mismatch": "复算结果与保存记录不一致，已停止恢复，请核对来源和软件版本。",
    "work_invalid_record": "工作记录格式或内容无效，请核对完整保存目录。",
    "work_stale_result": "影像、轮廓或分析状态已变化，请重新计算后保存。",
}


def error_message(code):
    """Never echo arbitrary exception text, source paths or research values."""
    if code in ERROR_MESSAGES:
        return ERROR_MESSAGES[code]
    if code.startswith("work_"):
        return "工作记录校验未通过，当前结果未恢复；请核对原病例和完整保存目录。"
    if code.startswith("roi_"):
        return "轮廓或来源校验未通过，请检查轮廓、层面和对应影像。"
    if code.startswith("feature_"):
        return "分析条件或来源校验未通过，请核对采集状态、曲线、分析窗口及参考结果。"
    if code.startswith("cohort_"):
        return "研究资料校验未通过，请核对配置、补录表和审核记录。"
    return "操作未完成，请检查所选资料；可从“文件→打开工作目录”查看运行日志。"
