"""Fixed human labels shared by desktop and research-report format v1."""
def unit_label(unit):
    return {'SYNTHETIC_AU':'合成信号单位','SYNTHETIC_AU/s':'合成信号单位/秒',
            'SYNTHETIC_AU*s':'合成信号单位·秒','MR_SIGNAL_AU':'MR信号单位',
            'MR_SIGNAL_AU/s':'MR信号单位/秒','MR_SIGNAL_AU*s':'MR信号单位·秒','1':'无量纲'}.get(unit,unit)


def measurement_label(name):
    special={'normalized_upslope':'心肌/血池斜率比','mr_mpri':'MR灌注储备指数（MPRI，半定量）',
        'rest_normalized_upslope':'静息心肌/血池斜率比','stress_normalized_upslope':'负荷心肌/血池斜率比',
        'rest_mbf':'静息心肌血流量（MBF）','stress_mbf':'负荷心肌血流量（MBF）','pet_mfr':'PET心肌血流储备（MFR）'}
    if name in special:return special[name]
    metrics={'baseline':'基线信号','peak':'峰值信号','peak_enhancement':'峰值增强','relative_peak_enhancement':'相对峰值增强',
        'peak_time':'达峰参考时间','ttp_from_window_origin':'窗口起点至达峰时间','max_upslope':'最大上升斜率','signed_enhancement_auc':'增强曲线积分'}
    parts=name.split('.',1)
    return ({'myocardium':'心肌','blood_pool':'血池'}.get(parts[0],parts[0])+metrics.get(parts[1],parts[1])) if len(parts)==2 else name
