"""Versioned patient population policy; pending independent data are not cases."""

from .inputs import require, opaque


def population_status(project, patients, eligible):
    version = project["schema_version"]
    counts = {p: sum(r["partition"] == p for r in patients)
              for p in ("development", "independent_test", "excluded")}
    status = {"protocol_version": version, "registered_patients": counts,
              "eligible_development_patients": eligible,
              "independent_evaluation_performed": False}
    if version == 1:
        return status, [], {"development", "independent_test"} <= {r["partition"] for r in patients}
    protocol = project.get("population_protocol", {})
    require(isinstance(protocol, dict) and opaque(protocol.get("protocol_id")), "cohort_population_protocol_required")
    for name in ("development_target_patients", "independent_target_patients"):
        require(type(protocol.get(name)) is int and protocol[name] > 0, "cohort_population_targets")
    state = protocol.get("independent_registration_status")
    require(state in ("pending", "partial", "registered"), "cohort_independent_registration_status")
    issues = []
    if protocol.get("development_population_reviewed") is not True or not opaque(protocol.get("development_population_reference")):
        issues.append("policy_pending:development_population")
    n, target = counts["independent_test"], protocol["independent_target_patients"]
    consistent = ((state == "pending" and n == 0) or (state == "partial" and 0 < n < target)
                  or (state == "registered" and n >= target))
    if not consistent or (n and not opaque(protocol.get("independent_registry_reference"))):
        issues.append("independent_registration_count_or_reference_mismatch")
    status.update({"protocol_id": protocol["protocol_id"],
                   "development_target_patients": protocol["development_target_patients"],
                   "independent_target_patients": target, "independent_registration_status": state,
                   "development_target_gap": max(0, protocol["development_target_patients"]-counts["development"]),
                   "independent_target_gap": max(0, target-n),
                   "independent_registered_to_target": state == "registered" and consistent,
                   "development_population_reference": protocol.get("development_population_reference"),
                   "independent_registry_reference": protocol.get("independent_registry_reference")})
    return status, issues, counts["development"] > 0
