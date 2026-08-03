from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any

try:
    from app.modules.assignment.algorithms.models import CandidateScore, TaskSchedule
    from app.modules.assignment.algorithms.schedule import ResourceAwareScheduler
except ModuleNotFoundError:
    from models import CandidateScore, TaskSchedule
    from schedule import ResourceAwareScheduler


# Trọng số mặc định khi không có state model (đồng bộ với StateModelService)
_DEFAULT_FEATURE_WEIGHTS: dict[str, float] = {
    "skillMatch": 0.55,
    "capacityFit": 0.25,
    "costFit": 0.10,
    "priorityFit": 0.10,
}

# Ánh xạ feature weights (state model) → strategy weights (evaluator)
# Mỗi feature trong state model ảnh hưởng đến 1 component của evaluator
_FEATURE_TO_STRATEGY: dict[str, str] = {
    "skillMatch": "skill",
    "capacityFit": "workload",
    "costFit": "cost",
    "priorityFit": "schedule",
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


SKILL_LEVEL = {
    "INTERN": 1.0,
    "JUNIOR": 2.0,
    "MID": 3.0,
    "INTERMEDIATE": 3.0,
    "MIDDLE": 3.0,
    "SENIOR": 4.0,
    "LEAD": 5.0,
    "EXPERT": 5.0,
}


class AssignmentEvaluator:
    def __init__(
        self,
        tasks: list[dict[str, Any]],
        resources: list[dict[str, Any]],
        targets: list[dict[str, Any]],
        definitions: list[dict[str, Any]],
        schedule: dict[str, TaskSchedule],
        predecessors: dict[str, list[str]] | None = None,
        constraints: Any = None,
        preferences: dict[str, Any] | None = None,
        strategy: str = "BALANCED",
        state_model: dict[str, Any] | None = None,
        metric_weights: dict[str, float] | None = None,
    ) -> None:
        self.tasks = tasks
        self.resources = resources
        self.targets = targets
        self.definitions = definitions
        self.schedule = schedule
        self.predecessors = predecessors or {}
        self.constraints = constraints
        self.preferences = preferences or {}
        self.strategy = strategy
        self.state_model = state_model or {}
        self.metric_weights = self._normalize_metric_weights(metric_weights)
        self.task_by_id = {str(task.get("taskId")): task for task in tasks if task.get("taskId")}
        self.resource_by_id = {str(resource.get("resourceId")): resource for resource in resources if resource.get("resourceId")}
        self.target_by_code = {str(target.get("kpiCode")): target for target in targets if target.get("kpiCode")}
        self.definition_by_code = {str(item.get("kpiCode")): item for item in definitions if item.get("kpiCode")}
        self.hard_constraints = list(getattr(constraints, "hardConstraints", []) or [])
        self.assignment_scheduler = ResourceAwareScheduler(tasks, schedule, self.predecessors)
        # Tích hợp state model: build lookup và resolve weights
        self._capability_index: dict[tuple[str, str], dict[str, Any]] = self._build_capability_index()
        self._resolved_feature_weights: dict[str, float] = self._resolve_feature_weights()

    def evaluate(self, assignment: dict[str, str]) -> CandidateScore:
        actual_schedule = self.assignment_scheduler.build(assignment)
        skill_score = self.skill_score(assignment)
        workload_score, load_info = self.workload_score(assignment, actual_schedule)
        cost_score, total_cost = self.cost_score(assignment, actual_schedule)
        schedule_score = self.schedule_score(assignment, actual_schedule)
        kpi_score, predicted = self.kpi_estimate_score(assignment)
        hard_violations = self.hard_violations(assignment, load_info, predicted)
        weights = self.strategy_weights()
        penalty = min(80.0, len(hard_violations) * 25.0)
        total = (
            weights["kpi"] * kpi_score
            + weights["skill"] * skill_score
            + weights["workload"] * workload_score
            + weights["cost"] * cost_score
            + weights["schedule"] * schedule_score
            - penalty
        )
        return CandidateScore(
            total_score=round(max(0.0, min(100.0, total)), 2),
            kpi_score=round(kpi_score, 2),
            skill_score=round(skill_score, 2),
            workload_score=round(workload_score, 2),
            cost_score=round(cost_score, 2),
            schedule_score=round(schedule_score, 2),
            hard_violations=hard_violations,
            diagnostics={
                "resourceLoads": load_info,
                "actualSchedule": self._schedule_diagnostics(actual_schedule),
                "makespan": self._makespan(actual_schedule),
                "baseMakespan": self._makespan(self.schedule),
                "estimatedKpis": predicted,
                "totalCost": round(total_cost, 2),
            },
        )

    def strategy_weights(self) -> dict[str, float]:
        """Trả về strategy weights đã được điều chỉnh bởi featureWeights của state model.

        Cách hoạt động:
          1. Lấy base weights theo strategy (KPI_FOCUSED, BALANCED, ...)
          2. Nếu state model có featureWeights (từ T4 feedback), blend vào
             theo ánh xạ _FEATURE_TO_STRATEGY (skillMatch→skill, capacityFit→workload, ...)
          3. Normalize để tổng = 1.0
          4. Nếu không có state model → trả về base weights như cũ
        """
        base_mapping = {
            "KPI_FOCUSED": {"kpi": 0.55, "skill": 0.20, "workload": 0.15, "cost": 0.05, "schedule": 0.05},
            "COST_FOCUSED": {"kpi": 0.25, "skill": 0.10, "workload": 0.15, "cost": 0.40, "schedule": 0.10},
            "SPEED_FOCUSED": {"kpi": 0.25, "skill": 0.15, "workload": 0.15, "cost": 0.05, "schedule": 0.40},
            "BALANCED":     {"kpi": 0.35, "skill": 0.20, "workload": 0.25, "cost": 0.10, "schedule": 0.10},
        }
        if self.metric_weights:
            return self.metric_weights
        base = dict(base_mapping.get(self.strategy, base_mapping["BALANCED"]))

        # Không có state model hoặc featureWeights → trả về base nguyên bản
        fw = self._resolved_feature_weights
        if not fw or fw == _DEFAULT_FEATURE_WEIGHTS:
            return base

        # Blend: mỗi feature weight ảnh hưởng component tương ứng trong evaluator
        # kpi weight giữ nguyên để không làm mất trọng tâm nghiệp vụ
        non_kpi_base_total = sum(v for k, v in base.items() if k != "kpi")
        if non_kpi_base_total <= 0:
            return base

        # Tính delta từ state model feature weights
        default_fw = _DEFAULT_FEATURE_WEIGHTS
        blended = dict(base)
        for feature_key, strategy_key in _FEATURE_TO_STRATEGY.items():
            if strategy_key not in blended:
                continue
            default_val = default_fw.get(feature_key, 0.0)
            current_val = fw.get(feature_key, default_val)
            # Tỷ lệ thay đổi so với default: >1 tăng, <1 giảm
            ratio = current_val / default_val if default_val > 0 else 1.0
            # Blend 30% từ state model, 70% từ strategy base (tránh thay đổi đột ngột)
            blended[strategy_key] = base[strategy_key] * (0.70 + 0.30 * ratio)

        # Giữ nguyên kpi weight, normalize phần còn lại
        kpi_w = blended["kpi"]
        non_kpi = {k: v for k, v in blended.items() if k != "kpi"}
        non_kpi_total = sum(non_kpi.values())
        target_non_kpi = 1.0 - kpi_w
        if non_kpi_total > 0 and target_non_kpi > 0:
            scale = target_non_kpi / non_kpi_total
            for k in non_kpi:
                blended[k] = round(non_kpi[k] * scale, 4)
        blended["kpi"] = round(kpi_w, 4)
        return blended

    def _normalize_metric_weights(self, weights: dict[str, float] | None) -> dict[str, float] | None:
        if not isinstance(weights, dict):
            return None
        allowed = ("kpi", "skill", "workload", "cost", "schedule")
        result: dict[str, float] = {}
        for key in allowed:
            if key in weights:
                value = safe_float(weights.get(key), -1.0)
                if value >= 0:
                    result[key] = value
        if not result:
            return None
        for key in allowed:
            result.setdefault(key, 0.0)
        total = sum(result.values())
        if total <= 0:
            return None
        return {key: round(value / total, 6) for key, value in result.items()}

    def resource_skill_score(self, task_id: str, resource_id: str) -> float:
        """Tính điểm skill fit giữa resource và task.

        Ưu tiên dùng skillMatch từ RESOURCE_TASK_CAPABILITY edge trong state model
        (đã tính composite: skill + capacity + cost với featureWeights được calibrate).
        Fallback về tính thủ công nếu không có capability edge.
        """
        # Ưu tiên 1: capability edge từ state model graph (pre-computed)
        capability = self._capability_index.get((resource_id, task_id))
        if capability:
            raw_skill_match = safe_float(capability.get("skillMatch"), -1.0)
            if raw_skill_match >= 0:
                return round(raw_skill_match * 100.0, 2)

        # Fallback: tính thủ công từ skill data thô (behavior cũ)
        task = self.task_by_id.get(task_id, {})
        required = self._required_skills(task.get("requiredSkills") or [])
        if not required:
            return 100.0
        resource = self.resource_by_id.get(resource_id, {})
        skill_levels = {}
        for skill in resource.get("skills") or []:
            name = str(skill.get("skillName", "")).strip()
            level = skill.get("level", 1)
            if name:
                skill_levels[name] = self._skill_level(level)
        scores = []
        for requirement in required:
            skill_name = requirement["skillName"]
            required_level = max(1.0, self._skill_level(requirement.get("level", 3)))
            have = skill_levels.get(skill_name, 0.0)
            scores.append(min(1.0, have / required_level))
        return (mean(scores) * 100.0) if scores else 100.0

    def resource_assignment_heuristic_score(
        self,
        task_id: str,
        resource_id: str,
        remaining_capacity: float | None = None,
    ) -> float:
        """Dynamic task-resource priority used by optimizer construction steps.

        The final objective is still ``evaluate()``. This lightweight score keeps
        greedy construction, pitch-adjustment, and candidate ordering aligned with
        the same metric weights configured by the user.
        """
        weights = self.strategy_weights()
        resource = self.resource_by_id.get(resource_id, {})
        task = self.task_by_id.get(task_id, {})
        duration = self.schedule.get(task_id).duration_hours if task_id in self.schedule else safe_float(task.get("estimatedHours"), 1.0)
        capacity = max(1.0, safe_float(resource.get("capacity"), 1.0))
        current_load = safe_float(resource.get("currentLoad"))
        available = remaining_capacity if remaining_capacity is not None else max(0.0, capacity - current_load)

        skill = self.resource_skill_score(task_id, resource_id)
        capacity_fit = max(0.0, min(100.0, (available / max(duration, 1.0)) * 100.0))
        slack_fit = max(0.0, min(100.0, ((available - duration) / capacity) * 100.0))

        rates = [safe_float(item.get("costPerHour"), 50.0) for item in self.resources]
        min_rate = min(rates) if rates else 50.0
        max_rate = max(rates) if rates else 50.0
        rate = safe_float(resource.get("costPerHour"), 50.0)
        if max_rate <= min_rate:
            cost = 100.0
        else:
            cost = max(0.0, min(100.0, 100.0 * (1.0 - ((rate - min_rate) / (max_rate - min_rate)))))

        schedule = 100.0 if available >= duration else max(0.0, min(100.0, available / max(duration, 1.0) * 100.0))
        kpi = min(100.0, sum(safe_float(impact.get("weight")) for impact in task.get("kpiImpacts") or []) * skill)

        return (
            weights["kpi"] * kpi
            + weights["skill"] * skill
            + weights["workload"] * ((capacity_fit + slack_fit) / 2.0)
            + weights["cost"] * cost
            + weights["schedule"] * schedule
        )

    def skill_score(self, assignment: dict[str, str]) -> float:
        scores = [self.resource_skill_score(task_id, resource_id) for task_id, resource_id in assignment.items()]
        return mean(scores) if scores else 0.0

    def workload_score(
        self,
        assignment: dict[str, str],
        actual_schedule: dict[str, TaskSchedule] | None = None,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        loads = {
            resource_id: {
                "capacity": max(safe_float(resource.get("capacity")), 1.0),
                "currentLoad": safe_float(resource.get("currentLoad")),
                "allocated": 0.0,
                "assignedTasksCount": 0.0,
            }
            for resource_id, resource in self.resource_by_id.items()
        }
        schedule = actual_schedule or self.schedule
        for task_id, resource_id in assignment.items():
            if resource_id not in loads:
                continue
            duration = schedule.get(task_id).duration_hours if task_id in schedule else safe_float(self.task_by_id.get(task_id, {}).get("estimatedHours"))
            loads[resource_id]["allocated"] += duration
            loads[resource_id]["assignedTasksCount"] += 1

        utilizations = []
        overload_penalty = 0.0
        for item in loads.values():
            utilization = ((item["currentLoad"] + item["allocated"]) / item["capacity"]) * 100.0
            item["utilizationRate"] = round(utilization, 2)
            utilizations.append(utilization)
            overload_penalty += max(0.0, utilization - 100.0)

        active = [value for value in utilizations if value > 0]
        if not active:
            return 100.0, loads
        balance = 100.0 / (1.0 + (pstdev(active) / max(mean(active), 1.0)))
        score = balance - min(60.0, overload_penalty)
        return max(0.0, min(100.0, score)), loads

    def cost_score(
        self,
        assignment: dict[str, str],
        actual_schedule: dict[str, TaskSchedule] | None = None,
    ) -> tuple[float, float]:
        rates = [safe_float(resource.get("costPerHour"), 50.0) for resource in self.resources]
        min_rate = min(rates) if rates else 50.0
        max_rate = max(rates) if rates else 50.0
        total_hours = 0.0
        total_cost = 0.0
        schedule = actual_schedule or self.schedule
        for task_id, resource_id in assignment.items():
            duration = schedule.get(task_id).duration_hours if task_id in schedule else safe_float(self.task_by_id.get(task_id, {}).get("estimatedHours"))
            rate = safe_float(self.resource_by_id.get(resource_id, {}).get("costPerHour"), 50.0)
            total_hours += duration
            total_cost += duration * rate
        if total_hours <= 0 or max_rate <= min_rate:
            return 100.0, total_cost
        avg_rate = total_cost / total_hours
        score = 100.0 * (1.0 - ((avg_rate - min_rate) / (max_rate - min_rate)))
        return max(0.0, min(100.0, score)), total_cost

    def schedule_score(self, assignment: dict[str, str], actual_schedule: dict[str, TaskSchedule]) -> float:
        if not assignment:
            return 0.0
        base_makespan = self._makespan(self.schedule)
        actual_makespan = self._makespan(actual_schedule)
        if actual_makespan <= 0:
            return 100.0
        if base_makespan <= 0:
            return 100.0
        return max(0.0, min(100.0, (base_makespan / actual_makespan) * 100.0))

    def kpi_estimate_score(self, assignment: dict[str, str]) -> tuple[float, dict[str, float]]:
        estimated: dict[str, float] = {code: 0.0 for code in self.target_by_code}
        for task_id, resource_id in assignment.items():
            task = self.task_by_id.get(task_id, {})
            skill_factor = self.resource_skill_score(task_id, resource_id) / 100.0
            for impact in task.get("kpiImpacts") or []:
                code = str(impact.get("kpiCode", "")).strip()
                if not code:
                    continue
                target = safe_float(self.target_by_code.get(code, {}).get("targetValue"), 100.0)
                estimated[code] = estimated.get(code, 0.0) + safe_float(impact.get("weight")) * target * skill_factor

        if not self.target_by_code:
            return 100.0, estimated
        weighted_score = 0.0
        weight_total = 0.0
        for code, target_doc in self.target_by_code.items():
            target = safe_float(target_doc.get("targetValue"), 0.0)
            weight = safe_float(target_doc.get("weight"), 1.0)
            direction = "MINIMIZE" if self.definition_by_code.get(code, {}).get("higherIsBetter") is False else "MAXIMIZE"
            predicted = estimated.get(code, 0.0)
            if target <= 0:
                score = 100.0
            elif direction == "MINIMIZE":
                score = 100.0 if predicted <= target else max(0.0, min(100.0, (target / max(predicted, 1.0)) * 100.0))
            else:
                score = max(0.0, min(100.0, (predicted / target) * 100.0))
            weighted_score += weight * score
            weight_total += weight
        return weighted_score / weight_total if weight_total else 100.0, estimated

    def hard_violations(
        self,
        assignment: dict[str, str],
        load_info: dict[str, dict[str, float]],
        predicted_kpis: dict[str, float],
    ) -> list[dict[str, Any]]:
        violations = []
        for constraint in self.hard_constraints:
            constraint_type = str(constraint.get("type", "")).upper()
            if constraint_type == "MAX_WORKLOAD_PER_RESOURCE":
                max_value = safe_float(constraint.get("value"), 100.0)
                unit = str(constraint.get("unit", "PERCENT")).upper()
                for resource_id, item in load_info.items():
                    value = item.get("currentLoad", 0.0) + item.get("allocated", 0.0) if unit == "HOURS" else item.get("utilizationRate", 0.0)
                    if value > max_value:
                        violations.append({"type": constraint_type, "resourceId": resource_id, "value": round(value, 2), "limit": max_value})
            elif constraint_type == "SKILL_MATCH_REQUIRED" and bool(constraint.get("value", True)):
                for task_id, resource_id in assignment.items():
                    if self.resource_skill_score(task_id, resource_id) < 100.0:
                        violations.append({"type": constraint_type, "taskId": task_id, "resourceId": resource_id})
            elif constraint_type == "MIN_KPI_ACHIEVEMENT":
                code = str(constraint.get("kpiCode", "")).strip()
                min_value = safe_float(constraint.get("minValue"))
                if code and predicted_kpis.get(code, 0.0) < min_value:
                    violations.append({"type": constraint_type, "kpiCode": code, "predictedValue": round(predicted_kpis.get(code, 0.0), 2), "minValue": min_value})
        return violations

    def _skill_level(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(5.0, float(value)))
        text = str(value).strip().upper()
        if text in SKILL_LEVEL:
            return SKILL_LEVEL[text]
        return safe_float(value, 1.0)

    def _required_skills(self, value: list[Any]) -> list[dict[str, Any]]:
        requirements = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("skillName", "")).strip()
                if name:
                    requirements.append({"skillName": name, "level": item.get("level", 3)})
                continue
            text = str(item).strip()
            if not text:
                continue
            if ":" in text:
                name, level = text.split(":", 1)
                requirements.append({"skillName": name.strip(), "level": level.strip()})
            else:
                requirements.append({"skillName": text, "level": 3})
        return requirements

    def _makespan(self, schedule: dict[str, TaskSchedule]) -> float:
        return max((item.planned_end_hour for item in schedule.values()), default=0.0)

    def _schedule_diagnostics(self, schedule: dict[str, TaskSchedule]) -> dict[str, dict[str, float]]:
        return {
            task_id: {
                "plannedStartHour": item.planned_start_hour,
                "plannedEndHour": item.planned_end_hour,
                "durationHours": item.duration_hours,
                "topoLevel": item.topo_level,
            }
            for task_id, item in schedule.items()
        }

    # ------------------------------------------------------------------
    # State model helpers
    # ------------------------------------------------------------------

    def _build_capability_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Parse RESOURCE_TASK_CAPABILITY edges từ state model graph thành dict
        (resourceId, taskId) → edge_attributes để tra cứu O(1) trong evaluate."""
        index: dict[tuple[str, str], dict[str, Any]] = {}
        graph_data = self.state_model.get("graphData") if isinstance(self.state_model, dict) else None
        if not isinstance(graph_data, dict):
            return index

        raw_nodes = graph_data.get("nodes") or []
        raw_edges = graph_data.get("links") or graph_data.get("edges") or []

        # Build node lookup: node_id → resource/task attributes
        node_by_id: dict[str, dict[str, Any]] = {}
        for node in raw_nodes:
            if isinstance(node, dict):
                node_id = str(node.get("id") or "")
                if node_id:
                    node_by_id[node_id] = node

        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            edge_type = str(edge.get("edge_type") or edge.get("edgeType") or "").upper()
            if edge_type != "RESOURCE_TASK_CAPABILITY":
                continue
            src_node = node_by_id.get(str(edge.get("source") or ""), {})
            tgt_node = node_by_id.get(str(edge.get("target") or ""), {})
            res_id = str(src_node.get("resourceId") or "")
            task_id = str(tgt_node.get("taskId") or tgt_node.get("taskCode") or "")
            if res_id and task_id:
                index[(res_id, task_id)] = edge
        return index

    def _resolve_feature_weights(self) -> dict[str, float]:
        """Lấy featureWeights đã được calibrate từ state model.
        Chỉ giữ lại keys có trong _DEFAULT_FEATURE_WEIGHTS, normalize về tổng 1.0.
        Trả về default nếu state model không có hoặc weights không hợp lệ."""
        stored = self.state_model.get("featureWeights") if isinstance(self.state_model, dict) else None
        if not isinstance(stored, dict) or not stored:
            return dict(_DEFAULT_FEATURE_WEIGHTS)

        result: dict[str, float] = {}
        for key in _DEFAULT_FEATURE_WEIGHTS:
            if key in stored:
                val = safe_float(stored[key], -1.0)
                if val >= 0:
                    result[key] = val

        if not result:
            return dict(_DEFAULT_FEATURE_WEIGHTS)

        # Điền missing keys bằng default rồi normalize
        for key, default_val in _DEFAULT_FEATURE_WEIGHTS.items():
            result.setdefault(key, default_val)

        total = sum(result.values())
        if total <= 0:
            return dict(_DEFAULT_FEATURE_WEIGHTS)
        return {k: round(v / total, 6) for k, v in result.items()}

    def state_model_info(self) -> dict[str, Any]:
        """Trả về thông tin debug về state model đang được sử dụng."""
        return {
            "hasStateModel": bool(self.state_model),
            "capabilityEdges": len(self._capability_index),
            "featureWeightsSource": "state_model" if self._resolved_feature_weights != _DEFAULT_FEATURE_WEIGHTS else "default",
            "featureWeights": self._resolved_feature_weights,
        }


def workload_balance_summary(utilization: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [safe_float(item.get("utilizationRate")) for item in utilization]
    avg = mean(rates) if rates else 0.0
    std = pstdev(rates) if len(rates) > 1 else 0.0
    variance = std**2
    return {
        "avgUtilization": round(avg, 2),
        "stdDeviation": round(std, 2),
        "variance": round(variance, 2),
        "balanceScore": max(0.0, round(100.0 - std, 2)),
        "imbalancedResources": [item["resourceId"] for item in utilization if safe_float(item.get("utilizationRate")) > 100.0],
    }
