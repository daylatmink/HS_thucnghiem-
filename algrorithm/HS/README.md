# Harmony Search CSV runner

Thu muc nay giu lai thuat toan HS goc va them runner doc dau vao tu CSV.

## Chay voi template mau

```powershell
python .\algrorithm\HS\csv_runner.py `
  --algorithm hs `
  --tasks ".\template\task-template (1).csv" `
  --resources .\template\resource-template.csv `
  --kpi-definitions .\template\kpi-definition-template.csv `
  --kpi-targets .\template\kpi-target-template.csv `
  --cycle .\template\cycle-template.csv `
  --output-json .\algrorithm\HS\output\result.json `
  --output-csv .\algrorithm\HS\output\assignment.csv
```

## Format cac cot dang list

- `dependencies`: cach nhau bang dau phay, vi du `TASK_A,TASK_B`
- `requiredSkills`: `SKILL_NAME:REQUIRED_LEVEL`, vi du `FASTAPI:MID,MONGODB:SENIOR`; neu khong ghi level thi mac dinh la `3`
- `kpiImpacts`: `KPI_CODE:weight`, vi du `PROGRESS_RATE:0.5,QUALITY_SCORE:0.5`
- `skills`: `SKILL_NAME:LEVEL`, vi du `FASTAPI:SENIOR,MONGODB:MID`

## Schedule hien tai

- Schedule ban dau duoc tao tu dependency va duration de lay topo order.
- Sau khi optimizer sinh assignment `task -> resource`, evaluator lap lich lai theo resource.
- Mot resource chi lam mot task tai mot thoi diem; task sau phai cho predecessor xong va resource ranh.
- `scheduleScore` = `baseMakespan / actualMakespan * 100`, clamp trong khoang `0..100`.
- Output JSON co `diagnostics.actualSchedule`, `makespan`, `baseMakespan`; output CSV dung actual schedule.

## Chay baseline

- `--algorithm random`: random multi-start, dung `--max-iterations` lam so lan thu.
- `--algorithm greedy`: greedy heuristic, chay mot lan theo topo order.

Vi du:

```powershell
python .\algrorithm\HS\csv_runner.py `
  --algorithm greedy `
  --tasks .\algrorithm\HS\examples\msrcpsp_10_5_8_5\tasks.csv `
  --resources .\algrorithm\HS\examples\msrcpsp_10_5_8_5\resources.csv `
  --kpi-definitions .\algrorithm\HS\examples\msrcpsp_10_5_8_5\kpi-definitions.csv `
  --kpi-targets .\algrorithm\HS\examples\msrcpsp_10_5_8_5\kpi-targets.csv `
  --cycle .\algrorithm\HS\examples\msrcpsp_10_5_8_5\cycle.csv
```
