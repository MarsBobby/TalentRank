@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo ============================================================
echo        TalentRank System Launcher ^& Pipeline Auto-Run
echo ============================================================

:: 1. 放弃脆弱的批处理路径检测，改用 Python 稳定执行冷启动自愈逻辑
if not exist "gold_standards" (
    echo [ENV] Target folder "gold_standards" not found. Running Python self-healing...
    python -c "import os, glob, json; gd='gold_standards'; dr='data'; os.makedirs(gd, exist_ok=True); m={'品牌':'branding','电商':'ecommerce','人力资源':'hr','生产':'manufacturing','研发':'rnd','销售':'sales'}; [json.dump({os.path.splitext(os.path.basename(f))[0]: (4 if i==0 else (3 if i==1 else 2)) for i, f in enumerate(glob.glob(os.path.join(dr, d, '*.docx')))}, open(os.path.join(gd, f'{k}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2) for k, d in m.items() if os.path.exists(os.path.join(dr, d))]"
    echo [ENV] Mock gold standards generated successfully.
    echo.
)

echo ============================================================
echo [SYSTEM] Starting evaluation pipelines for all 6 functions...
echo ============================================================

:: 2. 依次平铺执行 6 大职能，杜绝复杂的嵌套循环和符号污染
echo.
echo ------------------------------------------------------------
echo  [1/6] Processing Function Domain: [品牌]
echo ------------------------------------------------------------
python main.py --function "品牌" --data_dir D:/project_TalentSystem/data/branding
if errorlevel 1 goto :run_failed

echo.
echo ------------------------------------------------------------
echo  [2/6] Processing Function Domain: [人力资源]
echo ------------------------------------------------------------
python main.py --function "人力资源" --data_dir D:/project_TalentSystem/data/hr
if errorlevel 1 goto :run_failed

echo.
echo ------------------------------------------------------------
echo  [3/6] Processing Function Domain: [生产]
echo ------------------------------------------------------------
python main.py --function "生产" --data_dir D:/project_TalentSystem/data/manufacturing
if errorlevel 1 goto :run_failed

echo.
echo ------------------------------------------------------------
echo  [4/6] Processing Function Domain: [电商]
echo ------------------------------------------------------------
python main.py --function "电商" --data_dir D:/project_TalentSystem/data/ecommerce
if errorlevel 1 goto :run_failed

echo.
echo ------------------------------------------------------------
echo  [5/6] Processing Function Domain: [销售]
echo ------------------------------------------------------------
python main.py --function "销售" --data_dir D:/project_TalentSystem/data/sales
if errorlevel 1 goto :run_failed

echo.
echo ------------------------------------------------------------
echo  [6/6] Processing Function Domain: [研发]
echo ------------------------------------------------------------
python main.py --function "研发" --data_dir D:/project_TalentSystem/data/rnd
if errorlevel 1 goto :run_failed

echo.
echo ============================================================
echo [SUCCESS] Full-link modeling pipeline completed successfully!
echo Multi-dim feature matrices are saved in outputs/
echo ============================================================
pause
exit /b 0

:run_failed
echo.
echo ============================================================
echo [FAILED] Execution stopped. Check the Python Traceback above for details.
echo ============================================================
pause
exit /b 1