REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./pypolo2/configs/CONF.yaml

REM for seed in 0 3 7 11 13 15 18 20 32 42
for %%s in (0 3 7 11 15 18 20 32 42 ^
            50 55 60 66 70) do (
    for %%t in (MaximumCoverageSpray) do (
        (
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 18 
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 7 --sche_step 18 
            rem python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 18 
        ) || (
            REM Append the ERRORLEVEL (PID) to the pids variable
            set "pids=!pids!!ERRORLEVEL!!"
            echo Terminating processes...
            REM Terminate all background processes
            for %%p in (%pids%) do (
                taskkill /PID %%p /F >nul
            )
            exit /b 1
        )
    )
)

exit /b 1