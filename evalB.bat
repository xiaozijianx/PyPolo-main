REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./pypolo2/configs/CONF.yaml

REM for seed in 0 3 7 11 13 15 18 20 32 42
REM for %%s in (0 3 7 11 13 15 18 20 32 42 ^
REM            50 51 52 53 54 55 56 57 58 59) do (
for %%s in (0 3) do (
    for %%t in (SA_time_non_uniform) do (
        (
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 18
            REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 15
            REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 12
        ) 
        echo Terminating processes...
        REM Terminate all background processes
        for %%p in (%pids%) do (
            taskkill /PID %%p /F >nul
        )
    )
)

exit /b 1