REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./pypolo2/configs/CONF.yaml

for %%s in (7 11 18 20 25 36 42 50 60 72 80 85) do (
@REM for %%s in (20 40 60 80 100 150 200 250 300) do (
@REM for %%s in (400 500 600 700) do (    
    for %%t in (DualObjectScheduling) do (
        (
            @REM python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 4 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 1
            @REM python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 4 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 2
            @REM python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 4 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 3
            @REM python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 4 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 4
            @REM python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 4 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 5
            python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 1 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 2
            python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 2 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 2
            python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 2
            python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 4 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 2
            python .\main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 12 --adaptive_step 3 --R_change_interval 9 --sourcenum 2

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