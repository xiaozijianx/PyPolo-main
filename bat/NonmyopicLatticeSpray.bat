REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./pypolo2/configs/CONF.yaml

for %%s in (7 11 18 20 25 36 42 50 60 72 80 85) do (
@REM for %%s in (5 7 11 13 15 18 20 25 32 36) do (
@REM for %%s in (42 46 50 55 60 68 72 80 85 92) do (
    for %%t in (NonmyonicLatticeSpray) do (
        (
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --adaptive_step 4 --sourcenum 1 --R_change_interval 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --adaptive_step 4 --sourcenum 2 --R_change_interval 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --adaptive_step 4 --sourcenum 3 --R_change_interval 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --adaptive_step 4 --sourcenum 4 --R_change_interval 3
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --adaptive_step 4 --sourcenum 5 --R_change_interval 3
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 18 --adaptive_step 4 --sourcenum 6 --R_change_interval 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 18 --adaptive_step 4
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