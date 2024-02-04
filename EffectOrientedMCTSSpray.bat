REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./pypolo2/configs/CONF.yaml

REM for seed in 0 3 7 11 13 15 18 20 32 42
for %%s in (7 11 18 20 25 36 42 50 60 72 80 85) do (
    for %%t in (EffectOrientedMCTSSpray) do (
        (
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 1
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 2
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 3
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 4
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 5
            python main.py --config %config% --seed %%s --strategy_name %%t --team_size 3 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 6         
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --team_size 5 --sche_step 8 --R_change_interval 3 --adaptive_step 8 --sourcenum 3
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