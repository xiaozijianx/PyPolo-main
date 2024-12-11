REM @echo off
chcp 65001
setlocal enabledelayedexpansion
set "pids="
set config=./pypolo2/configs/CONF.yaml

REM for seed in 0 3 7 11 13 15 18 20 32 42
@REM for %%s in (7 11 18 20 25 36 42 50 60 72 80 85) do (
for %%s in (50 100 150 200 300 400 500 700 1000 1200) do (
    for %%t in (EffectOrientedMCTSNonMyopicSpray) do (
        (
            python .\main.py --config %config% --seed 0 --strategy_name %%t --sche_step 18 --team_size 2 --adaptive_step 18 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --sche_step 18 --team_size 3 --adaptive_step 18 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --sche_step 18 --team_size 4 --adaptive_step 18 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --sche_step 18 --team_size 5 --adaptive_step 18 --bound1 %%s
            python .\main.py --config %config% --seed 0 --strategy_name %%t --sche_step 18 --team_size 6 --adaptive_step 18 --bound1 %%s
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 2
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 3
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 4
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 5
            @REM python main.py --config %config% --seed %%s --strategy_name %%t --sche_step 8 --adaptive_step 8 --sourcenum 6         
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