REM pip needs to be installed
where /q pip

IF NOT ERRORLEVEL 1 (

    REM Check if Anaconda is installed
    where /q conda
    IF NOT ERRORLEVEL 1 (
        conda remove scipy
	conda remove numpy
      ) ELSE (
        ECHO.Anaconda is not installed, so proceeding with the installation.
      )

    pip uninstall spfeas
    pip uninstall mpglue

    pip install --no-cache-dir SpFeas-0.1.3-cp27-cp27m-win_amd64.whl
    pip install --no-cache-dir MpGlue-0.1.3-cp27-cp27m-win_amd64.whl

    pip install --upgrade --no-cache-dir numpy-1.11.3+mkl-cp27-cp27m-win_amd64.whl scipy-0.19.0-cp27-cp27m-win_amd64.whl

    REM Check if SpFeas installed correctly
    where /q spfeas
    IF NOT ERRORLEVEL 1 (
        ECHO.The installation has finished!
      ) ELSE (
        ECHO.SpFeas failed to install.
      )

  ) ELSE (
    ECHO.Download get-pip.py from https://pip.pypa.io/en/latest/installing/
)