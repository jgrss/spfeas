REM pip needs to be installed
where /q pip

IF NOT ERRORLEVEL 1 (

    REM Install compiled wheels that do not pip install
    pip install --no-cache-dir numpy‑1.12.0rc2+mkl‑cp27‑cp27m‑win32.whl numexpr-2.6.1-cp27-cp27m-win32.whl scikit_image-0.12.3-cp27-cp27m-win32.whl scipy-0.18.1-cp27-cp27m-win32.whl tables-3.3.0-cp27-cp27m-win32.whl Bottleneck-1.2.0-cp27-cp27m-win32.whl GDAL‑2.1.2‑cp27‑cp27m‑win32.whl statsmodels-0.8.0rc1-cp27-cp27m-win32.whl

    pip install --no-cache-dir beautifulsoup4 retrying six xmltodict colorama cython joblib matplotlib opencv-python pandas psutil PySAL PyYAML scikit-learn 

    REM Uninstall MpGlue if it already exists 
    where /q classify
    IF NOT ERRORLEVEL 1 (
        pip uninstall mpglue
      )

    where /q git
    IF NOT ERRORLEVEL 1 (
        pip install git+https://github.com/jgrss/mpglue.git
    ) ELSE (
        pip install MpGlue-0.0.5.tar.gz
    )

    REM Uninstall SpFeas if it already exists
    where /q spfeas
    IF NOT ERRORLEVEL 1 (
        pip uninstall spfeas
      ) 

    where /q git
    IF NOT ERRORLEVEL 1 (
        pip install git+https://github.com/jgrss/spfeas.git     
    ) ELSE (
        pip install SpFeas-0.0.3.tar.gz
    )

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
