REM pip needs to be installed
where /q pip

IF NOT ERRORLEVEL 1 (

    REM Check if Anaconda is installed
    where /q conda
    IF NOT ERRORLEVEL 1 (
        conda remove scipy
	conda remove numpy
      ) ELSE (
        ECHO.Anaconda is not installed.
      )

    REM Install compiled wheels that do not pip install
    pip install --upgrade --no-cache-dir numexpr-2.6.2-cp27-cp27m-win_amd64.whl scikit_image-0.12.3-cp27-cp27m-win_amd64.whl tables-3.3.0-cp27-cp27m-win_amd64.whl Bottleneck-1.2.0-cp27-cp27m-win_amd64.whl GDAL-2.1.3-cp27-cp27m-win_amd64.whl statsmodels-0.8.0rc1-cp27-cp27m-win_amd64.whl scikit_learn-0.18.1-cp27-cp27m-win_amd64.whl opencv_python-3.1.0-cp27-cp27m-win_amd64.whl Cython-0.25.2-cp27-cp27m-win_amd64.whl pandas-0.19.2-cp27-cp27m-win_amd64.whl

    pip install --upgrade --no-cache-dir beautifulsoup4 retrying six xmltodict colorama joblib matplotlib psutil PySAL PyYAML

    pip install --no-cache-dir SpFeas-0.1.1.tar.gz
    pip install --no-cache-dir MpGlue-0.1.0.tar.gz

    REM Check if SpFeas installed correctly
    where /q spfeas
    IF NOT ERRORLEVEL 1 (
        ECHO.The installation has finished!
      ) ELSE (
        ECHO.SpFeas failed to install.
      )

    pip install --upgrade --no-cache-dir numpy-1.12.1+mkl-cp27-cp27m-win_amd64.whl scipy-0.19.0-cp27-cp27m-win_amd64.whl

  ) ELSE (
    ECHO.Download get-pip.py from https://pip.pypa.io/en/latest/installing/
)