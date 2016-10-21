REM pip needs to be installed
where /q pip

IF NOT ERRORLEVEL 1 (

    pip install beautifulsoup4 retrying six tables xmltodict

    pip install numpy-1.11.2+mkl-cp27-cp27m-win32.whl colorama-0.3.7-py2.py3-none-any.whl Cython-0.24.1-cp27-cp27m-win32.whl joblib-0.10.2-py2.py3-none-any.whl matplotlib-2.0.0b4-cp27-cp27m-win32.whl numexpr-2.6.1-cp27-cp27m-win32.whl opencv_python-3.1.0-cp27-cp27m-win32.whl pandas-0.19.0-cp27-cp27m-win32.whl PySAL-1.12.0-py2-none-any.whl scikit_image-0.12.3-cp27-cp27m-win32.whl scikit_learn-0.18-cp27-cp27m-win32.whl scipy-0.18.1-cp27-cp27m-win32.whl 
   
    REM Uninstall MpGlue if it already exists 
    where /q classify
    IF NOT ERRORLEVEL 1 (
        pip uninstall mpglue
      )

    pip install git+https://github.com/jgrss/mpglue.git

    REM Uninstall SpFeas if it already exists
    where /q spfeas
    IF NOT ERRORLEVEL 1 (
        pip uninstall spfeas
      ) 

    pip install git+https://github.com/jgrss/spfeas.git     

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
