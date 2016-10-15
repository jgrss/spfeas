IF where pip > NUL (
    pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas PySAL retrying scikit-image scikit-learn scipy six tables xmltodict 
    
    pip install SpFeas-0.0.1.tar.gz

    IF where spfeas > NUL (
        ECHO.The installation has finished!
      ) ELSE (
        ECHO.SpFeas failed to install.
      )

  ) ELSE (
    ECHO.Download get-pip.py from https://pip.pypa.io/en/latest/installing/
  )
