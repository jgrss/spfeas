IF where pip > NUL (

    pip install beautifulsoup4 Bottleneck colorama cython joblib matplotlib numexpr numpy opencv-python pandas PySAL retrying scikit-image scikit-learn scipy six tables xmltodict 
   
    IF where classify > NUL (
        pip uninstall mpglue
      )

    pip install git+https://github.com/jgrss/mpglue.git

    IF where spfeas > NUL (
        pip uninstall spfeas
      ) 

    pip install git+https://github.com/jgrss/spfeas.git     

    IF where spfeas > NUL (
        ECHO.The installation has finished!
      ) ELSE (
        ECHO.SpFeas failed to install.
      )

  ) ELSE (
    ECHO.Download get-pip.py from https://pip.pypa.io/en/latest/installing/
  )
