import sys
from streamlit.web import cli as stcli


if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "streamlit_display.py", "Usr", "--server.port", "8888" ]
    sys.exit(stcli.main())