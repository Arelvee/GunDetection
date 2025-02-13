Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c venv\Scripts\activate && python main.py", 0