
import sys
import gui_app
from PySide6.QtWidgets import QApplication

def verify():
    print("Verifying gui_app...")
    try:
        app = QApplication(sys.argv)
        print("QApplication created.")
        window = gui_app.MainWindow()
        print("MainWindow instantiated.")
        if window:
            print("Verification Successful: Window object created.")
        else:
            print("Verification Failed: Window object is None.")
    except Exception as e:
        print(f"Verification Failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify()
