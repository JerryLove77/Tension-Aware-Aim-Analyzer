# filepath: main.py

import sys

from PySide6.QtWidgets import QApplication

from app.controller import MainController
from app.views.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    # Wire up the controller — keeps UI and business logic decoupled
    controller = MainController(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
