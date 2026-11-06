"""Dark charcoal theme with violet and cyan accents."""

BG = "#0b0b10"
PANEL = "#14141c"
PANEL_2 = "#1c1c27"
BORDER = "#2a2a38"
TEXT = "#ececf3"
MUTED = "#9a9ab0"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"
DANGER = "#f87171"
OK = "#34d399"

QSS = f"""
* {{ font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QWidget#sidebar {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
QPushButton#nav {{ text-align: left; padding: 10px 16px; border: none; border-radius: 8px; color: {MUTED};
                   background: transparent; font-size: 14px; }}
QPushButton#nav:hover {{ background: {PANEL_2}; color: {TEXT}; }}
QPushButton#nav:checked {{ background: {PANEL_2}; color: {TEXT}; border-left: 3px solid {VIOLET}; }}
QLabel#brand {{ font-size: 18px; font-weight: 700; padding: 18px 16px 8px 16px; }}
QLabel#h1 {{ font-size: 22px; font-weight: 700; }}
QLabel#h2 {{ font-size: 15px; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#badge {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 3px 10px; }}
QLabel#modeLocal {{ background: #0f2f2a; color: {OK}; border-radius: 10px; padding: 3px 10px; font-weight: 600; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 8px 14px; }}
QPushButton:hover {{ border-color: {VIOLET}; }}
QPushButton:disabled {{ color: #55556a; border-color: #222230; }}
QPushButton#primary {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {VIOLET}, stop:1 #6d28d9);
                       border: none; font-weight: 600; }}
QPushButton#primary:disabled {{ background: #2b2540; color: #77708f; }}
QPushButton#danger {{ background: #3a1620; border-color: #5b2230; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {PANEL_2}; border: 1px solid {BORDER};
    border-radius: 6px; padding: 5px 8px; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; selection-background-color: {VIOLET}; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {CYAN}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {VIOLET}; border-radius: 2px; }}
QProgressBar {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; text-align: center; height: 16px; }}
QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {VIOLET}, stop:1 {CYAN}); border-radius: 5px; }}
QTableWidget, QTextEdit, QPlainTextEdit, QListWidget {{ background: {PANEL}; border: 1px solid {BORDER};
    border-radius: 8px; gridline-color: {BORDER}; }}
QHeaderView::section {{ background: {PANEL_2}; border: none; padding: 6px; color: {MUTED}; }}
QScrollArea {{ border: none; background: transparent; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {MUTED}; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; }}
"""
