import sys
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QSplitter,
    QFrame,
    QVBoxLayout,
    QToolButton,
    QGraphicsDropShadowEffect,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPathItem,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import (
    QPixmap,
    QPainter,
    QPen,
    QColor,
    QCursor,
    QBrush,
    QLinearGradient,
    QPainterPath,
    QRadialGradient,
)

# --- STYLING ---
TOOLBAR_CONTAINER_STYLE = """
    QFrame {
        background-color: #333333;
        border-radius: 15px;
        border: 1px solid #444444;
    }
"""

BUTTON_STYLE = """
    QToolButton {
        background-color: transparent;
        color: white;
        border-radius: 10px;
        padding: 8px;
        font-size: 20px;
    }
    QToolButton:hover {
        background-color: #555555;
    }
    QToolButton:checked {
        background-color: #7289da; /* Blue highlight when selected */
        color: white;
    }
"""


class DrawingScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = None
        self.drawing = False
        self.erasing = False
        self.current_path_item = None

    def mousePressEvent(self, event):
        if self.current_tool == "pen" and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.current_path_item = QGraphicsPathItem()
            self.current_path_item.setPos(event.scenePos())
            path = QPainterPath()
            path.moveTo(0, 0)
            self.current_path_item.setPath(path)

            pen = QPen(Qt.GlobalColor.black, 2)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            self.current_path_item.setPen(pen)

            self.addItem(self.current_path_item)
            event.accept()
        elif (
            self.current_tool == "eraser"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.erasing = True
            self.erase_at(event.scenePos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_path_item:
            path = self.current_path_item.path()
            pos = self.current_path_item.mapFromScene(event.scenePos())
            path.lineTo(pos)
            self.current_path_item.setPath(path)
            event.accept()
        elif self.erasing:
            self.erase_at(event.scenePos())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False
            self.current_path_item = None
            event.accept()
        elif self.erasing and event.button() == Qt.MouseButton.LeftButton:
            self.erasing = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def erase_at(self, position):
        eraser_size = 20
        eraser_area = QRectF(
            position.x() - eraser_size / 2,
            position.y() - eraser_size / 2,
            eraser_size,
            eraser_size,
        )
        items_to_remove = self.items(eraser_area)
        for item in items_to_remove:
            if isinstance(item, QGraphicsPathItem):
                self.removeItem(item)


class DrawingView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = DrawingScene(self)
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.scene.changed.connect(self.update_scene_rect)

    def set_tool(self, tool_id):
        self.scene.current_tool = tool_id

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_scene_rect()

    def update_scene_rect(self, rects=None):
        items_rect = self.scene.itemsBoundingRect()
        view_rect = self.viewport().rect()

        width = max(items_rect.right(), view_rect.width()) + 200
        height = max(items_rect.bottom(), view_rect.height()) + 200

        self.scene.setSceneRect(0, 0, width, height)

    def setCursor(self, cursor):
        super().setCursor(cursor)
        self.viewport().setCursor(cursor)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drawtex")
        self.resize(1200, 800)

        # Custom cursors
        self.cursors = {
            "default": Qt.CursorShape.ArrowCursor,
            "pen": self._create_pen_cursor(),
            "eraser": self._create_eraser_cursor(),
            "rect": self._create_rect_cursor(),
        }

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(3)

        self.left_pane = DrawingView()
        self.left_pane.setStyleSheet("background-color: #ffffff;")

        self.left_pane.setCursor(self.cursors["default"])

        # Position the toolbar.
        self.left_layout = QVBoxLayout(self.left_pane)
        self.left_layout.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.left_layout.setContentsMargins(20, 20, 0, 0)

        self.toolbar_widget = self.create_floating_toolbar()

        self.toolbar_widget.setCursor(Qt.CursorShape.ArrowCursor)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.toolbar_widget.setGraphicsEffect(shadow)

        self.left_layout.addWidget(self.toolbar_widget)

        self.left_layout.addStretch()

        self.right_pane = QFrame()
        self.right_pane.setStyleSheet("background-color: #f4f4f4;")
        self.right_pane.setFrameShape(QFrame.Shape.StyledPanel)

        splitter.addWidget(self.left_pane)
        splitter.addWidget(self.right_pane)

        splitter.setStretchFactor(0, 9)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self.show()

    def create_floating_toolbar(self):
        container = QFrame()
        container.setStyleSheet(TOOLBAR_CONTAINER_STYLE)
        container.setFixedWidth(60)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(8)

        drawing_tools = [
            ("✏️", "Pen", "pen"),
            ("🧼", "Eraser", "eraser"),
            ("⛶", "Rect Select", "rect"),
        ]

        self.tool_buttons = []

        for icon, tooltip, tool_id in drawing_tools:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setCheckable(True)
            btn.setStyleSheet(BUTTON_STYLE)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

            btn.clicked.connect(
                lambda checked, t=tool_id, b=btn: self.change_tool(t, b)
            )

            layout.addWidget(btn)
            self.tool_buttons.append(btn)

        file_tools = [
            ("💾", "Save"),
            ("📁", "Open Save"),
        ]

        for icon, tooltip in file_tools:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(BUTTON_STYLE)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(btn)

        return container

    def change_tool(self, tool_id, clicked_btn):
        """Logic to switch cursors and highlight the active button"""

        for btn in self.tool_buttons:
            if btn != clicked_btn:
                btn.setChecked(False)

        if not clicked_btn.isChecked():
            self.left_pane.setCursor(self.cursors["default"])
            if hasattr(self.left_pane, "set_tool"):
                self.left_pane.set_tool("default")
            return

        if tool_id in self.cursors:
            self.left_pane.setCursor(self.cursors[tool_id])
            if hasattr(self.left_pane, "set_tool"):
                self.left_pane.set_tool(tool_id)

    def _create_pen_cursor(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(13, 13, 7, 7)

        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(14, 14, 5, 5)

        painter.end()

        return QCursor(pixmap, 16, 16)

    def _create_eraser_cursor(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.translate(16, 16)
        painter.rotate(-45)
        painter.translate(-16, -16)

        metal_grad = QLinearGradient(10, 0, 22, 0)
        metal_grad.setColorAt(0.0, QColor(100, 100, 100))
        metal_grad.setColorAt(0.4, QColor(220, 220, 220))
        metal_grad.setColorAt(1.0, QColor(80, 80, 80))

        painter.setPen(QPen(QColor(50, 50, 50), 0.5))
        painter.setBrush(QBrush(metal_grad))
        painter.drawRect(10, 14, 12, 8)

        painter.setPen(QPen(QColor(0, 0, 0, 50), 1))
        painter.drawLine(10, 16, 22, 16)
        painter.drawLine(10, 19, 22, 19)

        eraser_grad = QLinearGradient(10, 0, 22, 0)
        eraser_grad.setColorAt(0.0, QColor(200, 100, 120))
        eraser_grad.setColorAt(0.5, QColor(255, 180, 200))
        eraser_grad.setColorAt(1.0, QColor(180, 90, 110))

        painter.setPen(QPen(QColor(160, 80, 90), 0.5))
        painter.setBrush(QBrush(eraser_grad))
        painter.drawRoundedRect(10, 4, 12, 11, 3, 3)

        painter.end()

        return QCursor(pixmap, 4, 4)

    def _create_rect_cursor(self):
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 6
        corner_rad = 2
        gap_size = 8

        path = QPainterPath()

        left = margin
        right = 32 - margin
        top = margin
        bottom = 32 - margin
        mid = 16
        gap_half = gap_size / 2

        path.moveTo(left, mid - gap_half)
        path.lineTo(left, top + corner_rad)
        path.quadTo(left, top, left + corner_rad, top)
        path.lineTo(mid - gap_half, top)

        path.moveTo(mid + gap_half, top)
        path.lineTo(right - corner_rad, top)
        path.quadTo(right, top, right, top + corner_rad)
        path.lineTo(right, mid - gap_half)

        path.moveTo(right, mid + gap_half)
        path.lineTo(right, bottom - corner_rad)
        path.quadTo(right, bottom, right - corner_rad, bottom)
        path.lineTo(mid + gap_half, bottom)

        path.moveTo(mid - gap_half, bottom)
        path.lineTo(left + corner_rad, bottom)
        path.quadTo(left, bottom, left, bottom - corner_rad)
        path.lineTo(left, mid + gap_half)

        painter.setPen(QPen(QColor(255, 255, 255), 3))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(QPen(QColor(0, 0, 0), 1.5))
        painter.drawPath(path)

        painter.end()
        return QCursor(pixmap, 16, 16)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec())
