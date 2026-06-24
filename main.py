import sys
import time
import json
import os
import analyze

# Configure matplotlib to work with PyQt6
import matplotlib
matplotlib.use('QtAgg')  # Use QtAgg which works with PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
    QGraphicsRectItem,
    QGraphicsItem,
    QRadioButton,
    QButtonGroup,
    QLabel,
    QScrollArea,
    QWidget,
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QLineF, QThread, pyqtSignal, QSize
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
    QTransform,
    QIcon,
    QTabletEvent,
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

RADIO_STYLE = """
    QRadioButton {
        color: white;
        font-size: 14px;
        padding: 4px;
    }
    QRadioButton::indicator {
        width: 14px;
        height: 14px;
    }
"""

SCROLLBAR_STYLE = """
    QScrollBar:vertical {
        border: none;
        background: #f0f0f0;
        width: 14px;
        margin: 0px;
        border-radius: 0px;
    }
    QScrollBar::handle:vertical {
        background: #c0c0c0;
        min-height: 30px;
        border-radius: 7px;
        margin: 2px;
    }
    QScrollBar::handle:vertical:hover {
        background: #a0a0a0;
    }
    QScrollBar::handle:vertical:pressed {
        background: #808080;
    }
    QScrollBar::sub-line:vertical {
        height: 0px;
    }
    QScrollBar::add-line:vertical {
        height: 0px;
    }
    QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
        background: none;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: none;
    }

    QScrollBar:horizontal {
        border: none;
        background: #f0f0f0;
        height: 14px;
        margin: 0px;
        border-radius: 0px;
    }
    QScrollBar::handle:horizontal {
        background: #c0c0c0;
        min-width: 30px;
        border-radius: 7px;
        margin: 2px;
    }
    QScrollBar::handle:horizontal:hover {
        background: #a0a0a0;
    }
    QScrollBar::handle:horizontal:pressed {
        background: #808080;
    }
    QScrollBar::sub-line:horizontal {
        width: 0px;
    }
    QScrollBar::add-line:horizontal {
        width: 0px;
    }
    QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {
        background: none;
    }
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
    }
"""


# ==========================================
# Background Analysis Worker Thread
# ==========================================
class AnalysisWorker(QThread):
    """Worker thread to run analysis in background without freezing UI"""
    # Signals to communicate with main thread
    finished = pyqtSignal(str)  # Emits the LaTeX result
    error = pyqtSignal(str)     # Emits error message
    progress = pyqtSignal(str)  # Emits progress updates

    def __init__(self, vectors):
        super().__init__()
        self.vectors = vectors

    def run(self):
        """This runs in a background thread"""
        try:
            self.progress.emit("Analyzing strokes...")
            latex_result = analyze.analyze_vectors(self.vectors)

            if latex_result and latex_result.strip():
                self.finished.emit(latex_result)
            else:
                self.finished.emit("Analysis produced no output")

        except Exception as e:
            import traceback
            error_msg = f"Analysis error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class LatexDisplayWidget(QWidget):
    """Widget to display rendered LaTeX using matplotlib"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 10), facecolor='white')
        self.canvas = FigureCanvas(self.figure)

        # Create scroll area for the canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.canvas)
        self.scroll_area.setStyleSheet("background-color: white; border: none;")

        self.layout.addWidget(self.scroll_area)

        # Initial empty display
        self.display_latex("")

    def display_latex(self, latex_string):
        """Render LaTeX string to the display"""
        self.figure.clear()

        if not latex_string or latex_string.strip() == "":
            # Display placeholder
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Draw and click Analyze to begin",
                   ha='center', va='center', fontsize=14, color='gray')
            ax.axis('off')
        else:
            ax = self.figure.add_subplot(111)
            ax.axis('off')

            # Parse and render the content line by line
            try:
                import re

                y_position = 0.95

                # Split content by lines first
                lines = latex_string.split('\n')

                i = 0
                while i < len(lines):
                    line = lines[i].strip()

                    if not line:
                        i += 1
                        continue

                    # Check if this is a display math block
                    if line.startswith('$$') and line.endswith('$$'):
                        # Extract math content (remove $$ delimiters)
                        math_content = line[2:-2].strip()

                        # Test if matplotlib can render this math expression
                        can_render = False
                        try:
                            # Try to parse the math expression first
                            from matplotlib import mathtext
                            parser = mathtext.MathTextParser('path')
                            parser.parse(f'${math_content}$')
                            can_render = True
                        except Exception as test_err:
                            # Expression uses unsupported LaTeX commands
                            print(f"Math expression not supported by matplotlib: {test_err}")
                            can_render = False

                        if can_render:
                            # Render as proper math
                            ax.text(0.5, y_position, f'${math_content}$',
                                   transform=ax.transAxes,
                                   fontsize=14,
                                   verticalalignment='top',
                                   horizontalalignment='center',
                                   usetex=False)
                        else:
                            # Fallback: show LaTeX code as plain text (no math rendering)
                            # Truncate if too long
                            display_content = math_content
                            if len(display_content) > 80:
                                display_content = display_content[:77] + '...'

                            # Use plain text, NOT math mode (no $ delimiters)
                            ax.text(0.5, y_position, f'[Math] {display_content}',
                                   transform=ax.transAxes,
                                   fontsize=9,
                                   verticalalignment='top',
                                   horizontalalignment='center',
                                   family='monospace',
                                   color='darkblue',
                                   bbox=dict(boxstyle='round,pad=0.5',
                                            facecolor='lightblue',
                                            alpha=0.3))

                        y_position -= 0.15

                    elif '[DIAGRAM PLACEHOLDER]' in line:
                        # Diagram placeholder
                        ax.text(0.05, y_position, line,
                               transform=ax.transAxes,
                               fontsize=10,
                               verticalalignment='top',
                               horizontalalignment='left',
                               style='italic',
                               color='blue')
                        y_position -= 0.08

                    elif line:
                        # Regular text
                        ax.text(0.05, y_position, line,
                               transform=ax.transAxes,
                               fontsize=12,
                               verticalalignment='top',
                               horizontalalignment='left')
                        y_position -= 0.06

                    i += 1

            except Exception as e:
                # Fallback: display as plain text if everything fails
                ax.clear()
                ax.axis('off')
                # Remove any $ symbols that could cause math parsing
                safe_string = latex_string.replace('$', '').replace('\\', '\\\\')
                ax.text(0.05, 0.95, f"Output:\n\n{safe_string}",
                       transform=ax.transAxes,
                       fontsize=10,
                       verticalalignment='top',
                       horizontalalignment='left',
                       family='monospace')
                print(f"LaTeX rendering error: {e}")

        # Wrap canvas.draw in try/catch to prevent crashes during rendering
        try:
            self.canvas.draw()
        except Exception as draw_error:
            # Ultimate fallback: show error in plain text with no special chars
            print(f"Canvas draw error: {draw_error}")
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.axis('off')
            ax.text(0.5, 0.5, "Rendering Error\nSee console for details",
                   transform=ax.transAxes,
                   ha='center', va='center',
                   fontsize=12, color='red',
                   family='sans-serif')
            try:
                self.canvas.draw()
            except:
                pass  # Give up if even this fails


class StrokeItem(QGraphicsPathItem):
    def __init__(self, label="text", parent=None):
        super().__init__(parent)
        self.stroke_data = []
        self.label = label


class DrawingScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = None
        self.current_label = "text"
        self.drawing = False
        self.erasing = False
        self.selecting = False
        self.current_path_item = None
        self.selection_rect_item = None
        self.selection_start_pos = None
        self.last_point = None

    def mousePressEvent(self, event):
        if self.current_tool == "pen" and event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.current_path_item = StrokeItem(label=self.current_label)
            self.current_path_item.setPos(event.scenePos())
            path = QPainterPath()
            path.moveTo(0, 0)
            self.current_path_item.setPath(path)
            self.last_point = QPointF(0, 0)
            # Record start point and time
            self.current_path_item.stroke_data.append((0.0, 0.0, time.time()))

            pen = QPen(QColor(0, 0, 0), 3)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            self.current_path_item.setPen(pen)
            self.current_path_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True
            )

            self.addItem(self.current_path_item)
            event.accept()
        elif (
            self.current_tool == "eraser"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.erasing = True
            self.erase_at(event.scenePos())
            event.accept()
        elif (
            self.current_tool == "rect" and event.button() == Qt.MouseButton.LeftButton
        ):
            item = self.itemAt(event.scenePos(), QTransform())
            if item and item.isSelected():
                super().mousePressEvent(event)
            else:
                self.clearSelection()
                self.selecting = True
                self.selection_start_pos = event.scenePos()
                self.selection_rect_item = QGraphicsRectItem()
                self.selection_rect_item.setPen(
                    QPen(Qt.GlobalColor.black, 1, Qt.PenStyle.DashLine)
                )
                self.selection_rect_item.setBrush(QBrush(QColor(0, 0, 255, 50)))
                self.selection_rect_item.setRect(
                    QRectF(self.selection_start_pos, self.selection_start_pos)
                )
                self.addItem(self.selection_rect_item)
                event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing and self.current_path_item:
            pos = self.current_path_item.mapFromScene(event.scenePos())

            if (pos - self.last_point).manhattanLength() < 2.0:
                return

            path = self.current_path_item.path()
            mid_point = (self.last_point + pos) / 2
            path.quadTo(self.last_point, mid_point)

            self.last_point = pos
            self.current_path_item.stroke_data.append((pos.x(), pos.y(), time.time()))
            self.current_path_item.setPath(path)
            event.accept()
        elif self.erasing:
            self.erase_at(event.scenePos())
            event.accept()
        elif self.selecting and self.selection_rect_item:
            rect = QRectF(self.selection_start_pos, event.scenePos()).normalized()
            self.selection_rect_item.setRect(rect)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.MouseButton.LeftButton:
            if self.current_path_item:
                path = self.current_path_item.path()
                # Record end point and time
                pos = self.current_path_item.mapFromScene(event.scenePos())
                path.lineTo(pos)
                self.current_path_item.stroke_data.append(
                    (pos.x(), pos.y(), time.time())
                )
                self.current_path_item.setPath(path)
                self.smooth_stroke(self.current_path_item)
            self.drawing = False
            self.current_path_item = None
            event.accept()
        elif self.erasing and event.button() == Qt.MouseButton.LeftButton:
            self.erasing = False
            event.accept()
        elif self.selecting and event.button() == Qt.MouseButton.LeftButton:
            self.selecting = False
            if self.selection_rect_item:
                rect = self.selection_rect_item.rect()
                items = self.items(rect, Qt.ItemSelectionMode.ContainsItemShape)
                for item in items:
                    if item != self.selection_rect_item:
                        item.setSelected(True)
                        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
                self.removeItem(self.selection_rect_item)
                self.selection_rect_item = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for item in self.selectedItems():
                self.removeItem(item)
        else:
            super().keyPressEvent(event)

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

    def smooth_stroke(self, stroke_item):
        data = stroke_item.stroke_data
        if len(data) < 2:
            return

        path = QPainterPath()
        path.moveTo(data[0][0], data[0][1])

        for i in range(1, len(data)):
            p_last = QPointF(data[i - 1][0], data[i - 1][1])
            p_curr = QPointF(data[i][0], data[i][1])
            mid = (p_last + p_curr) / 2.0
            path.quadTo(p_last, mid)

        path.lineTo(data[-1][0], data[-1][1])
        stroke_item.setPath(path)

    def get_absolute_vectors(self):
        vectors = []
        for item in self.items():
            if isinstance(item, StrokeItem):
                item_x = item.pos().x()
                item_y = item.pos().y()

                stroke_points = []
                for x, y, t in item.stroke_data:
                    stroke_points.append((x + item_x, y + item_y, t))
                if stroke_points:
                    vectors.append(stroke_points)

        return vectors[::-1]


class DrawingView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = DrawingScene(self)
        self.setScene(self.scene)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._tool_cursor = Qt.CursorShape.ArrowCursor
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))

        self.scene.changed.connect(self.update_scene_rect)

    def set_tool(self, tool_id, cursor=None):
        self.scene.current_tool = tool_id
        if cursor:
            self._tool_cursor = cursor
            self.setCursor(cursor)

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

    def mousePressEvent(self, event):
        if (
            self.scene.current_tool == "rect"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            item = self.scene.itemAt(self.mapToScene(event.pos()), QTransform())
            if item and item.isSelected():
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.scene.current_tool == "rect":
            if not event.buttons() & Qt.MouseButton.LeftButton:
                item = self.scene.itemAt(self.mapToScene(event.pos()), QTransform())
                if item and item.isSelected():
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(self._tool_cursor)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if self.scene.current_tool == "rect":
            item = self.scene.itemAt(self.mapToScene(event.pos()), QTransform())
            if item and item.isSelected():
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(self._tool_cursor)

    def tabletEvent(self, event: QTabletEvent):
        """Convert tablet (stylus) events to mouse events for reliable pen input."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QMouseEvent

        etype = event.type()
        pos = event.position()
        global_pos = event.globalPosition()

        if etype == QEvent.Type.TabletPress:
            synth = QMouseEvent(
                QEvent.Type.MouseButtonPress,
                pos, global_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            self.mousePressEvent(synth)
            event.accept()
        elif etype == QEvent.Type.TabletMove:
            synth = QMouseEvent(
                QEvent.Type.MouseMove,
                pos, global_pos,
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            self.mouseMoveEvent(synth)
            event.accept()
        elif etype == QEvent.Type.TabletRelease:
            synth = QMouseEvent(
                QEvent.Type.MouseButtonRelease,
                pos, global_pos,
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
                event.modifiers(),
            )
            self.mouseReleaseEvent(synth)
            event.accept()
        else:
            super().tabletEvent(event)


def _create_tool_icon(name, size=32):
    """Create a modern outline-style icon for cross-platform toolbar buttons."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    fg = QColor(210, 215, 225)  # light foreground
    fg_dim = QColor(160, 165, 175)  # dimmer variant
    stroke_pen = QPen(fg, 1.6, cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin)
    s = size  # alias

    if name == "pen":
        # Minimal pen/pencil — diagonal line with tip
        p.setPen(QPen(fg, 1.8, cap=Qt.PenCapStyle.RoundCap))
        # Pencil body (diagonal)
        p.drawLine(int(s*0.68), int(s*0.12), int(s*0.22), int(s*0.58))
        # Pencil tip triangle
        tip = QPainterPath()
        tip.moveTo(s*0.22, s*0.58)
        tip.lineTo(s*0.14, s*0.78)
        tip.lineTo(s*0.34, s*0.70)
        tip.closeSubpath()
        p.setPen(QPen(fg, 1.2, cap=Qt.PenCapStyle.RoundCap))
        p.setBrush(fg_dim)
        p.drawPath(tip)
        # Small mark from tip
        p.setPen(QPen(fg, 1.4, cap=Qt.PenCapStyle.RoundCap))
        p.drawLine(int(s*0.14), int(s*0.78), int(s*0.12), int(s*0.86))

    elif name == "eraser":
        # Rounded eraser shape at an angle
        p.save()
        p.translate(s*0.5, s*0.5)
        p.rotate(-35)
        p.translate(-s*0.5, -s*0.5)
        # Body
        p.setPen(QPen(fg, 1.5, cap=Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(int(s*0.18), int(s*0.3), int(s*0.64), int(s*0.4), 4, 4)
        # Divider line
        p.setPen(QPen(fg_dim, 1.0))
        p.drawLine(int(s*0.42), int(s*0.3), int(s*0.42), int(s*0.7))
        p.restore()

    elif name == "rect":
        # Selection rectangle with corner handles
        p.setPen(QPen(fg_dim, 1.2, Qt.PenStyle.DashLine))
        r = 4
        bx, by, bw, bh = int(s*0.2), int(s*0.2), int(s*0.6), int(s*0.6)
        p.drawRect(bx, by, bw, bh)
        # Corner dots
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(fg)
        for cx, cy in [(bx, by), (bx+bw, by), (bx, by+bh), (bx+bw, by+bh)]:
            p.drawEllipse(cx - r//2, cy - r//2, r, r)

    elif name == "save":
        # Minimal floppy/download icon — arrow into tray
        p.setPen(stroke_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Down arrow
        mid = s // 2
        p.drawLine(mid, int(s*0.15), mid, int(s*0.58))
        # Arrow head
        p.drawLine(mid, int(s*0.58), int(s*0.35), int(s*0.43))
        p.drawLine(mid, int(s*0.58), int(s*0.65), int(s*0.43))
        # Tray
        path = QPainterPath()
        path.moveTo(s*0.2, s*0.55)
        path.lineTo(s*0.2, s*0.8)
        path.lineTo(s*0.8, s*0.8)
        path.lineTo(s*0.8, s*0.55)
        p.drawPath(path)

    elif name == "open":
        # Folder outline
        p.setPen(stroke_pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(s*0.15, s*0.75)
        path.lineTo(s*0.15, s*0.28)
        path.lineTo(s*0.38, s*0.28)
        path.lineTo(s*0.46, s*0.38)
        path.lineTo(s*0.85, s*0.38)
        path.lineTo(s*0.85, s*0.75)
        path.closeSubpath()
        p.drawPath(path)

    elif name == "analyze":
        # Sparkle/wand — three-pronged star
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 210, 70))
        cx, cy = s*0.45, s*0.4
        # Main 4-point star
        star = QPainterPath()
        star.moveTo(cx, cy - s*0.28)
        star.quadTo(cx + s*0.06, cy - s*0.06, cx + s*0.28, cy)
        star.quadTo(cx + s*0.06, cy + s*0.06, cx, cy + s*0.28)
        star.quadTo(cx - s*0.06, cy + s*0.06, cx - s*0.28, cy)
        star.quadTo(cx - s*0.06, cy - s*0.06, cx, cy - s*0.28)
        p.drawPath(star)
        # Small accent star
        sx, sy = s*0.75, s*0.72
        r2 = s*0.1
        star2 = QPainterPath()
        star2.moveTo(sx, sy - r2)
        star2.quadTo(sx + s*0.025, sy - s*0.025, sx + r2, sy)
        star2.quadTo(sx + s*0.025, sy + s*0.025, sx, sy + r2)
        star2.quadTo(sx - s*0.025, sy + s*0.025, sx - r2, sy)
        star2.quadTo(sx - s*0.025, sy - s*0.025, sx, sy - r2)
        p.drawPath(star2)

    p.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Drawtex")
        self.resize(1200, 800)

        # FLAG: Set mode and label here!
        self.data_collection_mode = False

        # Background worker for analysis
        self.analysis_worker = None

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
        self.left_pane.setStyleSheet(SCROLLBAR_STYLE)

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

        if self.data_collection_mode:
            self.label_toolbar = self.create_label_toolbar()
            self.left_layout.addWidget(self.label_toolbar)

        self.left_layout.addStretch()

        # Create LaTeX display widget for right pane
        self.latex_display = LatexDisplayWidget()
        self.right_pane = self.latex_display

        splitter.addWidget(self.left_pane)
        splitter.addWidget(self.right_pane)

        splitter.setSizes([840, 360])
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)
        self.show()

    def create_floating_toolbar(self):
        container = QFrame()
        container.setStyleSheet(TOOLBAR_CONTAINER_STYLE)
        container.setFixedWidth(60)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 10, 5, 10)
        layout.setSpacing(8)

        icon_size = QSize(24, 24)

        drawing_tools = [
            ("pen", "Pen"),
            ("eraser", "Eraser"),
            ("rect", "Rect Select"),
        ]

        self.tool_buttons = []

        for tool_id, tooltip in drawing_tools:
            btn = QToolButton()
            btn.setIcon(_create_tool_icon(tool_id))
            btn.setIconSize(icon_size)
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
            ("save", "Save"),
            ("open", "Open Save"),
        ]

        for icon_name, tooltip in file_tools:
            btn = QToolButton()
            btn.setIcon(_create_tool_icon(icon_name))
            btn.setIconSize(icon_size)
            btn.setToolTip(tooltip)
            btn.setStyleSheet(BUTTON_STYLE)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(btn)

        analyze_btn = QToolButton()
        analyze_btn.setIcon(_create_tool_icon("analyze"))
        analyze_btn.setIconSize(icon_size)
        analyze_btn.setToolTip("Analyze")
        analyze_btn.setStyleSheet(BUTTON_STYLE)
        analyze_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        analyze_btn.clicked.connect(self.run_analysis)
        layout.addWidget(analyze_btn)

        return container

    def create_label_toolbar(self):
        container = QFrame()
        container.setStyleSheet(TOOLBAR_CONTAINER_STYLE)
        container.setFixedWidth(100)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        lbl = QLabel("Label:")
        lbl.setStyleSheet("color: #aaaaaa; font-weight: bold; font-size: 12px;")
        layout.addWidget(lbl)

        self.label_group = QButtonGroup(self)
        modes = ["text", "math", "diagram"]

        for mode in modes:
            rb = QRadioButton(mode.capitalize())
            rb.setStyleSheet(RADIO_STYLE)
            layout.addWidget(rb)
            self.label_group.addButton(rb)
            if mode == "text":
                rb.setChecked(True)

        self.label_group.buttonToggled.connect(self.update_label_mode)
        return container

    def change_tool(self, tool_id, clicked_btn):
        """Logic to switch cursors and highlight the active button"""

        for btn in self.tool_buttons:
            if btn != clicked_btn:
                btn.setChecked(False)

        if hasattr(self.left_pane, "scene"):
            for item in self.left_pane.scene.selectedItems():
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
            self.left_pane.scene.clearSelection()

        if not clicked_btn.isChecked():
            if hasattr(self.left_pane, "set_tool"):
                self.left_pane.set_tool("default", self.cursors["default"])
            return

        if tool_id in self.cursors:
            if hasattr(self.left_pane, "set_tool"):
                self.left_pane.set_tool(tool_id, self.cursors[tool_id])

    def update_label_mode(self, btn, checked):
        if checked:
            self.left_pane.scene.current_label = btn.text().lower()

    def run_analysis(self):
        if self.data_collection_mode:
            collected_strokes = []
            # items() returns items in descending stacking order (top first).
            # We reverse it to get chronological order (bottom first).
            items = list(self.left_pane.scene.items())
            items.reverse()

            for item in items:
                if isinstance(item, StrokeItem):
                    # Convert local coordinates to absolute scene coordinates
                    scene_pos = item.pos()
                    abs_stroke = [
                        (x + scene_pos.x(), y + scene_pos.y(), t)
                        for x, y, t in item.stroke_data
                    ]
                    collected_strokes.append(
                        {"label": item.label, "points": abs_stroke}
                    )

            data_entry = {
                "session_timestamp": time.time(),
                "strokes": collected_strokes,
            }
            self._save_to_json(data_entry)
            self.left_pane.scene.clear()
            return

        # Check if analysis is already running
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            print("Analysis already in progress, please wait...")
            return

        # Get vectors to analyze
        try:
            vectors = self.left_pane.scene.get_absolute_vectors()
            if not vectors or len(vectors) == 0:
                self.latex_display.display_latex("No strokes to analyze")
                return
        except Exception as e:
            print(f"Error getting vectors: {e}")
            self.latex_display.display_latex(f"Error: {str(e)}")
            return

        # Show "Analyzing..." message
        self.latex_display.display_latex("Analyzing strokes...\n\nPlease wait, you can continue drawing.")

        # Create and start background worker
        self.analysis_worker = AnalysisWorker(vectors)
        self.analysis_worker.finished.connect(self._on_analysis_finished)
        self.analysis_worker.error.connect(self._on_analysis_error)
        self.analysis_worker.progress.connect(self._on_analysis_progress)
        self.analysis_worker.start()

        print("Analysis started in background. You can continue drawing...")

    def _on_analysis_finished(self, latex_result):
        """Called when background analysis completes successfully"""
        self.latex_display.display_latex(latex_result)
        print("Analysis complete!")

    def _on_analysis_error(self, error_msg):
        """Called when background analysis encounters an error"""
        self.latex_display.display_latex(f"Analysis error:\n\n{error_msg}")
        print(f"Analysis error: {error_msg}")

    def _on_analysis_progress(self, message):
        """Called when background analysis reports progress"""
        print(f"Progress: {message}")

    def _save_to_json(self, entry):
        filename = "collected_strokes.json"
        data = []

        if os.path.exists(filename):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError, PermissionError) as e:
                print(f"Warning: Could not read {filename}: {e}")
                data = []

        data.append(entry)

        try:
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved data to {filename}")
        except (IOError, PermissionError, OSError) as e:
            print(f"Error: Could not save to {filename}: {e}")

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
