# speech_segments_viewer.py
from __future__ import annotations

import sys
from typing import List, Literal, TypedDict

from PyQt6.QtCore import (
    QAbstractTableModel,
    QEvent,
    QMarginsF,
    QModelIndex,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QVariant,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QFont, QIcon, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QMainWindow,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


class SpeechSegment(TypedDict):
    num: int
    start_ms: float
    end_ms: float
    prob: float
    frame_start: int
    frame_end: int
    type: Literal["speech", "non-speech"]


class SpeechSegmentModel(QAbstractTableModel):
    COLUMNS = [
        "Action",
        "Num",
        "Start (s)",
        "End (s)",
        "Duration (s)",
        "Prob",
        "Type",
        "Frame Start",
        "Frame End",
    ]
    ACTION_COL = 0

    def __init__(self, segments: List[SpeechSegment], parent=None):
        super().__init__(parent)
        self.segments = segments

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.segments) if not parent.isValid() else 0

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.COLUMNS) if not parent.isValid() else 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        row = index.row()
        col = index.column()
        seg = self.segments[row]

        if col == self.ACTION_COL:
            if role == Qt.ItemDataRole.DecorationRole:
                return QIcon.fromTheme("media-playback-start")
            if role == Qt.ItemDataRole.ToolTipRole:
                return "Play / Jump to this segment"
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole:
            match col:
                case 1:
                    return str(seg["num"])
                case 2:
                    return f"{seg['start_ms'] / 1000:.3f}"
                case 3:
                    return f"{seg['end_ms'] / 1000:.3f}"
                case 4:
                    return f"{(seg['end_ms'] - seg['start_ms']) / 1000:.3f}"
                case 5:
                    return f"{seg['prob']:.3f}"
                case 6:
                    return seg["type"]
                case 7:
                    return str(seg["frame_start"])
                case 8:
                    return str(seg["frame_end"])

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (2, 3, 4, 5):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return int(Qt.AlignmentFlag.AlignCenter)

        elif role == Qt.ItemDataRole.ForegroundRole and col != self.ACTION_COL:
            if seg["type"] == "speech":
                return QColor("#1e7e34")
            else:
                return QColor("#6c757d")

        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            return self.COLUMNS[section]
        return QVariant()


class PlayButtonDelegate(QStyledItemDelegate):
    """Renders clickable play icon in action column"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.play_icon = QIcon.fromTheme("media-playback-start")

    def paint(
        self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex
    ):
        if not index.isValid():
            return

        # Draw background
        self.initStyleOption(option, index)
        style = QApplication.style()
        style.drawPrimitive(
            QApplication.style().PrimitiveElement.PE_PanelItemViewItem, option, painter
        )

        # Center icon
        rect = option.rect
        icon_size = 20
        x = rect.x() + (rect.width() - icon_size) // 2
        y = rect.y() + (rect.height() - icon_size) // 2

        mode = QIcon.Mode.Normal
        if option.state & QStyle.StateFlag.State_Selected:
            mode = QIcon.Mode.Selected
        if option.state & QStyle.StateFlag.State_MouseOver:
            mode = QIcon.Mode.Active

        self.play_icon.paint(
            painter,
            QRect(x, y, icon_size, icon_size),
            Qt.AlignmentFlag.AlignCenter,
            mode,
        )

    def editorEvent(self, event, model, option, index):
        # Fragile: better to connect via table.clicked(index)
        # For now keep as-is, but consider refactoring later
        if event.type() == QEvent.Type.MouseButtonRelease:
            if option.rect.contains(event.position().toPoint()):
                if hasattr(model.parent(), "segmentActivated"):
                    model.parent().segmentActivated.emit(index.row())
                return True
        return super().editorEvent(event, model, option, index)


class TimelineWidget(QWidget):
    segmentClicked = pyqtSignal(int)  # segment index clicked
    positionChanged = pyqtSignal(float)  # new time ms (if dragging playhead later)

    def __init__(self, segments: List[SpeechSegment], parent=None):
        super().__init__(parent)
        self.segments = segments
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.total_duration_ms = 0.0
        self.current_position_ms = 0.0
        self.update_duration()

        self.hovered_index = -1
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def update_duration(self):
        if not self.segments:
            self.total_duration_ms = 0.0
            return
        self.total_duration_ms = max((s["end_ms"] for s in self.segments), default=0.0)

    def set_current_position(self, ms: float):
        self.current_position_ms = max(0.0, min(ms, self.total_duration_ms))
        self.update()

    def paintEvent(self, event):
        if self.total_duration_ms <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        margin = QMarginsF(16, 16, 16, 24)
        content_rect = QRectF(
            margin.left(),
            margin.top(),
            w - margin.left() - margin.right(),
            h - margin.top() - margin.bottom(),
        )

        scale = content_rect.width() / self.total_duration_ms

        # Background
        painter.setBrush(QColor("#f8f9fa"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(content_rect, 8, 8)

        # Segments
        for i, seg in enumerate(self.segments):
            x1 = content_rect.left() + seg["start_ms"] * scale
            x2 = content_rect.left() + seg["end_ms"] * scale
            width = max(3.0, x2 - x1)

            color = QColor("#28a745") if seg["type"] == "speech" else QColor("#adb5bd")
            if i == self.hovered_index:
                color = color.lighter(125)

            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor("#2c3e50"), 1.0))
            rect = QRectF(
                x1, content_rect.top() + 16, width, content_rect.height() - 32
            )
            painter.drawRoundedRect(rect, 5, 5)

        # Playhead
        if self.current_position_ms > 0:
            px = content_rect.left() + self.current_position_ms * scale
            painter.setPen(QPen(QColor("#dc3545"), 2.5, Qt.PenStyle.SolidLine))
            painter.drawLine(
                int(px),
                int(content_rect.top() + 8),
                int(px),
                int(content_rect.bottom() - 8),
            )

            # Triangle head
            tri = [
                QPointF(px - 8, content_rect.top()),
                QPointF(px + 8, content_rect.top()),
                QPointF(px, content_rect.top() + 16),
            ]
            painter.setBrush(QColor("#dc3545"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(tri)

        # Time labels
        painter.setPen(QColor("#495057"))
        painter.setFont(QFont("Segoe UI", 9))
        step_ms = max(5000, (self.total_duration_ms // 10 // 5000 + 1) * 5000)
        for t in range(0, int(self.total_duration_ms) + 1, int(step_ms)):
            x = content_rect.left() + t * scale
            painter.drawLine(
                int(x),
                int(content_rect.bottom() - 10),
                int(x),
                int(content_rect.bottom()),
            )
            painter.drawText(
                QRectF(x - 40, content_rect.bottom() - 4, 80, 20),
                Qt.AlignmentFlag.AlignCenter,
                f"{t / 1000:.1f}s",
            )

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.total_duration_ms <= 0:
            return
        x = event.position().x()
        margin_left = 16
        content_w = self.width() - 32
        scale = content_w / self.total_duration_ms

        new_hover = -1
        for i, seg in enumerate(self.segments):
            x1 = 16 + seg["start_ms"] * scale
            x2 = 16 + seg["end_ms"] * scale
            if x1 <= x <= x2:
                new_hover = i
                break

        if new_hover != self.hovered_index:
            self.hovered_index = new_hover
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self.hovered_index >= 0:
            self.segmentClicked.emit(self.hovered_index)


class SpeechSegmentsViewer(QMainWindow):
    segmentActivated = pyqtSignal(int)  # index

    def __init__(self, segments: List[SpeechSegment]):
        super().__init__()
        self.setWindowTitle("Speech Segments Viewer")
        self.resize(1180, 780)
        self.segments = segments
        self.current_segment_index = -1

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 8, 12, 12)
        main_layout.setSpacing(10)

        # Toolbar
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        act_jump = toolbar.addAction(QIcon.fromTheme("go-jump"), "Jump to selected")
        act_jump.triggered.connect(self._on_jump_clicked)

        act_play = toolbar.addAction(QIcon.fromTheme("media-playback-start"), "Play")
        act_play.setEnabled(False)  # placeholder

        toolbar.addSeparator()

        # Header
        header = QLabel("Speech / Non-speech Segments")
        header.setStyleSheet(
            "font-size: 17pt; font-weight: 600; color: #212529; padding: 4px 0;"
        )
        main_layout.addWidget(header)

        # Timeline + position label
        tl_container = QWidget()
        tl_layout = QVBoxLayout(tl_container)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(4)

        self.position_label = QLabel("Position: 0.000 s")
        self.position_label.setStyleSheet("font-size: 11pt; color: #343a40;")
        tl_layout.addWidget(self.position_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.timeline = TimelineWidget(self.segments, self)
        self.timeline.segmentClicked.connect(self._on_segment_clicked)
        tl_layout.addWidget(self.timeline)

        main_layout.addWidget(tl_container)

        # Table
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(300)

        model = SpeechSegmentModel(self.segments, self)
        self.table.setModel(model)

        # Action column delegate
        delegate = PlayButtonDelegate(self.table)
        self.table.setItemDelegateForColumn(SpeechSegmentModel.ACTION_COL, delegate)

        # Column widths
        self.table.setColumnWidth(SpeechSegmentModel.ACTION_COL, 50)
        self.table.setColumnWidth(1, 60)  # Num
        self.table.setColumnWidth(2, 90)  # Start
        self.table.setColumnWidth(3, 90)
        self.table.setColumnWidth(4, 90)
        self.table.setColumnWidth(5, 70)  # Prob

        # Style
        self.table.setStyleSheet("""
            QTableView {
                background-color: #ffffff;
                alternate-background-color: #f8f9fa;
                font-family: "Segoe UI", sans-serif;
                font-size: 13px;
                selection-background-color: #cce5ff;
                selection-color: #212529;
            }
            QHeaderView::section {
                background-color: #e9ecef;
                padding: 6px 8px;
                border: 1px solid #ced4da;
                font-weight: 600;
            }
        """)

        self.table.doubleClicked.connect(self._on_double_clicked)
        self.table.selectionModel().currentRowChanged.connect(self._on_row_changed)

        scroll = QScrollArea()
        scroll.setWidget(self.table)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll, stretch=1)

    def _on_segment_clicked(self, idx: int):
        self._activate_segment(idx)

    def _on_double_clicked(self, index: QModelIndex):
        if index.isValid():
            self._activate_segment(index.row())

    def _on_row_changed(self, current: QModelIndex, previous: QModelIndex):
        if current.isValid():
            self._update_position_from_segment(current.row())

    def _on_jump_clicked(self):
        sel = self.table.selectionModel()
        if sel.hasSelection():
            row = sel.currentIndex().row()
            self._activate_segment(row)

    def _activate_segment(self, idx: int):
        if 0 <= idx < len(self.segments):
            self.current_segment_index = idx
            self.table.selectRow(idx)
            self.table.scrollTo(
                self.table.model().index(idx, 0), QTableView.ScrollHint.PositionAtCenter
            )
            seg = self.segments[idx]
            mid_ms = (seg["start_ms"] + seg["end_ms"]) / 2
            self.timeline.set_current_position(mid_ms)
            self.position_label.setText(
                f"Position: {mid_ms / 1000:.3f} s  —  Segment #{seg['num']}"
            )
            self.segmentActivated.emit(idx)

    def _update_position_from_segment(self, idx: int):
        if 0 <= idx < len(self.segments):
            seg = self.segments[idx]
            mid_ms = (seg["start_ms"] + seg["end_ms"]) / 2
            self.timeline.set_current_position(mid_ms)
            self.position_label.setText(
                f"Position: {mid_ms / 1000:.3f} s  —  Segment #{seg['num']}"
            )

    @classmethod
    def run_example(cls):
        app = QApplication(sys.argv)

        example_segments: List[SpeechSegment] = [
            {
                "num": 1,
                "start_ms": 0.0,
                "end_ms": 1450.0,
                "prob": 0.98,
                "frame_start": 0,
                "frame_end": 45,
                "type": "speech",
            },
            {
                "num": 2,
                "start_ms": 1800.0,
                "end_ms": 2200.0,
                "prob": 0.12,
                "frame_start": 56,
                "frame_end": 68,
                "type": "non-speech",
            },
            {
                "num": 3,
                "start_ms": 2450.0,
                "end_ms": 6200.0,
                "prob": 0.95,
                "frame_start": 76,
                "frame_end": 193,
                "type": "speech",
            },
            {
                "num": 4,
                "start_ms": 6500.0,
                "end_ms": 7200.0,
                "prob": 0.31,
                "frame_start": 202,
                "frame_end": 224,
                "type": "non-speech",
            },
            {
                "num": 5,
                "start_ms": 8500.0,
                "end_ms": 9800.0,
                "prob": 0.89,
                "frame_start": 265,
                "frame_end": 306,
                "type": "speech",
            },
        ]

        viewer = cls(example_segments)
        viewer.segmentActivated.connect(
            lambda i: print(
                f"Activated segment index {i} (#{viewer.segments[i]['num']})"
            )
        )
        viewer.show()
        sys.exit(app.exec())


if __name__ == "__main__":
    SpeechSegmentsViewer.run_example()
