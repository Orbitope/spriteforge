# Spriteforge — retro sprite restoration studio
# Copyright (C) 2026 Matthew Wesley Burke
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
PySide6 GUI application — Premium Retro 2D Sprite Studio (Phase 6).
Features sleek dark mode / glassmorphism styling, Convert (deterministic pixelation), Restore (AI VQ-GAN), and Evaluate (calibration grid) tabs.
"""

from __future__ import annotations

import re
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QCheckBox,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QMessageBox, QDialog, QDialogButtonBox,
    QLineEdit, QListWidget, QListWidgetItem, QColorDialog, QInputDialog,
)

from typing import TYPE_CHECKING

from spriteforge.core.pipeline import convert_image_to_sprite
from spriteforge.core.io import load_image_float32, save_image_float32
from spriteforge.core.palette import (
    extract_palette_kmeans,
    nearest_neighbor_snap,
    remove_background_flood,
)
from spriteforge.core import palette_library

# torch and the neural models are heavy; they are imported lazily inside the
# Restore tab's methods so launching the app (Convert-only for most users) stays
# fast and doesn't pull in torch at all.
if TYPE_CHECKING:
    from spriteforge.model.vqgan import SpriteVQGAN


DARK_THEME_QSS = """
QMainWindow, QWidget {
    background-color: #131418;
    color: #F0F0F0;
    font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif;
    font-size: 13px;
}
QTabWidget::pane {
    border: 1px solid #2D303E;
    background-color: #1A1C23;
    border-radius: 8px;
    margin-top: -1px;
}
QTabBar::tab {
    background-color: #131418;
    color: #A0A0A0;
    padding: 12px 24px;
    border: 1px solid #2D303E;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 4px;
    font-weight: 600;
}
QTabBar::tab:selected {
    background-color: #1A1C23;
    color: #6C5CE7;
    font-weight: bold;
    border-bottom: 2px solid #6C5CE7;
}
QTabBar::tab:hover {
    color: #F0F0F0;
    background-color: #22242D;
}
QGroupBox {
    background-color: #1E2028;
    border: 1px solid #2D303E;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 16px;
    font-weight: bold;
    color: #00CEC9;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
}
QPushButton {
    background-color: #2D303E;
    color: #F0F0F0;
    border: 1px solid #3D4052;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #3D4052;
    border-color: #6C5CE7;
}
QPushButton#primaryButton {
    background-color: #6C5CE7;
    color: #FFFFFF;
    border: none;
    font-size: 14px;
    padding: 10px 20px;
}
QPushButton#primaryButton:hover {
    background-color: #5A4BCE;
}
QPushButton#accentButton {
    background-color: #00CEC9;
    color: #131418;
    border: none;
    font-weight: bold;
}
QPushButton#accentButton:hover {
    background-color: #00B5B0;
}
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #131418;
    border: 1px solid #2D303E;
    border-radius: 6px;
    padding: 6px;
    color: #F0F0F0;
}
QSlider::groove:horizontal {
    border: 1px solid #2D303E;
    height: 6px;
    background: #131418;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #6C5CE7;
    border: 1px solid #5A4BCE;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QCheckBox {
    spacing: 8px;
    color: #F0F0F0;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3D4052;
    border-radius: 4px;
    background: #131418;
}
QCheckBox::indicator:checked {
    background: #6C5CE7;
    border-color: #6C5CE7;
}
QLabel#previewLabel {
    background-color: #131418;
    border: 2px dashed #2D303E;
    border-radius: 8px;
    padding: 10px;
}
"""


def numpy_to_qpixmap(arr_rgba: np.ndarray, scale_size: int = 320) -> QPixmap:
    """Convert float32 RGBA numpy array [0, 1] to a crisp nearest-neighbor QPixmap preview."""
    u8 = np.clip(arr_rgba * 255.0 + 0.5, 0, 255).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)  # QImage needs a C-contiguous buffer; a torch .permute()'d
    # array (e.g. the VQ-GAN's raw, unsnapped output) is not, and QImage() raises otherwise.
    h, w, c = u8.shape
    qimg = QImage(u8.data, w, h, w * c, QImage.Format_RGBA8888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(scale_size, scale_size, Qt.KeepAspectRatio, Qt.FastTransformation)


def _parse_color_input(text: str) -> tuple[int, int, int] | None:
    """Parse a color string in hex (#rgb, #rrggbb) or rgba(r,g,b,a) form.
    Returns (r, g, b) as 0-255 ints, or None if unrecognisable."""
    text = text.strip()
    m = re.fullmatch(r"#([0-9a-fA-F]{3})", text)
    if m:
        h = m.group(1)
        return tuple(int(c * 2, 16) for c in h)
    m = re.fullmatch(r"#([0-9a-fA-F]{6})", text)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    m = re.fullmatch(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*[\d.]+)?\s*\)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _make_swatch_icon(r: int, g: int, b: int) -> QIcon:
    pm = QPixmap(24, 24)
    pm.fill(QColor(r, g, b))
    return QIcon(pm)


def _populate_swatch_checklist(list_widget: QListWidget, colors: np.ndarray, checked: np.ndarray | None = None) -> None:
    """Fill `list_widget` with checkable colored swatches for `colors` (K, 3) float32
    in [0, 1]. `checked` is an optional (K,) bool array (defaults to all-checked).
    Signals are blocked during the rebuild so this doesn't fire itemChanged and
    trigger a live reconversion for every item added."""
    list_widget.blockSignals(True)
    list_widget.clear()
    for i, rgb in enumerate(colors):
        u8 = np.clip(np.asarray(rgb) * 255.0 + 0.5, 0, 255).astype(int)
        r, g, b = int(u8[0]), int(u8[1]), int(u8[2])
        item = QListWidgetItem(_make_swatch_icon(r, g, b), f"  #{r:02x}{g:02x}{b:02x}")
        item.setData(Qt.UserRole, (r, g, b))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        is_checked = True if checked is None else bool(checked[i])
        item.setCheckState(Qt.Checked if is_checked else Qt.Unchecked)
        list_widget.addItem(item)
    list_widget.blockSignals(False)


def _checklist_colors(list_widget: QListWidget, checked_only: bool) -> np.ndarray | None:
    """Colors currently in `list_widget` (all, or just the checked ones). None if
    the list is empty or (when checked_only) nothing is checked."""
    colors = []
    for i in range(list_widget.count()):
        item = list_widget.item(i)
        if checked_only and item.checkState() != Qt.Checked:
            continue
        colors.append(item.data(Qt.UserRole))
    if not colors:
        return None
    return np.array(colors, dtype=np.float32) / 255.0


class PaletteManagerDialog(QDialog):
    """Define, import, sub-select, save, and delete palettes for reuse.

    Swatches are checkable: checked colors are the *enabled subset* used for a
    sprite's imputation (get_enabled_palette_float32), while the full set is still
    available (get_full_palette_float32). Saving/importing/deleting go through
    spriteforge.core.palette_library, so user palettes persist in ~/.spriteforge
    and appear in the tab dropdowns immediately (no app restart)."""

    def __init__(self, parent=None, initial_colors: list[str] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Palette Manager")
        self.setMinimumSize(440, 580)
        self.library_changed = False  # tells the caller to refresh its combo
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Build a palette below (type/pick colors, or load/import one), "
            "then name it and save it to your library for reuse."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(intro)

        # ── color input row ──────────────────────────────────────────────────
        input_row = QHBoxLayout()
        self.edit_color = QLineEdit()
        self.edit_color.setPlaceholderText("#rrggbb  or  rgba(r,g,b)  or  #rgb")
        self.edit_color.returnPressed.connect(self._add_from_text)
        input_row.addWidget(self.edit_color, 1)
        btn_add = QPushButton("Add")
        btn_add.clicked.connect(self._add_from_text)
        input_row.addWidget(btn_add)
        btn_picker = QPushButton("🎨 Pick…")
        btn_picker.clicked.connect(self._add_from_picker)
        input_row.addWidget(btn_picker)
        layout.addLayout(input_row)

        # ── library load / import row ────────────────────────────────────────
        lib_row = QHBoxLayout()
        btn_lib = QPushButton("📚 Load from library…")
        btn_lib.clicked.connect(self._load_from_library)
        lib_row.addWidget(btn_lib)
        btn_import = QPushButton("📥 Import file…")
        btn_import.clicked.connect(self._import_file)
        lib_row.addWidget(btn_import)
        layout.addLayout(lib_row)

        # ── swatch list (checkable = enabled subset) ─────────────────────────
        layout.addWidget(QLabel("Checked colors are used for imputation:"))
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(24, 24))
        layout.addWidget(self.list_widget)

        # ── list actions ─────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Check all")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_row.addWidget(btn_all)
        btn_none = QPushButton("Uncheck all")
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        btn_row.addWidget(btn_none)
        btn_del = QPushButton("Remove selected")
        btn_del.clicked.connect(self._remove_selected)
        btn_row.addWidget(btn_del)
        btn_clear = QPushButton("Clear all")
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)

        # ── save-to-library row ──────────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.addWidget(QLabel("Save as:"))
        self.edit_preset_name = QLineEdit()
        self.edit_preset_name.setPlaceholderText("e.g. my-forest-palette")
        save_row.addWidget(self.edit_preset_name, 1)
        layout.addLayout(save_row)

        save_btn_row = QHBoxLayout()
        btn_save = QPushButton("💾 Save all colors")
        btn_save.setToolTip("Save every color above under the name to the left (overwrites if it already exists).")
        btn_save.clicked.connect(lambda: self._save_to_library(enabled_only=False))
        save_btn_row.addWidget(btn_save)
        btn_save_sel = QPushButton("💾 Save checked only")
        btn_save_sel.setToolTip("Save just the checked (sub-selected) colors as a new palette.")
        btn_save_sel.clicked.connect(lambda: self._save_to_library(enabled_only=True))
        save_btn_row.addWidget(btn_save_sel)
        btn_delete = QPushButton("🗑 Delete from library…")
        btn_delete.clicked.connect(self._delete_from_library)
        save_btn_row.addWidget(btn_delete)
        layout.addLayout(save_btn_row)

        # ── standard OK / Cancel ─────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        if initial_colors:
            for c in initial_colors:
                parsed = _parse_color_input(c)
                if parsed:
                    self._push_color(*parsed)

    # ── internal helpers ─────────────────────────────────────────────────────

    def _make_swatch(self, r: int, g: int, b: int) -> QIcon:
        from PySide6.QtGui import QPixmap as _QPixmap, QColor as _QColor
        pm = _QPixmap(24, 24)
        pm.fill(_QColor(r, g, b))
        return QIcon(pm)

    def _push_color(self, r: int, g: int, b: int) -> None:
        hex_str = f"#{r:02x}{g:02x}{b:02x}"
        item = QListWidgetItem(self._make_swatch(r, g, b), f"  {hex_str}")
        item.setData(Qt.UserRole, (r, g, b))
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        self.list_widget.addItem(item)

    def _push_palette_float32(self, colors: np.ndarray) -> None:
        for rgb in colors:
            u8 = np.clip(np.asarray(rgb) * 255.0 + 0.5, 0, 255).astype(int)
            self._push_color(int(u8[0]), int(u8[1]), int(u8[2]))

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def _add_from_text(self) -> None:
        text = self.edit_color.text()
        parsed = _parse_color_input(text)
        if parsed is None:
            QMessageBox.warning(self, "Invalid color", f"Could not parse: {text!r}\n\nUse #rrggbb, #rgb, or rgba(r,g,b)")
            return
        self._push_color(*parsed)
        self.edit_color.clear()

    def _add_from_picker(self) -> None:
        color = QColorDialog.getColor(parent=self)
        if color.isValid():
            self._push_color(color.red(), color.green(), color.blue())

    def _load_from_library(self) -> None:
        names = [p.name for p in palette_library.list_palettes()]
        if not names:
            QMessageBox.information(self, "Empty library", "No palettes available yet.")
            return
        name, ok = QInputDialog.getItem(self, "Load palette", "Palette:", names, 0, False)
        if not ok or not name:
            return
        try:
            named = palette_library.load_named(name)
        except ValueError as e:
            QMessageBox.warning(self, "Load failed", str(e))
            return
        self._clear()
        self._push_palette_float32(named.colors)
        self.edit_preset_name.setText(named.name)  # editing this one -> Save overwrites it

    def _import_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Palette File", "",
            "Palette Files (*.json *.hex *.pal *.gpl *.png)",
        )
        if not file_path:
            return
        try:
            colors = palette_library.import_palette(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", f"Could not import palette:\n{e}")
            return
        self._clear()
        self._push_palette_float32(colors)
        # Suggest the file's stem as the save name; not yet in the library until Saved.
        self.edit_preset_name.setText(Path(file_path).stem)

    def _remove_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def _clear(self) -> None:
        self.list_widget.clear()

    def _save_to_library(self, enabled_only: bool) -> None:
        name = self.edit_preset_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Enter a name before saving.")
            return
        colors = self.get_enabled_palette_float32() if enabled_only else self.get_full_palette_float32()
        if colors is None:
            QMessageBox.warning(self, "Empty palette", "Add at least one color before saving.")
            return
        try:
            path = palette_library.save_user_palette(name, colors)
        except ValueError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self.library_changed = True
        QMessageBox.information(self, "Saved", f"Palette '{path.stem}' saved to your library.")

    def _delete_from_library(self) -> None:
        user_names = palette_library.list_user_palettes()
        if not user_names:
            QMessageBox.information(self, "Nothing to delete", "You have no saved palettes. Built-in presets can't be deleted.")
            return
        name, ok = QInputDialog.getItem(self, "Delete palette", "Your palettes:", user_names, 0, False)
        if not ok or not name:
            return
        try:
            palette_library.delete_user_palette(name)
        except ValueError as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return
        self.library_changed = True
        QMessageBox.information(self, "Deleted", f"Deleted palette '{name}'.")

    # ── public API ───────────────────────────────────────────────────────────

    def _colors_rgb(self, checked_only: bool) -> list[tuple[int, int, int]]:
        result = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if checked_only and item.checkState() != Qt.Checked:
                continue
            result.append(item.data(Qt.UserRole))
        return result

    def get_full_palette_float32(self) -> np.ndarray | None:
        """All colors as float32 (N, 3) in [0, 1], or None if empty."""
        colors = self._colors_rgb(checked_only=False)
        if not colors:
            return None
        return np.array(colors, dtype=np.float32) / 255.0

    def get_enabled_palette_float32(self) -> np.ndarray | None:
        """Only the checked colors as float32 (M, 3) in [0, 1] — the sub-selected
        subset used for imputation. Falls back to the full palette if nothing is
        checked, so an all-unchecked state never yields an empty palette."""
        colors = self._colors_rgb(checked_only=True)
        if not colors:
            return self.get_full_palette_float32()
        return np.array(colors, dtype=np.float32) / 255.0


class StageAStudioTab(QWidget):
    """Tab 1: Convert — deterministic pixelation and palette reduction."""
    def __init__(self):
        super().__init__()
        self.current_img: np.ndarray | None = None
        self.generated_sprite: np.ndarray | None = None
        self.custom_palette: np.ndarray | None = None  # float32 (N,3) backing "Custom (unsaved)"
        self.active_palette_full: np.ndarray | None = None  # full candidate set behind the checklist
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # ── Left: controls, grouped into everyday basics + collapsible advanced ──
        control_col = QVBoxLayout()

        basics = QGroupBox("Convert")
        bform = QFormLayout(basics)

        self.btn_load = QPushButton("Load image…")
        self.btn_load.clicked.connect(self.load_image)
        bform.addRow(self.btn_load)

        self.combo_size = QComboBox()
        self.combo_size.addItems(["16x16", "32x32", "48x48"])
        self.combo_size.setCurrentIndex(1)
        self.combo_size.currentTextChanged.connect(lambda _: self.convert_sprite())
        bform.addRow("Size:", self.combo_size)

        self.combo_palette_mode = QComboBox()
        self._refresh_palette_combo()
        self.combo_palette_mode.currentTextChanged.connect(self.on_palette_mode_changed)
        bform.addRow("Palette:", self.combo_palette_mode)

        # Only meaningful for the per-image extraction modes; disabled for a named
        # palette (its color count is fixed), so it never reads as a dead control.
        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(2, 64)
        self.spin_colors.setValue(16)
        self.spin_colors.valueChanged.connect(lambda _: self.convert_sprite())
        bform.addRow("Colors:", self.spin_colors)

        self.btn_manage_palettes = QPushButton("Define / manage palettes…")
        self.btn_manage_palettes.clicked.connect(self.open_palette_manager)
        bform.addRow(self.btn_manage_palettes)

        # Embedded, always-visible sub-select: uncheck a color and the sprite
        # reconverts immediately, no dialog to open or save required.
        bform.addRow(QLabel("Colors in use — uncheck to exclude:"))
        self.list_subselect = QListWidget()
        self.list_subselect.setIconSize(QSize(20, 20))
        self.list_subselect.setMaximumHeight(140)
        self.list_subselect.itemChanged.connect(self._on_subselect_changed)
        bform.addRow(self.list_subselect)

        control_col.addWidget(basics)

        # Collapsible "Advanced" — rarely-touched knobs, hidden by default.
        self.btn_advanced = QPushButton("▸ Advanced")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setStyleSheet("text-align: left; padding: 4px;")
        self.btn_advanced.toggled.connect(self._toggle_advanced)
        control_col.addWidget(self.btn_advanced)

        self.advanced_box = QGroupBox()
        aform = QFormLayout(self.advanced_box)
        self.chk_dither = QCheckBox("Dithering")
        self.chk_dither.toggled.connect(self._on_advanced_changed)
        aform.addRow(self.chk_dither)
        self.spin_dither_str = QDoubleSpinBox()
        self.spin_dither_str.setRange(0.0, 0.5)
        self.spin_dither_str.setValue(0.05)
        self.spin_dither_str.setSingleStep(0.01)
        self.spin_dither_str.setEnabled(False)
        self.spin_dither_str.valueChanged.connect(lambda _: self.convert_sprite())
        aform.addRow("    Strength:", self.spin_dither_str)
        self.chk_despeckle = QCheckBox("Despeckle stray pixels")
        self.chk_despeckle.setChecked(True)
        self.chk_despeckle.toggled.connect(self._on_advanced_changed)
        aform.addRow(self.chk_despeckle)
        self.spin_despeckle_area = QSpinBox()
        self.spin_despeckle_area.setRange(1, 10)
        self.spin_despeckle_area.setValue(2)
        self.spin_despeckle_area.valueChanged.connect(lambda _: self.convert_sprite())
        aform.addRow("    Min area:", self.spin_despeckle_area)
        self.chk_remove_bg = QCheckBox("Remove background first")
        self.chk_remove_bg.toggled.connect(lambda _: self.convert_sprite())
        aform.addRow(self.chk_remove_bg)
        self.advanced_box.setVisible(False)
        control_col.addWidget(self.advanced_box)

        control_col.addStretch()

        self.btn_export = QPushButton("Export PNG…")
        self.btn_export.setObjectName("accentButton")
        self.btn_export.clicked.connect(self.export_sprite)
        control_col.addWidget(self.btn_export)

        control_wrap = QWidget()
        control_wrap.setLayout(control_col)
        layout.addWidget(control_wrap, 1)

        self._sync_colors_enabled()

        # ── Right: input vs result previews ─────────────────────────────────────
        preview_group = QGroupBox("Preview")
        preview_layout = QHBoxLayout(preview_group)
        preview_layout.setAlignment(Qt.AlignTop)

        vbox_in = QVBoxLayout()
        vbox_in.addWidget(QLabel("Input"))
        self.lbl_in_preview = QLabel("No image loaded")
        self.lbl_in_preview.setObjectName("previewLabel")
        self.lbl_in_preview.setAlignment(Qt.AlignCenter)
        self.lbl_in_preview.setMinimumSize(320, 320)
        vbox_in.addWidget(self.lbl_in_preview)
        vbox_in.addStretch()
        preview_layout.addLayout(vbox_in)

        vbox_out = QVBoxLayout()
        vbox_out.addWidget(QLabel("Sprite (zoomed)"))
        self.lbl_out_preview = QLabel("Load an image to start")
        self.lbl_out_preview.setObjectName("previewLabel")
        self.lbl_out_preview.setAlignment(Qt.AlignCenter)
        self.lbl_out_preview.setMinimumSize(320, 320)
        vbox_out.addWidget(self.lbl_out_preview)
        vbox_out.addStretch()
        preview_layout.addLayout(vbox_out)

        layout.addWidget(preview_group, 2)

    def _toggle_advanced(self, checked: bool) -> None:
        self.btn_advanced.setText(("▾ " if checked else "▸ ") + "Advanced")
        self.advanced_box.setVisible(checked)

    def _on_advanced_changed(self) -> None:
        """A dither/despeckle toggle changed: sync the dependent spinbox's enabled
        state (so a disabled feature's knob isn't editable) and reconvert."""
        self.spin_dither_str.setEnabled(self.chk_dither.isChecked())
        self.spin_despeckle_area.setEnabled(self.chk_despeckle.isChecked())
        self.convert_sprite()

    def _sync_colors_enabled(self) -> None:
        """Colors only affects the per-image extraction modes; grey it out otherwise."""
        mode = self.combo_palette_mode.currentText()
        self.spin_colors.setEnabled(mode in ("per-image-kmeans", "per-image-median"))

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Input Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if file_path:
            try:
                self.current_img = load_image_float32(file_path)
                pix = numpy_to_qpixmap(self.current_img, 320)
                self.lbl_in_preview.setPixmap(pix)
                self.lbl_in_preview.setText("")
                self.convert_sprite()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def _refresh_palette_combo(self) -> None:
        """(Re)populate the palette dropdown from the library (builtin + user),
        preserving the current selection. Call after the manager dialog changes
        the library so new palettes appear without an app restart."""
        combo = self.combo_palette_mode
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        items = ["per-image-kmeans", "per-image-median"]
        for p in palette_library.list_palettes():
            items.append(f"palette:{p.name}")
        # Keep the transient "just defined, not yet saved" entry alive across a
        # refresh (e.g. triggered by saving a *different* palette in the manager).
        if self.custom_palette is not None:
            items.append("Custom (unsaved)")
        combo.addItems(items)
        idx = combo.findText(prev)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def open_palette_manager(self) -> None:
        """Define, import, or edit a palette — separate from picking a mode above.
        The full set (and whatever was checked in the manager) becomes the
        starting point for the embedded, always-visible sub-select checklist."""
        dlg = PaletteManagerDialog(self)
        accepted = dlg.exec() == QDialog.Accepted
        if dlg.library_changed:
            self._refresh_palette_combo()
        if not accepted:
            return
        full = dlg.get_full_palette_float32()
        if full is None:
            return
        checked_mask = np.array(
            [dlg.list_widget.item(i).checkState() == Qt.Checked for i in range(dlg.list_widget.count())],
            dtype=bool,
        )
        self.custom_palette = full
        if self.combo_palette_mode.findText("Custom (unsaved)") < 0:
            self.combo_palette_mode.addItem("Custom (unsaved)")
        self.combo_palette_mode.blockSignals(True)
        self.combo_palette_mode.setCurrentText("Custom (unsaved)")
        self.combo_palette_mode.blockSignals(False)
        self.active_palette_full = full
        _populate_swatch_checklist(self.list_subselect, full, checked=checked_mask)
        self.convert_sprite(resync_palette=False)  # checklist is already set up above

    def on_palette_mode_changed(self, text: str) -> None:
        self._sync_colors_enabled()
        self.convert_sprite()

    def _on_subselect_changed(self, item: QListWidgetItem) -> None:
        """A checkbox in the embedded sub-select list was toggled. Reconvert live
        using only the checked colors — no dialog, no save, no re-extraction."""
        self.convert_sprite(resync_palette=False)

    def convert_sprite(self, resync_palette: bool = True):
        """resync_palette=True (the default — full "Generate" click, image load, or
        mode change) re-derives the full candidate palette for the current mode and
        resets the checklist to all-checked. resync_palette=False (a sub-select
        checkbox toggle) reuses the already-populated checklist and just narrows to
        the checked subset — this is what makes unchecking a color instant."""
        if self.current_img is None:
            return
        src = remove_background_flood(self.current_img) if self.chk_remove_bg.isChecked() else self.current_img
        size_val = int(self.combo_size.currentText().split("x")[0])
        combo_val = self.combo_palette_mode.currentText()
        colors_val = self.spin_colors.value()
        dither_val = self.chk_dither.isChecked()
        dither_str = self.spin_dither_str.value()
        despeckle_val = self.chk_despeckle.isChecked()
        despeckle_min = self.spin_despeckle_area.value()
        mode_val = "median-cut" if combo_val == "per-image-median" else "kmeans"

        try:
            if resync_palette:
                if combo_val.startswith("palette:"):
                    full_pal = palette_library.load_named(combo_val[len("palette:"):]).colors
                elif combo_val == "Custom (unsaved)":
                    if self.custom_palette is None:
                        return
                    full_pal = self.custom_palette
                else:
                    # per-image-kmeans / per-image-median: extract now so the
                    # checklist can show and sub-select the colors it found.
                    _, full_pal = convert_image_to_sprite(
                        src, target_size=size_val, palette_mode=mode_val, colors=colors_val,
                        dither=dither_val, dither_strength=dither_str,
                        despeckle=despeckle_val, despeckle_min_area=despeckle_min,
                        return_palette=True,
                    )
                self.active_palette_full = full_pal
                _populate_swatch_checklist(self.list_subselect, full_pal)

            palette_arr = _checklist_colors(self.list_subselect, checked_only=True)
            if palette_arr is None:
                palette_arr = self.active_palette_full  # nothing checked -> fall back to full

            self.generated_sprite = convert_image_to_sprite(
                src,
                target_size=size_val,
                palette_mode=mode_val,
                colors=colors_val,
                palette=palette_arr,
                dither=dither_val,
                dither_strength=dither_str,
                despeckle=despeckle_val,
                despeckle_min_area=despeckle_min,
            )
            pix = numpy_to_qpixmap(self.generated_sprite, 320)
            self.lbl_out_preview.setPixmap(pix)
            self.lbl_out_preview.setText("")
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", f"Conversion failed:\n{e}")

    def export_sprite(self):
        if self.generated_sprite is None:
            QMessageBox.warning(self, "Warning", "No sprite generated yet!")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sprite PNG", "sprite_out.png", "PNG Images (*.png)")
        if file_path:
            save_image_float32(self.generated_sprite, file_path)
            QMessageBox.information(self, "Success", f"Sprite saved successfully to:\n{file_path}")


class StageBStudioTab(QWidget):
    """Tab 2: Restore — VQ-GAN or E1 Palette UNet neural restoration."""
    def __init__(self):
        super().__init__()
        self.model: SpriteVQGAN | None = None
        self.e1_model = None  # PaletteUNet, loaded lazily
        self.current_img: np.ndarray | None = None
        self.restored_sprite: np.ndarray | None = None
        self.restored_sprite_raw: np.ndarray | None = None
        self.custom_palette_file: str | None = None
        self.custom_palette: np.ndarray | None = None  # float32 (N,3) backing "Custom (unsaved)"
        self.active_palette_full: np.ndarray | None = None  # full candidate set behind the checklist
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        control_group = QGroupBox("Restore Controls")
        form_layout = QFormLayout(control_group)

        # ── Engine selector ───────────────────────────────────────────────────
        self.combo_engine = QComboBox()
        self.combo_engine.addItems(["VQ-GAN", "Palette UNet (E1)"])
        self.combo_engine.currentTextChanged.connect(self._on_engine_changed)
        form_layout.addRow("Engine:", self.combo_engine)

        self.btn_load_ckpt = QPushButton("🤖 Load Checkpoint (.pt)...")
        self.btn_load_ckpt.clicked.connect(self.load_checkpoint)
        form_layout.addRow(self.btn_load_ckpt)

        self.lbl_ckpt_status = QLabel("No Checkpoint Loaded")
        self.lbl_ckpt_status.setStyleSheet("color: #FF7675;")
        form_layout.addRow("Status:", self.lbl_ckpt_status)

        self.btn_load_img = QPushButton("📁 Load Degraded Input...")
        self.btn_load_img.clicked.connect(self.load_image)
        form_layout.addRow(self.btn_load_img)

        self.chk_remove_bg = QCheckBox("Remove background before restoring")
        form_layout.addRow(self.chk_remove_bg)

        self.btn_restore = QPushButton("✨ Run Neural Restoration")
        self.btn_restore.setObjectName("primaryButton")
        # Lambda, not a direct connect: run_restoration gained a resync_palette
        # kwarg, and QPushButton.clicked emits a bool that would otherwise land
        # in that slot position. A full click should always resync (re-extract).
        self.btn_restore.clicked.connect(lambda: self.run_restoration())
        form_layout.addRow(self.btn_restore)

        # ── Palette: snaps VQ-GAN output, conditions E1 ───────────────────────
        self.combo_palette_mode = QComboBox()
        self._refresh_palette_combo()
        self.combo_palette_mode.setCurrentText("Source image (k-means)")
        self.combo_palette_mode.currentTextChanged.connect(self.on_palette_mode_changed)
        form_layout.addRow("Palette:", self.combo_palette_mode)

        # A dedicated action, separate from picking a mode above: define new colors,
        # import a file, sub-select, and save/delete named palettes in your library.
        self.btn_manage_palettes = QPushButton("🎨 Define / manage palettes…")
        self.btn_manage_palettes.clicked.connect(self.open_palette_manager)
        form_layout.addRow(self.btn_manage_palettes)

        # Embedded, always-visible sub-select: uncheck a color and the restoration
        # reruns immediately, no dialog to open or save required.
        subselect_container = QWidget()
        subselect_layout = QVBoxLayout(subselect_container)
        subselect_layout.setContentsMargins(0, 0, 0, 0)
        subselect_layout.addWidget(QLabel("Uncheck to exclude a color (updates live):"))
        self.list_subselect = QListWidget()
        self.list_subselect.setIconSize(QSize(20, 20))
        self.list_subselect.setMaximumHeight(140)
        self.list_subselect.itemChanged.connect(self._on_subselect_changed)
        subselect_layout.addWidget(self.list_subselect)
        form_layout.addRow(subselect_container)

        # For E1 the palette conditions the model; for VQ-GAN it snaps the output.
        # This hint updates with the engine (see _on_engine_changed).
        self.lbl_palette_hint = QLabel("")
        self.lbl_palette_hint.setWordWrap(True)
        self.lbl_palette_hint.setStyleSheet("color: #888; font-size: 11px;")
        form_layout.addRow("", self.lbl_palette_hint)

        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(2, 64)
        self.spin_colors.setValue(16)
        form_layout.addRow("Max Colors:", self.spin_colors)

        self.btn_export = QPushButton("💾 Export Restored Sprite...")
        self.btn_export.setObjectName("accentButton")
        self.btn_export.clicked.connect(self.export_sprite)
        form_layout.addRow(self.btn_export)

        layout.addWidget(control_group, 1)

        preview_group = QGroupBox("Restore Preview")
        preview_layout = QHBoxLayout(preview_group)

        vbox_in = QVBoxLayout()
        vbox_in.addWidget(QLabel("Degraded / Noisy Input:"))
        self.lbl_in_preview = QLabel("No Image Loaded")
        self.lbl_in_preview.setObjectName("previewLabel")
        self.lbl_in_preview.setAlignment(Qt.AlignCenter)
        self.lbl_in_preview.setMinimumSize(320, 320)
        vbox_in.addWidget(self.lbl_in_preview)
        preview_layout.addLayout(vbox_in)

        vbox_out = QVBoxLayout()
        self.lbl_out_header = QLabel("Restored Sprite (VQ-GAN):")
        vbox_out.addWidget(self.lbl_out_header)
        self.lbl_out_preview = QLabel("Load Checkpoint & Click Restore")
        self.lbl_out_preview.setObjectName("previewLabel")
        self.lbl_out_preview.setAlignment(Qt.AlignCenter)
        self.lbl_out_preview.setMinimumSize(320, 320)
        vbox_out.addWidget(self.lbl_out_preview)
        preview_layout.addLayout(vbox_out)

        layout.addWidget(preview_group, 2)

    def _on_engine_changed(self, engine: str) -> None:
        is_vqgan = engine == "VQ-GAN"
        # Palette drives BOTH engines now: it snaps VQ-GAN output, and conditions
        # the E1 model. So the combo stays enabled for E1 too.
        self.combo_palette_mode.setEnabled(True)
        self.spin_colors.setEnabled(True)
        self.lbl_palette_hint.setText(
            "Snaps the model's output to the chosen palette."
            if is_vqgan else
            "Conditions the model. The palette is adapted to the model's trained color count."
        )
        self.lbl_out_header.setText(
            "Restored Sprite (VQ-GAN):" if is_vqgan else "Restored Sprite (E1 Palette UNet):"
        )
        if is_vqgan:
            loaded = self.model is not None
            self.lbl_ckpt_status.setText(
                f"Loaded" if loaded else "No Checkpoint Loaded"
            )
            self.lbl_ckpt_status.setStyleSheet("color: #00CEC9;" if loaded else "color: #FF7675;")
        else:
            loaded = self.e1_model is not None
            self.lbl_ckpt_status.setText(
                "E1 loaded (16 colors)" if loaded else "No E1 Checkpoint Loaded"
            )
            self.lbl_ckpt_status.setStyleSheet("color: #00CEC9;" if loaded else "color: #FF7675;")

    def load_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Model Checkpoint", "checkpoints", "PyTorch Checkpoints (*.pt)")
        if not file_path:
            return
        try:
            import torch  # lazy: only when a checkpoint is actually loaded
            ckpt = torch.load(file_path, map_location="cpu", weights_only=False)
            if "num_colors" in ckpt:
                # E1 Palette UNet checkpoint
                from spriteforge.model.palette_unet import PaletteUNet, PaletteUNetConfig
                config = PaletteUNetConfig(num_colors=ckpt["num_colors"], hidden_channels=ckpt["hidden_channels"])
                self.e1_model = PaletteUNet(config)
                self.e1_model.load_state_dict(ckpt["model_state_dict"])
                self.e1_model.eval()
                self.combo_engine.setCurrentText("Palette UNet (E1)")
                self.lbl_ckpt_status.setText(f"E1 loaded ({ckpt['num_colors']} colors, epoch {ckpt['epoch']})")
                self.lbl_ckpt_status.setStyleSheet("color: #00CEC9;")
            else:
                # VQ-GAN checkpoint
                from spriteforge.model.config import get_config
                from spriteforge.model.vqgan import SpriteVQGAN
                config_name = ckpt.get("config_name", "32")
                config = get_config(config_name)
                self.model = SpriteVQGAN(config)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                self.combo_engine.setCurrentText("VQ-GAN")
                self.lbl_ckpt_status.setText(f"VQ-GAN loaded ({config.target_size}x{config.target_size})")
                self.lbl_ckpt_status.setStyleSheet("color: #00CEC9;")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load checkpoint:\n{e}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Degraded Image", "", "Images (*.png *.jpg *.bmp *.webp)")
        if file_path:
            try:
                self.current_img = load_image_float32(file_path)
                pix = numpy_to_qpixmap(self.current_img, 320)
                self.lbl_in_preview.setPixmap(pix)
                self.lbl_in_preview.setText("")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")

    def _refresh_palette_combo(self) -> None:
        """(Re)populate the palette dropdown from the library (builtin + user),
        preserving the selection. Called after the manager dialog edits the
        library so new palettes appear without an app restart."""
        combo = self.combo_palette_mode
        prev = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        items = ["None (raw output)", "Source image (k-means)", "Restored output (k-means)"]
        for p in palette_library.list_palettes():
            items.append(f"palette:{p.name}")
        items.append("Custom file...")
        # Keep the transient "just defined, not yet saved" entry alive across a
        # refresh (e.g. triggered by saving a *different* palette in the manager).
        if self.custom_palette is not None:
            items.append("Custom (unsaved)")
        combo.addItems(items)
        idx = combo.findText(prev)
        combo.setCurrentIndex(idx if idx >= 0 else combo.findText("Source image (k-means)"))
        combo.blockSignals(False)

    def open_palette_manager(self) -> None:
        """Define, import, or edit a palette — separate from picking a mode above.
        The full set (and whatever was checked in the manager) becomes the
        starting point for the embedded, always-visible sub-select checklist."""
        dlg = PaletteManagerDialog(self)
        accepted = dlg.exec() == QDialog.Accepted
        if dlg.library_changed:
            self._refresh_palette_combo()
        if not accepted:
            return
        full = dlg.get_full_palette_float32()
        if full is None:
            return
        checked_mask = np.array(
            [dlg.list_widget.item(i).checkState() == Qt.Checked for i in range(dlg.list_widget.count())],
            dtype=bool,
        )
        self.custom_palette = full
        if self.combo_palette_mode.findText("Custom (unsaved)") < 0:
            self.combo_palette_mode.addItem("Custom (unsaved)")
        self.combo_palette_mode.blockSignals(True)
        self.combo_palette_mode.setCurrentText("Custom (unsaved)")
        self.combo_palette_mode.blockSignals(False)
        self.active_palette_full = full
        _populate_swatch_checklist(self.list_subselect, full, checked=checked_mask)
        if self.current_img is not None and (self.model is not None or self.e1_model is not None):
            self.run_restoration(resync_palette=False)  # checklist is already set up above

    def on_palette_mode_changed(self, text: str):
        if text == "Custom file...":
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Palette File", "",
                "Palette Files (*.json *.hex *.pal *.gpl *.png)",
            )
            if file_path:
                self.custom_palette_file = file_path
            else:
                self.combo_palette_mode.setCurrentText("Source image (k-means)")

    def _on_subselect_changed(self, item: QListWidgetItem) -> None:
        """A checkbox in the embedded sub-select list was toggled. Rerun the
        restoration live using only the checked colors — no dialog, no save."""
        self.run_restoration(resync_palette=False)

    def _explicit_palette_or_none(self) -> np.ndarray | None:
        """The fixed palette array for the current selection, or None for the
        dynamic k-means/raw modes. Shared by the VQ-GAN snap and E1 conditioning."""
        mode = self.combo_palette_mode.currentText()
        if mode.startswith("palette:"):
            return palette_library.load_named(mode[len("palette:"):]).colors
        if mode == "Custom file...":
            return palette_library.import_palette(self.custom_palette_file) if self.custom_palette_file else None
        if mode == "Custom (unsaved)":
            return self.custom_palette
        return None

    def _resolve_full_palette(
        self, source_resized: np.ndarray, raw_output: np.ndarray | None, num_colors: int
    ) -> np.ndarray | None:
        """Full candidate colors for the current Palette selection — used to (re)populate
        the sub-select checklist on a resync. Only VQ-GAN's "None (raw output)" has
        nothing to sub-select (the output isn't restricted to any palette); E1 always
        needs a concrete palette (its own default is k-means from the source), so we
        compute that default here too, rather than leaving it opaque inside the model."""
        mode = self.combo_palette_mode.currentText()
        if mode == "None (raw output)":
            return None if self.combo_engine.currentText() == "VQ-GAN" else extract_palette_kmeans(source_resized, k=num_colors)
        if mode == "Source image (k-means)":
            return extract_palette_kmeans(source_resized, k=num_colors)
        if mode == "Restored output (k-means)":
            basis = raw_output if raw_output is not None else source_resized
            return extract_palette_kmeans(basis, k=num_colors)
        return self._explicit_palette_or_none()

    def run_restoration(self, resync_palette: bool = True):
        """resync_palette=True (the default — full "Run" click, engine/checkpoint
        change) re-derives the full candidate palette and resets the checklist to
        all-checked. resync_palette=False (a sub-select checkbox toggle) reuses the
        already-populated checklist and just narrows to the checked subset."""
        if self.current_img is None:
            QMessageBox.warning(self, "Warning", "Please load an input image first!")
            return
        engine = self.combo_engine.currentText()
        try:
            if engine == "Palette UNet (E1)":
                self._run_e1_restoration(resync_palette=resync_palette)
            else:
                self._run_vqgan_restoration(resync_palette=resync_palette)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Restoration failed:\n{e}")

    def _run_vqgan_restoration(self, resync_palette: bool = True) -> None:
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a VQ-GAN checkpoint first!")
            return
        import torch
        from spriteforge.core.resize import resize_to_target
        target_size = self.model.config.target_size
        src = remove_background_flood(self.current_img) if self.chk_remove_bg.isChecked() else self.current_img
        resized_in = resize_to_target(src, target_size=target_size)
        tensor_in = torch.from_numpy(resized_in).permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            out_tensor, _, _ = self.model(tensor_in)
        raw_output = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        self.restored_sprite_raw = raw_output

        colors = self.spin_colors.value()
        if resync_palette:
            full_pal = self._resolve_full_palette(resized_in, raw_output, colors)
            self.active_palette_full = full_pal
            if full_pal is not None:
                _populate_swatch_checklist(self.list_subselect, full_pal)
            else:
                self.list_subselect.clear()

        if self.active_palette_full is None:
            self.restored_sprite = raw_output  # "None (raw output)" on VQ-GAN: no snap
        else:
            palette_arr = _checklist_colors(self.list_subselect, checked_only=True)
            if palette_arr is None:
                palette_arr = self.active_palette_full
            self.restored_sprite = nearest_neighbor_snap(raw_output, palette_arr)

        pix = numpy_to_qpixmap(self.restored_sprite, 320)
        self.lbl_out_preview.setPixmap(pix)
        self.lbl_out_preview.setText("")

    def _run_e1_restoration(self, resync_palette: bool = True) -> None:
        if self.e1_model is None:
            QMessageBox.warning(self, "Warning", "Please load a Palette UNet (E1) checkpoint first!")
            return
        from spriteforge.core.resize import resize_to_target
        from spriteforge.model.palette_infer import restore_sprite
        src = remove_background_flood(self.current_img) if self.chk_remove_bg.isChecked() else self.current_img
        resized_in = resize_to_target(src, target_size=32)
        img_rgba = resized_in.copy()
        img_rgba[..., :3] *= img_rgba[..., 3:4]  # premultiply alpha, matching training
        num_colors = self.e1_model.config.num_colors

        if resync_palette:
            full_pal = self._resolve_full_palette(img_rgba, None, num_colors)
            self.active_palette_full = full_pal
            _populate_swatch_checklist(self.list_subselect, full_pal)

        # A user-selected (possibly sub-selected) palette conditions the model.
        # restore_sprite adapts any palette width to the model's trained K via
        # _fit_palette_width, so a checked subset works even if it's smaller than K.
        palette_arr = _checklist_colors(self.list_subselect, checked_only=True)
        if palette_arr is None:
            palette_arr = self.active_palette_full
        out = restore_sprite(self.e1_model, num_colors, img_rgba, palette=palette_arr)
        self.restored_sprite_raw = out
        self.restored_sprite = out
        pix = numpy_to_qpixmap(self.restored_sprite, 320)
        self.lbl_out_preview.setPixmap(pix)
        self.lbl_out_preview.setText("")

    def export_sprite(self):
        if self.restored_sprite is None:
            QMessageBox.warning(self, "Warning", "No restored sprite yet! Run restoration first.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Restored Sprite PNG", "restored_sprite.png", "PNG Images (*.png)"
        )
        if file_path:
            # Save exact model-resolution RGBA — preserves the alpha matte, no interpolation.
            # This is post-palette-snap per the "Palette Snap" combo box above (defaults to
            # source-image-extracted, not the raw model output).
            save_image_float32(self.restored_sprite, file_path)
            QMessageBox.information(self, "Success", f"Restored sprite saved to:\n{file_path}")


class SpriteforgeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spriteforge")
        self.resize(1000, 700)
        self.setStyleSheet(DARK_THEME_QSS)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel("Spriteforge")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #6C5CE7; margin-bottom: 8px;")
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(StageAStudioTab(), "Convert")
        tabs.addTab(StageBStudioTab(), "Restore")
        layout.addWidget(tabs)

        self.setCentralWidget(central_widget)


def run_app():
    app = QApplication(sys.argv)
    window = SpriteforgeMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
