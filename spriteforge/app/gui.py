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
from PySide6.QtGui import QImage, QPixmap, QIcon, QFont, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QComboBox, QSlider, QCheckBox,
    QFileDialog, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QMessageBox, QSplitter, QScrollArea, QFrame, QDialog, QDialogButtonBox,
    QLineEdit, QListWidget, QListWidgetItem, QColorDialog, QInputDialog,
)

from spriteforge.core.pipeline import convert_image_to_sprite
from spriteforge.core.io import load_image_float32, save_image_float32
from spriteforge.core.degrade import degrade, DegradeRanges
from spriteforge.core.palette import (
    extract_palette_kmeans,
    nearest_neighbor_snap,
    remove_background_flood,
)
from spriteforge.core import palette_library
from spriteforge.model.config import get_config
from spriteforge.model.vqgan import SpriteVQGAN
import torch


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
        self.custom_palette: np.ndarray | None = None  # float32 (N,3) from editor
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)

        # Left Panel: Controls
        control_group = QGroupBox("Convert Controls")
        form_layout = QFormLayout(control_group)

        self.btn_load = QPushButton("📁 Load Input Image...")
        self.btn_load.clicked.connect(self.load_image)
        form_layout.addRow(self.btn_load)

        self.combo_size = QComboBox()
        self.combo_size.addItems(["16x16", "32x32", "48x48"])
        self.combo_size.setCurrentIndex(1)
        form_layout.addRow("Target Size:", self.combo_size)

        self.combo_palette_mode = QComboBox()
        self._refresh_palette_combo()
        self.combo_palette_mode.currentTextChanged.connect(self.on_palette_mode_changed)
        form_layout.addRow("Palette Mode:", self.combo_palette_mode)

        # A dedicated action, separate from picking a mode above: define new colors,
        # import a file, sub-select, and save/delete named palettes in your library.
        self.btn_manage_palettes = QPushButton("🎨 Define / manage palettes…")
        self.btn_manage_palettes.clicked.connect(self.open_palette_manager)
        form_layout.addRow(self.btn_manage_palettes)

        self.spin_colors = QSpinBox()
        self.spin_colors.setRange(2, 64)
        self.spin_colors.setValue(16)
        form_layout.addRow("Max Colors:", self.spin_colors)

        self.chk_dither = QCheckBox("Enable Bayer Ordered Dithering")
        form_layout.addRow(self.chk_dither)

        self.spin_dither_str = QDoubleSpinBox()
        self.spin_dither_str.setRange(0.0, 0.5)
        self.spin_dither_str.setValue(0.05)
        self.spin_dither_str.setSingleStep(0.01)
        form_layout.addRow("Dither Strength:", self.spin_dither_str)

        self.chk_despeckle = QCheckBox("Enable Despeckling")
        self.chk_despeckle.setChecked(True)
        form_layout.addRow(self.chk_despeckle)

        self.spin_despeckle_area = QSpinBox()
        self.spin_despeckle_area.setRange(1, 10)
        self.spin_despeckle_area.setValue(2)
        form_layout.addRow("Min Speckle Area:", self.spin_despeckle_area)

        self.chk_remove_bg = QCheckBox("Remove background before converting")
        form_layout.addRow(self.chk_remove_bg)

        self.btn_convert = QPushButton("⚡ Generate Retro Sprite")
        self.btn_convert.setObjectName("primaryButton")
        self.btn_convert.clicked.connect(self.convert_sprite)
        form_layout.addRow(self.btn_convert)

        self.btn_export = QPushButton("💾 Export Sprite PNG...")
        self.btn_export.setObjectName("accentButton")
        self.btn_export.clicked.connect(self.export_sprite)
        form_layout.addRow(self.btn_export)

        layout.addWidget(control_group, 1)

        # Right Panel: Live Previews
        preview_group = QGroupBox("Pixel Art Studio Preview")
        preview_layout = QHBoxLayout(preview_group)

        vbox_in = QVBoxLayout()
        vbox_in.addWidget(QLabel("Original High-Res Input:"))
        self.lbl_in_preview = QLabel("No Image Loaded")
        self.lbl_in_preview.setObjectName("previewLabel")
        self.lbl_in_preview.setAlignment(Qt.AlignCenter)
        self.lbl_in_preview.setMinimumSize(320, 320)
        vbox_in.addWidget(self.lbl_in_preview)
        preview_layout.addLayout(vbox_in)

        vbox_out = QVBoxLayout()
        vbox_out.addWidget(QLabel("Generated Retro Sprite (Sharp 10x):"))
        self.lbl_out_preview = QLabel("Click Generate")
        self.lbl_out_preview.setObjectName("previewLabel")
        self.lbl_out_preview.setAlignment(Qt.AlignCenter)
        self.lbl_out_preview.setMinimumSize(320, 320)
        vbox_out.addWidget(self.lbl_out_preview)
        preview_layout.addLayout(vbox_out)

        layout.addWidget(preview_group, 2)

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
        """Define, import, or edit a palette — separate from picking a mode above."""
        dlg = PaletteManagerDialog(self)
        accepted = dlg.exec() == QDialog.Accepted
        if dlg.library_changed:
            self._refresh_palette_combo()
        if not accepted:
            return
        pal = dlg.get_enabled_palette_float32()
        if pal is None or len(pal) == 0:
            return
        self.custom_palette = pal
        if self.combo_palette_mode.findText("Custom (unsaved)") < 0:
            self.combo_palette_mode.addItem("Custom (unsaved)")
        self.combo_palette_mode.setCurrentText("Custom (unsaved)")  # triggers convert_sprite via signal

    def on_palette_mode_changed(self, text: str) -> None:
        self.convert_sprite()

    def convert_sprite(self):
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

        # Resolve the palette. Named library palettes and the just-defined "Custom
        # (unsaved)" entry both produce an explicit array passed via `palette=`;
        # per-image modes let the pipeline extract. One unified conversion call.
        palette_arr = None
        mode_val = "kmeans"
        if combo_val == "per-image-median":
            mode_val = "median-cut"
        elif combo_val.startswith("palette:"):
            try:
                palette_arr = palette_library.load_named(combo_val[len("palette:"):]).colors
            except ValueError as e:
                QMessageBox.critical(self, "Palette Error", str(e))
                return
        elif combo_val == "Custom (unsaved)":
            if self.custom_palette is None:
                return
            palette_arr = self.custom_palette

        try:
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
        self.custom_palette: np.ndarray | None = None
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
        self.btn_restore.clicked.connect(self.run_restoration)
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
        """Define, import, or edit a palette — separate from picking a mode above."""
        dlg = PaletteManagerDialog(self)
        accepted = dlg.exec() == QDialog.Accepted
        if dlg.library_changed:
            self._refresh_palette_combo()
        if not accepted:
            return
        pal = dlg.get_enabled_palette_float32()
        if pal is None or len(pal) == 0:
            return
        self.custom_palette = pal
        if self.combo_palette_mode.findText("Custom (unsaved)") < 0:
            self.combo_palette_mode.addItem("Custom (unsaved)")
        self.combo_palette_mode.setCurrentText("Custom (unsaved)")

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

    def _apply_palette_snap(self, raw_output: np.ndarray, source_resized: np.ndarray) -> np.ndarray:
        """Snap the VQ-GAN's raw continuous output to a palette. Defaults to a palette
        extracted from the ORIGINAL (pre-restoration) source image rather than the
        model's own output — see the palette combo comment above for why."""
        mode = self.combo_palette_mode.currentText()
        colors = self.spin_colors.value()

        if mode == "None (raw output)":
            return raw_output
        if mode == "Source image (k-means)":
            palette = extract_palette_kmeans(source_resized, k=colors)
        elif mode == "Restored output (k-means)":
            palette = extract_palette_kmeans(raw_output, k=colors)
        else:
            palette = self._explicit_palette_or_none()
            if palette is None:
                return raw_output

        return nearest_neighbor_snap(raw_output, palette)

    def run_restoration(self):
        if self.current_img is None:
            QMessageBox.warning(self, "Warning", "Please load an input image first!")
            return
        engine = self.combo_engine.currentText()
        try:
            from spriteforge.core.resize import resize_to_target
            if engine == "Palette UNet (E1)":
                self._run_e1_restoration()
            else:
                self._run_vqgan_restoration()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Restoration failed:\n{e}")

    def _run_vqgan_restoration(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "Warning", "Please load a VQ-GAN checkpoint first!")
            return
        from spriteforge.core.resize import resize_to_target
        target_size = self.model.config.target_size
        src = remove_background_flood(self.current_img) if self.chk_remove_bg.isChecked() else self.current_img
        resized_in = resize_to_target(src, target_size=target_size)
        tensor_in = torch.from_numpy(resized_in).permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            out_tensor, _, _ = self.model(tensor_in)
        raw_output = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        self.restored_sprite_raw = raw_output
        self.restored_sprite = self._apply_palette_snap(raw_output, resized_in)
        pix = numpy_to_qpixmap(self.restored_sprite, 320)
        self.lbl_out_preview.setPixmap(pix)
        self.lbl_out_preview.setText("")

    def _run_e1_restoration(self) -> None:
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
        # A user-selected palette conditions the model; None falls back to the
        # per-source k-means default. restore_sprite adapts any palette width to the
        # model's trained K via _fit_palette_width, so a sub-selected subset works.
        palette = self._explicit_palette_or_none()
        out = restore_sprite(self.e1_model, num_colors, img_rgba, palette=palette)
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


class CalibrationTab(QWidget):
    """Tab 3: Evaluate — calibration grid and dataset verification."""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        self.btn_gen_grid = QPushButton("🔬 Generate Calibration Grid...")
        self.btn_gen_grid.setObjectName("primaryButton")
        self.btn_gen_grid.clicked.connect(self.generate_grid)
        top_bar.addWidget(self.btn_gen_grid)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.lbl_grid_preview = QLabel("Click button above to select a clean sprite and generate a 14-primitive degradation grid.")
        self.lbl_grid_preview.setObjectName("previewLabel")
        self.lbl_grid_preview.setAlignment(Qt.AlignCenter)
        self.lbl_grid_preview.setMinimumSize(600, 300)
        layout.addWidget(self.lbl_grid_preview, 1)

    def generate_grid(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Clean Sprite", "data_private", "PNG Images (*.png)")
        if file_path:
            try:
                img = load_image_float32(file_path)
                from spriteforge.core.resize import resize_to_target
                real_downscaled = resize_to_target(img, target_size=32, method="area")
                
                rng = np.random.default_rng(seed=42)
                ranges = DegradeRanges()
                samples = [real_downscaled]
                for _ in range(5):
                    samples.append(degrade(real_downscaled, rng=rng, ranges=ranges))
                
                grid = np.concatenate(samples, axis=1)
                pix = numpy_to_qpixmap(grid, scale_size=640)
                self.lbl_grid_preview.setPixmap(pix)
                self.lbl_grid_preview.setText("")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Grid generation failed:\n{e}")


class SpriteforgeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spriteforge — Retro 2D Sprite Generator & AI Studio")
        self.resize(1000, 700)
        self.setStyleSheet(DARK_THEME_QSS)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header title
        header = QLabel("👾 Spriteforge — Pixel Art Sprite Studio")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #6C5CE7; margin-bottom: 10px;")
        layout.addWidget(header)

        tabs = QTabWidget()
        tabs.addTab(StageAStudioTab(), "🎨 Convert")
        tabs.addTab(StageBStudioTab(), "🧠 Restore")
        tabs.addTab(CalibrationTab(), "🔬 Evaluate")
        layout.addWidget(tabs)

        self.setCentralWidget(central_widget)


def run_app():
    app = QApplication(sys.argv)
    window = SpriteforgeMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_app()
