"""Camera Kelvin controls and the source-space retouch workspace (2.1.0: AI removal)."""
import copy
import time
from pathlib import Path
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QCheckBox, QListWidget, QLabel
from .widgets import AdjustSlider
from . import host
from .i18n import tr

#: Tools of the retouch panel, in the order of its menu (``update_tool`` maps them to the canvas).
RETOUCH_TOOLS = ('heal', 'clone', 'remove', 'object')


class RevisionMixin:
    def build_camera_wb(self, layout):
        self.camera_wb_label = QLabel(tr('打开 RAW 后读取相机白平衡'))
        self.camera_wb_label.setWordWrap(True)
        self.camera_wb_label.setObjectName('subtle')
        layout.addWidget(self.camera_wb_label)
        self.kelvin = AdjustSlider(tr('色温  K'), 1500, 25000, 10)
        self.kelvin.changed.connect(self.kelvin_changed)
        self.kelvin.committed.connect(self.commit)
        layout.addWidget(self.kelvin)

    def kelvin_changed(self, value):
        if not self.refreshing and self.edits['white_balance']['camera_kelvin'] is not None:
            self.edits['white_balance']['kelvin'] = value
            self.changed()

    def refresh_camera_wb(self):
        settings = self.edits['white_balance']
        base = settings['camera_kelvin']
        self.kelvin.setEnabled(base is not None)
        self.kelvin.setValue(settings['kelvin'] or 5500)
        self.kelvin.default_value = base or 5500
        self.kelvin.title.setText(tr('色温  K') if base else tr('色温  K · 无可用基准'))
        if base is None:
            text = tr('未读取到可用的相机色温；可用相对冷暖和吸管调整。')
        elif settings['estimated']:
            text = tr('相机未记录有效 K 值 · 估算基准 ≈ {base:.0f} K\n原始通道增益已保留；估算值可能与 LR 不同。', base=base)
        else:
            text = tr('相机记录：{base:.0f} K · 在原始白平衡基础上调整', base=base)
        if self.edits['wb_gain'] != [1., 1., 1.]:
            text += tr('\n当前还应用了吸管取样校正。')
        self.camera_wb_label.setText(text)
        self.camera_wb_label.setToolTip(self.info.get('wb_warning', ''))

    def build_retouch(self):
        layout = self.panel(tr('修复'))
        title = QLabel(tr('修复与仿制  /  RETOUCH'))
        title.setObjectName('section')
        layout.addWidget(title)
        self.retouch_tool = QComboBox()
        self.retouch_tool.addItems([tr('污点修复 · 邻域修补'), tr('仿制图章 · 对齐取样'),
                                    tr('AI 移除 · 涂抹物体'), tr('AI 移除 · 点选物体')])
        self.retouch_tool.currentIndexChanged.connect(self.update_tool)
        self.retouch_tool.currentIndexChanged.connect(self.retouch_tool_changed)
        layout.addWidget(self.retouch_tool)
        self.retouch_tip = QLabel()
        self.retouch_tip.setWordWrap(True)
        self.retouch_tip.setObjectName('subtle')
        layout.addWidget(self.retouch_tip)
        self.retouch_controls = {}
        for key, title, lo, hi, default in [('size', tr('画笔直径 %'), .2, 20, 3),
              ('feather', tr('羽化'), 0, 100, 50), ('opacity', tr('不透明度'), 0, 100, 100)]:
            c = AdjustSlider(title, lo, hi, .2 if key == 'size' else 1)
            c.setValue(default)
            c.default_value = default
            c.changed.connect(self.update_tool)
            self.retouch_controls[key] = c
            layout.addWidget(c)
        self.retouch_hint = QLabel(tr('每个笔划都可单独隐藏、删除或撤销。'))
        self.retouch_hint.setWordWrap(True)
        self.retouch_hint.setObjectName('subtle')
        layout.addWidget(self.retouch_hint)
        self.retouch_list = QListWidget()
        self.retouch_list.setMaximumHeight(235)
        self.retouch_list.currentRowChanged.connect(self.select_retouch)
        layout.addWidget(self.retouch_list)
        self.retouch_enabled = QCheckBox(tr('启用选中笔划'))
        self.retouch_enabled.toggled.connect(self.toggle_retouch)
        layout.addWidget(self.retouch_enabled)
        row = QHBoxLayout()
        row.addWidget(self.button(tr('删除选中'), self.remove_retouch))
        row.addWidget(self.button(tr('清除取样点'), self.clear_clone_source))
        layout.addLayout(row)
        layout.addStretch()
        self.canvas.clone_sampled.connect(self.clone_sampled)
        self.removal_queue, self.removal_running, self.removal_overlay = [], False, None
        self.retouch_tool_changed()

    def retouch_tool_changed(self, *_):
        tool = RETOUCH_TOOLS[self.retouch_tool.currentIndex()]
        # A picked object brings its own outline: brush size and feather do not apply.
        for key in ('size', 'feather'):
            self.retouch_controls[key].setEnabled(tool != 'object')
        if tool in ('remove', 'object'):
            text = tr('AI 移除：涂抹要去掉的物体（多涂一圈边缘和影子），松开后由本地 LaMa 模型按周围画面补全。') if tool == 'remove' else \
                tr('AI 移除：单击要去掉的物体，自动识别它的轮廓后补全；识别不准时改用涂抹。')
            text += tr('\n结果在添加时计算并随工程保存（原图上约 1–5 秒），可隐藏、删除或撤销。')
        else:
            text = tr('污点修复：在灰尘、小污点上点击或涂抹。\n仿制图章：{OPTION} + 单击取样，再涂抹目标区域。\n取样位置随笔触对齐；重新 {OPTION} 取样可更换来源。', OPTION=host.OPTION)
        self.retouch_tip.setText(text + tr('\n{NAVIGATION}；修复在裁切与调色之前应用。', NAVIGATION=host.NAVIGATION))

    # 2.1.0: AI removal.  Each removal is filled on the original photograph on its own thread (not
    # the image job queue, so previews keep updating), one at a time, in the order they were made.
    # It is part of the work state (Activity.REMOVAL): closing, saving and every export wait for it.
    # A removal belongs to its photo: made before another photo was opened, it still lands there.

    def start_removal(self, stroke=None, picked=None):
        """Queue an AI removal: a painted ``stroke`` or ``picked = (point, alpha)``, an object mask
        on the preview."""
        from . import remove
        if self.source is None:
            return
        if not remove.available():
            return self.error(tr('缺少 AI 移除模型，请重新安装 LUMEN RAW。'))
        if len(self.edits['retouch']) + len(self.removal_queue) >= 500:
            self.statusBar().showMessage(tr('修复笔划达到 500 个上限，请删除不需要的笔划。'))
            return
        if stroke is not None:
            op = dict(kind='remove', points=copy.deepcopy(stroke['points']), radius=stroke['radius'],
                      feather=self.retouch_controls['feather'].spin.value())
            self.canvas.pending.append(stroke)
        else:
            point, alpha = picked
            op = dict(kind='remove', points=[list(point)], radius=.005, feather=0., picked=True, alpha=alpha)
        op.update(enabled=True, opacity=self.retouch_controls['opacity'].spin.value())
        self.removal_queue.append((self.source_path, op, stroke))
        self.canvas.update()
        if not self.removal_running:
            self.next_removal()

    def next_removal(self):
        from . import remove, lens, engine, viewport
        from .nl_panel import run_async, _Relay
        from .scheduler import Activity as A
        while self.removal_queue and self.removal_queue[0][0] != self.source_path and \
                self.removal_queue[0][0] not in self.documents:
            self.finish_removal(self.removal_queue.pop(0)[2])    # its photo left the 选片集
        if not self.removal_queue or self.closing:
            self.removal_running = False
            self.work.end(A.REMOVAL)
            return
        path, op, stroke = self.removal_queue[0]
        self.removal_running = True
        self.work.begin(A.REMOVAL, force=True)
        if not hasattr(self, 'removal_relay'):
            self.removal_relay = _Relay(self)
        if path == self.source_path and self.source is not None:
            if self.detail_source is None:
                self.detail_source = viewport.DetailSource(self.source_path)
            holder, edits = self.detail_source, copy.deepcopy(self.edits)
        else:
            holder, edits = viewport.DetailSource(path), copy.deepcopy(self.documents[path]['edits'])
        before, gains = edits['retouch'], engine._gain(edits)
        started = time.perf_counter()
        waiting = len(self.removal_queue) - 1
        self.statusBar().showMessage(tr('正在读取原图并 AI 移除…') + (tr('（另有 {count} 处排队）', count=waiting) if waiting else ''))
        def work():
            frame = lens.view(holder.level(0), edits)
            h, w = frame.shape[:2]
            if op.get('picked'):
                coverage, area = remove.object_coverage(op['alpha'], h, w)
            else:
                coverage, area = remove.brush_coverage(op, h, w)
            return remove.fill(frame, before, coverage, area, gains)
        def done():
            self.removal_queue.pop(0)
            self.finish_removal(stroke)
            self.next_removal()
        def ready(patch):
            if not self.closing:
                item = {k: v for k, v in op.items() if k != 'alpha'} | dict(patch=patch)
                self.add_removal(path, item, f'{time.perf_counter() - started:.1f}')
            done()
        def fail(error):
            if not self.closing:
                text = str(error).strip().splitlines()[-1][:160] if str(error).strip() else type(error).__name__
                self.statusBar().showMessage(tr('AI 移除失败：') + text)
            done()
        run_async(self.removal_relay, work, ready, fail)

    def add_removal(self, path, item, seconds):
        """A finished removal into its photo: the open one, or the one it was made on, kept in the
        选片集 with its undo history, when another photo was opened meanwhile."""
        from . import model
        if path == self.source_path and self.source is not None:
            if len(self.edits['retouch']) < 500:
                self.commit()
                self.edits['retouch'].append(item)
                self.refresh_retouch(len(self.edits['retouch']) - 1)
                self.changed()
                self.commit()
                self.statusBar().showMessage(tr('已 AI 移除 · {seconds} 秒 · 可隐藏、删除或撤销', seconds=seconds))
            return
        document = self.documents.get(path)
        if document is None or len(document['edits']['retouch']) >= 500:
            return
        before = copy.deepcopy(document['edits'])
        document['edits']['retouch'].append(item)
        document.setdefault('history', model.History(before)).push(document['edits'])
        film = self.film_item(path)
        if film is not None and not film.text().endswith(' •'):
            film.setText(film.text() + ' •')
        self.statusBar().showMessage(tr('已 AI 移除 {name} · {seconds} 秒 · 打开这张照片可隐藏、删除或撤销',
                                        name=Path(path).name, seconds=seconds))

    def finish_removal(self, stroke):
        if stroke is not None and stroke in self.canvas.pending:
            self.canvas.pending.remove(stroke)
        self.sync_pending_removals()

    def sync_pending_removals(self):
        """Pending strokes and the object being removed, drawn only over the photo they belong to."""
        mine = [item for item in self.removal_queue if item[0] == self.source_path]
        self.canvas.pending = [stroke for _, _, stroke in mine if stroke is not None]
        if not any(op.get('picked') for _, op, _ in mine):
            self.removal_overlay = None
        self.update_overlay()
        self.canvas.update()

    def removal_busy(self):
        """True (with a note in the status bar) while an AI removal is still being filled."""
        from .scheduler import Activity as A
        if self.work.active(A.REMOVAL):
            self.statusBar().showMessage(tr('正在 AI 移除，请稍候再试。'))
            return True
        return False

    def pick_removal(self, point):
        """A click with the object-removal tool: find the object on the preview, then remove it."""
        from . import engine, selection
        from .scheduler import Activity as A
        if self.source is None or not self.work.can_start(A.SELECTION):
            return
        self.work.begin(A.SELECTION)
        token, source, edits = self.document_token, self.source, copy.deepcopy(self.edits)
        key = (token, engine._key(edits.get('develop', {}), edits.get('lens'), edits.get('retouch'), engine.process_version(edits)))
        self.statusBar().showMessage(tr('正在识别物体…'))
        def work():
            rgb = engine.selection_view(source, edits)
            return selection.object_mask(rgb, [list(point) + [1]], key)
        def ready(alpha):
            self.work.end(A.SELECTION)
            if token != self.document_token:
                return
            if float((alpha > .5).mean()) < .00005:
                self.statusBar().showMessage(tr('没有识别到物体；可改用涂抹移除。'))
                return
            if float((alpha > .5).mean()) > .6:
                self.statusBar().showMessage(tr('识别到的区域超过画面的六成，未移除；请点在物体上，或改用涂抹移除。'))
                return
            self.removal_overlay = alpha
            self.update_overlay()
            self.start_removal(picked=(point, alpha))
        def fail(text):
            self.work.end(A.SELECTION)
            if token == self.document_token:
                self.error(tr('物体识别失败：') + text)
        self.job(work, ready, fail)

    def clear_clone_source(self):
        self.canvas.clone_source = self.canvas.clone_offset = None
        self.canvas.update()
        self.retouch_hint.setText(tr('{OPTION} + 单击设置新的图章取样点。', OPTION=host.OPTION))

    def clone_sampled(self, point):
        self.retouch_hint.setText(tr('已取样：在目标区域涂抹。') if point else tr('请先按住 {OPTION} 并单击画面，设置取样来源。', OPTION=host.OPTION))

    def add_retouch(self, stroke):
        if len(self.edits['retouch']) >= 500:
            self.statusBar().showMessage(tr('修复笔划达到 500 个上限，请删除不需要的笔划。'))
            return
        op = {k: copy.deepcopy(v) for k, v in stroke.items() if k != 'erase'}
        op.update(enabled=True, feather=self.retouch_controls['feather'].spin.value(),
                  opacity=self.retouch_controls['opacity'].spin.value())
        self.edits['retouch'].append(op)
        self.refresh_retouch(len(self.edits['retouch']) - 1)
        self.changed()
        self.commit()

    def refresh_retouch(self, row=None):
        if row is None:
            row = self.retouch_list.currentRow()
        self.retouch_list.blockSignals(True)
        self.retouch_list.clear()
        names = dict(heal=tr('污点修复'), clone=tr('仿制图章'), remove=tr('AI 移除'))
        self.retouch_list.addItems([f'{i + 1:02d}  {names[op["kind"]]}{tr(" · 物体") if op.get("picked") else ""}{"" if op["enabled"] else tr(" · 已隐藏")}'
                                  for i, op in enumerate(self.edits['retouch'])])
        self.retouch_list.setCurrentRow(min(row, len(self.edits['retouch']) - 1))
        self.retouch_list.blockSignals(False)
        self.select_retouch()

    def select_retouch(self, *_):
        i = self.retouch_list.currentRow()
        valid = 0 <= i < len(self.edits['retouch'])
        self.retouch_enabled.blockSignals(True)
        self.retouch_enabled.setEnabled(valid)
        self.retouch_enabled.setChecked(valid and self.edits['retouch'][i]['enabled'])
        self.retouch_enabled.blockSignals(False)

    def toggle_retouch(self, enabled):
        i = self.retouch_list.currentRow()
        if 0 <= i < len(self.edits['retouch']) and not self.refreshing:
            self.edits['retouch'][i]['enabled'] = enabled
            self.refresh_retouch(i)
            self.changed()
            self.commit()

    def remove_retouch(self):
        i = self.retouch_list.currentRow()
        if 0 <= i < len(self.edits['retouch']):
            self.edits['retouch'].pop(i)
            self.refresh_retouch(i)
            self.changed()
            self.commit()
