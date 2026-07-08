"""
水质数据串口采集工具
- 通过串口发送启动指令 (AA AA BB BB)，接收 JSON 数据
- 用户补充实测 COD/UV254
- 自动追加到 Excel 表，便于后续训练
"""

import json
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd
import serial
import serial.tools.list_ports

# ======================== 配置 ========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL = BASE_DIR / 'collected_data.xlsx'
DEFAULT_BAUDRATE = 115200
START_CMD = bytes([0xAA, 0xAA, 0xBB, 0xBB])  # 启动指令
TIMEOUT_SEC = 10  # 接收超时（秒）

# Excel 列顺序（与训练数据格式一致）
EXCEL_COLUMNS = [
    'timestamp', 'device_id',
    'base_550', 'base_254',
    'comp_550', 'comp_254',
    'raw_550', 'raw_254',
    'Temp',
    'COD_lab', 'UV254_lab',         # 用户实测的 COD/UV254
]


class SerialCollectorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('水质数据串口采集系统')
        self.root.geometry('720x640')
        self.root.resizable(True, True)

        self.ser: serial.Serial | None = None
        self.received_data: dict | None = None
        self.running = False

        self._build_ui()
        self._refresh_ports()

    # ──────────────────── UI 搭建 ────────────────────
    def _build_ui(self):
        # 样式
        style = ttk.Style()
        style.theme_use('clam')

        pad = {'padx': 5, 'pady': 3}

        # ── 串口配置区 ──
        cfg_frame = ttk.LabelFrame(self.root, text='串口配置')
        cfg_frame.pack(fill=tk.X, **pad)

        row1 = ttk.Frame(cfg_frame)
        row1.pack(fill=tk.X, **pad)
        ttk.Label(row1, text='端口:').pack(side=tk.LEFT)
        self.port_var = tk.StringVar()
        self.port_cb = ttk.Combobox(row1, textvariable=self.port_var, width=12, state='readonly')
        self.port_cb.pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text='刷新', command=self._refresh_ports, width=6).pack(side=tk.LEFT)

        ttk.Label(row1, text='  波特率:').pack(side=tk.LEFT)
        self.baud_var = tk.StringVar(value=str(DEFAULT_BAUDRATE))
        baud_cb = ttk.Combobox(row1, textvariable=self.baud_var, width=10,
                               values=['9600', '19200', '38400', '57600', '115200', '230400', '460800'])
        baud_cb.pack(side=tk.LEFT, padx=5)

        ttk.Label(row1, text='  超时(s):').pack(side=tk.LEFT)
        self.timeout_var = tk.StringVar(value=str(TIMEOUT_SEC))
        ttk.Entry(row1, textvariable=self.timeout_var, width=5).pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(cfg_frame)
        row2.pack(fill=tk.X, **pad)
        self.btn_open = ttk.Button(row2, text='打开串口', command=self._toggle_serial)
        self.btn_open.pack(side=tk.LEFT, padx=5)
        self.btn_test = ttk.Button(row2, text='⚡ 开始测试', command=self._start_test, state=tk.DISABLED)
        self.btn_test.pack(side=tk.LEFT, padx=5)

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value='请先打开串口')
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # ── 接收数据显示区 ──
        data_frame = ttk.LabelFrame(self.root, text='传感器接收数据')
        data_frame.pack(fill=tk.X, **pad)

        self.data_text = tk.Text(data_frame, height=10, width=85, font=('Consolas', 10))
        self.data_text.pack(**pad)

        # ── 实测数据输入区 ──
        input_frame = ttk.LabelFrame(self.root, text='实验室实测数据')
        input_frame.pack(fill=tk.X, **pad)

        ir = ttk.Frame(input_frame)
        ir.pack(fill=tk.X, **pad)
        ttk.Label(ir, text='COD (mg/L):').pack(side=tk.LEFT)
        self.cod_var = tk.StringVar()
        ttk.Entry(ir, textvariable=self.cod_var, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(ir, text='  UV254 (cm⁻¹):').pack(side=tk.LEFT)
        self.uv254_var = tk.StringVar()
        ttk.Entry(ir, textvariable=self.uv254_var, width=12).pack(side=tk.LEFT, padx=5)
        self.btn_save = ttk.Button(ir, text='💾 保存到 Excel', command=self._save_to_excel, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=15)

        # ── 历史记录表 ──
        hist_frame = ttk.LabelFrame(self.root, text='历史记录（最近 20 条）')
        hist_frame.pack(fill=tk.BOTH, expand=True, **pad)

        columns = ('时间', 'COD实测', 'UV254实测', '温度')
        self.tree = ttk.Treeview(hist_frame, columns=columns, show='headings', height=8)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=105, anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self._refresh_history()

    # ──────────────────── 串口操作 ────────────────────
    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_cb['values'] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        if not ports:
            self.port_var.set('')
            self._set_status('未检测到串口')

    def _toggle_serial(self):
        if self.ser and self.ser.is_open:
            self._close_serial()
        else:
            self._open_serial()

    def _open_serial(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning('提示', '请选择串口')
            return
        try:
            baud = int(self.baud_var.get())
            timeout = int(self.timeout_var.get())
            # 显式设置 DTR/RTS，避免部分 STM32 虚拟串口被误复位
            self.ser = serial.Serial(
                port, baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout,
                write_timeout=1,
            )
            self.ser.dtr = False
            self.ser.rts = False
            self.btn_open.config(text='关闭串口')
            self.btn_test.config(state=tk.NORMAL)
            self._set_status(f'串口 {port} 已打开 ({baud}bps, 8N1)')
        except PermissionError:
            messagebox.showerror(
                '串口被占用',
                f'无法打开 {port}：端口被其他程序占用。\n\n'
                '请关闭以下可能占用串口的程序后重试：\n'
                '  • SSCOM / 串口调试助手\n'
                '  • STM32CubeProgrammer / ST-LINK Utility\n'
                '  • Arduino IDE 串口监视器\n'
                '  • VS Code 调试器（Serial Monitor 面板）\n'
                '  • 其他 Python 进程'
            )
        except Exception as e:
            messagebox.showerror('串口错误', f'无法打开串口:\n{e}')

    def _close_serial(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = None
        self.btn_open.config(text='打开串口')
        self.btn_test.config(state=tk.DISABLED)
        self._set_status('串口已关闭')

    def _set_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ──────────────────── 数据采集流程 ────────────────────
    def _start_test(self):
        if not self.ser or not self.ser.is_open:
            messagebox.showwarning('提示', '请先打开串口')
            return

        self.btn_test.config(state=tk.DISABLED)
        self.btn_open.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.data_text.delete('1.0', tk.END)
        self.received_data = None

        self._set_status('正在发送启动指令 AA AA BB BB ...')
        threading.Thread(target=self._test_worker, daemon=True).start()

    def _test_worker(self):
        try:
            # 1) 清空缓冲后发送启动指令（HEX 字节，非 ASCII）
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.write(START_CMD)
            self.ser.flush()  # 确保数据完全发出
            self._log(f'>>> 已发送启动指令 (HEX): {" ".join(f"{b:02X}" for b in START_CMD)}')
            self._log(f'    (共 {len(START_CMD)} 字节, 二进制模式)')

            # 给 STM32 一点时间处理指令
            time.sleep(0.1)

            # 2) 等待接收数据
            self._set_status('等待接收数据...')
            timeout = int(self.timeout_var.get())
            raw = self._read_until_json(timeout)

            if raw is None:
                self._log('❌ 接收超时，未收到有效 JSON 数据')
                self._set_status('接收失败：超时')
                self._enable_ui_after_test()
                return

            # 3) 解析 JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                self._log(f'❌ JSON 解析失败，原始数据:\n{raw}')
                self._set_status('接收失败：JSON 格式错误')
                self._enable_ui_after_test()
                return

            self.received_data = data
            self._log(f'✅ 接收成功 ({len(raw)} 字节)')
            self._log(json.dumps(data, indent=2, ensure_ascii=False))

            # 4) 显示解析结果
            self._display_parsed(data)

            self._set_status('✅ 数据接收完成，请输入实测 COD/UV254 后保存')
            self.root.after(0, lambda: self.btn_save.config(state=tk.NORMAL))

        except Exception as e:
            self._log(f'❌ 错误: {e}')
            self._set_status(f'错误: {e}')
        finally:
            self._enable_ui_after_test()

    def _read_until_json(self, timeout: float) -> str | None:
        """从串口读取数据直到收到完整 JSON，或超时

        部分 Windows 虚拟串口驱动 (如 STM32 VCP) 的 in_waiting
        可能始终返回 0，此时降级为逐字节阻塞读取。
        """
        deadline = time.time() + timeout
        buf_bytes = b''
        used_fallback = False

        while time.time() < deadline:
            if self.ser.in_waiting > 0:
                chunk = self.ser.read(self.ser.in_waiting)
                buf_bytes += chunk
            else:
                # in_waiting 不可靠时的降级方案：逐字节读取
                used_fallback = True
                try:
                    byte = self.ser.read(1)
                    if byte:
                        buf_bytes += byte
                    else:
                        time.sleep(0.01)
                        continue
                except serial.SerialTimeoutException:
                    time.sleep(0.01)
                    continue

            # 尝试解码并提取 JSON
            text = buf_bytes.decode('utf-8', errors='replace')
            json_str = self._extract_json(text)
            if json_str:
                return json_str

            time.sleep(0.01)

        # 超时后输出调试信息
        if buf_bytes:
            text = buf_bytes.decode('utf-8', errors='replace')
            self._log(f'⚠ 超时前共收到 {len(buf_bytes)} 字节原始数据:')
            hex_str = ' '.join(f'{b:02X}' for b in buf_bytes[:128])
            self._log(f'  HEX: {hex_str}')
            self._log(f'  文本: {repr(text[:200])}')
            if used_fallback:
                self._log(f'  (使用了逐字节读取降级方案)')
        else:
            self._log('⚠ 未收到任何数据（串口无响应）')
            self._log('  可能原因: 1) STM32 未连接  2) TX/RX 接反  3) 波特率不匹配')
        return None

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """从文本中提取第一个完整 JSON 对象"""
        # 找第一个 { 和匹配的 }
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None

    def _display_parsed(self, data: dict):
        """在文本区整理显示关键字段"""
        self.data_text.insert(tk.END, '\n── 解析结果 ──\n')
        fields = [
            ('base_550', '基准 550'), ('base_254', '基准 254'),
            ('comp_550', '补偿 550'), ('comp_254', '补偿 254'),
            ('raw_550', '原始 550'), ('raw_254', '原始 254'),
            ('Temp', '温度 ℃'), ('COD', 'COD 传感器'), ('UV254', 'UV254 传感器'),
            ('device_id', '设备 ID'), ('status', '状态'),
        ]
        for key, label in fields:
            if key in data:
                self.data_text.insert(tk.END, f'  {label:12s}: {data[key]}\n')
        self.data_text.see(tk.END)

    def _enable_ui_after_test(self):
        self.root.after(0, lambda: self.btn_open.config(state=tk.NORMAL))
        self.root.after(0, lambda: self.btn_test.config(state=tk.NORMAL))

    # ──────────────────── 数据保存 ────────────────────
    def _save_to_excel(self):
        if self.received_data is None:
            messagebox.showwarning('提示', '请先完成测试接收数据')
            return

        # 校验用户输入
        try:
            cod_lab = float(self.cod_var.get().strip())
            uv254_lab = float(self.uv254_var.get().strip())
        except ValueError:
            messagebox.showwarning('输入错误', '请输入有效的 COD 和 UV254 数值')
            return

        d = self.received_data
        row = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device_id': d.get('device_id', ''),
            'base_550': d.get('base_550', ''),
            'base_254': d.get('base_254', ''),
            'comp_550': d.get('comp_550', ''),
            'comp_254': d.get('comp_254', ''),
            'raw_550': d.get('raw_550', ''),
            'raw_254': d.get('raw_254', ''),
            'Temp': d.get('Temp', ''),
            'COD_lab': cod_lab,
            'UV254_lab': uv254_lab,
        }

        try:
            self._append_to_excel(row)
            self._log(f'\n💾 已保存 → {DEFAULT_EXCEL.name}')
            self._log(f'   COD实测={cod_lab}, UV254实测={uv254_lab}')
            self._set_status(f'已保存: COD={cod_lab}, UV254={uv254_lab}')

            # 清空输入，准备下一次
            self.cod_var.set('')
            self.uv254_var.set('')
            self.btn_save.config(state=tk.DISABLED)
            self.received_data = None

            self._refresh_history()
            messagebox.showinfo('保存成功', f'数据已追加到 {DEFAULT_EXCEL.name}')
        except Exception as e:
            messagebox.showerror('保存失败', str(e))

    def _append_to_excel(self, row: dict):
        """追加一行数据到 Excel 文件"""
        df_new = pd.DataFrame([row], columns=EXCEL_COLUMNS)

        if DEFAULT_EXCEL.exists():
            df_old = pd.read_excel(DEFAULT_EXCEL)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_excel(DEFAULT_EXCEL, index=False)

    # ──────────────────── 日志与历史 ────────────────────
    def _log(self, msg: str):
        """向文本区追加日志（线程安全）"""

        def _append():
            self.data_text.insert(tk.END, msg + '\n')
            self.data_text.see(tk.END)

        self.root.after(0, _append)

    def _refresh_history(self):
        """刷新历史记录表格"""
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not DEFAULT_EXCEL.exists():
            return

        try:
            df = pd.read_excel(DEFAULT_EXCEL)
            # 取最近 20 条
            for _, row in df.tail(20).iloc[::-1].iterrows():
                values = (
                    str(row.get('timestamp', ''))[:19],
                    row.get('COD_lab', ''),
                    row.get('UV254_lab', ''),
                    row.get('Temp', ''),
                )
                self.tree.insert('', tk.END, values=values)
        except Exception:
            pass

    def on_close(self):
        self._close_serial()
        self.root.destroy()


# ======================== 入口 ========================
if __name__ == '__main__':
    root = tk.Tk()
    app = SerialCollectorApp(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()
