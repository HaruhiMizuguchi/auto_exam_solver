import json, tkinter as tk
from tkinter import messagebox
from ui_capture import capture_region_to_image
from ocr_solver import solve_from_image


def run_flow():
    img = capture_region_to_image()
    if img is None:
        messagebox.showinfo('auto_exam_solver', 'キャンセルしました')
        return
    try:
        result_json = solve_from_image(img)
        result = json.loads(result_json)
        msg = f"回答: {result.get('answer_label')}

根拠:
{result.get('reason')}"
        messagebox.showinfo('auto_exam_solver', msg)
    except Exception as e:
        messagebox.showerror('auto_exam_solver', f'エラー: {e}')


if __name__ == '__main__':
    root = tk.Tk()
    root.title('auto_exam_solver')
    root.geometry('260x120')
    btn = tk.Button(root, text='画面の問題を解く（範囲選択）', command=run_flow)
    btn.pack(expand=True, fill='both', padx=12, pady=12)
    root.mainloop()
