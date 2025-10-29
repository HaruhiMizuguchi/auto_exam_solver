# ocr_solver.py
import os, re, pytesseract
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OCR_LANG = os.getenv('OCR_LANG', 'eng')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Windows で Tesseract のパスが必要なら設定:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(img: Image.Image) -> str:
    # 解像度を上げると精度が上がることが多い
    scale = 1.5
    w, h = img.size
    img = img.resize((int(w*scale), int(h*scale)))
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    return text

def parse_question_choices(text: str):
    # よくあるフォーマットを包括的に拾う（ア/イ/ウ/エ、A/B/C/D、1)/2)など）
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    choice_pat = re.compile(r'^((?:[ア-エ]|[A-D]|[1-9]|①|②|③|④|a|b|c|d)[\)|．\.\s])')
    question_lines, choice_lines = [], []
    hit_choice = False
    for ln in lines:
        if choice_pat.match(ln):
            hit_choice = True
            choice_lines.append(ln)
        else:
            (choice_lines if hit_choice else question_lines).append(ln)

    # 選択肢行を「ラベル:本文」に分割
    choices = []
    for ln in choice_lines:
        m = choice_pat.match(ln)
        if m:
            label = m.group(1).strip()
            label = label.strip(').．. ').upper()
            body = ln[m.end():].strip()
            choices.append((label, body))
        else:
            # 前行の続きとして結合
            if choices:
                choices[-1] = (choices[-1][0], (choices[-1][1] + ' ' + ln).strip())

    question = '\n'.join(question_lines).strip()
    return question, choices

def ask_llm(question: str, choices: list[tuple[str,str]]):
    client = OpenAI(api_key=OPENAI_API_KEY)
    choice_text = '\n'.join([f"{k}. {v}" for k,v in choices]) if choices else '(選択肢なし)'
    system = 'あなたは厳密で説明力の高い試験対策アシスタントです。根拠を筋道立てて示し、最終的に一つの回答ラベルをJSONで返してください。'
    user = f"""問題:
{question}

選択肢:
{choice_text}

出力フォーマット:
```json
{{
  "answer_label": "<ラベル(例: A/ア/1など)>",
  "reason": "<100〜300字で根拠>"
}}
```"""

    resp = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[{'role':'system','content':system},
                  {'role':'user','content':user}],
        response_format={'type':'json_object'},
        temperature=0.2,
    )
    return resp.choices[0].message.content  # JSON文字列

def solve_from_image(img: Image.Image):
    text = ocr_image(img)
    q, ch = parse_question_choices(text)
    return ask_llm(q, ch)