import os, re, json, pytesseract
from PIL import Image
from dotenv import load_dotenv

from openai import OpenAI as OpenAIClient
import google.generativeai as genai

load_dotenv()
OCR_LANG = os.getenv('OCR_LANG', 'eng')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
DEFAULT_CONFIG = {
    'llm_provider': 'openai',
    'openai_model': 'gpt-4.1-mini',
    'gemini_model': 'gemini-1.5-flash'
}

def load_config():
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return {**DEFAULT_CONFIG, **cfg}
    except Exception:
        return DEFAULT_CONFIG


def ocr_image(img: Image.Image) -> str:
    scale = 1.5
    w, h = img.size
    img = img.resize((int(w*scale), int(h*scale)))
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    return text


def parse_question_choices(text: str):
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

    choices = []
    for ln in choice_lines:
        m = choice_pat.match(ln)
        if m:
            label = m.group(1).strip()
            label = label.strip(').．. ').upper()
            body = ln[m.end():].strip()
            choices.append((label, body))
        else:
            if choices:
                choices[-1] = (choices[-1][0], (choices[-1][1] + ' ' + ln).strip())

    question = '\n'.join(question_lines).strip()
    return question, choices


def _ask_openai(model: str, question: str, choices: list[tuple[str,str]]):
    client = OpenAIClient(api_key=OPENAI_API_KEY)
    choice_text = '\n'.join([f"{k}. {v}" for k,v in choices]) if choices else '(選択肢なし)'
    system = 'あなたは厳密で説明力の高い試験対策アシスタントです。根拠を筋道立てて示し、最終的に一つの回答ラベルをJSONで返してください。'
    user = f"""問題:\n{question}\n\n選択肢:\n{choice_text}\n\n出力フォーマット:\n```json\n{{\n  \"answer_label\": \"<ラベル(例: A/ア/1など)>\",\n  \"reason\": \"<100〜300字で根拠>\"\n}}\n```\n"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{'role':'system','content':system},
                  {'role':'user','content':user}],
        response_format={'type':'json_object'},
        temperature=0.2,
    )
    return resp.choices[0].message.content


def _ask_gemini(model: str, question: str, choices: list[tuple[str,str]]):
    genai.configure(api_key=GEMINI_API_KEY)
    m = genai.GenerativeModel(model)
    choice_text = '\n'.join([f"{k}. {v}" for k,v in choices]) if choices else '(選択肢なし)'
    prompt = f"""あなたは厳密で説明力の高い試験対策アシスタントです。\n次の問題に対し、根拠を筋道立てて示したうえで、最終回答を JSON で返してください。\n\n問題:\n{question}\n\n選択肢:\n{choice_text}\n\n出力フォーマット(厳守):\n{{\n  \"answer_label\": \"<ラベル>\",\n  \"reason\": \"<100〜300字>\"\n}}\n（JSONのみを出力。前後の文章やコードブロックは不要。）\n"""
    resp = m.generate_content(prompt, generation_config={
        'temperature': 0.2,
        'max_output_tokens': 512,
        'response_mime_type': 'application/json'
    })
    return resp.text


def ask_llm(question: str, choices: list[tuple[str,str]]):
    cfg = load_config()
    provider = cfg.get('llm_provider', 'openai').lower()
    if provider == 'openai':
        model = cfg.get('openai_model', 'gpt-4.1-mini')
        return _ask_openai(model, question, choices)
    elif provider == 'gemini':
        model = cfg.get('gemini_model', 'gemini-1.5-flash')
        return _ask_gemini(model, question, choices)
    else:
        raise ValueError(f'Unsupported llm_provider: {provider}')


def solve_from_image(img: Image.Image):
    text = ocr_image(img)
    q, ch = parse_question_choices(text)
    return ask_llm(q, ch)
