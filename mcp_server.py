import json, sys, base64
from io import BytesIO
from PIL import Image
from ocr_solver import solve_from_image


def write(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + '
')
    sys.stdout.flush()

write({
  'protocol': 'MCP/2025-06-18',
  'capabilities': {'tools': True},
  'tools': [{
    'name': 'solve_from_png_base64',
    'description': 'PNG(Base64)画像からOCR→LLMで回答と根拠を返す',
    'inputSchema': {
      'type': 'object',
      'properties': { 'png_base64': {'type': 'string'} },
      'required': ['png_base64']
    }
  }]
})

for line in sys.stdin:
    req = json.loads(line)
    if req.get('type') == 'tool_call' and req.get('name') == 'solve_from_png_base64':
        b64 = req['arguments']['png_base64']
        img = Image.open(BytesIO(base64.b64decode(b64)))
        out = solve_from_image(img)
        write({'type':'tool_result','content':out})
