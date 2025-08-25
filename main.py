import os
import json
import re
import logging
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from together import Together

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

app = FastAPI(
    title="AI 기획안 분석 서비스",
    description="기획안 이미지와 설명을 입력하면 AI가 분석 결과를 반환하는 API입니다.",
    version="FINAL"
)

class PlanInput(BaseModel):
    image_url: str = Field(..., min_length=1)
    description: str = Field(...)

MODEL_NAME = "meta-llama/Llama-Vision-Free"
TEMPERATURE = 0.2
MAX_TOKENS = 1200
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.getenv("TOGETHER_API_KEY")
if not API_KEY:
    raise SystemExit("ERROR: TOGETHER_API_KEY가 .env 파일에 설정되어 있지 않습니다.")
client = Together(api_key=API_KEY)

SYSTEM_MESSAGE_TEMPLATE = """너는 주어진 기획안을 분석하고 컨설팅하는 전문 기획자다. 이미지(기획안)와 사용자 설명을 기반으로 응답을 생성한다. 오직 아래의 정확한 JSON 형식으로만 응답해야 한다.

**응답 형식:**
{{"title": "AI가 생성한 행사명", "considerations":["핵심 주제: 구체적인 실행안 (예시 포함)","..."], "slogans":["..."], "user_evaluation":{{"positive_percentage":"NN%","negative_reasons":["..."]}}}}

**지침:**
1.  **title (행사명 생성):**
    * 이미지와 설명 내용을 종합하여 가장 적합하고 창의적인 행사명을 생성한다.
2.  **considerations (고려할 점):**
    * 각 항목은 반드시 **'핵심 주제: 구체적인 실행안 (상세 내용 포함)'** 형식이어야 한다.
    * 여기서 '핵심 주제'는 '지역 특성 연계'나 '홍보 방안'과 같은 아이디어의 핵심 키워드를 의미한다.
3.  **user_evaluation (사용자 예상 평가):**
    * 'negative_reasons' 항목은 기획안의 약점이나 현실적인 우려 사항을 1~2개 반드시 찾아내야 한다. 이 항목은 비워둘 수 없다.
    * 'positive_percentage' 항목은 위에서 지적한 'negative_reasons'의 심각성을 고려하여 최종 긍정 평가 비율을 산출한다. 70%, 80% 같은 상징적인 숫자를 피하고, 분석에 기반한 다양한 수치를 제공해야 한다.

**중요: 너의 출력물은 인사, 추가 설명 등 다른 텍스트를 제외하고 오직 JSON 객체 하나여야만 한다.**
"""

def extract_json_from_string(text: str) -> str:
    text = text.strip().replace('{{', '{').replace('}}', '}')
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json_str
    return text

def post_process_response(eval_json: dict, input_data: dict) -> dict:
    if not isinstance(eval_json, dict): return {"title": "N/A", "considerations": [], "slogans": [], "user_evaluation": {"positive_percentage": "N/A", "negative_reasons": ["응답 파싱 실패"]}}
    
    title = eval_json.get("title", "제목 생성 실패")
    unwanted_keywords = ["기획안 평가", "기획안"]
    for keyword in unwanted_keywords:
        title = title.replace(keyword, "")
    final_title = title.strip()

    cons = eval_json.get("considerations", [])
    cleaned_cons = []
    if isinstance(cons, list):
        temp_cons = []
        for item in cons:
            clean_item = str(item).strip()
            if clean_item.startswith("핵심 주제:"):
                clean_item = clean_item[len("핵심 주제:"):].lstrip()
            elif clean_item.startswith("주제:"):
                 clean_item = clean_item[len("주제:"):].lstrip()
            temp_cons.append(clean_item)
        cleaned_cons = list(dict.fromkeys(temp_cons))[:5]
    
    slogans = eval_json.get("slogans", [])
    cleaned_slogans = []
    if isinstance(slogans, list):
        cleaned_slogans = list(dict.fromkeys([str(s).strip().replace('"', '') for s in slogans if s and isinstance(s, str)]))[:3]
    
    evaluation = eval_json.get("user_evaluation", {})
    if isinstance(evaluation, dict):
        seed_text = input_data.get("image_url", "") + input_data.get("description", "")
        pseudo_random_int = sum(ord(c) for c in seed_text) % 21  
        base_score = 60 + pseudo_random_int 
        
        neg_reasons = evaluation.get("negative_reasons", [])
        cleaned_reasons = []
        if isinstance(neg_reasons, list):
            cleaned_reasons = list(dict.fromkeys([r.strip() for r in neg_reasons if r.strip()]))[:3]

        penalty = len(cleaned_reasons) * 7 
        final_score = base_score - penalty
        
        if final_score < 20: final_score = 20 

        evaluation["positive_percentage"] = f"{final_score}%"
        evaluation["negative_reasons"] = cleaned_reasons

    final_json = {
        "title": final_title,
        "considerations": cleaned_cons,
        "slogans": cleaned_slogans,
        "user_evaluation": evaluation
    }
    
    return final_json

def generate_ai_evaluation(input_data: dict):
    system_message = SYSTEM_MESSAGE_TEMPLATE
    user_text = (f"Analyze the event proposal based on the following information.\n\n"
                 f"Image URL: {input_data.get('image_url')}\n"
                 f"Description: {input_data.get('description')}\n\n"
                 "Please provide your analysis in the specified JSON format.")
    
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_text}]
    try:
        resp = client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
        return resp.choices[0].message.content or ""
    except Exception as e:
        logging.error(f"AI API 호출 중 오류 발생: {e}")
        raise HTTPException(status_code=503, detail="AI 서비스에 접근할 수 없습니다.")

@app.post("/evaluate-plan/")
async def evaluate_plan(plan_input: PlanInput):
    input_dict = plan_input.dict()
    raw_response = generate_ai_evaluation(input_dict)
    if not raw_response:
        raise HTTPException(status_code=504, detail="AI 모델로부터 유효한 응답을 받지 못했습니다.")
    
    json_str = extract_json_from_string(raw_response)
    
    try:
        eval_json = json.loads(json_str)
    except json.JSONDecodeError:
        logging.error(f"JSON 파싱 오류. 원본 응답: {raw_response}")
        raise HTTPException(status_code=500, detail="AI 모델 응답을 처리하는 중 오류가 발생했습니다.")
        
    final_evaluation = post_process_response(eval_json, input_dict)
    return final_evaluation

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI 기획안 분석 서비스가 실행 중입니다."}