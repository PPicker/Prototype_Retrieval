import base64
import os
from google import genai
from google.genai import types

def translate(input_text):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-2.0-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Can you translate Korean to English which will be used for CLIP text encoder??"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Okay, I understand. I will translate Korean text into English, focusing on clarity and capturing the core meaning for use with a CLIP text encoder.  I will avoid overly poetic or nuanced translations and prioritize accuracy and directness.

Please provide the Korean text you would like me to translate. I'm ready!
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""파란색 체크 무늬 오버핏 셔츠"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Blue checkered oversized shirt
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""검은색 스트레이트핏 데님"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Black straight-fit denim jeans
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""회색 셔츠"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Gray shirt
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""소매에 단추가 있는 오버핏 셔츠"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Oversized shirt with buttoned sleeves
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""Translate Kor to Eng for CLIP text Encoder"""),
        ],
    )

    text_list = []  # 모든 청크를 저장할 리스트

    for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
        text_list.append(chunk.text)  # 청크를 리스트에 추가

    full_text = ''.join(text_list)  # 모든 청크를 하나의 문자열로 합침
    return full_text



if __name__ == '__main__':
    print(translate('회색 오버핏 셔츠'))