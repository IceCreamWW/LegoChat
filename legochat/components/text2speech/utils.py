import re

def extract_tts_text(text):
    punctuation = r"[，。！？,.!?:：；;；、\n\t\r•]"
    for i in range(len(text), 10, -1):
        prefix = text[:i]
        if re.search(punctuation + r"$", prefix) and len(prefix) > 10:
            return prefix, text[i:]
    return "", text

# def extract_tts_text(text, min_chunk_length=10):
#     soft_punctuation = [
#         "，",",",";","；",'、',

#     ]
#     solid_punctuation = [
#         "。","！","？",".","!","?","：",":","\n","\t","\r","•","/","|",
#     ]
#     def in_neibor(text, neibor_nums):
#         for i in range(neibor_nums):
#             if text[i] in solid_punctuation:
#                 return True
#         return False
    
#     tts_text = ""
#     remain_text = text
#     for i in range(len(text)):
#         if text[i] in solid_punctuation or (text[i] in soft_punctuation and i>min_chunk_length and not in_neibor(text[i+1:], min_chunk_length)):
#             tts_text = text[:i+1]
#             remain_text = text[i+1:]
#             break

#     print(f"Extracted TTS text: {tts_text}, Remaining text: {remain_text}")
#     return tts_text, remain_text

PRIORITY = [
    ("\n\r", 0, False),
    (":：\t", 10, False),
    (".", 10, True),
    ("；;。!！？?", 10, False),
    ("，,、", 10, False),
]

def extract_tts_text_v1(text: str):
    # text = text.strip()

    try:
        for p, min_len, check_numel in PRIORITY:
            tts_text, text = sub_extract(text, p, min_len, check_numel)
            # print("Extracted TTS text:", tts_text.__repr__(), "Remaining text:", text.__repr__())
            if tts_text:
                return tts_text, text
        return "", text
    except Exception as e:
        print("Error in extract_tts_text:", e)
        raise e
        return extract_tts_text_v0(text)

def sub_extract(text, p, min_len=10, check_numel=False):
    # print(text.__repr__(), p, min_len, check_numel)
    if check_numel:
        punctuation = rf"(?<!\d)[{re.escape(p)}]"
    else:
        punctuation = rf"[{re.escape(p)}]"

    parts = re.split(rf"(?<={punctuation})", text)
    parts = [p for p in parts if p]

    if parts and not re.match(rf".*{punctuation}$", parts[-1]):
        complete_parts = parts[:-1]
        remainder = parts[-1]
    else:
        complete_parts = parts
        remainder = ""

    merged_parts = [""]
    for p in complete_parts:
        if len(merged_parts[-1]) < min_len:
            merged_parts[-1] += p
        else:
            merged_parts.append(p)

    merged_parts = [p for p in merged_parts if p]
    if len(merged_parts) and len(merged_parts[-1]) < min_len:
        remainder = merged_parts[-1] + remainder
        merged_parts = merged_parts[:-1]
    
    # print(f"Extracted TTS text: {merged_parts}, Remaining text: {remainder}")
    # return "".join(merged_parts), remainder
    if len(merged_parts) == 0:
        return "", text
    return merged_parts[0], "".join(merged_parts[1:]) + remainder

# def extract_tts_text(text: str):
#     parts = re.split(r"(?<=[.,!?。，！？、\t\r\n•:])", text)
#     parts = [p for p in parts if p]

#     if parts and not re.match(r".*[.,!?。，！？、\t\r\n•]$", parts[-1]):
#         complete_parts = parts[:-1]
#         remainder = parts[-1]
#     else:
#         complete_parts = parts
#         remainder = ""

#     merged_parts = [""]
#     for p in complete_parts:
#         if len(merged_parts[-1]) < 20:
#             merged_parts[-1] += p
#         else:
#             merged_parts.append(p)

#     merged_parts = [p for p in merged_parts if p]
#     if len(merged_parts) and len(merged_parts[-1]) < 20:
#         remainder = merged_parts[-1] + remainder
#         merged_parts = merged_parts[:-1]
    
#     print(f"Extracted TTS text: {merged_parts}, Remaining text: {remainder}")
#     return "".join(merged_parts), remainder

if __name__ == '__main__':
    z = """
1. cat - 猫
2. dog - 狗
3. bird - 鸟
4. book - 书
5. pencil - 铅笔
6. friend - 朋友
7. teacher - 老师
8. school - 学校
9. apple - 苹果
10. family - 家庭
"""

    for _ in range(10):
        ttt, z = extract_tts_text(z)
        print(ttt)