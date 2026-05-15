import json
import ast
from pathlib import Path

from tqdm import tqdm

DEFAULT_JSONL = Path(__file__).resolve().parent / "data" / "english_pii_43k.jsonl"


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# =====================================================================
# 【全量 PII 映射字典】(保留英文标签，中文注释)
# =====================================================================
PII_CONFIG = {
    "PREFIX": ("PREFIX", False),
    "FIRSTNAME": ("FIRSTNAME", False),
    "LASTNAME": ("LASTNAME", False),
    "MIDDLENAME": ("MIDDLENAME", False),
    "AGE": ("AGE", False),
    "GENDER": ("GENDER", False),
    "SEX": ("SEX", False),
    "DOB": ("DOB", False),
    "EYECOLOR": ("EYECOLOR", False),
    "HEIGHT": ("HEIGHT", False),
    "PHONENUMBER": ("PHONENUMBER", True),
    "PHONEIMEI": ("PHONEIMEI", True),
    "MASKEDNUMBER": ("MASKEDNUMBER", True),
    "EMAIL": ("EMAIL", False), 
    "CITY": ("CITY", False),
    "STATE": ("STATE", False),
    "COUNTY": ("COUNTY", False),
    "STREET": ("STREET", False),
    "SECONDARYADDRESS": ("SECONDARYADDRESS", False),
    "BUILDINGNUMBER": ("BUILDINGNUMBER", False),
    "ZIPCODE": ("ZIPCODE", True),
    "ORDINALDIRECTION": ("ORDINALDIRECTION", False),
    "NEARBYGPSCOORDINATE": ("NEARBYGPSCOORDINATE", False),
    "COMPANYNAME": ("COMPANYNAME", False),
    "JOBTITLE": ("JOBTITLE", False),
    "JOBTYPE": ("JOBTYPE", False),
    "JOBAREA": ("JOBAREA", False),
    "USERNAME": ("USERNAME", False),
    "PASSWORD": ("PASSWORD", True),
    "PIN": ("PIN", True),
    "URL": ("URL", False),
    "IP": ("IP", False),
    "IPV4": ("IPV4", False),
    "IPV6": ("IPV6", True),
    "MAC": ("MAC", True),
    "USERAGENT": ("USERAGENT", False),
    "ACCOUNTNAME": ("ACCOUNTNAME", False),
    "ACCOUNTNUMBER": ("ACCOUNTNUMBER", True),
    "IBAN": ("IBAN", True),
    "BIC": ("BIC", False),
    "CREDITCARDISSUER": ("CREDITCARDISSUER", False),
    "CREDITCARDNUMBER": ("CREDITCARDNUMBER", True),
    "CREDITCARDCVV": ("CREDITCARDCVV", True),
    "AMOUNT": ("AMOUNT", False),
    "CURRENCY": ("CURRENCY", False),
    "CURRENCYNAME": ("CURRENCYNAME", False),
    "CURRENCYCODE": ("CURRENCYCODE", False),
    "CURRENCYSYMBOL": ("CURRENCYSYMBOL", False),
    "ETHEREUMADDRESS": ("ETHEREUMADDRESS", True),
    "BITCOINADDRESS": ("BITCOINADDRESS", True),
    "LITECOINADDRESS": ("LITECOINADDRESS", True),
    "SSN": ("SSN", True),
    "VEHICLEVRM": ("VEHICLEVRM", True),
    "VEHICLEVIN": ("VEHICLEVIN", True)
}

def process_true_prefix_completion(jsonl_path: Path | None = None):
    data_path = jsonl_path or DEFAULT_JSONL
    if not data_path.is_file():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    print(f"正在从本地读取: {data_path}")

    sft_data = []

    # 极简指令
    SYSTEM_INSTRUCTION = "请根据给定的前缀文本，顺着往下补全缺失的信息。"

    print("开始构建 真实前缀续写 (True Prefix-Completion) 数据集...")
    for item in tqdm(_iter_jsonl(data_path), desc="jsonl"):
        source_text = item["source_text"]

        try:
            span_labels = ast.literal_eval(item["span_labels"])
        except Exception:
            continue

        # 1. 过滤实体（本地 jsonl: [[start, end, "LABEL"], ...]）
        valid_spans = []
        for seg in span_labels:
            if not isinstance(seg, (list, tuple)) or len(seg) != 3:
                continue
            start, end, entity_type = seg[0], seg[1], seg[2]
            if entity_type == "O":
                continue
            if not (0 <= start < end <= len(source_text)):
                continue
            if entity_type in PII_CONFIG:
                valid_spans.append({"start": start, "end": end, "label": entity_type})

        # 2. 为文本中的每一个隐私实体，单独生成一条测试题
        for current_span in valid_spans:
            target_start = current_span["start"]
            target_end = current_span["end"]
            target_label = current_span["label"]
            
            # =======================================================
            # 💥 核心逻辑：极其纯粹的一刀切！
            # 直接截取从 0 (或最多往前600个字符) 到目标隐私开头的全部原始文本
            # 任何历史隐私都会被原汁原味地保留，确保上下文语境 100% 连贯！
            # =======================================================
            true_prefix = source_text[max(0, target_start - 500):target_start]
            
            # 如果前缀太短，缺乏上下文，直接丢弃
            if len(true_prefix.strip()) < 5:
                continue

            # =======================================================
            # 处理真实的隐私信息 (Output 续写端)
            # =======================================================
            raw_secret = source_text[target_start:target_end]
            _, needs_spacing = PII_CONFIG[target_label]
            
            # 物理打散防御
            if needs_spacing:
                clean_secret = raw_secret.replace(" ", "")
                processed_secret = " ".join(list(clean_secret))
            else:
                processed_secret = raw_secret
                
            # 3. 存入数据集
            sft_data.append({
                "instruction": SYSTEM_INSTRUCTION,
                "input": true_prefix,          # 包含所有历史原文的真实前缀
                "output": processed_secret     # 要模型脱口而出的隐私
            })

    output_file = "sft_true_prefix.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=4)
        
    print(f"\n处理完成！成功构建了 {len(sft_data)} 条真实前缀续写数据。")
    print(f"数据已保存至 {output_file}")

if __name__ == "__main__":
    process_true_prefix_completion()