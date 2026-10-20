import sys, json
sys.path.insert(0, sys.argv[1])
import ler_dev as L
prereg, qwen_tok_dir, tiny_dir, fake = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]

# 1. protocol strings
missing = L.check_against_protocol(open(prereg, encoding="utf-8").read())
print("1. strings missing from protocol:", missing)
assert missing == []

# 2. S3 listing parser (real ListObjectsV2 shape, with namespace)
xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Name>physionet-open</Name>
<Prefix>challenge-2019/1.0.0/training/</Prefix><KeyCount>2</KeyCount><MaxKeys>1000</MaxKeys>
<IsTruncated>true</IsTruncated><NextContinuationToken>abc==</NextContinuationToken>
<Contents><Key>challenge-2019/1.0.0/training/training_setA/p000001.psv</Key><Size>9000</Size></Contents>
<Contents><Key>challenge-2019/1.0.0/training/training_setA/p000002.psv</Key><Size>3900</Size></Contents>
</ListBucketResult>"""
keys, tok = L.parse_s3_listing(xml)
print("2. s3:", keys, tok)
assert len(keys) == 2 and tok == "abc=="
keys, tok = L.parse_s3_listing(xml.replace(b"<IsTruncated>true", b"<IsTruncated>false"))
assert tok is None

# 3. IDs mapping
maps = L.parse_ids_mapping(open(f"{fake}/diabetes130/IDS_mapping.csv").read())
print("3. maps:", {k: len(v) for k, v in maps.items()}, maps["discharge_disposition_id"]["19"])
assert set(maps) == {"admission_type_id", "discharge_disposition_id", "admission_source_id"}
assert maps["admission_source_id"]["1"] == "Physician Referral"
assert "Medicaid only, hospice" in maps["discharge_disposition_id"]["19"]

# 4. tokenizers: Yes/No variants, rendering, template system text
from transformers import AutoTokenizer
for name, d in (("qwen", qwen_tok_dir), ("tiny", tiny_dir)):
    try:
        tk = AutoTokenizer.from_pretrained(d)
    except Exception as e:
        print("4.", name, "tokenizer not loadable:", e); continue
    v, s = L.find_word_variants(tk)
    r = L.render("Question: Will this patient die?\nAnswer with Yes or No.", "instruct", tk)
    rb = L.render("Question: x?\nAnswer with Yes or No.", "base", tk)
    print(f"4. {name}: variants {v} {s} | system text: {L.template_system_text(tk)!r}")
    print("   instruct ends:", repr(r[-60:]), "| base ends:", repr(rb[-30:]))
    ids = tk.encode(r, add_special_tokens=False)
    print("   last tokens:", [tk.decode([i]) for i in ids[-6:]])
    assert r.endswith("Answer:") and rb.endswith("\nAnswer:")
print("all unit tests passed")
