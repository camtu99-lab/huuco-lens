import base64
import csv
import html
import io
import json
import math
import random
import re
import time
import unicodedata
from datetime import datetime

import streamlit as st
from google import genai
from google.genai import errors, types
from PIL import Image
from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

RDLogger.DisableLog("rdApp.*")

MODEL = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_PER_SESSION = int(st.secrets.get("MAX_PER_SESSION", 15))
BENCH_MAX = int(st.secrets.get("BENCH_MAX", 30))

APP_NAME = st.secrets.get("APP_NAME", "MolSnap")
APP_ICON = st.secrets.get("APP_ICON", "🧬")

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")

RULES = """Bạn là giáo viên Hóa hữu cơ Việt Nam kiêm chuyên gia nhận dạng công thức. Quy ước trong SGK Việt Nam: công thức khai triển/thu gọn (CH3-CH2-OH, CH3COOH), công thức khung (mỗi đầu mút/đỉnh là C, H gắn vào C không vẽ), vòng benzen vẽ lục giác với 3 liên kết đôi xen kẽ hoặc có vòng tròn bên trong, C6H5- hoặc Ph là phenyl, nhóm chức như -OH, -CHO, -COOH, -COO-, -CO-, -NH2, -NO2, -C≡N, -X. Đếm cẩn thận số C, vị trí nhánh và nhóm thế trên vòng (o-, m-, p-). Danh pháp dùng cách đọc tiếng Việt theo SGK (metan, etanol, propan-2-ol, axit etanoic, etanal, benzen, phenol, anilin, toluen, etyl axetat...), kèm tên thường gọi nếu có."""

FMT = """Chỉ trả về MỘT đối tượng JSON, không markdown, không lời dẫn, các khóa:
{"smiles":"SMILES chuẩn (không có H tường minh)","do_tin_cay":số 0-1,"ten_danh_phap":"tên thay thế/IUPAC đọc kiểu SGK Việt Nam","ten_thuong_goi":"tên thông thường hoặc rỗng","loai_hop_chat":"ví dụ: ancol no đơn chức","nhom_chuc":["..."],"trang_thai_nhiet_do":"thể ở điều kiện thường, nhận xét nhiệt độ sôi/tan","tinh_chat_vat_ly":["..."],"tinh_chat_hoa_hoc":["phản ứng đặc trưng kèm phương trình ngắn, điều kiện"],"dieu_che_ung_dung":["..."],"dong_phan":"nhận xét ngắn nếu có ý nghĩa","luu_y":["điều chưa chắc chắn trong ảnh, nét vẽ mơ hồ, cảnh báo an toàn"]}
Nếu ảnh không phải công thức hóa học hoặc không đọc được, đặt "smiles":"" và giải thích trong luu_y."""

SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


@st.cache_resource
def get_client():
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def to_jpeg(file_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img.thumbnail((1600, 1600))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ask_ai(contents, temperature=None):
    """Gọi Gemini, tự thử lại khi máy chủ quá tải (503) hoặc hết lượt tạm thời (429),
    rồi chuyển sang mô hình dự phòng."""
    models = [MODEL]
    fb = st.secrets.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
    if fb and fb != MODEL:
        models.append(fb)
    last = None
    for m in models:
        for i in range(3):
            try:
                resp = get_client().models.generate_content(
                    model=m,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        **({} if temperature is None else {"temperature": temperature}),
                    ),
                )
                text = (resp.text or "").replace("```json", "").replace("```", "").strip()
                return json.loads(text[text.index("{"): text.rindex("}") + 1])
            except errors.ServerError as e:
                last = e
                time.sleep(2 * (i + 1))
            except errors.ClientError as e:
                code = getattr(e, "code", None)
                if code == 429:
                    last = e
                    time.sleep(3 * (i + 1))
                elif code == 404:
                    last = e
                    break  # mô hình này không còn dùng được, chuyển sang mô hình tiếp theo
                else:
                    raise
    raise last


def analyze_image(file_bytes: bytes):
    prompt = (
        RULES
        + "\n\nĐây là ảnh chụp công thức cấu tạo viết tay. Hãy nhận dạng đúng một chất hữu cơ trong ảnh. "
        + "Chép ĐÚNG như hình vẽ, KHÔNG tự sửa lỗi hóa trị và không thêm hoặc bớt nguyên tử. "
        + "Nếu hình vẽ sai (ví dụ nguyên tử C có 5 liên kết) vẫn ghi SMILES đúng theo hình và nêu rõ lỗi trong luu_y.\n"
        + FMT
    )
    return ask_ai([types.Part.from_bytes(data=to_jpeg(file_bytes), mime_type="image/jpeg"), prompt], temperature=0)


def analyze_smiles(smiles: str):
    prompt = (
        RULES
        + f"\n\nChất có SMILES: {smiles}\nGiữ nguyên SMILES này trong khóa \"smiles\", đặt do_tin_cay=1 và cung cấp tên cùng tính chất.\n"
        + FMT
    )
    return ask_ai([prompt])


def chem_numbers(smiles: str):
    """Tính trực tiếp bằng RDKit, không phụ thuộc AI."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    pretty = "".join(c.translate(SUB) if c.isdigit() else c for c in formula)
    molh = Chem.AddHs(mol)
    cnt = {}
    for a in molh.GetAtoms():
        cnt[a.GetSymbol()] = cnt.get(a.GetSymbol(), 0) + 1
    dbe = None
    if cnt.get("C"):
        x = sum(cnt.get(k, 0) for k in ("F", "Cl", "Br", "I"))
        dbe = (2 * cnt["C"] + 2 + cnt.get("N", 0) + cnt.get("P", 0) - cnt.get("H", 0) - x) / 2
    return mol, {"formula": pretty, "mass": Descriptors.MolWt(mol), "dbe": dbe}


def bullets(title, items):
    if items:
        st.markdown(f"**{title}**")
        for t in [items] if isinstance(items, str) else items:
            st.markdown(f"- {t}")


def render(r: dict, from_image: bool):
    smiles = (r.get("smiles") or "").strip()
    if not smiles:
        st.warning("Chưa nhận dạng được công thức trong ảnh.")
    mol, num = chem_numbers(smiles) if smiles else (None, None)
    if smiles and mol is None:
        st.error("Mã SMILES không hợp lệ. Hãy sửa ở ô bên dưới rồi bấm Phân tích lại.")
    c1, c2 = st.columns(2)
    with c1:
        if mol is not None:
            st.image(Draw.MolToImage(mol, size=(420, 300)))
    with c2:
        if smiles:
            st.markdown("**SMILES**")
            st.code(smiles, language=None)
        if r.get("ten_danh_phap"):
            st.markdown(f"**Tên (danh pháp):** {r['ten_danh_phap']}")
        if r.get("ten_thuong_goi"):
            st.markdown(f"**Tên thường gọi:** {r['ten_thuong_goi']}")
        if r.get("loai_hop_chat"):
            st.markdown(f"**Loại hợp chất:** {r['loai_hop_chat']}")
        if num:
            st.markdown(f"**Công thức phân tử:** {num['formula']}")
            st.markdown(f"**Khối lượng mol:** {num['mass']:.2f} g/mol")
            if num["dbe"] is not None:
                k = num["dbe"]
                st.markdown(f"**Độ bất bão hòa (k):** {int(k) if k == int(k) else k}")
    if from_image and isinstance(r.get("do_tin_cay"), (int, float)):
        p = max(0.0, min(1.0, float(r["do_tin_cay"])))
        st.progress(p, text=f"Độ tin cậy nhận dạng: {round(p * 100)}%")
    if r.get("nhom_chuc"):
        st.markdown("**Nhóm chức:** " + " · ".join(r["nhom_chuc"]))
    bullets("Trạng thái, nhiệt độ", r.get("trang_thai_nhiet_do"))
    bullets("Tính chất vật lí", r.get("tinh_chat_vat_ly"))
    bullets("Tính chất hóa học", r.get("tinh_chat_hoa_hoc"))
    bullets("Điều chế và ứng dụng", r.get("dieu_che_ung_dung"))
    bullets("Đồng phân", r.get("dong_phan"))
    if r.get("luu_y"):
        st.info("Lưu ý: " + " • ".join([r["luu_y"]] if isinstance(r["luu_y"], str) else r["luu_y"]))



# ---------------- Phản ứng hóa học ----------------
RULES_RX = """Bạn là giáo viên Hóa học Việt Nam. Hãy đọc phương trình phản ứng (viết tay hoặc gõ) và phân loại theo cách của SGK Việt Nam (Hóa 8-12). Các loại thường gặp: phản ứng thế (thế halogen, thế nitro, thế brom vào phenol...), cộng (cộng H2, Br2, HX, H2O), tách (tách H2O, tách HX, tách H2), trùng hợp, trùng ngưng, oxi hóa hoàn toàn (đốt cháy), oxi hóa không hoàn toàn, oxi hóa - khử, hóa hợp, phân hủy, trao đổi, trung hòa, este hóa, thủy phân, xà phòng hóa, tráng bạc, lên men, cracking, đề hiđro hóa, hiđro hóa. Một phản ứng có thể thuộc nhiều loại cùng lúc (ví dụ: cộng đồng thời là oxi hóa - khử); hãy nêu loại chính trước. Đọc kỹ hệ số và điều kiện (nhiệt độ, xúc tác, áp suất). Với muối/bazơ vô cơ dùng SMILES dạng ion, ví dụ NaOH là [Na+].[OH-], HCl là Cl, H2O là O, O2 là O=O, H2 là [H][H]."""

RFMT = """Chỉ trả về MỘT đối tượng JSON, không markdown, không lời dẫn:
{"phuong_trinh":"phương trình đọc được, viết dạng chữ, dùng -> giữa hai vế",
"chat_tham_gia":[{"he_so":số nguyên,"ten":"tên chất","cong_thuc":"công thức phân tử","smiles":"SMILES"}],
"san_pham":[{"he_so":số nguyên,"ten":"","cong_thuc":"","smiles":""}],
"dieu_kien":"nhiệt độ, xúc tác... hoặc rỗng",
"loai_phan_ung":["loại chính","loại phụ nếu có"],
"giai_thich":"2-4 câu: dựa vào sự thay đổi nào của cấu tạo/nhóm chức/số oxi hóa mà kết luận loại phản ứng",
"phuong_trinh_can_bang":"phương trình đã cân bằng đúng nếu ảnh chưa cân bằng hoặc sai, ngược lại để rỗng",
"luu_y":["điều chưa chắc chắn khi đọc ảnh, ví dụ hệ số hoặc chất bị mờ"]}
Nếu không phải phương trình phản ứng hoặc không đọc được, trả về chat_tham_gia và san_pham rỗng và giải thích trong luu_y."""


def analyze_reaction(img_bytes, text):
    head = RULES_RX + "\n\n"
    if img_bytes:
        prompt = head + "Đây là ảnh chụp một phương trình phản ứng hóa học.\n" + RFMT
        return ask_ai([types.Part.from_bytes(data=to_jpeg(img_bytes), mime_type="image/jpeg"), prompt])
    return ask_ai([head + f"Phương trình: {text}\n" + RFMT])


def atom_balance(species):
    """Đếm nguyên tử và điện tích mỗi vế bằng RDKit. Trả về None nếu có SMILES lỗi."""
    tot, charge = {}, 0
    for coef, smi in species:
        mol = Chem.MolFromSmiles(smi or "")
        if mol is None:
            return None
        for a in Chem.AddHs(mol).GetAtoms():
            tot[a.GetSymbol()] = tot.get(a.GetSymbol(), 0) + coef
            charge += coef * a.GetFormalCharge()
    return tot, charge


def to_int(v):
    try:
        return max(1, int(v))
    except Exception:
        return 1


def render_reaction(r: dict):
    left, right = r.get("chat_tham_gia") or [], r.get("san_pham") or []
    if r.get("phuong_trinh"):
        st.markdown("**Phương trình đọc được**")
        st.code(r["phuong_trinh"], language=None)
    types_ = r.get("loai_phan_ung") or []
    if types_:
        st.success("**Loại phản ứng:** " + " · ".join(types_))
    if r.get("giai_thich"):
        st.markdown("**Vì sao:** " + r["giai_thich"])
    if r.get("dieu_kien"):
        st.markdown("**Điều kiện:** " + r["dieu_kien"])

    rows, mols, legends = [], [], []
    for role, lst in (("Tham gia", left), ("Sản phẩm", right)):
        for sp in lst:
            m = Chem.MolFromSmiles(sp.get("smiles") or "")
            mass = f"{Descriptors.MolWt(m):.2f}" if m is not None else "—"
            rows.append({"Vai trò": role, "Hệ số": to_int(sp.get("he_so")), "Tên": sp.get("ten", ""),
                         "Công thức": sp.get("cong_thuc", ""), "SMILES": sp.get("smiles", ""), "M (g/mol)": mass})
            if m is not None and m.GetNumHeavyAtoms() > 1:
                mols.append(m)
                legends.append(sp.get("ten") or sp.get("cong_thuc") or "")
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    if mols:
        try:
            st.image(Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(260, 190), legends=legends))
        except Exception:
            pass

    if left and right:
        bl = atom_balance([(to_int(s.get("he_so")), s.get("smiles")) for s in left])
        br = atom_balance([(to_int(s.get("he_so")), s.get("smiles")) for s in right])
        if bl is None or br is None:
            st.info("Không kiểm tra được cân bằng vì có chất có SMILES không hợp lệ.")
        elif bl == br:
            st.success("Phương trình **đã cân bằng** (RDKit đếm nguyên tử và điện tích hai vế bằng nhau).")
        else:
            diff = [f"{k}: {bl[0].get(k, 0)} ≠ {br[0].get(k, 0)}" for k in sorted(set(bl[0]) | set(br[0]))
                    if bl[0].get(k, 0) != br[0].get(k, 0)]
            if bl[1] != br[1]:
                diff.append(f"điện tích: {bl[1]} ≠ {br[1]}")
            st.warning("Phương trình **chưa cân bằng** theo hệ số đọc được. Lệch: " + "; ".join(diff))
            if r.get("phuong_trinh_can_bang"):
                st.markdown("**Gợi ý cân bằng:**")
                st.code(r["phuong_trinh_can_bang"], language=None)
    if r.get("luu_y"):
        st.info("Lưu ý: " + " • ".join([r["luu_y"]] if isinstance(r["luu_y"], str) else r["luu_y"]))


def reaction_ui():
    top = st.container()  # lỗi và lý do sai hiện ở đầu tab
    src = st.radio("Nguồn phương trình", ["🖼️ Tải ảnh", "📷 Chụp ảnh", "⌨️ Gõ phương trình"], horizontal=True, key="rx_src")
    img, txt = None, None
    if src.startswith("🖼️"):
        f = st.file_uploader("Chọn ảnh phương trình", type=["jpg", "jpeg", "png", "webp"], key="rx_up")
        img = f.getvalue() if f else None
    elif src.startswith("📷"):
        f = st.camera_input("Chụp phương trình", key="rx_cam")
        img = f.getvalue() if f else None
    else:
        txt = st.text_input("Phương trình", placeholder="Ví dụ: CH2=CH2 + Br2 -> CH2Br-CH2Br", key="rx_txt").strip()
    if st.button("Nhận dạng và phân loại phản ứng", type="primary", disabled=not (img or txt), key="rx_go"):
        if st.session_state.n >= MAX_PER_SESSION:
            show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
        else:
            with st.spinner("Đang phân tích…"):
                try:
                    st.session_state.rx = analyze_reaction(img, txt)
                    st.session_state.n += 1
                except Exception as e:
                    st.session_state.rx = None
                    show_top_error(top, "Có lỗi khi phân tích, hãy thử lại hoặc chụp rõ hơn.", e)
    if st.session_state.get("rx"):
        st.divider()
        render_reaction(st.session_state.rx)



# ---------------- Đồng phân ----------------
RULES_ISO = """Bạn là giáo viên Hóa hữu cơ Việt Nam. Xác định đồng phân theo SGK: đồng phân cấu tạo (mạch cacbon, vị trí nhóm chức hoặc liên kết bội, nhóm chức) và đồng phân lập thể (hình học cis-trans, quang học). Tên gọi dùng danh pháp tiếng Việt theo SGK (butan-1-ol, đimetyl ete, 2-metylpropan-2-ol...)."""

IFMT = """Chỉ trả về MỘT đối tượng JSON, không markdown, không lời dẫn:
{"chat_goc":{"ten":"","smiles":""},"cong_thuc_phan_tu":"","tong_so_dong_phan_cau_tao":số nguyên hoặc null,
"dong_phan":[{"ten":"tên theo SGK","smiles":"SMILES","nhom":"loại hợp chất/nhóm chức","loai_so_voi_chat_goc":"giống chất gốc | mạch cacbon | vị trí nhóm chức hoặc liên kết bội | nhóm chức | khác | không áp dụng","ghi_chu":""}],
"nhan_xet":"2-3 câu: các loại đồng phân có thể có và cách đếm","luu_y":[]}
Quy tắc: liệt kê TẤT CẢ đồng phân cấu tạo ứng với công thức phân tử, kể cả chính chất gốc (tối đa 30; nếu nhiều hơn thì liệt kê 30 chất tiêu biểu và nói rõ trong nhan_xet). Mỗi đồng phân đúng một SMILES, không trùng nhau, đúng công thức phân tử, không dùng ký hiệu lập thể (@, /, \\) trong SMILES. Nếu đầu vào không đọc được thì để dong_phan rỗng và giải thích trong luu_y."""

PRETTY = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def formula_of(m):
    return rdMolDescriptors.CalcMolFormula(m)


def pretty(f):
    return "".join(c.translate(PRETTY) if c.isdigit() else c for c in f)


def noiso(m):
    return Chem.MolToSmiles(m, isomericSmiles=False)


def stereo_count(m):
    try:
        opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, tryEmbedding=False, maxIsomers=32)
        return max(1, len(list(EnumerateStereoisomers(m, options=opts))))
    except Exception:
        return 1


def mirror(smi):
    return smi.replace("@@", "\x00").replace("@", "@@").replace("\x00", "@")


def relation(m1, m2):
    f1, f2 = formula_of(m1), formula_of(m2)
    if f1 != f2:
        return "khac_cong_thuc"
    if noiso(m1) != noiso(m2):
        return "cau_tao"
    i1, i2 = Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)
    if i1 == i2:
        return "cung_chat"
    mir = Chem.MolFromSmiles(mirror(i1))
    if mir is not None and Chem.MolToSmiles(mir) == i2:
        return "doi_quang"
    if not Chem.FindMolChiralCenters(m1) and not Chem.FindMolChiralCenters(m2):
        return "hinh_hoc"
    return "lap_the"


def analyze_isomers(img, text, flt):
    extra = ("\nĐiều kiện lọc của người dùng: " + flt) if flt else ""
    head = RULES_ISO + "\n\n"
    if img:
        c = [types.Part.from_bytes(data=to_jpeg(img), mime_type="image/jpeg"),
             head + "Đây là ảnh một chất hoặc một công thức phân tử. Hãy liệt kê các đồng phân." + extra + "\n" + IFMT]
    else:
        c = [head + "Chất/công thức cần xét: " + text + ". Hãy liệt kê các đồng phân." + extra + "\n" + IFMT]
    return ask_ai(c)


def validate_isomers(r):
    parsed, bad = [], []
    for c in r.get("dong_phan") or []:
        m = Chem.MolFromSmiles(c.get("smiles") or "")
        if m is None:
            bad.append((c.get("ten", ""), c.get("smiles", ""), "SMILES không hợp lệ"))
        else:
            parsed.append((c, m))
    mg = Chem.MolFromSmiles((r.get("chat_goc") or {}).get("smiles") or "")
    if mg is not None:
        target = formula_of(mg)
    elif parsed:
        target = Counter(formula_of(m) for _, m in parsed).most_common(1)[0][0]
    else:
        target = None
    good, seen = [], set()
    for c, m in parsed:
        if formula_of(m) != target:
            bad.append((c.get("ten", ""), c.get("smiles", ""), "khác công thức phân tử: " + pretty(formula_of(m))))
        elif noiso(m) in seen:
            bad.append((c.get("ten", ""), c.get("smiles", ""), "trùng với một đồng phân đã có"))
        else:
            seen.add(noiso(m))
            good.append((c, m))
    return target, good, bad


def render_isomers(r):
    target, good, bad = validate_isomers(r)
    if not good:
        st.warning("Chưa lập được danh sách đồng phân hợp lệ.")
    else:
        stereo = [stereo_count(m) for _, m in good]
        c1, c2, c3 = st.columns(3)
        c1.metric("Công thức phân tử", pretty(target))
        c2.metric("Đồng phân cấu tạo", len(good))
        c3.metric("Tính cả lập thể", sum(stereo))
        rows = []
        for i, ((c, m), sc) in enumerate(zip(good, stereo), 1):
            rows.append({"STT": i, "Tên": c.get("ten", ""), "Nhóm": c.get("nhom", ""),
                         "So với chất gốc": c.get("loai_so_voi_chat_goc", ""),
                         "Lập thể": "không" if sc == 1 else f"{sc} đồng phân lập thể",
                         "SMILES": Chem.MolToSmiles(m)})
        st.dataframe(rows, hide_index=True, width="stretch")
        try:
            st.image(Draw.MolsToGridImage([m for _, m in good], molsPerRow=4, subImgSize=(230, 180),
                                          legends=[f"{i}. {c.get('ten', '')}" for i, (c, _) in enumerate(good, 1)]))
        except Exception:
            pass
        n_ai = r.get("tong_so_dong_phan_cau_tao")
        if isinstance(n_ai, int) and n_ai != len(good):
            st.warning(f"AI nói có {n_ai} đồng phân cấu tạo nhưng danh sách hợp lệ chỉ có {len(good)}. "
                       "Có thể danh sách còn thiếu hoặc con số bị sai, hãy đếm lại bằng tay.")
    if r.get("nhan_xet"):
        st.markdown("**Nhận xét:** " + r["nhan_xet"])
    if bad:
        with st.expander(f"Đã loại bỏ {len(bad)} mục AI đưa ra (sai hoặc trùng)"):
            for n, s_, why in bad:
                st.markdown(f"- {n} `{s_}`: {why}")
    if r.get("luu_y"):
        st.info("Lưu ý: " + " • ".join([r["luu_y"]] if isinstance(r["luu_y"], str) else r["luu_y"]))
    st.caption("RDKit chỉ kiểm tra từng chất có đúng công thức phân tử và loại chất trùng. RDKit không đảm bảo danh sách đã đủ, "
               "nên với công thức nhiều cacbon hãy đối chiếu SGK.")


def to_smiles(img, text):
    if text:
        if Chem.MolFromSmiles(text) is not None:
            return text
    prompt = RULES + '\n\nChuyển chất sau thành SMILES chuẩn. Chỉ trả JSON {"smiles":"","ten":""}, để rỗng nếu không đọc được.'
    if img:
        r = ask_ai([types.Part.from_bytes(data=to_jpeg(img), mime_type="image/jpeg"), prompt])
    elif text:
        r = ask_ai([prompt + "\nChất: " + text])
    else:
        return ""
    return (r.get("smiles") or "").strip()


REL = {
    "cung_chat": ("warning", "Hai công thức biểu diễn **cùng một chất**, không phải đồng phân."),
    "khac_cong_thuc": ("error", "**Không phải đồng phân** vì khác công thức phân tử."),
    "cau_tao": ("success", "Hai chất là **đồng phân cấu tạo** của nhau."),
    "hinh_hoc": ("success", "Hai chất là **đồng phân hình học (cis–trans)** của nhau."),
    "doi_quang": ("success", "Hai chất là **đồng phân quang học (đối quang)** của nhau."),
    "lap_the": ("success", "Hai chất là **đồng phân lập thể** (không phải đối quang) của nhau."),
}


def compare_ui_result(s1, s2):
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        st.error("Không đọc được một trong hai chất. Hãy kiểm tra lại ảnh hoặc nhập SMILES.")
        return
    c1, c2 = st.columns(2)
    for col, m, s_, tag in ((c1, m1, s1, "A"), (c2, m2, s2, "B")):
        with col:
            st.markdown(f"**Chất {tag}**")
            st.image(Draw.MolToImage(m, size=(320, 240)))
            st.markdown(f"{pretty(formula_of(m))} · M = {Descriptors.MolWt(m):.2f} g/mol")
            st.code(s_, language=None)
    rel = relation(m1, m2)
    kind, msg = REL[rel]
    getattr(st, kind)(msg)
    if rel == "khac_cong_thuc":
        st.markdown(f"{pretty(formula_of(m1))} ≠ {pretty(formula_of(m2))}")
        return
    if rel == "cung_chat":
        return
    try:
        r = ask_ai([RULES_ISO + f'\n\nChất A: {s1}\nChất B: {s2}\nRDKit đã xác nhận quan hệ: {msg}\n'
                    'Hãy đặt tên hai chất và giải thích ngắn gọn theo SGK. Chỉ trả JSON '
                    '{"ten_A":"","ten_B":"","loai_dong_phan":"ví dụ: đồng phân nhóm chức","giai_thich":"2-3 câu: điểm khác nhau về cấu tạo hoặc không gian"}'])
        st.markdown(f"**Tên:** A = {r.get('ten_A', '')}; B = {r.get('ten_B', '')}")
        if r.get("loai_dong_phan"):
            st.markdown("**Loại đồng phân:** " + r["loai_dong_phan"])
        if r.get("giai_thich"):
            st.markdown(r["giai_thich"])
    except Exception as e:
        st.caption(f"Không lấy được phần giải thích: {type(e).__name__}")


def isomer_ui():
    top = st.container()  # lỗi và lý do sai hiện ở đầu tab
    t1, t2 = st.tabs(["Liệt kê đồng phân", "So sánh hai chất"])
    with t1:
        src = st.radio("Nguồn", ["⌨️ Gõ công thức / tên / SMILES", "🖼️ Tải ảnh", "📷 Chụp ảnh"], horizontal=True, key="iso_src")
        img, txt = None, None
        if src.startswith("🖼️"):
            f = st.file_uploader("Ảnh công thức", type=["jpg", "jpeg", "png", "webp"], key="iso_up")
            img = f.getvalue() if f else None
        elif src.startswith("📷"):
            f = st.camera_input("Chụp công thức", key="iso_cam")
            img = f.getvalue() if f else None
        else:
            txt = st.text_input("Công thức phân tử, tên hoặc SMILES", placeholder="Ví dụ: C4H10O", key="iso_txt").strip()
        flt = st.text_input("Điều kiện lọc (tùy chọn)", placeholder="Ví dụ: chỉ ancol; chỉ đồng phân đơn chức", key="iso_flt").strip()
        if st.button("Liệt kê đồng phân", type="primary", disabled=not (img or txt), key="iso_go"):
            if st.session_state.n >= MAX_PER_SESSION:
                show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
            else:
                with st.spinner("Đang phân tích…"):
                    try:
                        st.session_state.iso = analyze_isomers(img, txt, flt)
                        st.session_state.n += 1
                    except Exception as e:
                        st.session_state.iso = None
                        show_top_error(top, "Có lỗi khi phân tích, hãy thử lại.", e)
        if st.session_state.get("iso"):
            st.divider()
            render_isomers(st.session_state.iso)
    with t2:
        st.caption("Mỗi chất nhập bằng ảnh, SMILES hoặc tên.")
        ca, cb = st.columns(2)
        inp = []
        for col, tag in ((ca, "A"), (cb, "B")):
            with col:
                st.markdown(f"**Chất {tag}**")
                f = st.file_uploader(f"Ảnh chất {tag}", type=["jpg", "jpeg", "png", "webp"], key=f"cmp_up{tag}")
                t = st.text_input(f"Hoặc SMILES/tên chất {tag}", key=f"cmp_t{tag}").strip()
                inp.append((f.getvalue() if f else None, t))
        ok = all(i or t for i, t in inp)
        if st.button("So sánh", type="primary", disabled=not ok, key="cmp_go"):
            if st.session_state.n >= MAX_PER_SESSION:
                show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
            else:
                with st.spinner("Đang phân tích…"):
                    try:
                        smi = [to_smiles(i, t) for i, t in inp]
                        st.session_state.n += 1
                        st.session_state.cmp = smi
                    except Exception as e:
                        st.session_state.cmp = None
                        show_top_error(top, "Có lỗi khi đọc chất, hãy thử lại.", e)
        if st.session_state.get("cmp"):
            st.divider()
            compare_ui_result(*st.session_state.cmp)



# ---------------- Dụng cụ thí nghiệm ----------------
RULES_EQ = """Bạn là giáo viên Hóa học Việt Nam hướng dẫn thực hành thí nghiệm ở trường phổ thông. Hãy nhận biết dụng cụ thí nghiệm theo tên gọi trong SGK Việt Nam, ví dụ: ống nghiệm, cốc thủy tinh, bình tam giác, bình cầu (đáy tròn, đáy bằng, có nhánh), phễu lọc, phễu chiết, đèn cồn, kẹp ống nghiệm, giá sắt, kiềng đun, lưới amiăng, ống đong, pipet, buret, bình định mức, cối và chày sứ, chén sứ, đũa thủy tinh, thìa lấy hóa chất, nhiệt kế, ống sinh hàn, bình Kíp, ống dẫn khí, nút cao su, giấy lọc, cân, bình tia, giá để ống nghiệm. Chỉ nêu những gì thấy rõ trong ảnh; nếu không chắc thì giảm độ tin cậy và nói rõ."""

EFMT = """Chỉ trả về MỘT đối tượng JSON, không markdown, không lời dẫn:
{"loai_anh":"dụng cụ riêng lẻ | bộ lắp thí nghiệm | không phải dụng cụ thí nghiệm",
"dung_cu":[{"ten":"tên theo SGK","ten_khac":"tên gọi khác hoặc tên tiếng Anh, có thể rỗng","vi_tri":"vị trí trong ảnh, ví dụ: bên trái, ở giữa","chat_lieu":"thủy tinh | sứ | kim loại | nhựa | cao su | ...","cong_dung":"1-2 câu","cach_dung":["bước hoặc lưu ý sử dụng"],"luu_y_an_toan":["điều cần chú ý để an toàn"],"do_tin_cay":số 0-1}],
"thi_nghiem":null hoặc {"ten":"thí nghiệm có thể đang lắp, ví dụ: điều chế khí oxi từ KMnO4","mo_ta":"1-2 câu","nhan_xet_lap_rap":["điểm đúng hoặc điểm có thể sai/nguy hiểm trong cách lắp, chỉ nêu khi nhìn thấy rõ"]},
"luu_y":["điều chưa chắc chắn, ví dụ ảnh mờ, bị che khuất"]}
Nếu ảnh có nhiều dụng cụ thì liệt kê từng dụng cụ nhận ra được. Chỉ đặt thi_nghiem khi ảnh là một bộ lắp ráp rõ ràng. Nếu không phải dụng cụ thí nghiệm, để dung_cu rỗng và giải thích trong luu_y."""


def analyze_equipment(img, text):
    head = RULES_EQ + "\n\n"
    if img:
        return ask_ai([types.Part.from_bytes(data=to_jpeg(img), mime_type="image/jpeg"),
                       head + "Đây là ảnh dụng cụ thí nghiệm hóa học. Hãy nhận biết.\n" + EFMT])
    return ask_ai([head + "Hãy giải thích dụng cụ có tên: " + text + ".\n" + EFMT])


def render_equipment(r):
    kind = r.get("loai_anh")
    tools = r.get("dung_cu") or []
    if kind:
        st.markdown(f"**Loại ảnh:** {kind}")
    if not tools:
        st.warning("Chưa nhận ra dụng cụ thí nghiệm nào trong ảnh.")
    for t in tools:
        with st.container(border=True):
            st.markdown(f"### {t.get('ten', 'Dụng cụ')}")
            if t.get("ten_khac"):
                st.caption(f"Tên khác: {t['ten_khac']}")
            meta = [x for x in (t.get("vi_tri"), t.get("chat_lieu")) if x]
            if meta:
                st.markdown("**Vị trí / chất liệu:** " + " · ".join(meta))
            conf = t.get("do_tin_cay")
            if isinstance(conf, (int, float)):
                p = max(0.0, min(1.0, float(conf)))
                st.progress(p, text=f"Độ tin cậy: {round(p * 100)}%")
                if p < 0.6:
                    st.warning("Độ tin cậy thấp, hãy hỏi giáo viên hoặc chụp lại rõ hơn.")
            if t.get("cong_dung"):
                st.markdown("**Công dụng:** " + t["cong_dung"])
            bullets("Cách dùng", t.get("cach_dung"))
            if t.get("luu_y_an_toan"):
                st.warning("**An toàn:** " + " • ".join([t["luu_y_an_toan"]] if isinstance(t["luu_y_an_toan"], str) else t["luu_y_an_toan"]))
    tn = r.get("thi_nghiem")
    if isinstance(tn, dict) and (tn.get("ten") or tn.get("mo_ta")):
        st.divider()
        st.markdown("### Bộ thí nghiệm trong ảnh")
        if tn.get("ten"):
            st.markdown("**Có thể là:** " + tn["ten"])
        if tn.get("mo_ta"):
            st.markdown(tn["mo_ta"])
        bullets("Nhận xét về cách lắp", tn.get("nhan_xet_lap_rap"))
    if r.get("luu_y"):
        st.info("Lưu ý: " + " • ".join([r["luu_y"]] if isinstance(r["luu_y"], str) else r["luu_y"]))
    st.caption("Kết quả do AI nhận biết qua ảnh nên có thể nhầm, nhất là khi ảnh mờ hoặc dụng cụ bị che. "
               "Khi làm thí nghiệm thật, hãy làm theo hướng dẫn của giáo viên và quy định an toàn của phòng thực hành.")


def equipment_ui():
    top = st.container()  # lỗi và lý do sai hiện ở đầu tab
    src = st.radio("Nguồn", ["🖼️ Tải ảnh", "📷 Chụp ảnh", "⌨️ Gõ tên dụng cụ"], horizontal=True, key="eq_src")
    img, txt = None, None
    if src.startswith("📷"):
        f = st.camera_input("Chụp dụng cụ hoặc bộ thí nghiệm", key="eq_cam")
        img = f.getvalue() if f else None
    elif src.startswith("🖼️"):
        f = st.file_uploader("Ảnh dụng cụ hoặc bộ thí nghiệm", type=["jpg", "jpeg", "png", "webp"], key="eq_up")
        img = f.getvalue() if f else None
    else:
        txt = st.text_input("Tên dụng cụ", placeholder="Ví dụ: ống sinh hàn", key="eq_txt").strip()
    if st.button("Nhận biết dụng cụ", type="primary", disabled=not (img or txt), key="eq_go"):
        if st.session_state.n >= MAX_PER_SESSION:
            show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
        else:
            with st.spinner("Đang phân tích…"):
                try:
                    st.session_state.eq = analyze_equipment(img, txt)
                    st.session_state.n += 1
                except Exception as e:
                    st.session_state.eq = None
                    show_top_error(top, "Có lỗi khi phân tích, hãy thử lại hoặc chụp rõ hơn.", e)
    if st.session_state.get("eq"):
        st.divider()
        render_equipment(st.session_state.eq)



# ---------------- Chẩn đoán lỗi & chấm bài ----------------
GROUPS = [
    ("ancol", "[CX4][OX2H]"), ("phenol", "c[OX2H]"), ("ete", "[CX4][OX2][CX4]"),
    ("anđehit", "[CX3;H1,H2;!$(C[OX2,NX3])]=O"), ("xeton", "[#6][CX3](=O)[#6]"),
    ("axit cacboxylic", "[CX3](=O)[OX2H1]"), ("este", "[CX3](=O)[OX2][#6]"),
    ("amin", "[NX3;!$(NC=[O,N,S])][#6]"), ("nitro", "[N+](=O)[O-]"),
    ("anken (C=C)", "[CX3]=[CX3]"), ("ankin (C≡C)", "[CX2]#[CX2]"),
    ("vòng benzen", "c1ccccc1"), ("dẫn xuất halogen", "[#6][F,Cl,Br,I]"),
]
_GP = [(n, Chem.MolFromSmarts(p)) for n, p in GROUPS]


def groups_of(m):
    return [n for n, p in _GP if m.HasSubstructMatch(p)]


def atom_counts(m):
    c = {}
    for a in Chem.AddHs(m).GetAtoms():
        c[a.GetSymbol()] = c.get(a.GetSymbol(), 0) + 1
    return c


def diff_text(ms, mt):
    a, b = atom_counts(ms), atom_counts(mt)
    order = ["C", "H"] + sorted(k for k in set(a) | set(b) if k not in ("C", "H"))
    out = []
    for k in order:
        d = a.get(k, 0) - b.get(k, 0)
        if d > 0:
            out.append(f"thừa {d} nguyên tử {k}")
        elif d < 0:
            out.append(f"thiếu {-d} nguyên tử {k}")
    return ", ".join(out)


def parse_student(smi):
    """Trả về (mol, thông báo lỗi bằng tiếng Việt)."""
    smi = (smi or "").strip()
    if not smi:
        return None, "Chưa đọc được công thức em vẽ."
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return None, "Công thức bị sai cú pháp (thiếu hoặc thừa ngoặc, vòng chưa đóng hoặc ký hiệu lạ)."
    try:
        Chem.SanitizeMol(m)
    except Chem.AtomValenceException as e:
        mm = re.search(r"atom # (\d+) (\w+), (\d+)", str(e))
        if mm:
            return None, f"Nguyên tử {mm.group(2)} thứ {int(mm.group(1)) + 1} có hóa trị {mm.group(3)}, vượt quá hóa trị cho phép."
        return None, "Có nguyên tử vượt quá hóa trị cho phép."
    except Chem.KekulizeException:
        return None, "Vòng thơm vẽ sai: không sắp xếp được các nối đôi xen kẽ trong vòng."
    except Exception:
        return None, "Công thức có lỗi hóa trị."
    return m, None


def draw_student(smi, size=(320, 220)):
    """Vẽ lại đúng hình học sinh đã vẽ, kể cả khi sai hóa trị; khoanh đỏ nguyên tử sai nếu có."""
    m = Chem.MolFromSmiles((smi or "").strip(), sanitize=False)
    if m is None:
        return None
    bad = []
    try:
        Chem.SanitizeMol(m)
    except Chem.AtomValenceException as e:
        mm = re.search(r"atom # (\d+)", str(e))
        if mm:
            bad = [int(mm.group(1))]
    except Exception:
        pass
    try:
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
        return Draw.MolToImage(m, size=size, highlightAtoms=bad, highlightColor=(1.0, 0.55, 0.55))
    except Exception:
        return None


def grade_structure(smi, target_smi):
    mt = Chem.MolFromSmiles(target_smi)
    ms, err = parse_student(smi)
    tips = ["Nhớ hóa trị: C có 4 liên kết, N có 3, O có 2, H và halogen có 1."]
    if ms is None:
        return {"kq": "loi", "nx": [err], "goi_y": tips}
    fs, ft = formula_of(ms), formula_of(mt)
    if noiso(ms) == noiso(mt):
        return {"kq": "dung", "nx": ["Chính xác. Công thức cấu tạo em vẽ đúng."], "goi_y": []}
    gs, gt = set(groups_of(ms)), set(groups_of(mt))
    if fs == ft:
        nx = [f"Công thức phân tử đúng ({pretty(fs)}) nhưng cấu tạo chưa đúng yêu cầu: em đã vẽ một đồng phân khác."]
        if gs != gt:
            nx.append("Nhóm chức khác với chất cần tìm. Hình em vẽ có: " + (", ".join(sorted(gs)) or "không có nhóm chức đặc trưng")
                      + ". Đề cần: " + (", ".join(sorted(gt)) or "không có nhóm chức đặc trưng") + ".")
            nx.append("Loại đồng phân em đã vẽ: đồng phân nhóm chức.")
        else:
            nx.append("Nhóm chức đúng nhưng mạch cacbon hoặc vị trí nhóm thế/liên kết bội chưa đúng.")
        return {"kq": "dong_phan", "nx": nx, "goi_y": ["Đếm lại số cacbon ở mạch chính và vị trí gắn nhóm thế."]}
    nx = [f"Công thức phân tử của hình em vẽ là {pretty(fs)}. So với chất cần tìm, em {diff_text(ms, mt)}."]
    if gs != gt:
        nx.append("Nhóm chức chưa khớp: hình em vẽ có " + (", ".join(sorted(gs)) or "không có nhóm chức đặc trưng")
                  + "; đề cần " + (", ".join(sorted(gt)) or "không có nhóm chức đặc trưng") + ".")
    return {"kq": "sai_cong_thuc", "nx": nx, "goi_y": ["Đếm lại số nguyên tử C và số H gắn vào mỗi C."]}


def norm_name(x):
    x = unicodedata.normalize("NFD", (x or "").lower().replace("đ", "d"))
    x = "".join(c for c in x if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", x)


BANK = [
    ("metan", "C", []), ("etan", "CC", []), ("propan", "CCC", []), ("butan", "CCCC", ["n-butan"]),
    ("2-metylpropan", "CC(C)C", ["isobutan"]), ("pentan", "CCCCC", ["n-pentan"]),
    ("2-metylbutan", "CCC(C)C", ["isopentan"]), ("2,2-đimetylpropan", "CC(C)(C)C", ["neopentan"]),
    ("eten", "C=C", ["etilen"]), ("propen", "C=CC", ["propilen"]), ("but-1-en", "C=CCC", ["1-buten"]),
    ("but-2-en", "CC=CC", ["2-buten"]), ("etin", "C#C", ["axetilen"]), ("propin", "C#CC", []),
    ("benzen", "c1ccccc1", []), ("toluen", "Cc1ccccc1", ["metylbenzen"]), ("stiren", "C=Cc1ccccc1", ["vinylbenzen"]),
    ("phenol", "Oc1ccccc1", ["hiđroxibenzen"]), ("anilin", "Nc1ccccc1", ["phenylamin"]),
    ("metanol", "CO", ["ancol metylic"]), ("etanol", "CCO", ["ancol etylic"]),
    ("propan-1-ol", "CCCO", ["ancol propylic"]), ("propan-2-ol", "CC(C)O", ["ancol isopropylic"]),
    ("butan-1-ol", "CCCCO", []), ("etylen glicol", "OCCO", ["etan-1,2-điol"]),
    ("glixerol", "OCC(O)CO", ["glixerin", "propan-1,2,3-triol"]),
    ("metanal", "C=O", ["fomanđehit", "anđehit fomic"]), ("etanal", "CC=O", ["anđehit axetic", "axetanđehit"]),
    ("propanon", "CC(C)=O", ["axeton"]), ("axit metanoic", "OC=O", ["axit fomic"]),
    ("axit etanoic", "CC(=O)O", ["axit axetic"]), ("axit propanoic", "CCC(=O)O", ["axit propionic"]),
    ("axit benzoic", "OC(=O)c1ccccc1", []), ("metyl axetat", "COC(C)=O", ["metyl etanoat"]),
    ("etyl axetat", "CCOC(C)=O", ["etyl etanoat"]), ("metylamin", "CN", ["metanamin"]), ("etylamin", "CCN", ["etanamin"]),
    ("glyxin", "NCC(=O)O", ["axit aminoaxetic"]), ("alanin", "CC(N)C(=O)O", ["axit 2-aminopropanoic"]),
    ("đimetyl ete", "COC", []), ("đietyl ete", "CCOCC", []), ("etyl clorua", "CCCl", ["cloetan"]),
    ("clorofom", "ClC(Cl)Cl", ["triclometan"]),
]

GRAPH_PROMPT = RULES + """

Đây là ảnh BÀI LÀM của học sinh và sẽ được chấm điểm. Hãy CHÉP LẠI TRUNG THỰC cấu trúc học sinh đã vẽ dưới dạng ĐỒ THỊ nguyên tử - liên kết. TUYỆT ĐỐI không sửa bài, không đoán học sinh định vẽ chất nào và không tự cân lại hóa trị. Nếu hình vẽ sai (ví dụ nguyên tử C nối với 5 nét) thì vẫn chép đúng 5 nét đó.

Quy tắc:
- Mỗi nguyên tử trong hình là một đối tượng {"id":số thứ tự bắt đầu từ 1,"el":"C" hoặc "O", "N", "Cl", "Br"...,"h":số H}.
- Công thức khung (đường gấp khúc): mỗi đầu mút và mỗi chỗ gấp khúc là một nguyên tử C với "h":null (H ngầm định, không vẽ).
- Công thức thu gọn hoặc khai triển: "h" là SỐ H được ghi kèm hoặc vẽ nối trực tiếp vào nguyên tử đó. Ví dụ CH3 thì h=3; nếu có 4 nét nối từ C tới 4 chữ H thì h=4. Đếm đúng từng nét, không làm tròn theo hóa trị. Nguyên tử không ghi và không vẽ H nào thì "h":null.
- KHÔNG tạo đối tượng riêng cho các H đã tính vào "h".
- Mỗi liên kết giữa hai nguyên tử (không kể H) là [id1,id2,bậc], bậc 1, 2 hoặc 3 tính theo số nét song song.
- Vòng benzen vẽ có vòng tròn bên trong: ghi 6 nguyên tử C và các liên kết xen kẽ bậc 1 và 2.
Chỉ trả về JSON: {"atoms":[...],"bonds":[[1,2,1],...],"ghi_chu":"điều chưa rõ hoặc mơ hồ trong hình, có thể rỗng"}"""


def graph_to_smiles(g):
    """Dựng phân tử từ đồ thị AI chép lại, giữ nguyên số H đã vẽ để lỗi hóa trị không bị che mất."""
    atoms, bonds = g.get("atoms") or [], g.get("bonds") or []
    if not atoms:
        return ""
    rw, idx = Chem.RWMol(), {}
    for a in atoms:
        el = str(a.get("el", "C")).strip()
        at = Chem.Atom(el[:1].upper() + el[1:].lower())
        h = a.get("h")
        if isinstance(h, (int, float)) and not isinstance(h, bool):
            at.SetNumExplicitHs(int(h))
            at.SetNoImplicit(True)
        idx[a["id"]] = rw.AddAtom(at)
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for b in bonds:
        i, j, o = b[0], b[1], int(b[2])
        if i != j and rw.GetBondBetweenAtoms(idx[i], idx[j]) is None:
            rw.AddBond(idx[i], idx[j], kinds[o])
    m = rw.GetMol()
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return Chem.MolToSmiles(m)


def transcribe_student(img):
    r = ask_ai([types.Part.from_bytes(data=to_jpeg(img), mime_type="image/jpeg"), GRAPH_PROMPT], temperature=0)
    try:
        smi = graph_to_smiles(r)
    except Exception:
        raise ValueError("Không dựng được cấu trúc từ ảnh, hãy chụp rõ hơn.")
    return smi, r.get("ghi_chu", "")


def analyze_image_strict(img):
    """Đọc ảnh trung thực (đồ thị nguyên tử) rồi mới gọi tên và tính chất nếu công thức hợp lệ."""
    smi, note = transcribe_student(img)
    res = {"smiles": smi}
    if smi and parse_student(smi)[0] is not None:
        res = analyze_smiles(smi)
        res["smiles"] = smi
        res.pop("do_tin_cay", None)
    if note:
        old = res.get("luu_y") or []
        res["luu_y"] = ([old] if isinstance(old, str) else list(old)) + [note]
    return res


RANDOM_Q = "🎲 Ngẫu nhiên"


def new_question(kind):
    st.session_state.pq = {"kind": kind, "idx": random.randrange(len(BANK)), "result": None}
    st.session_state["pr_pick"] = RANDOM_Q


def practice_ui():
    st.session_state.setdefault("pq", None)
    st.session_state.setdefault("ph", [])
    top = st.container()  # kết quả chấm và lý do sai hiện ở đầu tab
    c1, c2 = st.columns([3, 2])
    with c1:
        kind = st.radio("Dạng bài", ["✍️ Vẽ công thức cấu tạo", "🏷️ Gọi tên chất"], horizontal=True, key="pr_kind")
    with c2:
        who = st.text_input("Tên học sinh (không bắt buộc)", key="pr_name")
    if st.session_state.pq is None or st.session_state.pq["kind"] != kind:
        new_question(kind)
    names = [b[0] for b in BANK]
    pick = st.selectbox("Chọn đề cụ thể (hoặc để ngẫu nhiên)", [RANDOM_Q] + names, key="pr_pick")
    if pick != RANDOM_Q and names.index(pick) != st.session_state.pq["idx"]:
        st.session_state.pq = {"kind": kind, "idx": names.index(pick), "result": None}
    q = st.session_state.pq
    ten, smi, alias = BANK[q["idx"]]
    mt = Chem.MolFromSmiles(smi)

    with st.container(border=True):
        if kind.startswith("✍️"):
            st.markdown(f"#### Viết công thức cấu tạo của: **{ten}**")
            if st.checkbox("Gợi ý: hiện công thức phân tử", key=f"pr_hint_{q['idx']}"):
                st.markdown(f"Công thức phân tử: **{pretty(formula_of(mt))}**")
            how = st.radio("Nộp bài bằng", ["🖼️ Ảnh bài làm", "📷 Chụp ảnh", "⌨️ Nhập SMILES"], horizontal=True, key="pr_how")
            img, ans = None, ""
            if how.startswith("🖼️"):
                f = st.file_uploader("Ảnh bài làm", type=["jpg", "jpeg", "png", "webp"], key="pr_up")
                img = f.getvalue() if f else None
            elif how.startswith("📷"):
                f = st.camera_input("Chụp bài làm", key="pr_cam")
                img = f.getvalue() if f else None
            else:
                ans = st.text_input("SMILES bài làm", key="pr_smi").strip()
            go = st.button("Nộp bài", type="primary", disabled=not (img or ans), key="pr_go")
        else:
            st.markdown("#### Gọi tên chất sau")
            st.image(Draw.MolToImage(mt, size=(360, 240)))
            ans = st.text_input("Tên chất (theo SGK)", key="pr_name_ans").strip()
            img = None
            go = st.button("Nộp bài", type="primary", disabled=not ans, key="pr_go2")

    if go and q["result"] is None:
        if img and st.session_state.n >= MAX_PER_SESSION:
            show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
        else:
            try:
                note, given = "", ans
                if kind.startswith("✍️"):
                    if img:
                        with st.spinner("Đang đọc bài làm…"):
                            given, note = transcribe_student(img)
                        st.session_state.n += 1
                    res = grade_structure(given, smi)
                else:
                    ok = norm_name(ans) in {norm_name(x) for x in [ten] + alias}
                    res = {"kq": "dung" if ok else "sai",
                           "nx": ["Chính xác." if ok else "Chưa đúng. Hãy xác định mạch chính, nhóm chức và vị trí nhóm thế rồi gọi tên lại."],
                           "goi_y": []}
                res.update({"given": given, "note": note})
                q["result"] = res
                st.session_state.ph.append({
                    "Thời gian": datetime.now().strftime("%H:%M:%S"), "Học sinh": who, "Dạng bài": "vẽ CTCT" if kind.startswith("✍️") else "gọi tên",
                    "Đề": ten, "Bài làm": given, "Kết quả": res["kq"]})
            except Exception as e:
                show_top_error(top, "Có lỗi khi chấm bài, hãy thử lại.", e)

    with top:
        res = q["result"]
        if res:
            head = {"dung": "Đúng rồi!", "dong_phan": "CHƯA ĐÚNG: đã vẽ một đồng phân khác", "sai_cong_thuc": "CHƯA ĐÚNG: sai công thức phân tử",
                    "loi": "CHƯA ĐÚNG: công thức có lỗi", "sai": "CHƯA ĐÚNG"}[res["kq"]]
            if res["kq"] == "dung":
                st.success("**" + head + "**")
                for t in res["nx"]:
                    st.markdown("- " + t)
            else:
                st.markdown(errbanner(head, *res["nx"]), unsafe_allow_html=True)
            st.markdown("**Đề:** " + (f"viết công thức cấu tạo của **{ten}**" if kind.startswith("✍️") else "gọi tên chất trong hình"))
            for t in res["goi_y"]:
                st.markdown("- 💡 " + t)
            if res.get("note"):
                st.caption("Ghi chú khi đọc ảnh: " + res["note"])
            if kind.startswith("✍️") and res.get("given"):
                im = draw_student(res["given"])
                if im is not None:
                    st.markdown("**Hình em vẽ (máy đọc lại đúng như hình, giữ nguyên lỗi" + (", nguyên tử sai khoanh đỏ" if res["kq"] == "loi" else "") + "):**")
                    st.image(im)
            with st.expander("Xem đáp án"):
                st.image(Draw.MolToImage(mt, size=(300, 200)))
                st.markdown(f"**{ten}** · {pretty(formula_of(mt))} · `{smi}`")
                gl = groups_of(mt)
                if gl:
                    st.markdown("Nhóm chức/đặc điểm: " + ", ".join(gl))
            st.button("➡️ Câu tiếp theo", on_click=new_question, args=(kind,), key="pr_next")

    hist = st.session_state.ph
    if hist:
        st.divider()
        n_ok = sum(1 for h in hist if h["Kết quả"] == "dung")
        a, b, c = st.columns(3)
        a.metric("Đã làm", len(hist))
        b.metric("Đúng", n_ok)
        c.metric("Tỉ lệ đúng", f"{round(100 * n_ok / len(hist))}%")
        st.dataframe(hist, hide_index=True, width="stretch")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
        st.download_button("⬇️ Tải phiếu kết quả (CSV, nộp cho giáo viên)", buf.getvalue().encode("utf-8-sig"),
                           file_name="phieu_ket_qua.csv", mime="text/csv", key="pr_dl")
    st.caption("Phần chấm công thức do RDKit so sánh cấu tạo nên chính xác. Phần đọc ảnh bài làm do AI thực hiện nên có thể đọc sai nét viết khó.")


# ---------------- Đánh giá độ chính xác (giáo viên / nhóm nghiên cứu) ----------------
def parse_truth(text):
    rows = {}
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) >= 2 and parts[0] and parts[1]:
            rows[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return rows


def bench_row(pred, truth):
    mp, mt = Chem.MolFromSmiles(pred or ""), Chem.MolFromSmiles(truth)
    if mp is None or mt is None:
        return {"hop_le": False, "khop_cong_thuc": False, "khop_cau_tao": False, "khop_hoan_toan": False}
    return {"hop_le": True, "khop_cong_thuc": formula_of(mp) == formula_of(mt),
            "khop_cau_tao": noiso(mp) == noiso(mt),
            "khop_hoan_toan": Chem.MolToSmiles(mp) == Chem.MolToSmiles(mt)}


def bench_summary(rows):
    n = len(rows)
    pct = lambda k: round(100 * sum(1 for r in rows if r[k]) / n, 1) if n else 0
    return {"n": n, "hop_le": pct("hop_le"), "khop_cong_thuc": pct("khop_cong_thuc"),
            "khop_cau_tao": pct("khop_cau_tao"), "khop_hoan_toan": pct("khop_hoan_toan"),
            "tb_giay": round(sum(r["giay"] for r in rows) / n, 1) if n else 0}


def bench_ui():
    code = st.secrets.get("TEACHER_CODE", "")
    if code and st.text_input("Mã giáo viên", type="password", key="bm_code") != code:
        st.info("Tab này dành cho giáo viên hoặc nhóm nghiên cứu. Nhập mã giáo viên để dùng.")
        return
    st.markdown("Tải lên một bộ ảnh đã biết đáp án để đo **độ chính xác nhận dạng**. Kết quả có thể đưa vào báo cáo đề tài.")
    files = st.file_uploader("Ảnh kiểm thử", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="bm_files")
    up = st.file_uploader("File đáp án CSV (tên_file, SMILES, nhóm)", type=["csv", "txt"], key="bm_csv")
    txt = st.text_area("Hoặc dán đáp án, mỗi dòng: tên_file, SMILES, nhóm", height=130, key="bm_truth",
                       placeholder="03_toluen.jpg, Cc1ccccc1, benzen\n06_etyl_axetat.jpg, CCOC(C)=O, công thức khung")
    method = st.radio("Phương pháp đọc ảnh", ["Đồ thị nguyên tử (trung thực với bài vẽ)", "SMILES trực tiếp"], horizontal=True, key="bm_method")
    src = up.getvalue().decode("utf-8-sig", errors="ignore") if up else txt
    truth = parse_truth(src)
    todo = [f for f in (files or []) if f.name in truth][:BENCH_MAX]
    if files and truth:
        st.caption(f"Có {len(todo)}/{len(files)} ảnh khớp tên trong file đáp án (tối đa {BENCH_MAX} ảnh mỗi lần chạy).")
    if st.button("Chạy đánh giá", type="primary", disabled=not todo, key="bm_go"):
        rows, bar = [], st.progress(0.0, text="Đang chạy…")
        for i, f in enumerate(todo):
            t0 = time.time()
            pred, err = "", ""
            try:
                pred = transcribe_student(f.getvalue())[0] if method.startswith("Đồ thị") else to_smiles(f.getvalue(), "")
            except Exception as e:
                err = type(e).__name__
            tr, grp = truth[f.name]
            r = {"tep": f.name, "nhom": grp, "dap_an": tr, "phuong_phap": "do_thi" if method.startswith("Đồ thị") else "smiles", "ket_qua_may": pred, "loi": err, "giay": round(time.time() - t0, 1)}
            r.update(bench_row(pred, tr))
            rows.append(r)
            bar.progress((i + 1) / len(todo), text=f"Đã xử lý {i + 1}/{len(todo)}")
        st.session_state.bm = rows
    rows = st.session_state.get("bm")
    if rows:
        sm = bench_summary(rows)
        a, b, c, d = st.columns(4)
        a.metric("Số ảnh", sm["n"])
        b.metric("Khớp cấu tạo", f"{sm['khop_cau_tao']}%")
        c.metric("Khớp công thức PT", f"{sm['khop_cong_thuc']}%")
        d.metric("SMILES hợp lệ", f"{sm['hop_le']}%")
        st.caption(f"Khớp hoàn toàn (kể cả lập thể): {sm['khop_hoan_toan']}% · thời gian trung bình: {sm['tb_giay']} giây/ảnh")
        grp = sorted({r["nhom"] for r in rows if r["nhom"]})
        if grp:
            st.markdown("**Theo nhóm nét vẽ**")
            st.dataframe([{"Nhóm": g, **{k: v for k, v in bench_summary([r for r in rows if r["nhom"] == g]).items()
                                         if k in ("n", "khop_cau_tao", "khop_cong_thuc", "hop_le")}} for g in grp],
                         hide_index=True, width="stretch")
        st.dataframe(rows, hide_index=True, width="stretch")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        st.download_button("⬇️ Tải bảng kết quả (CSV)", buf.getvalue().encode("utf-8-sig"), file_name="ket_qua_danh_gia.csv",
                           mime="text/csv", key="bm_dl")
    st.caption("Khớp cấu tạo: SMILES máy đọc được và đáp án giống nhau về cấu tạo (RDKit chuẩn hóa trước khi so sánh).")


# ---------------- Giao diện ----------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&family=Caveat:wght@600;700&display=swap');
:root{--ink:#14224a;--muted:#5d6b8c;--blue:#2f80ff;--violet:#7b4dff;--teal:#10b98a;--line:#e3e9f6}
.stApp,.stApp div,.stApp p,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp li,.stApp a,.stApp h1,.stApp h2,.stApp h3,.stApp h4{font-family:'Be Vietnam Pro',system-ui,sans-serif}
.stApp{color-scheme:light;
background:radial-gradient(900px 520px at 78% -8%,rgba(122,140,255,.55),transparent 62%),radial-gradient(700px 500px at 4% 105%,rgba(47,128,255,.35),transparent 60%),linear-gradient(165deg,#0b1438 0%,#1a2f7a 55%,#7f9be0 100%)}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{max-width:1180px!important;margin:10px auto 28px;padding:18px 22px 26px!important;color:var(--ink);
background:rgba(244,247,255,.95);border:1px solid rgba(255,255,255,.7);border-radius:26px;box-shadow:0 24px 70px rgba(5,10,40,.45)}

/* ---- thanh bên ---- */
section[data-testid="stSidebar"]{background:linear-gradient(185deg,#070d2b 0%,#0c1a52 60%,#0a1440 100%);border-right:1px solid rgba(255,255,255,.06)}
section[data-testid="stSidebar"][aria-expanded="true"]{width:262px!important;min-width:262px!important}
.sbrand{display:flex;align-items:center;gap:10px;margin:2px 0 20px}
.sbrand svg{width:46px;height:46px;flex:none}
.sbrand b{font-size:1.75rem;font-weight:800;color:#fff;letter-spacing:-.01em}
.sbrand b i{font-style:normal;color:#38d5ff}
.st-key-nav_side div[data-testid="stRadioGroup"]{flex-direction:column;align-items:stretch;gap:6px}
.st-key-nav_side.st-key-nav_side label[data-testid="stRadioOption"]{width:100%;margin:0;padding:12px 14px;border-radius:12px;background:transparent;transition:background .15s}
.st-key-nav_side.st-key-nav_side label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]){display:none}
.st-key-nav_side.st-key-nav_side label[data-testid="stRadioOption"] p{color:#cdd7ff;font-weight:600;font-size:.93rem}
.st-key-nav_side.st-key-nav_side label[data-testid="stRadioOption"]:hover{background:rgba(255,255,255,.07)}
.st-key-nav_side.st-key-nav_side label[data-testid="stRadioOption"]:has(input:checked){background:linear-gradient(90deg,#2f80ff,#7b4dff);box-shadow:0 8px 22px rgba(63,100,255,.45)}
.st-key-nav_side.st-key-nav_side label[data-testid="stRadioOption"]:has(input:checked) p{color:#fff}
.squote{font-family:'Caveat',cursive!important;font-size:1.5rem;line-height:1.15;color:#c9d4ff;margin:34px 6px 0;transform:rotate(-6deg);opacity:.95}
.sdna{margin-top:18px;opacity:.55}

/* ---- thanh trên và banner ---- */
.topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 4px 12px}
.topbar .tag{font-weight:600;color:var(--muted);font-size:1rem}
.topbar .ask{display:inline-flex;align-items:center;gap:8px;background:#fff;border:1px solid var(--line);border-radius:999px;padding:8px 18px;
font-weight:700;color:var(--ink)!important;text-decoration:none!important;box-shadow:0 4px 14px rgba(30,60,140,.10)}
.hero{position:relative;overflow:hidden;border-radius:22px;padding:26px 30px 24px;color:#fff;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.15fr);gap:8px;align-items:center;
background:radial-gradient(520px 260px at 80% 42%,rgba(70,120,255,.55),transparent 70%),linear-gradient(110deg,#0b1038 0%,#1a1f6b 55%,#2a2f9a 100%)}
.hero .brand{display:flex;gap:14px;align-items:center}
.hero .brand > svg{width:68px;height:68px;flex:none}
.hero .t{font-size:3.2rem;font-weight:800;line-height:1;letter-spacing:-.02em;color:#fff}
.hero .t b{color:#38d5ff;font-weight:800}
.hero p{margin:10px 0 0;font-size:1.08rem;line-height:1.5;color:#e3e9ff;max-width:32em}
.hero .chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
.hero .chips span{background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.16);border-radius:999px;padding:7px 16px;font-weight:600;font-size:.92rem}
.hero .art{position:relative;min-height:200px}
.hero .art > svg{position:absolute;inset:0;width:100%;height:100%}
.hero .script{position:absolute;right:-6px;top:14%;font-family:'Caveat',cursive!important;font-size:1.75rem;font-weight:700;line-height:1.05;transform:rotate(-8deg);color:#fff}
.hero .script svg{display:block;width:130px;height:12px;margin-top:4px}

/* ---- thẻ chức năng ---- */
.st-key-nav_top{background:#fff;border-radius:18px;padding:10px 12px 0;box-shadow:0 4px 18px rgba(30,60,140,.08);margin:12px 0}
.st-key-nav_top.st-key-nav_top div[data-testid="stRadioGroup"]{flex-wrap:nowrap;overflow-x:auto;gap:6px;padding-bottom:14px}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]{padding:9px 12px!important;font-size:.9rem;border-color:transparent!important;box-shadow:none!important;white-space:nowrap;position:relative}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:nth-child(1){background:#e8f1ff}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:nth-child(2){background:#ffe9f0}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:nth-child(3){background:#efe9ff}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:nth-child(4){background:#dff7ee}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:nth-child(5){background:#fff0d9}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:nth-child(6){background:#e8f1ff}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:has(input:checked){background:#fff!important;box-shadow:0 6px 18px rgba(16,185,138,.25)!important;border-color:#bff0e0!important}
.st-key-nav_top.st-key-nav_top label[data-testid="stRadioOption"]:has(input:checked)::after{content:"";position:absolute;left:16%;right:16%;bottom:-8px;width:auto;height:4px;margin:0;border-radius:4px;background:var(--teal)}
.st-key-page_card{background:#fff;border-radius:20px;padding:20px 24px 18px;border:1px solid #eef2fb;box-shadow:0 6px 24px rgba(30,60,140,.08)}
.pagehead{position:relative;display:flex;gap:18px;align-items:center;border-radius:18px;padding:16px 22px;margin:0 0 18px;overflow:hidden;background:linear-gradient(100deg,var(--a1),var(--a2));border:1px solid rgba(20,34,74,.05)}
.pagehead .ic{flex:none;width:70px;height:70px;border-radius:18px;display:grid;place-items:center;font-size:2rem;background:linear-gradient(145deg,var(--c1),var(--c2));box-shadow:0 10px 22px var(--sh)}
.pagehead b{display:block;font-size:1.5rem;font-weight:800;color:var(--ink);line-height:1.2}
.pagehead .d{display:block;color:var(--muted);margin-top:4px;font-size:.98rem}
.pagehead .note{margin-left:auto;padding-left:12px;font-family:'Caveat',cursive!important;font-size:1.8rem;font-weight:700;line-height:1;color:var(--c1);transform:rotate(-6deg);white-space:nowrap}
.fcard{border-radius:18px;padding:18px 18px 14px;margin-bottom:8px;background:linear-gradient(150deg,var(--a1),#fff);border:1px solid var(--line);min-height:150px}
.fcard .ic{width:50px;height:50px;border-radius:14px;display:grid;place-items:center;font-size:1.5rem;background:linear-gradient(145deg,var(--c1),var(--c2));box-shadow:0 8px 18px var(--sh);margin-bottom:10px}
.fcard b{display:block;font-size:1.08rem;color:var(--ink)}
.fcard span{display:block;margin-top:4px;color:var(--muted);font-size:.92rem;line-height:1.45}
.usage{display:flex;align-items:center;gap:10px;margin-top:20px;padding:12px 16px;border-radius:14px;background:#f4f7ff;border:1px solid var(--line);color:var(--muted);font-size:.92rem}
.usage b{color:var(--teal)}

/* ---- ô nhập liệu ---- */
.block-container div[data-testid="stRadioGroup"]{gap:10px;flex-wrap:wrap}
.block-container label[data-testid="stRadioOption"]{display:inline-flex;align-items:center;flex-wrap:nowrap;margin:0;padding:9px 18px 9px 16px;background:#fff;border:1.5px solid var(--line);border-radius:999px;box-shadow:0 2px 8px rgba(30,60,140,.06);transition:border-color .15s,box-shadow .15s}
.block-container label[data-testid="stRadioOption"] > div > div:not([data-testid="stMarkdownContainer"]){display:none}
.block-container label[data-testid="stRadioOption"] p{color:var(--ink);font-weight:600}
.block-container label[data-testid="stRadioOption"]:hover{border-color:#9fe3cc}
.block-container label[data-testid="stRadioOption"]:has(input:checked){background:#effcf7;border-color:var(--teal);box-shadow:0 0 0 3px rgba(16,185,138,.15)}
.block-container label[data-testid="stRadioOption"]:has(input:checked)::after{content:"✓";display:inline-flex;align-items:center;justify-content:center;width:20px;height:20px;margin-left:10px;border-radius:50%;background:var(--teal);color:#fff;font-size:.72rem;font-weight:800}
.block-container div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p{font-weight:700;border-left:4px solid var(--blue);padding-left:8px;line-height:1.1}
[data-testid="stFileUploaderDropzone"]{display:flex;flex-direction:row-reverse;align-items:center;justify-content:space-between;padding:22px 26px;border:2px dashed #9db8f5;border-radius:18px;background:linear-gradient(100deg,#f3f7ff,#f8f5ff)}
[data-testid="stFileUploaderDropzone"] svg{color:#7a5cff}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span,[data-testid="stFileUploaderDropzoneInstructions"] > div > small{display:none}
[data-testid="stFileUploaderDropzoneInstructions"] > div::before{content:"Kéo thả ảnh vào đây";display:block;font-weight:800;font-size:1.15rem;color:var(--ink)}
[data-testid="stFileUploaderDropzoneInstructions"] > div::after{content:"hoặc nhấn để chọn file · JPG, PNG, WEBP · tối đa 200MB";display:block;margin-top:2px;color:var(--muted);font-size:.9rem}
[data-testid="stFileUploaderDropzone"] button{font-size:0;padding:.7rem 1.9rem;border:0;border-radius:999px;background:linear-gradient(90deg,#1f7bff,#7a3cff);color:#fff;box-shadow:0 8px 20px rgba(60,90,255,.35)}
[data-testid="stFileUploaderDropzone"] button > *{display:none}
[data-testid="stFileUploaderDropzone"] button::after{content:"Tải ảnh lên";font-size:1rem;font-weight:700;color:#fff}
[data-testid="stCameraInput"]{max-width:440px;margin:0 auto}
.block-container [data-testid="stBaseButton-primary"]{padding:.6rem 1.8rem;border:0;border-radius:999px;background:linear-gradient(90deg,#1f7bff,#7a3cff);color:#fff;font-weight:700;box-shadow:0 8px 20px rgba(60,90,255,.30)}
.block-container [data-testid="stBaseButton-primary"]:disabled{opacity:.45;box-shadow:none}
.block-container [data-testid="stBaseButton-secondary"]{border:1.5px solid var(--line);border-radius:999px;font-weight:600;color:var(--ink)}
button{transition:transform .15s ease,box-shadow .15s ease}
button:hover{transform:translateY(-1px)}
button[data-baseweb="tab"]{font-size:1.02rem;font-weight:600;padding:12px 14px}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--teal)}
div[data-baseweb="tab-highlight"]{background-color:var(--teal)}
[data-testid="stMetric"]{background:#f7f9ff;border:1px solid var(--line);border-radius:14px;padding:12px 16px}
.errtop{color:#d00000;font-weight:800;font-size:1.08rem;line-height:1.5;background:rgba(220,0,0,.08);border:2px solid rgba(220,0,0,.45);border-radius:12px;padding:12px 16px;margin:6px 0 12px}

@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
@media(max-width:900px){.hero{grid-template-columns:1fr}.hero .art{display:none}.pagehead .note{display:none}}
@media(max-width:640px){.block-container{padding:12px 12px 20px!important;border-radius:18px}.hero{padding:18px}.hero .t{font-size:2.3rem}.hero .brand > svg{width:48px;height:48px}.topbar .tag{display:none}.pagehead{padding:12px 14px}.pagehead .ic{width:54px;height:54px;font-size:1.5rem}.st-key-page_card{padding:14px}}
</style>
"""

ACC = {  # (đậm, nhạt, nền 1, nền 2, bóng)
    "home": ("#7b4dff", "#a88bff", "#efe9ff", "#f8f5ff", "rgba(123,77,255,.35)"),
    "struct": ("#2f80ff", "#6aa6ff", "#e4eeff", "#f5f9ff", "rgba(47,128,255,.35)"),
    "rx": ("#ec4a7e", "#ff8fb0", "#ffe6ee", "#fff7fa", "rgba(236,74,126,.30)"),
    "iso": ("#7b4dff", "#a88bff", "#ece6ff", "#f8f5ff", "rgba(123,77,255,.35)"),
    "equip": ("#0f9d78", "#4ddcb2", "#d9f5ea", "#f2fcf8", "rgba(16,185,138,.35)"),
    "practice": ("#f08a00", "#ffc65a", "#ffefd2", "#fffaf0", "rgba(240,138,0,.30)"),
    "bench": ("#2f80ff", "#6aa6ff", "#e4eeff", "#f5f9ff", "rgba(47,128,255,.35)"),
}


def acc_style(pid):
    c1, c2, a1, a2, sh = ACC[pid]
    return f"--c1:{c1};--c2:{c2};--a1:{a1};--a2:{a2};--sh:{sh}"


def helix_svg(uid):
    """Logo xoắn kép (DNA) vẽ bằng đường sin, nghiêng như logo mẫu."""
    a, b, rungs = [], [], []
    for i in range(0, 29):
        y = 4 + i * 2
        s = math.sin(2 * math.pi * (y - 4) / 56 * 1.5)
        a.append(f"{32 + 15 * s:.1f},{y}")
        b.append(f"{32 - 15 * s:.1f},{y}")
        if i % 3 == 1 and abs(s) > 0.35:
            rungs.append(f'<line x1="{32 + 15 * s:.1f}" y1="{y}" x2="{32 - 15 * s:.1f}" y2="{y}"/>')
    return (
        f'<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg"><defs>'
        f'<linearGradient id="{uid}a" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffb02e"/><stop offset="1" stop-color="#ff5d8f"/></linearGradient>'
        f'<linearGradient id="{uid}b" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#38d5ff"/><stop offset="1" stop-color="#5b6bff"/></linearGradient></defs>'
        f'<g transform="rotate(28 32 32)" fill="none" stroke-linecap="round" stroke-linejoin="round">'
        f'<g stroke="#ffe7a3" stroke-width="2.2" opacity=".85">{"".join(rungs)}</g>'
        f'<polyline points="{" ".join(a)}" stroke="url(#{uid}a)" stroke-width="5"/>'
        f'<polyline points="{" ".join(b)}" stroke="url(#{uid}b)" stroke-width="5"/></g></svg>'
    )


HERO_ART = (
    '<svg viewBox="0 0 720 230" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg"><defs>'
    '<radialGradient id="hg" cx=".6" cy=".4" r=".6"><stop offset="0" stop-color="#3b7bff" stop-opacity=".7"/><stop offset="1" stop-color="#3b7bff" stop-opacity="0"/></radialGradient>'
    '<linearGradient id="hcam" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#46c0ff"/><stop offset="1" stop-color="#2a5bff"/></linearGradient>'
    '<linearGradient id="hliqp" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c08cff"/><stop offset="1" stop-color="#6a3cff"/></linearGradient>'
    '<linearGradient id="hliqb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#58d6ff"/><stop offset="1" stop-color="#2f5dff"/></linearGradient></defs>'
    '<ellipse cx="320" cy="100" rx="250" ry="120" fill="url(#hg)"/>'
    '<polygon points="38,40 54,31 70,40 70,58 54,67 38,58" fill="none" stroke="#fff" stroke-opacity=".28" stroke-width="2"/>'
    '<polygon points="96,18 108,11 120,18 120,32 108,39 96,32" fill="none" stroke="#fff" stroke-opacity=".22" stroke-width="2"/>'
    # sách mở
    '<path d="M150 198 Q220 176 292 198 L292 224 Q220 204 150 222 Z" fill="#f2f5ff"/>'
    '<path d="M292 198 Q364 176 434 198 L434 222 Q364 204 292 224 Z" fill="#d9e3ff"/>'
    '<path d="M292 198 L292 224" stroke="#9fb2ee" stroke-width="2"/>'
    # bình cầu
    '<rect x="108" y="86" width="24" height="34" rx="3" fill="#fff" fill-opacity=".12" stroke="#fff" stroke-opacity=".7" stroke-width="2.5"/>'
    '<rect x="104" y="82" width="32" height="7" rx="3" fill="#fff" fill-opacity=".6"/>'
    '<circle cx="120" cy="152" r="38" fill="#fff" fill-opacity=".10" stroke="#fff" stroke-opacity=".7" stroke-width="2.5"/>'
    '<path d="M83 156 a37 37 0 0 0 74 0 z" fill="url(#hliqb)"/>'
    # bình tam giác
    '<path d="M223 66 h24 v36 l44 84 a10 10 0 0 1 -9 15 h-94 a10 10 0 0 1 -9 -15 l44 -84 z" fill="#fff" fill-opacity=".10" stroke="#fff" stroke-opacity=".75" stroke-width="2.5" stroke-linejoin="round"/>'
    '<path d="M205 142 h60 l25 44 q3 8 -6 8 h-98 q-9 0 -6 -8 z" fill="url(#hliqp)"/>'
    '<rect x="219" y="60" width="32" height="8" rx="3" fill="#fff" fill-opacity=".65"/>'
    '<circle cx="232" cy="168" r="4" fill="#fff" fill-opacity=".5"/><circle cx="252" cy="158" r="3" fill="#fff" fill-opacity=".5"/>'
    # phân tử
    '<g stroke="#fff" stroke-opacity=".75" stroke-width="3"><line x1="300" y1="58" x2="318" y2="90"/><line x1="318" y1="90" x2="294" y2="114"/><line x1="318" y1="90" x2="338" y2="64"/></g>'
    '<circle cx="300" cy="58" r="7" fill="#fff"/><circle cx="318" cy="90" r="10" fill="#38d5ff"/><circle cx="294" cy="114" r="8" fill="#ff4fa3"/><circle cx="338" cy="64" r="6" fill="#ffb02e"/>'
    # thẻ camera
    '<g transform="rotate(-5 415 86)"><rect x="345" y="30" width="140" height="112" rx="22" fill="#10215e" stroke="url(#hcam)" stroke-width="5"/>'
    '<rect x="357" y="42" width="116" height="88" rx="14" fill="#1a3fa8" fill-opacity=".85"/>'
    '<rect x="380" y="62" width="70" height="14" rx="5" fill="#fff"/><rect x="372" y="70" width="86" height="50" rx="10" fill="#fff"/>'
    '<circle cx="415" cy="95" r="16" fill="url(#hcam)"/><circle cx="415" cy="95" r="8" fill="#0f2a80"/><circle cx="411" cy="91" r="3" fill="#fff" fill-opacity=".8"/></g>'
    '<g fill="#fff"><circle cx="262" cy="30" r="2.5" opacity=".8"/><circle cx="500" cy="190" r="2" opacity=".6"/><circle cx="60" cy="180" r="2.5" opacity=".5"/></g></svg>'
)

_m = re.match(r"^(.*?[a-zà-ỹ])([A-Z][^\s]*)$", APP_NAME)
BRAND_HTML = f"{html.escape(_m.group(1))}<b>{html.escape(_m.group(2))}</b>" if _m else html.escape(APP_NAME)

SWOOSH = '<svg viewBox="0 0 130 12"><path d="M3 9 C35 2 85 2 127 6" fill="none" stroke="#ffd35a" stroke-width="3.5" stroke-linecap="round"/></svg>'
DNA_DECO = (
    '<svg class="sdna" viewBox="0 0 220 90" xmlns="http://www.w3.org/2000/svg" fill="none" stroke-linecap="round">'
    '<path d="M5 70 C45 10 85 100 125 40 S190 20 215 10" stroke="#38d5ff" stroke-width="3"/>'
    '<path d="M5 40 C45 100 85 10 125 70 S190 80 215 60" stroke="#7b4dff" stroke-width="3"/></svg>'
)

st.markdown(CSS, unsafe_allow_html=True)

# ---------------- Điều hướng ----------------
TOP = ["struct", "rx", "iso", "equip", "practice", "bench"]
SIDE = ["home"] + TOP
SIDE_LABEL = {
    "home": ":material/home: Trang chủ", "struct": ":material/image_search: Phân tích ảnh",
    "rx": ":material/compare_arrows: Phản ứng", "iso": ":material/hexagon: Đồng phân",
    "equip": ":material/science: Dụng cụ hóa học", "practice": ":material/school: Luyện tập & chấm bài",
    "bench": ":material/bar_chart: Đánh giá",
}
TOP_LABEL = {
    "struct": "✍️ Công thức cấu tạo", "rx": "🧪 Phương trình phản ứng", "iso": "🔀 Đồng phân",
    "equip": "🔬 Dụng cụ hóa học", "practice": "🎓 Luyện tập & chấm bài", "bench": "📊 Đánh giá",
}


def go(p):
    st.session_state.page = p
    st.session_state.nav_side = p
    st.session_state.nav_top = p if p in TOP else None


def _from_side():
    go(st.session_state.nav_side)


def _from_top():
    if st.session_state.nav_top:
        go(st.session_state.nav_top)


if "page" not in st.session_state:
    go("home")

with st.sidebar:
    st.markdown(f'<div class="sbrand">{helix_svg("sb")}<b>{BRAND_HTML}</b></div>', unsafe_allow_html=True)
    st.radio("Điều hướng", SIDE, format_func=SIDE_LABEL.get, key="nav_side", on_change=_from_side, label_visibility="collapsed")
    st.markdown('<div class="squote">Khoa học không chỉ là kiến thức mà còn là khả năng khám phá thế giới!</div>' + DNA_DECO, unsafe_allow_html=True)

st.markdown(
    '<div class="topbar"><div class="tag">Nhìn thấy thế giới – Hiểu rõ hơn</div>'
    '<a class="ask" href="https://gemini.google.com" target="_blank" rel="noopener">✨ Hỏi Gemini</a></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="hero"><div class="l"><div class="brand">{helix_svg("hr")}<div><div class="t">{BRAND_HTML}</div>'
    "<p>Chụp là biết, luyện là nhớ: công thức, phản ứng, đồng phân, dụng cụ, chấm bài tự động</p></div></div>"
    '<div class="chips"><span>⚡ Nhanh chóng</span><span>🎯 Chính xác</span><span>👍 Tiện lợi</span><span>🕒 Học mọi lúc</span></div></div>'
    f'<div class="art">{HERO_ART}<div class="script">Học thông minh<br>cùng {html.escape(APP_NAME)}!{SWOOSH}</div></div></div>',
    unsafe_allow_html=True,
)

pw = st.secrets.get("APP_PASSWORD", "")
if pw and st.text_input("Mật khẩu lớp học", type="password") != pw:
    st.stop()

st.session_state.setdefault("n", 0)
st.session_state.setdefault("res", None)

st.radio("Chức năng", TOP, index=None, format_func=TOP_LABEL.get, key="nav_top", on_change=_from_top, horizontal=True, label_visibility="collapsed")


def intro(icon, title, text, pid="home", note=""):
    note_html = f'<div class="note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="pagehead" style="{acc_style(pid)}"><div class="ic">{icon}</div>'
        f'<div><b>{title}</b><span class="d">{text}</span></div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def errbanner(*lines):
    body = "<br>".join(html.escape(str(x)) for x in lines if x)
    return f'<div class="errtop">❌ {body}</div>'


def show_top_error(top, msg, exc=None):
    """Hiện lỗi ở ĐẦU tab, chữ đỏ in đậm."""
    with top:
        st.markdown(errbanner(msg), unsafe_allow_html=True)
        if exc is not None:
            st.caption(f"Chi tiết kỹ thuật: {type(exc).__name__}: {str(exc)[:400]}")


def structure_ui():
    top = st.container()  # lỗi và lý do sai hiện ở đầu tab
    mode = st.radio("Cách nhập", ["🖼️ Tải ảnh lên", "📷 Chụp ảnh", "⌨️ Nhập SMILES"], horizontal=True, key="st_mode")
    img_bytes, smi_in = None, None
    if mode.startswith("📷"):
        f = st.camera_input("Chụp công thức (viết rõ nét, nền sáng, mỗi ảnh một chất)", key="st_cam")
        img_bytes = f.getvalue() if f else None
    elif mode.startswith("🖼️"):
        f = st.file_uploader("Chọn ảnh công thức (viết rõ nét, nền sáng, mỗi ảnh một chất)", type=["jpg", "jpeg", "png", "webp"], key="st_up")
        img_bytes = f.getvalue() if f else None
    else:
        smi_in = st.text_input("Mã SMILES", placeholder="Ví dụ: CC(=O)O", key="st_smi").strip()

    if st.button("Phân tích", type="primary", disabled=not (img_bytes or smi_in), key="st_go"):
        if st.session_state.n >= MAX_PER_SESSION:
            show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
        else:
            with st.spinner("Đang phân tích…"):
                try:
                    if img_bytes:
                        st.session_state.res = (analyze_image_strict(img_bytes), True)
                        st.session_state.n += 1
                    elif parse_student(smi_in)[0] is None:
                        st.session_state.res = ({"smiles": smi_in}, False)  # SMILES sai: chấm lỗi ngay, không gọi AI
                    else:
                        st.session_state.res = (analyze_smiles(smi_in), False)
                        st.session_state.n += 1
                except Exception as e:
                    st.session_state.res = None
                    show_top_error(top, "Có lỗi khi phân tích, hãy thử lại hoặc chụp rõ hơn.", e)

    if st.session_state.res:
        r, from_img = st.session_state.res
        smiles_r = (r.get("smiles") or "").strip()
        _, err_r = parse_student(smiles_r) if smiles_r else (None, None)
        if err_r:
            with top:
                st.markdown(errbanner("CÔNG THỨC VẼ SAI: " + err_r), unsafe_allow_html=True)
                im = draw_student(smiles_r)
                if im is not None:
                    st.markdown("**Hình máy đọc lại đúng như hình vẽ (nguyên tử sai khoanh đỏ):**")
                    st.image(im)
                if r.get("luu_y"):
                    st.caption("Ghi chú khi đọc ảnh: " + " • ".join([r["luu_y"]] if isinstance(r["luu_y"], str) else r["luu_y"]))
            st.info("Vì công thức sai hóa trị nên chưa thể gọi tên hay tính khối lượng mol. Hãy sửa lại hình vẽ hoặc sửa SMILES ở ô bên dưới.")
        else:
            st.divider()
            render(r, from_img)
        st.divider()
        fix = st.text_input("Nhận dạng sai? Sửa SMILES rồi phân tích lại", value=r.get("smiles", ""), key="st_fix").strip()
        if st.button("Phân tích lại từ SMILES đã sửa", key="st_fix_go") and fix:
            if st.session_state.n >= MAX_PER_SESSION:
                show_top_error(top, "Đã hết số lượt của phiên này.")
            else:
                with st.spinner("Đang phân tích…"):
                    try:
                        st.session_state.res = (analyze_smiles(fix), False)
                        st.session_state.n += 1
                        st.rerun()
                    except Exception as e:
                        show_top_error(top, "Có lỗi khi phân tích lại.")
    st.caption("Kết quả do AI gợi ý, có thể sai ở nét vẽ khó đọc. Công thức phân tử và khối lượng mol được tính trực tiếp từ SMILES bằng RDKit. Hãy đối chiếu SGK trước khi dùng làm đáp án.")


FEATS = [
    ("struct", "✍️", "Công thức cấu tạo", "Chụp công thức vẽ tay → mã SMILES, tên gọi, khối lượng mol, tính chất."),
    ("rx", "🧪", "Phương trình phản ứng", "Nhận dạng loại phản ứng và kiểm tra phương trình đã cân bằng chưa."),
    ("iso", "🔀", "Đồng phân", "Liệt kê đồng phân hoặc so sánh hai chất có phải đồng phân không."),
    ("equip", "🔬", "Dụng cụ hóa học", "Biết tên, công dụng, cách dùng và lưu ý an toàn của dụng cụ."),
    ("practice", "🎓", "Luyện tập & chấm bài", "Ra đề, nộp bài vẽ tay và nhận nhận xét lỗi sai tự động."),
    ("bench", "📊", "Đánh giá", "Chạy bộ ảnh có đáp án để đo độ chính xác nhận dạng."),
]


def home_ui():
    intro("🏠", f"Chào mừng đến với {html.escape(APP_NAME)}", "Chọn một chức năng bên dưới hoặc ở thanh bên trái để bắt đầu.", "home", "Bắt đầu nào!")
    cols = []
    for i, (pid, ic, title, desc) in enumerate(FEATS):
        if i % 3 == 0:
            cols = st.columns(3)
        with cols[i % 3]:
            st.markdown(f'<div class="fcard" style="{acc_style(pid)}"><div class="ic">{ic}</div><b>{title}</b><span>{desc}</span></div>', unsafe_allow_html=True)
            st.button("Dùng ngay", key=f"home_{pid}", on_click=go, args=(pid,))


page = st.session_state.page
with st.container(key="page_card"):
    if page == "home":
        home_ui()
    elif page == "struct":
        intro("✍️", "Công thức cấu tạo viết tay", "Chụp ảnh công thức vẽ tay kiểu SGK → mã SMILES, tên gọi, khối lượng mol, tính chất.", "struct", "Chụp là biết!")
        structure_ui()
    elif page == "rx":
        intro("🧪", "Phương trình phản ứng", "Đọc phương trình, cho biết đây là loại phản ứng nào và kiểm tra đã cân bằng chưa.", "rx", "Cân bằng là xong!")
        reaction_ui()
        st.caption("Kết quả do AI gợi ý, có thể sai ở nét viết khó đọc. Việc kiểm tra cân bằng nguyên tử do RDKit tính trực tiếp. Hãy đối chiếu SGK.")
    elif page == "iso":
        intro("🔀", "Đồng phân", "Liệt kê đồng phân của một công thức hoặc so sánh hai chất xem có phải đồng phân không.", "iso", "Đếm cho đủ!")
        isomer_ui()
    elif page == "equip":
        intro("🔬", "Dụng cụ hóa học", "Chụp ảnh dụng cụ hoặc bộ thí nghiệm để biết tên, công dụng, cách dùng và lưu ý an toàn.", "equip", "Quan sát là hiểu!")
        equipment_ui()
    elif page == "practice":
        intro("🎓", "Luyện tập và chấm bài tự động", "Ra đề, nộp bài bằng ảnh vẽ tay, máy chấm và chỉ ra lỗi sai (hóa trị, thiếu thừa nguyên tử, đồng phân, nhóm chức).", "practice", "Luyện là nhớ!")
        practice_ui()
    else:
        intro("📊", "Đánh giá độ chính xác", "Dành cho giáo viên và nhóm nghiên cứu: chạy bộ ảnh có đáp án để lấy số liệu cho báo cáo.", "bench", "Số liệu nói lên tất cả!")
        bench_ui()
    st.markdown(
        f'<div class="usage">📋 Lượt phân tích đã dùng trong phiên này: <b>{st.session_state.n}/{MAX_PER_SESSION}</b></div>',
        unsafe_allow_html=True,
    )
