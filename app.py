import base64
import csv
import html
import io
import json
import random
import re
import time
import unicodedata
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
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

APP_NAME = st.secrets.get("APP_NAME", "HữuCơ Lens")
APP_ICON = st.secrets.get("APP_ICON", "⚗️")

st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="wide")

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


# ---- Sinh đồng phân cấu tạo bằng thuật toán (không dùng AI) ----
VALENCE = {"C": 4, "N": 3, "O": 2, "S": 2, "F": 1, "Cl": 1, "Br": 1, "I": 1}
_BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
ISO_TIME = float(st.secrets.get("ISO_TIME", 15))


class TooBig(Exception):
    pass


def _canon(atoms, edges, strict=False):
    rw = Chem.RWMol()
    for e in atoms:
        rw.AddAtom(Chem.Atom(e))
    for i, j, o in edges:
        rw.AddBond(i, j, _BT[o])
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    if strict:
        for b in m.GetBonds():
            if b.GetBondType() == Chem.BondType.TRIPLE and b.IsInRing():
                raise ValueError("liên kết ba trong vòng")
    return Chem.MolToSmiles(m, isomericSmiles=False)


def enumerate_isomers(counts, nH, time_limit=15, max_graphs=200000, strict=False):
    """Liệt kê TẤT CẢ đồng phân cấu tạo của công thức (mọi đồ thị liên thông đúng hóa trị, không trùng nhau).
    strict=True: chỉ mạch hở hoặc 1 vòng, không có O–O, không liên kết ba trong vòng."""
    els = sorted(counts)
    N = sum(counts.values())
    sigma = sum(VALENCE[e] * counts[e] for e in els)
    if sigma - nH < 0 or (sigma - nH) % 2:
        return []
    B = (sigma - nH) // 2
    if B < N - 1:
        return []
    t0, memo, total = time.time(), {}, [0]

    def build(sub):
        if sub in memo:
            return memo[sub]
        n = sum(sub)
        res = {}
        if n == 1:
            e = els[[i for i, c in enumerate(sub) if c][0]]
            res[_canon([e], [])] = ([e], [])
            memo[sub] = res
            return res
        upper = B - (N - n)
        for i, c in enumerate(sub):
            if not c:
                continue
            prev = list(sub)
            prev[i] -= 1
            e, vn = els[i], VALENCE[els[i]]
            for atoms, edges in list(build(tuple(prev)).values()):
                T = sum(o for _, _, o in edges)
                if T + 1 > upper:
                    continue
                used = [0] * len(atoms)
                for a, b, o in edges:
                    used[a] += o
                    used[b] += o
                free = [VALENCE[atoms[k]] - used[k] for k in range(len(atoms))]
                k = len(atoms)

                def rec(j, rem_new, added, att):
                    if time.time() - t0 > time_limit:
                        raise TooBig("time")
                    if j == k:
                        if att and T + added <= upper and (n < N or T + added == B):
                            na, ne = atoms + [e], edges + [(a, k, o) for a, o in att]
                            if strict:
                                if len(ne) - len(na) + 1 > 1:
                                    return
                                if e == "O" and any(atoms[a] == "O" for a, _ in att):
                                    return
                            try:
                                key = _canon(na, ne, strict)
                            except Exception:
                                return
                            if key not in res:
                                res[key] = (na, ne)
                                total[0] += 1
                                if total[0] > max_graphs:
                                    raise TooBig("size")
                        return
                    rec(j + 1, rem_new, added, att)
                    for o in (1, 2, 3):
                        if o <= rem_new and o <= free[j] and T + added + o <= upper:
                            rec(j + 1, rem_new - o, added + o, att + [(j, o)])

                rec(0, vn, 0, [])
        memo[sub] = res
        return res

    return sorted(build(tuple(counts[e] for e in els)).keys())


def parse_formula(s):
    s = (s or "").translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")).replace(" ", "")
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", s)
    if not toks or "".join(a + b for a, b in toks) != s:
        raise ValueError("không phải công thức phân tử")
    c = {}
    for a, b in toks:
        c[a] = c.get(a, 0) + (int(b) if b else 1)
    nH = c.pop("H", 0)
    bad = [e for e in c if e not in VALENCE]
    if bad or not c:
        raise ValueError("nguyên tố chưa hỗ trợ: " + ", ".join(bad or ["?"]))
    return c, nH


def is_simple_formula(s):
    s = (s or "").translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")).replace(" ", "")
    toks = re.findall(r"([A-Z][a-z]?)(\d*)", s)
    if not toks or "".join(a + b for a, b in toks) != s:
        return False
    syms = [a for a, _ in toks]
    return len(syms) == len(set(syms)) and (any(b for _, b in toks) or "H" in syms)


def hill_key(counts, nH):
    parts = []
    if "C" in counts:
        parts.append(("C", counts["C"]))
    if nH:
        parts.append(("H", nH))
    parts += [(e, counts[e]) for e in sorted(counts) if e != "C"]
    return "".join(f"{e}{n if n > 1 else ''}" for e, n in parts)


def mol_counts(m):
    c = atom_counts(m)
    nH = c.pop("H", 0)
    return c, nH


# Phân loại theo chương trình phổ thông (dùng riêng cho tab đồng phân)
ISO_GROUPS = [
    ("ancol bậc I", "[CX4;H2,H3][OX2H]"), ("ancol bậc II", "[CX4;H1]([#6])([#6])[OX2H]"),
    ("ancol bậc III", "[CX4;H0]([#6])([#6])([#6])[OX2H]"), ("phenol", "c[OX2H]"),
    ("ete", "[#6;!$([#6]=O)][OX2][#6;!$([#6]=O)]"), ("anđehit", "[CX3;H1,H2;!$(C[OX2,NX3])]=O"),
    ("xeton", "[#6][CX3](=O)[#6]"), ("axit cacboxylic", "[CX3](=O)[OX2H1]"), ("este", "[CX3](=O)[OX2][#6]"),
    ("amin bậc I", "[NX3;H2;!$(NC=[O,N,S])][#6]"), ("amin bậc II", "[NX3;H1;!$(NC=[O,N,S])]([#6])[#6]"),
    ("amin bậc III", "[NX3;H0;!$(NC=[O,N,S])]([#6])([#6])[#6]"), ("amit", "[NX3][CX3]=O"),
    ("nitro", "[N+](=O)[O-]"), ("nitrin", "[CX2]#[NX1]"), ("anken (C=C)", "[CX3]=[CX3]"), ("ankin (C≡C)", "[CX2]#[CX2]"),
    ("hợp chất thơm", "a"), ("dẫn xuất halogen", "[#6][F,Cl,Br,I]"),
]
_ISO_GP = [(n, Chem.MolFromSmarts(p)) for n, p in ISO_GROUPS]
UNSTABLE = [
    ("enol (OH trên C=C)", "[OX2H][CX3]=[CX3]"), ("ynol (OH trên C≡C)", "[OX2H][CX2]#[CX2]"),
    ("hai nhóm OH trên cùng một C", "[CX4]([OX2H])[OX2H]"), ("hemiaxetal", "[CX4]([OX2H])[OX2][#6]"),
    ("OH và halogen trên cùng một C", "[CX4]([OX2H])[F,Cl,Br,I]"), ("OH và N trên cùng một C", "[CX4]([OX2H])[NX3]"),
]
_UNST = [(n, Chem.MolFromSmarts(p)) for n, p in UNSTABLE]


_PAIR = Chem.MolFromSmarts("[CX3](=O)[OX2,NX3]")
_NITRO = Chem.MolFromSmarts("[N+](=O)[O-]")


def func_units(m):
    """Số nhóm chức chứa dị tố: axit/este/amit và nitro mỗi nhóm chỉ tính 1 dù có nhiều nguyên tử."""
    n_het = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("O", "N", "S", "F", "Cl", "Br", "I"))
    return n_het - len(m.GetSubstructMatches(_PAIR)) - 2 * len(m.GetSubstructMatches(_NITRO))


def is_mono_open(m):
    """Đơn chức (tối đa 1 nhóm chức chứa dị tố) và không có vòng no/không thơm (vòng benzen vẫn được giữ)."""
    return func_units(m) <= 1 and not any(a.IsInRing() and not a.GetIsAromatic() for a in m.GetAtoms())


def iso_labels(m):
    lab = [n for n, p in _ISO_GP if m.HasSubstructMatch(p)]
    has_ring = any(not a.GetIsAromatic() and a.IsInRing() for a in m.GetAtoms())
    if not lab:
        if all(a.GetSymbol() == "C" for a in m.GetAtoms()):
            return ["xicloankan"] if has_ring else ["ankan"]
        lab = ["khác"]
    if has_ring:
        lab.append("có vòng không thơm")
    return lab


@st.cache_data(show_spinner=False, max_entries=64)
def iso_enumerate(fkey, strict):
    counts, nH = parse_formula(fkey)
    try:
        return {"ok": True, "smiles": enumerate_isomers(counts, nH, time_limit=ISO_TIME, strict=strict)}
    except TooBig as e:
        return {"ok": False, "why": str(e)}


def stereo_split(m):
    """(tổng, hình học, quang học): số dạng lập thể; hình học = cis/trans bỏ qua quang học; quang học = bỏ qua cis/trans."""
    try:
        opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True, tryEmbedding=False, maxIsomers=64)
        isos = list(EnumerateStereoisomers(m, options=opts))
    except Exception:
        return 1, 1, 1
    geo, opt = set(), set()
    for x in isos:
        a = Chem.Mol(x)  # hình học: giữ cis/trans của nối đôi và cấu hình tương đối trên vòng; bỏ tâm bất đối ngoài vòng và gộp cặp đối quang
        for at in a.GetAtoms():
            if not at.IsInRing():
                at.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        s1 = Chem.MolToSmiles(a)
        mir = Chem.MolFromSmiles(mirror(s1))
        geo.add(min(s1, Chem.MolToSmiles(mir)) if mir is not None else s1)
        b = Chem.Mol(x)  # quang học: giữ mọi tâm bất đối, bỏ cis/trans của nối đôi
        for bd in b.GetBonds():
            if bd.GetBondType() == Chem.BondType.DOUBLE:
                bd.SetStereo(Chem.BondStereo.STEREONONE)
                bd.SetBondDir(Chem.BondDir.NONE)
        opt.add(Chem.MolToSmiles(b))
    return max(1, len(isos)), max(1, len(geo)), max(1, len(opt))


# Từ vựng nhóm mà AI được phép dùng; mỗi từ mở rộng thành các nhãn do bộ máy phân loại
GROUP_VOCAB = {
    "ankan": ["ankan"], "xicloankan": ["xicloankan"], "anken": ["anken (C=C)"], "ankin": ["ankin (C≡C)"],
    "thơm": ["hợp chất thơm"], "ancol": ["ancol bậc I", "ancol bậc II", "ancol bậc III"],
    "ancol bậc I": ["ancol bậc I"], "ancol bậc II": ["ancol bậc II"], "ancol bậc III": ["ancol bậc III"],
    "phenol": ["phenol"], "ete": ["ete"], "anđehit": ["anđehit"], "xeton": ["xeton"],
    "axit cacboxylic": ["axit cacboxylic"], "este": ["este"],
    "amin": ["amin bậc I", "amin bậc II", "amin bậc III"], "amin bậc I": ["amin bậc I"],
    "amin bậc II": ["amin bậc II"], "amin bậc III": ["amin bậc III"], "amit": ["amit"], "nitro": ["nitro"],
    "nitrin": ["nitrin"], "dẫn xuất halogen": ["dẫn xuất halogen"],
}
PRESETS = {
    "mono": {"ben": True, "chuc": "don", "mach": "ho_hoac_thom", "nhom": [], "nhom_ten": [], "kieu": "bat_ky", "lap_the": "khong", "defaults": []},
    "pho": {"ben": True, "chuc": "bat_ky", "mach": "bat_ky", "nhom": [], "nhom_ten": [], "kieu": "bat_ky", "lap_the": "khong", "defaults": []},
    "math": {"ben": False, "chuc": "bat_ky", "mach": "bat_ky", "nhom": [], "nhom_ten": [], "kieu": "bat_ky", "lap_the": "khong", "defaults": []},
}


def norm_params(raw):
    """Kiểm tra tham số AI trích ra: chỉ nhận giá trị trong danh mục. Giá trị AI để trống thì dùng mặc định và ghi rõ."""
    raw = raw or {}
    warn, defaults = [], []
    names = []
    for g in raw.get("nhom") or []:
        key = str(g).strip().lower().replace("đ", "đ")
        hit = next((k for k in GROUP_VOCAB if k.lower() == key), None)
        if hit:
            names.append(hit)
        else:
            warn.append(f"Bỏ qua nhóm không có trong danh mục: {g}")
    p = {"ben": True, "nhom_ten": names, "nhom": [GROUP_VOCAB[n] for n in names]}
    kieu = raw.get("kieu_nhom")
    p["kieu"] = kieu if kieu in ("bat_ky", "tat_ca") else "bat_ky"
    for k, ok, dflt in (("chuc", ("don", "da", "bat_ky"), "don"), ("mach", ("ho", "vong", "bat_ky"), "ho_hoac_thom"),
                        ("lap_the", ("khong", "hinh_hoc", "quang_hoc", "tat_ca"), "khong")):
        v = raw.get(k)
        if v in ok:
            p[k] = v
        else:
            p[k] = dflt
            defaults.append(k)
    if "xicloankan" in names and "mach" in defaults:
        p["mach"] = "bat_ky"
    p["defaults"] = defaults
    return p, warn


def pass_filters(m, labels, p):
    u = func_units(m)
    if (p["chuc"] == "don" and u > 1) or (p["chuc"] == "da" and u < 2):
        return False
    rings = m.GetRingInfo().NumRings()
    nonarom = any(a.IsInRing() and not a.GetIsAromatic() for a in m.GetAtoms())
    if (p["mach"] == "ho" and rings) or (p["mach"] == "vong" and not rings) or (p["mach"] == "ho_hoac_thom" and nonarom):
        return False
    if p["nhom"]:
        hit = [bool(set(labels) & set(g)) for g in p["nhom"]]
        return all(hit) if p["kieu"] == "tat_ca" else any(hit)
    return True


PARSE_PROMPT = """Bạn là bộ PHÂN TÍCH CÂU HỎI Hóa học. Nhiệm vụ DUY NHẤT: trích tham số từ câu hỏi về đồng phân. TUYỆT ĐỐI không đếm, không liệt kê chất, không trả lời câu hỏi.
Từ vựng nhóm (chỉ dùng đúng các chuỗi này): ankan, xicloankan, anken, ankin, thơm, ancol, ancol bậc I, ancol bậc II, ancol bậc III, phenol, ete, anđehit, xeton, axit cacboxylic, este, amin, amin bậc I, amin bậc II, amin bậc III, amit, nitro, nitrin, dẫn xuất halogen.
Chỉ trả về MỘT đối tượng JSON, không markdown:
{"cong_thuc":"công thức phân tử viết liền, ví dụ C4H8O2, hoặc null",
"chat_goc":"tên hoặc SMILES của chất nếu câu hỏi nhắc một chất cụ thể thay vì công thức, hoặc null",
"nhom":["nhóm mà đồng phân cần tìm phải thuộc, lấy từ từ vựng; để [] nếu câu hỏi không nêu nhóm"],
"kieu_nhom":"bat_ky nếu chỉ cần thuộc ít nhất một nhóm, tat_ca nếu phải có đồng thời mọi nhóm, null nếu không nêu",
"chuc":"don nếu câu hỏi nói đơn chức, da nếu nói đa chức hoặc tạp chức, bat_ky nếu nói rõ tính mọi loại, null nếu không nhắc",
"mach":"ho nếu mạch hở, vong nếu mạch vòng, bat_ky nếu nói rõ tính cả hai, null nếu không nhắc",
"lap_the":"khong nếu nói không tính đồng phân lập thể, hinh_hoc nếu tính thêm đồng phân hình học (cis-trans), quang_hoc nếu tính thêm đồng phân quang học, tat_ca nếu tính mọi đồng phân lập thể, null nếu không nhắc",
"can_hoi_lai":"một câu hỏi làm rõ nếu thiếu công thức hoặc chất, hoặc câu hỏi mơ hồ; nếu không thì để rỗng",
"tom_tat":"câu hỏi được hiểu lại ngắn gọn bằng tiếng Việt"}
Không đoán những gì câu hỏi không nói: để null hoặc []. Ví dụ: 'Có bao nhiêu đồng phân este của C4H8O2?' thì nhom=["este"], cong_thuc="C4H8O2", các mục còn lại null."""


def parse_question(q):
    raw = ask_ai([PARSE_PROMPT + "\n\nCâu hỏi: " + q], temperature=0)
    p, warn = norm_params(raw)
    return p, warn, raw


def resolve_formula(img, txt):
    """Trả về (counts, nH, SMILES chất gốc hoặc None, lỗi hoặc None)."""
    if img:
        smi, _ = transcribe_student(img)
        m, err = parse_student(smi)
        if m is None:
            return None, None, None, "Hình vẽ sai nên không xác định được công thức: " + (err or "không đọc được hình")
    elif is_simple_formula(txt):
        try:
            c, h = parse_formula(txt)
            return c, h, None, None
        except ValueError as e:
            return None, None, None, f"Công thức không hợp lệ ({e})."
    else:
        m, err = parse_student(txt)
        if m is None:
            smi = to_smiles(None, txt)  # tên chất hoặc công thức thu gọn: nhờ AI đổi sang SMILES
            m, err = parse_student(smi)
            if m is None:
                return None, None, None, "Không hiểu chất đã nhập. Hãy gõ công thức phân tử (ví dụ C4H10O), SMILES hoặc tên chất."
    c, h = mol_counts(m)
    if any(e not in VALENCE for e in c):
        return None, None, None, "Chất có nguyên tố chưa được hỗ trợ."
    return c, h, Chem.MolToSmiles(m, isomericSmiles=False), None


def ai_names(smiles_list):
    """Nhờ AI đặt tên. Cấu trúc do thuật toán sinh nên chỉ phần tên có thể sai."""
    lines = "\n".join(f"{i + 1}. {s_}" for i, s_ in enumerate(smiles_list))
    r = ask_ai([RULES_ISO + "\n\nĐặt tên theo danh pháp SGK cho từng chất sau (theo đúng thứ tự). "
                'Chỉ trả JSON {"ten":["tên 1","tên 2",...]} có đúng ' + str(len(smiles_list)) + " phần tử.\n" + lines], temperature=0)
    names = r.get("ten") or []
    return {s_: (names[i] if i < len(names) else "") for i, s_ in enumerate(smiles_list)}


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


def build_iso_result(fkey, orig, p):
    """Bộ máy hóa học: sinh toàn bộ đồng phân, kiểm tra và đếm theo tham số p. AI không tham gia."""
    out = iso_enumerate(fkey, p["ben"])
    if not out["ok"]:
        return {"ok": False, "why": out["why"], "fkey": fkey}
    kept, excluded, hidden = [], {}, 0
    for smi in out["smiles"]:
        m = Chem.MolFromSmiles(smi)
        if p["ben"]:
            bad = next((n for n, pt in _UNST if m.HasSubstructMatch(pt)), None)
            if bad:
                excluded[bad] = excluded.get(bad, 0) + 1
                continue
        labels = iso_labels(m)
        if not pass_filters(m, labels, p):
            hidden += 1
            continue
        tot, geo, opt = stereo_split(m)
        kept.append({"smiles": smi, "labels": labels, "stereo": tot, "geo": geo, "opt": opt})
    return {"ok": True, "fkey": fkey, "params": p, "kept": kept, "excluded": excluded, "hidden": hidden,
            "total_math": len(out["smiles"]), "orig": orig}


LB_CHUC = {"don": "đơn chức", "da": "đa chức hoặc tạp chức", "bat_ky": "mọi loại chức"}
LB_MACH = {"ho": "mạch hở", "vong": "mạch vòng", "ho_hoac_thom": "mạch hở hoặc vòng benzen", "bat_ky": "mọi loại mạch"}
LB_LT = {"khong": "không tính đồng phân lập thể", "hinh_hoc": "kể cả đồng phân hình học (cis–trans)",
         "quang_hoc": "kể cả đồng phân quang học", "tat_ca": "kể cả mọi đồng phân lập thể"}


def answer_of(r):
    k, lt = r["kept"], r["params"]["lap_the"]
    return {"hinh_hoc": sum(x["geo"] for x in k), "quang_hoc": sum(x["opt"] for x in k),
            "tat_ca": sum(x["stereo"] for x in k)}.get(lt, len(k))


def render_iso(r, names):
    kept, p = r["kept"], r["params"]
    nh = " hoặc ".join(p["nhom_ten"]) if p["nhom_ten"] else "mọi nhóm"
    st.success(f"**Kết quả: {answer_of(r)} đồng phân** của {pretty(r['fkey'])} (nhóm: {nh} · {LB_CHUC[p['chuc']]} · "
               f"{LB_MACH[p['mach']]} · {LB_LT[p['lap_the']]}). Con số này do bộ máy hóa học sinh cấu trúc và đếm, không phải AI đoán.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đồng phân cấu tạo", len(kept))
    c2.metric("Kể cả hình học", sum(k["geo"] for k in kept))
    c3.metric("Kể cả quang học", sum(k["opt"] for k in kept))
    c4.metric("Kể cả lập thể (đủ)", sum(k["stereo"] for k in kept))
    st.caption(f"Đã ẩn {r['hidden']} đồng phân không thỏa bộ lọc và loại {sum(r['excluded'].values()) if p['ben'] else 0} cấu trúc kém bền "
               f"(công thức {pretty(r['fkey'])}).")
    comp = {}
    for k in kept:
        for lb in k["labels"]:
            d = comp.setdefault(lb, [0, 0])
            d[0] += 1
            d[1] += k["stereo"]
    st.markdown("**Số đồng phân theo nhóm** (một chất có nhiều nhóm sẽ được đếm ở mỗi nhóm)")
    st.dataframe([{"Nhóm": n, "Đồng phân cấu tạo": v[0], "Tính cả lập thể": v[1]} for n, v in sorted(comp.items())],
                 hide_index=True, width="stretch")

    pick = st.multiselect("Chỉ hiện các nhóm (để trống = hiện tất cả)", sorted(comp), key="iso_filter")
    shown = [k for k in kept if not pick or any(lb in pick for lb in k["labels"])]
    if pick:
        st.caption(f"Có {len(shown)} đồng phân cấu tạo thuộc ít nhất một trong các nhóm đã chọn.")

    orig = r.get("orig")
    if orig:
        hit = next((i for i, k in enumerate(kept, 1) if k["smiles"] == orig), None)
        if hit:
            st.info(f"Chất bạn nhập là đồng phân số {hit} trong danh sách (nhóm: {' + '.join(kept[hit - 1]['labels'])}).")
    rows = []
    for i, k in enumerate(shown, 1):
        rows.append({"STT": i, "Nhóm": " + ".join(k["labels"]), "Lập thể": "không" if k["stereo"] == 1 else f"{k['stereo']} đồng phân lập thể",
                     **({"Tên (AI gợi ý)": names.get(k["smiles"], "")} if names else {}), "SMILES": k["smiles"]})
    st.dataframe(rows[:200], hide_index=True, width="stretch")
    if len(rows) > 200:
        st.caption(f"Chỉ hiện 200 dòng đầu trong tổng {len(rows)}. Hãy lọc theo nhóm để xem các chất cần tìm, hoặc tải file CSV.")
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        st.download_button("⬇️ Tải danh sách đồng phân (CSV)", buf.getvalue().encode("utf-8-sig"),
                           file_name=f"dong_phan_{r['fkey']}.csv", mime="text/csv", key="iso_dl")
    ms = [Chem.MolFromSmiles(k["smiles"]) for k in shown[:48]]
    if ms:
        try:
            st.image(Draw.MolsToGridImage(ms, molsPerRow=4, subImgSize=(230, 180),
                                          legends=[str(i) for i in range(1, len(ms) + 1)]))
        except Exception:
            pass
        if len(shown) > 48:
            st.caption("Hình chỉ hiện 48 chất đầu.")
    if r["params"]["ben"] and r["excluded"]:
        with st.expander(f"Đã loại {sum(r['excluded'].values())} cấu trúc kém bền theo quy ước SGK"):
            for n, v in r["excluded"].items():
                st.markdown(f"- {n}: {v}")
            st.caption("Chọn phạm vi \"Toán học\" để xem cả những cấu trúc này.")
    if names:
        st.caption("Tên do AI gợi ý nên có thể sai. Cấu trúc và số lượng đồng phân do thuật toán sinh, không phải AI.")
    st.caption("Cách làm: sinh mọi cấu trúc nối liền đúng hóa trị (C=4, N=3, O=2, halogen=1), loại trùng bằng RDKit. "
               "Thuật toán đã được đối chiếu với số liệu đã biết: ankan C₈H₁₈=18, C₉H₂₀=35, ancol+ete C₆H₁₄O=32, "
               "C₆H₁₂=25, amin C₄H₁₁N=8, C₆H₆=217 (toán học).")


def render_understood(u, r):
    """Cho người dùng thấy máy đã hiểu câu hỏi thế nào để kiểm tra lại (chỉ khi dùng chế độ hỏi bằng tiếng Việt)."""
    if not u:
        return
    p = r["params"]
    dflt = set(p.get("defaults", []))
    mark = lambda k: " *(mặc định, câu hỏi không nêu)*" if k in dflt else ""
    with st.container(border=True):
        st.markdown("**Máy hiểu câu hỏi của bạn như sau** (hãy kiểm tra, nếu sai thì diễn đạt lại)")
        if u.get("tom_tat"):
            st.markdown("- Tóm tắt: " + str(u["tom_tat"]))
        st.markdown("- Công thức phân tử: **" + pretty(r["fkey"]) + "**")
        if u.get("chat_goc") and r.get("orig"):
            st.markdown(f"- Chất được nhắc tới: {u['chat_goc']} → `{r['orig']}` (hãy xem hình để chắc đây đúng chất bạn muốn)")
            try:
                st.image(Draw.MolToImage(Chem.MolFromSmiles(r["orig"]), size=(260, 180)))
            except Exception:
                pass
        nhom_txt = "không giới hạn" if not p["nhom_ten"] else (" và " if p["kieu"] == "tat_ca" else " hoặc ").join(p["nhom_ten"])
        st.markdown("- Nhóm cần tìm: **" + nhom_txt + "**")
        st.markdown(f"- Chức: **{LB_CHUC[p['chuc']]}**{mark('chuc')}")
        st.markdown(f"- Mạch: **{LB_MACH[p['mach']]}**{mark('mach')}")
        st.markdown(f"- Lập thể: **{LB_LT[p['lap_the']]}**{mark('lap_the')}")
        for w in u.get("warn") or []:
            st.caption(w)


def isomer_ui():
    top = st.container()  # lỗi và lý do sai hiện ở đầu tab
    t1, t2 = st.tabs(["Liệt kê đồng phân", "So sánh hai chất"])
    with t1:
        src = st.radio("Cách nhập", ["💬 Hỏi bằng tiếng Việt", "⌨️ Gõ công thức / tên / SMILES", "🖼️ Tải ảnh", "📷 Chụp ảnh"],
                       horizontal=True, key="iso_src")
        q_mode = src.startswith("💬")
        img, txt, q, scope = None, None, "", None
        if q_mode:
            q = st.text_area("Câu hỏi", height=90, key="iso_q",
                             placeholder="Ví dụ: Có bao nhiêu đồng phân este của C4H8O2?\nHoặc: C5H10 mạch hở, kể cả đồng phân hình học, có bao nhiêu đồng phân?").strip()
            st.caption("AI chỉ đọc câu hỏi và chuyển thành tham số (công thức, nhóm chức, mạch, lập thể). Việc sinh cấu trúc, kiểm tra và đếm do bộ máy hóa học làm.")
        elif src.startswith("🖼️"):
            f = st.file_uploader("Ảnh công thức", type=["jpg", "jpeg", "png", "webp"], key="iso_up")
            img = f.getvalue() if f else None
        elif src.startswith("📷"):
            f = st.camera_input("Chụp công thức", key="iso_cam")
            img = f.getvalue() if f else None
        else:
            txt = st.text_input("Công thức phân tử, tên hoặc SMILES", placeholder="Ví dụ: C4H10O", key="iso_txt").strip()
        if not q_mode:
            scope = st.radio("Phạm vi", ["Đơn chức, mạch hở (thường gặp trong đề SGK)",
                                         "Phổ thông (cả đa chức và 1 vòng, loại cấu trúc kém bền)",
                                         "Toán học (mọi cấu trúc đúng hóa trị)"], key="iso_scope")
        want_names = st.checkbox("Nhờ AI gọi tên (tên có thể sai, cấu trúc luôn do bộ máy sinh)", key="iso_names")
        if st.button("Hiểu câu hỏi và đếm" if q_mode else "Liệt kê đồng phân", type="primary",
                     disabled=not (q or img or txt), key="iso_go"):
            ai_used = bool(q_mode or img or want_names or (txt and not is_simple_formula(txt) and Chem.MolFromSmiles(txt) is None))
            if ai_used and st.session_state.n >= MAX_PER_SESSION:
                show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
            else:
                with st.spinner("Đang phân tích…"):
                    try:
                        st.session_state.iso = None
                        counts = nH = orig = err = None
                        und = None
                        if q_mode:
                            p, warn, raw = parse_question(q)
                            ftxt, gc = raw.get("cong_thuc"), raw.get("chat_goc")
                            und = {"tom_tat": raw.get("tom_tat"), "warn": warn, "cong_thuc": ftxt, "chat_goc": gc}
                            if ftxt and is_simple_formula(str(ftxt)):
                                counts, nH = parse_formula(str(ftxt))
                            elif gc:
                                counts, nH, orig, err = resolve_formula(None, str(gc))
                            else:
                                err = raw.get("can_hoi_lai") or "Câu hỏi chưa nêu công thức phân tử hoặc tên chất. Hãy bổ sung."
                        else:
                            counts, nH, orig, err = resolve_formula(img, txt)
                            p = dict(PRESETS["mono" if scope.startswith("Đơn") else ("pho" if scope.startswith("Phổ") else "math")])
                        if err:
                            show_top_error(top, err)
                        else:
                            n_heavy = sum(counts.values())
                            fkey = hill_key(counts, nH)
                            if n_heavy > 12:
                                show_top_error(top, f"Công thức {pretty(fkey)} có {n_heavy} nguyên tử khác H, quá lớn để liệt kê chính xác (tối đa 12).")
                            else:
                                res = build_iso_result(fkey, orig, p)
                                if not res["ok"]:
                                    show_top_error(top, f"Công thức {pretty(fkey)} có quá nhiều đồng phân để sinh trong {ISO_TIME:.0f} giây. "
                                                        "Hãy nêu rõ nhóm chức hoặc dùng công thức nhỏ hơn.")
                                elif not res["kept"]:
                                    st.session_state.iso = res
                                    st.session_state.iso_nm = {}
                                    st.session_state.iso_und = und
                                else:
                                    nm = {}
                                    if want_names:
                                        if len(res["kept"]) <= 60:
                                            nm = ai_names([k["smiles"] for k in res["kept"]])
                                        else:
                                            st.info("Có hơn 60 đồng phân nên không gọi tên bằng AI. Hãy lọc theo nhóm rồi thử lại.")
                                    st.session_state.iso = res
                                    st.session_state.iso_nm = nm
                                    st.session_state.iso_und = und
                                if ai_used:
                                    st.session_state.n += 1
                    except Exception as e:
                        st.session_state.iso = None
                        show_top_error(top, "Có lỗi khi phân tích, hãy thử lại.", e)
        if st.session_state.get("iso"):
            st.divider()
            render_understood(st.session_state.get("iso_und"), st.session_state.iso)
            if st.session_state.iso["kept"]:
                render_iso(st.session_state.iso, st.session_state.get("iso_nm") or {})
            else:
                st.warning(f"Không có đồng phân nào của {pretty(st.session_state.iso['fkey'])} thỏa các điều kiện trên. "
                           f"Đã ẩn {st.session_state.iso['hidden']} đồng phân không thỏa bộ lọc.")
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


# ---------------- Sơ đồ tư duy ----------------
MM_COLORS = ["#e11d48", "#f59e0b", "#16a34a", "#0ea5e9", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"]
MM_FONT = "Arial, Helvetica, sans-serif"

MM_PROMPT = """Bạn là giáo viên Hóa học THPT Việt Nam. Hãy lập SƠ ĐỒ TƯ DUY cho chủ đề được yêu cầu, đúng nội dung SGK Việt Nam.
Chỉ trả về MỘT đối tượng JSON, không markdown:
{"chu_de":"tên chủ đề ngắn gọn","chat_smiles":"SMILES nếu chủ đề là MỘT chất cụ thể (ví dụ etanol), nếu là nhóm chất hoặc khái niệm thì null",
"nhanh":[{"ten":"tên nhánh, tối đa 24 ký tự","y":["ý ngắn, tối đa 80 ký tự"]}]}
Quy tắc:
- Chọn các nhánh phù hợp chủ đề, ví dụ: Khái niệm và cấu tạo, Danh pháp, Tính chất vật lí, Tính chất hóa học, Điều chế, Ứng dụng, Nhận biết, Lưu ý. Với chủ đề phản ứng thì dùng: Định nghĩa, Điều kiện, Ví dụ, Quy tắc, Nhận biết.
- Mỗi ý ngắn gọn, dễ nhớ. Viết công thức bằng chữ thường (CH3-CH2-OH). Phương trình viết dạng: A + B → C (điều kiện).
- KHÔNG nêu số lượng đồng phân hay khối lượng mol (bộ máy tính toán sẽ thêm riêng). Không bịa số liệu."""


def _wrap(text, width):
    lines, cur = [], ""
    for w in str(text).split():
        if len(cur) + len(w) + (1 if cur else 0) <= width:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _tint(hexc, a=0.12):
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return f"rgba({r},{g},{b},{a})"


def mindmap_layout(title, branches):
    """Tính vị trí mọi khung và đường nối (đơn vị px). Dùng chung cho SVG, PowerPoint, draw.io và FreeMind."""
    LEAF_W, LH = 30, 19
    cw_leaf, cw_br, cw_root = 7.9, 9.8, 14.0
    root_lines = _wrap(title, 16)
    rw = max(len(x) for x in root_lines) * cw_root + 56
    rh = len(root_lines) * 28 + 36
    items = []
    for i, b in enumerate(branches):
        blines = _wrap(b["ten"], 18)
        bw = max(len(x) for x in blines) * cw_br + 36
        bh = len(blines) * 22 + 20
        leaves = []
        for t in b.get("y", []):
            ll = _wrap(t, LEAF_W)
            leaves.append((t, ll, max(len(x) for x in ll) * cw_leaf + 24, len(ll) * LH + 14))
        block = sum(l[3] for l in leaves) + 8 * max(0, len(leaves) - 1)
        items.append({"b": b, "blines": blines, "bw": bw, "bh": bh, "leaves": leaves, "block": block,
                      "h": max(block, bh), "color": b.get("color") or MM_COLORS[i % len(MM_COLORS)]})
    half = (len(items) + 1) // 2
    sides = {"R": items[:half], "L": items[half:]}
    GAP, MARG = 28, 50
    tot = {k: sum(x["h"] for x in v) + GAP * max(0, len(v) - 1) for k, v in sides.items()}
    H = max(max(tot.values()), rh) + 2 * MARG
    reach = 0
    for v in sides.values():
        for x in v:
            reach = max(reach, 70 + x["bw"] + 55 + max([l[2] for l in x["leaves"]] or [0]))
    W = 2 * (rw / 2 + reach + MARG)
    cx, cy = W / 2, H / 2
    nodes = [{"id": "n0", "kind": "root", "x": cx - rw / 2, "y": cy - rh / 2, "w": rw, "h": rh, "text": title,
              "lines": root_lines, "color": "#0e8f6e", "side": None, "parent": None}]
    edges, k = [], 1
    for side, v in sides.items():
        sgn = 1 if side == "R" else -1
        y = cy - tot[side] / 2
        for x in v:
            col, yb = x["color"], y + x["h"] / 2
            bx0 = cx + sgn * (rw / 2 + 70)
            bx1 = bx0 + sgn * x["bw"]
            bid = f"n{k}"
            k += 1
            nodes.append({"id": bid, "kind": "branch", "x": min(bx0, bx1), "y": yb - x["bh"] / 2, "w": x["bw"], "h": x["bh"],
                          "text": x["b"]["ten"], "lines": x["blines"], "color": col, "side": side, "parent": "n0"})
            edges.append({"a": "n0", "b": bid, "color": col, "width": 5, "x1": cx + sgn * rw / 2, "y1": cy, "x2": bx0, "y2": yb, "side": side})
            ly = y + (x["h"] - x["block"]) / 2
            lx0 = bx1 + sgn * 55
            for text, ll, lw, lh in x["leaves"]:
                lid = f"n{k}"
                k += 1
                nodes.append({"id": lid, "kind": "leaf", "x": lx0 if side == "R" else lx0 - lw, "y": ly, "w": lw, "h": lh,
                              "text": text, "lines": ll, "color": col, "side": side, "parent": bid})
                edges.append({"a": bid, "b": lid, "color": col, "width": 2.5, "x1": bx1, "y1": yb, "x2": lx0, "y2": ly + lh / 2, "side": side})
                ly += lh + 8
            y += x["h"] + GAP
    return {"W": W, "H": H, "nodes": nodes, "edges": edges}


def svg_from_layout(L):
    esc = html.escape
    W, H = L["W"], L["H"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" '
           f'font-family="{MM_FONT}"><defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0e8f6e"/>'
           f'<stop offset="1" stop-color="#14b8a6"/></linearGradient></defs><rect width="100%" height="100%" fill="#ffffff"/>']
    for e in L["edges"]:
        xm = (e["x1"] + e["x2"]) / 2
        out.append(f'<path d="M{e["x1"]:.1f} {e["y1"]:.1f} C{xm:.1f} {e["y1"]:.1f} {xm:.1f} {e["y2"]:.1f} {e["x2"]:.1f} {e["y2"]:.1f}" '
                   f'fill="none" stroke="{e["color"]}" stroke-width="{e["width"]}" stroke-linecap="round"/>')
    texts = []
    for n in L["nodes"]:
        x, y, w, h, col = n["x"], n["y"], n["w"], n["h"], n["color"]
        cxn, cyn = x + w / 2, y + h / 2
        if n["kind"] == "root":
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{h / 2:.1f}" fill="url(#rg)"/>')
            ty = cyn - len(n["lines"]) * 28 / 2 + 21
            for j, ln in enumerate(n["lines"]):
                texts.append(f'<text x="{cxn:.1f}" y="{ty + j * 28:.1f}" text-anchor="middle" font-size="24" font-weight="800" fill="#fff">{esc(ln)}</text>')
        elif n["kind"] == "branch":
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="14" fill="{col}"/>')
            ty = cyn - len(n["lines"]) * 22 / 2 + 16
            for j, ln in enumerate(n["lines"]):
                texts.append(f'<text x="{cxn:.1f}" y="{ty + j * 22:.1f}" text-anchor="middle" font-size="17" font-weight="700" fill="#fff">{esc(ln)}</text>')
        else:
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10" fill="{_tint(col)}" stroke="{col}" stroke-width="1.5"/>')
            for j, ln in enumerate(n["lines"]):
                texts.append(f'<text x="{x + 12:.1f}" y="{y + 7 + 19 * (j + 1) - 5:.1f}" font-size="15" fill="#1f2937">{esc(ln)}</text>')
    return "".join(out + texts) + "</svg>"


def mindmap_svg(title, branches):
    L = mindmap_layout(title, branches)
    return svg_from_layout(L), L["W"], L["H"]


def _blend(hexc, a=0.12):
    r, g, b = int(hexc[1:3], 16), int(hexc[3:5], 16), int(hexc[5:7], 16)
    return "{:02x}{:02x}{:02x}".format(*(round(c * a + 255 * (1 - a)) for c in (r, g, b)))


def mm_pptx_bytes(L):
    """PowerPoint: mỗi khung là hình có chữ sửa được, đường nối dính vào khung nên kéo khung thì đường nối đi theo."""
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
        from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
        from pptx.util import Emu, Pt
    except ImportError:
        return None
    E = 9525
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(int(L["W"] * E)), Emu(int(L["H"] * E))
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rgb = lambda h: RGBColor.from_string(h.lstrip("#").upper())
    shapes = {}
    for n in L["nodes"]:
        shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Emu(int(n["x"] * E)), Emu(int(n["y"] * E)), Emu(int(n["w"] * E)), Emu(int(n["h"] * E)))
        shp.adjustments[0] = min(0.5, (n["h"] / 2 if n["kind"] == "root" else (14 if n["kind"] == "branch" else 10)) / min(n["w"], n["h"]))
        shp.shadow.inherit = False
        shp.fill.solid()
        if n["kind"] == "leaf":
            shp.fill.fore_color.rgb = RGBColor.from_string(_blend(n["color"]).upper())
            shp.line.color.rgb = rgb(n["color"])
            shp.line.width = Pt(1.2)
        else:
            shp.fill.fore_color.rgb = rgb(n["color"])
            shp.line.fill.background()
        tf = shp.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = Emu(int(10 * E))
        tf.margin_top = tf.margin_bottom = Emu(int(4 * E))
        para = tf.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if n["kind"] == "leaf" else PP_ALIGN.CENTER
        run = para.add_run()
        run.text = n["text"]
        run.font.name = "Arial"
        run.font.size = Pt({"root": 18, "branch": 13, "leaf": 11}[n["kind"]])
        run.font.bold = n["kind"] != "leaf"
        run.font.color.rgb = RGBColor(0x1F, 0x29, 0x37) if n["kind"] == "leaf" else RGBColor(255, 255, 255)
        shapes[n["id"]] = shp
    tree = slide.shapes._spTree
    for e in L["edges"]:
        c = slide.shapes.add_connector(MSO_CONNECTOR.CURVE, Emu(int(e["x1"] * E)), Emu(int(e["y1"] * E)), Emu(int(e["x2"] * E)), Emu(int(e["y2"] * E)))
        right = e["side"] == "R"
        c.begin_connect(shapes[e["a"]], 3 if right else 1)
        c.end_connect(shapes[e["b"]], 1 if right else 3)
        c.line.color.rgb = rgb(e["color"])
        c.line.width = Pt(e["width"] * 0.75)
        tree.remove(c._element)
        tree.insert(2, c._element)  # đưa đường nối xuống dưới các khung
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def mm_drawio(L):
    """File .drawio (mở bằng app.diagrams.net hoặc draw.io Desktop)."""
    def attr(t):
        return html.escape(html.escape(str(t), quote=False), quote=True)
    cells = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
    for n in L["nodes"]:
        if n["kind"] == "root":
            st_ = "rounded=1;arcSize=50;whiteSpace=wrap;html=1;fillColor=#0e8f6e;strokeColor=none;fontColor=#ffffff;fontStyle=1;fontSize=18;"
        elif n["kind"] == "branch":
            st_ = f"rounded=1;arcSize=30;whiteSpace=wrap;html=1;fillColor={n['color']};strokeColor=none;fontColor=#ffffff;fontStyle=1;fontSize=13;"
        else:
            st_ = (f"rounded=1;arcSize=20;whiteSpace=wrap;html=1;fillColor=#{_blend(n['color'])};strokeColor={n['color']};"
                   "fontColor=#1f2937;fontSize=11;align=left;spacingLeft=6;")
        cells.append(f'<mxCell id="{n["id"]}" value="{attr(n["text"])}" style="{st_}" vertex="1" parent="1">'
                     f'<mxGeometry x="{n["x"]:.0f}" y="{n["y"]:.0f}" width="{n["w"]:.0f}" height="{n["h"]:.0f}" as="geometry"/></mxCell>')
    for i, e in enumerate(L["edges"]):
        ex, nx = ("1", "0") if e["side"] == "R" else ("0", "1")
        cells.append(f'<mxCell id="e{i}" style="curved=1;endArrow=none;html=1;strokeColor={e["color"]};strokeWidth={e["width"]};'
                     f'exitX={ex};exitY=0.5;entryX={nx};entryY=0.5;" edge="1" parent="1" source="{e["a"]}" target="{e["b"]}">'
                     '<mxGeometry relative="1" as="geometry"/></mxCell>')
    return ('<mxfile host="app.diagrams.net"><diagram name="So do tu duy" id="mm1"><mxGraphModel grid="1" gridSize="10" page="0">'
            "<root>" + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")


def mm_freemind(L):
    """File .mm (FreeMind) mở được bằng XMind, Freeplane, MindManager, Mindomo."""
    q = lambda t: html.escape(str(t), quote=True)
    root = L["nodes"][0]
    out = [f'<map version="1.0.1"><node TEXT="{q(root["text"])}">']
    for b in [n for n in L["nodes"] if n["kind"] == "branch"]:
        pos = "right" if b["side"] == "R" else "left"
        out.append(f'<node TEXT="{q(b["text"])}" POSITION="{pos}" COLOR="{b["color"]}">')
        out += [f'<node TEXT="{q(l["text"])}"/>' for l in L["nodes"] if l["parent"] == b["id"]]
        out.append("</node>")
    out.append("</node></map>")
    return "".join(out)


def rows_to_branches(rows):
    """Bảng chỉnh sửa (Nhánh, Ý) -> danh sách nhánh. Dòng không ghi nhánh thì thuộc nhánh ngay trên."""
    order, data = [], {}
    for r_ in rows:
        n = str(r_.get("Nhánh") or "").strip()
        y = str(r_.get("Ý") or "").strip()
        if not n and not y:
            continue
        n = n or (order[-1] if order else "Nhánh mới")
        if n not in data:
            data[n] = []
            order.append(n)
        if y:
            data[n].append(y[:100])
    return [{"ten": n[:28], "y": data[n], **({"color": "#475569"} if n.startswith("Số liệu (máy tính)") else {})} for n in order[:10]]


def mm_outline(title, branches):
    lines = [f"# {title}"]
    for b in branches:
        lines.append(f"\n## {b['ten']}")
        lines += [f"- {t}" for t in b.get("y", [])]
    return "\n".join(lines)


def mm_clean(raw):
    """Giữ sơ đồ gọn: tối đa 8 nhánh, 5 ý mỗi nhánh, cắt câu quá dài."""
    out = []
    for b in (raw.get("nhanh") or [])[:8]:
        ten = str(b.get("ten", "")).strip()[:28]
        ys = [str(t).strip()[:100] for t in (b.get("y") or []) if str(t).strip()][:5]
        if ten and ys:
            out.append({"ten": ten, "y": ys})
    return out


def mm_engine_facts(smiles):
    """Số liệu do RDKit và bộ máy đồng phân tính, không phải AI."""
    m, _ = parse_student(smiles or "")
    if m is None:
        return None
    f = formula_of(m)
    facts = [f"Công thức phân tử: {pretty(f)}", f"Khối lượng mol: {Descriptors.MolWt(m):.2f} g/mol",
             "Nhóm chức / loại: " + ", ".join(iso_labels(m))]
    counts, nH = mol_counts(m)
    if sum(counts.values()) <= 10 and all(e in VALENCE for e in counts):
        try:
            res = build_iso_result(hill_key(counts, nH), None, dict(PRESETS["mono"]))
            if res["ok"] and res["kept"]:
                k = res["kept"]
                facts.append(f"Số đồng phân cấu tạo đơn chức mạch hở của {pretty(f)}: {len(k)} (kể cả lập thể: {sum(x['stereo'] for x in k)})")
        except Exception:
            pass
    return facts


def mm_component(svg, W, H):
    scale = min(1.0, 1000.0 / W)
    page = ('<div style="font-family:Arial,sans-serif"><button id="dl" style="background:#1f7a4d;color:#fff;border:0;border-radius:10px;'
            'padding:9px 16px;font-weight:700;cursor:pointer">⬇️ Tải ảnh PNG</button>'
            '<div style="overflow:auto;border:1px solid #d6ece5;border-radius:12px;margin-top:8px;background:#fff">'
            '<div style="width:max-content;max-width:100%">' + svg.replace("<svg ", '<svg style="max-width:100%;height:auto;display:block" ', 1) +
            '</div></div><script>document.getElementById("dl").onclick=function(){'
            'var s=document.querySelector("svg"),x=new XMLSerializer().serializeToString(s),i=new Image();'
            'i.onload=function(){var c=document.createElement("canvas");c.width=' + str(int(W * 2)) + ';c.height=' + str(int(H * 2)) + ';'
            'var g=c.getContext("2d");g.fillStyle="#fff";g.fillRect(0,0,c.width,c.height);g.drawImage(i,0,0,c.width,c.height);'
            'c.toBlob(function(b){var a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="so_do_tu_duy.png";a.click();});};'
            'i.src="data:image/svg+xml;charset=utf-8,"+encodeURIComponent(x);};</script></div>')
    height = int(H * scale) + 90
    if hasattr(st, "iframe"):  # st.components.v1.html sẽ bị gỡ; nội dung trong page đã được escape, không chứa HTML từ AI
        st.iframe(page, height=height)
    else:
        components.html(page, height=height, scrolling=True)


MM_EXAMPLES = ["Ancol", "Anđehit và xeton", "Axit cacboxylic", "Este và lipit", "Amin và amino axit", "Benzen và đồng đẳng",
               "Phenol", "Etanol", "Phản ứng cộng"]


def _set_topic(t):
    st.session_state["mm_topic"] = t


def mindmap_ui():
    top = st.container()
    st.session_state.setdefault("mm", None)
    st.text_input("Chủ đề hoặc tên chất", key="mm_topic", placeholder="Ví dụ: Ancol, Etanol, Phản ứng thế, Este và lipit")
    cols = st.columns(len(MM_EXAMPLES))
    for c, t in zip(cols, MM_EXAMPLES):
        c.button(t, key=f"mm_ex_{t}", on_click=_set_topic, args=(t,), width="stretch")
    level = st.radio("Mức chi tiết", ["Gọn (4 nhánh, mỗi nhánh 2-3 ý)", "Đầy đủ (6 nhánh, mỗi nhánh 3-4 ý)"], horizontal=True, key="mm_level")
    topic = st.session_state.get("mm_topic", "").strip()
    if st.button("Tạo sơ đồ tư duy", type="primary", disabled=not topic, key="mm_go"):
        if st.session_state.n >= MAX_PER_SESSION:
            show_top_error(top, "Bạn đã dùng hết số lượt của phiên này. Hãy tải lại trang để tiếp tục.")
        else:
            with st.spinner("Đang vẽ sơ đồ…"):
                try:
                    want = "4 nhánh, mỗi nhánh 2-3 ý" if level.startswith("Gọn") else "6 nhánh, mỗi nhánh 3-4 ý"
                    raw = ask_ai([MM_PROMPT + f"\nSố nhánh và ý: {want}.\n\nChủ đề: {topic}"], temperature=0.2)
                    branches = mm_clean(raw)
                    if not branches:
                        show_top_error(top, "Chưa tạo được sơ đồ cho chủ đề này, hãy thử diễn đạt rõ hơn.")
                    else:
                        smi, facts = (raw.get("chat_smiles") or "").strip(), None
                        if smi and smi.lower() != "null":
                            facts = mm_engine_facts(smi)
                        if facts:
                            branches.append({"ten": "Số liệu (máy tính)", "y": facts, "color": "#475569"})
                        st.session_state.mm = {"title": str(raw.get("chu_de") or topic)[:60], "branches": branches,
                                               "smiles": smi if facts else "", "rev": int(time.time() * 1000)}
                        st.session_state.n += 1
                except Exception as e:
                    st.session_state.mm = None
                    show_top_error(top, "Có lỗi khi tạo sơ đồ, hãy thử lại.", e)
    r = st.session_state.get("mm")
    if r:
        st.divider()
        rev = r["rev"]
        title = st.text_input("Chủ đề ở giữa sơ đồ", value=r["title"], key=f"mm_title_{rev}").strip() or r["title"]
        with st.expander("✏️ Chỉnh sửa nội dung (sửa chữ, thêm hoặc xóa dòng, sơ đồ cập nhật ngay)", expanded=False):
            base = [{"Nhánh": b["ten"], "Ý": t} for b in r["branches"] for t in (b["y"] or [""])]
            edited = st.data_editor(base, num_rows="dynamic", hide_index=True, key=f"mm_edit_{rev}", width="stretch",
                                    column_config={"Nhánh": st.column_config.TextColumn("Nhánh", width="medium"),
                                                   "Ý": st.column_config.TextColumn("Ý", width="large")})
            st.caption("Muốn thêm nhánh mới: thêm dòng mới và gõ tên nhánh mới. Dòng để trống cột Nhánh sẽ thuộc nhánh ngay phía trên.")
        rows = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
        branches = rows_to_branches(rows)
        if not branches:
            st.warning("Sơ đồ đang trống, hãy nhập ít nhất một nhánh.")
        else:
            L = mindmap_layout(title, branches)
            svg = svg_from_layout(L)
            mm_component(svg, L["W"], L["H"])
            st.markdown("**Tải về để chỉnh sửa tiếp trên máy**")
            c1, c2, c3 = st.columns(3)
            pp = mm_pptx_bytes(L)
            if pp:
                c1.download_button("📊 PowerPoint (.pptx)", pp, file_name="so_do_tu_duy.pptx", key="mm_pptx",
                                   mime="application/vnd.openxmlformats-officedocument.presentationml.presentation", width="stretch")
            else:
                c1.caption("Thiếu thư viện python-pptx nên chưa xuất được PowerPoint.")
            c2.download_button("🧩 draw.io (.drawio)", mm_drawio(L).encode("utf-8"), file_name="so_do_tu_duy.drawio", key="mm_dio", width="stretch")
            c3.download_button("🗺️ XMind / FreeMind (.mm)", mm_freemind(L).encode("utf-8"), file_name="so_do_tu_duy.mm", key="mm_fm", width="stretch")
            d1, d2 = st.columns(2)
            d1.download_button("🖼️ SVG (vector, in sắc nét)", svg.encode("utf-8"), file_name="so_do_tu_duy.svg", mime="image/svg+xml", key="mm_svg", width="stretch")
            d2.download_button("📝 Dàn ý (Markdown)", mm_outline(title, branches).encode("utf-8"), file_name="so_do_tu_duy.md", key="mm_md", width="stretch")
            st.caption("PowerPoint: mỗi khung là hình có chữ sửa được và đường nối dính vào khung, kéo khung thì đường nối đi theo. "
                       "draw.io mở bằng app.diagrams.net. File .mm mở bằng XMind, Freeplane, MindManager.")
        if r.get("smiles"):
            with st.expander("Cấu trúc chất (kiểm tra xem đúng chất bạn muốn chưa)"):
                m = Chem.MolFromSmiles(r["smiles"])
                if m is not None:
                    st.image(Draw.MolToImage(m, size=(300, 200)))
                st.code(r["smiles"], language=None)
        st.caption("Nội dung lý thuyết do AI soạn theo SGK nên cần đối chiếu sách. Nhánh \"Số liệu (máy tính)\" do RDKit và bộ máy đếm đồng phân tính, không phải AI.")


# ---------------- Giao diện ----------------
CSS = """
<style>
.block-container{max-width:1000px;padding-top:1.2rem}
.hero{position:relative;overflow:hidden;color:#fff;border-radius:18px;padding:24px 28px;margin-bottom:14px;
background:linear-gradient(120deg,#1f7a4d,#0f9d8a,#2b6cb0,#1f7a4d);background-size:300% 300%;animation:gradmove 12s ease infinite}
.hero .t{position:relative;z-index:2;font-size:2rem;font-weight:800;line-height:1.2;
background:linear-gradient(90deg,#fff 0%,#b8ffe0 25%,#fff 50%,#b8ffe0 75%,#fff 100%);background-size:200% auto;
-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;animation:shine 5s linear infinite}
.hero p{position:relative;z-index:2;display:inline-block;margin:6px 0 0;font-size:1rem;opacity:.95;animation:reveal 2.6s steps(44,end) .4s both}
.hero .mol{position:absolute;z-index:1;opacity:.2;font-size:2.6rem;animation:floaty 7s ease-in-out infinite;user-select:none}
.hero .m1{right:6%;top:12%}.hero .m2{right:18%;bottom:8%;font-size:1.8rem;animation-delay:1.2s}
.hero .m3{right:30%;top:20%;font-size:2.2rem;animation-delay:2.4s}.hero .m4{right:42%;bottom:12%;font-size:1.5rem;animation-delay:3.2s}
@keyframes gradmove{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes shine{to{background-position:200% center}}
@keyframes reveal{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}
@keyframes floaty{0%,100%{transform:translateY(0) rotate(0)}50%{transform:translateY(-16px) rotate(18deg)}}
@keyframes fadein{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
div[data-baseweb="tab-panel"]{animation:fadein .45s ease}
.intro{animation:fadein .6s ease}
button:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(31,122,77,.25)}
button{transition:transform .15s ease,box-shadow .15s ease}
button[data-baseweb="tab"]{font-size:1.02rem;font-weight:600;padding:12px 14px}
button[data-baseweb="tab"][aria-selected="true"]{color:#1f7a4d}
div[data-baseweb="tab-highlight"]{background-color:#1f7a4d}
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:#1f7a4d;border-color:#1f7a4d;color:#fff;border-radius:10px;font-weight:600}
[data-testid="stCameraInput"]{max-width:440px;margin:0 auto}
.errtop{color:#d00000;font-weight:800;font-size:1.08rem;line-height:1.5;background:rgba(220,0,0,.08);border:2px solid rgba(220,0,0,.45);border-radius:12px;padding:12px 16px;margin:6px 0 12px}
@media(prefers-color-scheme:dark){.errtop{color:#ff6b6b;background:rgba(255,80,80,.12);border-color:rgba(255,107,107,.55)}}
.intro{display:flex;gap:14px;align-items:center;background:rgba(31,122,77,.10);border:1px solid rgba(31,122,77,.28);border-radius:14px;padding:12px 16px;margin:6px 0 16px}
.intro .ic{font-size:2rem;line-height:1}
.intro b{font-size:1.05rem}
.intro span{opacity:.8;font-size:.95rem}
@media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}.hero .t{background:none!important;-webkit-text-fill-color:#fff!important}}
@media(max-width:640px){.hero .t{font-size:1.5rem}.hero{padding:16px 18px}.hero .mol{display:none}button[data-baseweb="tab"]{padding:10px 10px;font-size:.95rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    f'<div class="hero"><span class="mol m1">⬡</span><span class="mol m2">⚗️</span><span class="mol m3">🧪</span><span class="mol m4">⚛️</span>'
    f'<div class="t">{APP_ICON} {APP_NAME}</div>'
    "<p>Chụp là biết, luyện là nhớ: công thức, phản ứng, đồng phân, dụng cụ, chấm bài tự động</p></div>",
    unsafe_allow_html=True,
)

pw = st.secrets.get("APP_PASSWORD", "")
if pw and st.text_input("Mật khẩu lớp học", type="password") != pw:
    st.stop()

st.session_state.setdefault("n", 0)
st.session_state.setdefault("res", None)


def intro(icon, title, text):
    st.markdown(
        f'<div class="intro"><div class="ic">{icon}</div><div><b>{title}</b><br><span>{text}</span></div></div>',
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


tab1, tab2, tab3, tab4, tab7, tab5, tab6 = st.tabs(
    ["✍️ Công thức cấu tạo", "🧪 Phương trình phản ứng", "🔀 Đồng phân", "🔬 Dụng cụ hóa học", "🧠 Sơ đồ tư duy", "🎓 Luyện tập & chấm bài", "📊 Đánh giá"]
)
with tab1:
    intro("✍️", "Công thức cấu tạo viết tay", "Chụp ảnh công thức vẽ tay kiểu SGK → mã SMILES, tên gọi, khối lượng mol, tính chất.")
    structure_ui()
with tab2:
    intro("🧪", "Phương trình phản ứng", "Đọc phương trình, cho biết đây là loại phản ứng nào và kiểm tra đã cân bằng chưa.")
    reaction_ui()
    st.caption("Kết quả do AI gợi ý, có thể sai ở nét viết khó đọc. Việc kiểm tra cân bằng nguyên tử do RDKit tính trực tiếp. Hãy đối chiếu SGK.")
with tab3:
    intro("🔀", "Đồng phân", "Liệt kê đồng phân của một công thức hoặc so sánh hai chất xem có phải đồng phân không.")
    isomer_ui()
with tab4:
    intro("🔬", "Dụng cụ hóa học", "Chụp ảnh dụng cụ hoặc bộ thí nghiệm để biết tên, công dụng, cách dùng và lưu ý an toàn.")
    equipment_ui()
with tab7:
    intro("🧠", "Sơ đồ tư duy", "Nhập chủ đề hoặc tên chất, máy vẽ sơ đồ tư duy để ôn tập. Số liệu của chất (công thức, khối lượng mol, số đồng phân) do máy tính toán.")
    mindmap_ui()
with tab5:
    intro("🎓", "Luyện tập và chấm bài tự động", "Ra đề, nộp bài bằng ảnh vẽ tay, máy chấm và chỉ ra lỗi sai (hóa trị, thiếu thừa nguyên tử, đồng phân, nhóm chức).")
    practice_ui()
with tab6:
    intro("📊", "Đánh giá độ chính xác", "Dành cho giáo viên và nhóm nghiên cứu: chạy bộ ảnh có đáp án để lấy số liệu cho báo cáo.")
    bench_ui()

st.caption(f"Lượt phân tích đã dùng trong phiên này: {st.session_state.n}/{MAX_PER_SESSION}")
