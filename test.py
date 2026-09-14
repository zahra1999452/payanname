import zipfile
import shutil
import tempfile
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path


# ============================================================
# تنظیمات
# ============================================================

INPUT_FILE = "E:\\rename\\yolo_work_out\\resrcher\\marine_debris_detection\\models\\پایان_نامه .docx"
OUTPUT_FILE = "thesis_fixed.docx"

# Namespace های Word
NS_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

W = "{%s}" % NS_W
M = "{%s}" % NS_M


# ============================================================
# تبدیل دستورات LaTeX به متن مناسب
# ============================================================

def latex_cleanup(s):
    s = s.strip()

    # حذف $$ ... $$
    s = re.sub(r"^\$\$(.*?)\$\$$", r"\1", s, flags=re.S)

    # حذف \[ ... \]
    s = re.sub(r"^\\\[(.*?)\\\]$", r"\1", s, flags=re.S)

    # حذف \( ... \)
    s = re.sub(r"^\\\((.*?)\\\)$", r"\1", s, flags=re.S)

    # دستورات ساده
    replacements = {
        r"\rightarrow": "→",
        r"\to": "→",
        r"\leftarrow": "←",
        r"\leftrightarrow": "↔",

        r"\times": "×",
        r"\cdot": "·",

        r"\leq": "≤",
        r"\le": "≤",
        r"\geq": "≥",
        r"\ge": "≥",
        r"\neq": "≠",

        r"\approx": "≈",
        r"\sim": "∼",

        r"\cap": "∩",
        r"\cup": "∪",

        r"\in": "∈",
        r"\notin": "∉",

        r"\infty": "∞",

        r"\pm": "±",

        r"\ldots": "…",
        r"\cdots": "⋯",

        r"\%": "%",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    # حذف \boxed{...}
    s = remove_simple_command(s, r"\boxed")

    # حذف \mathrm{...}
    s = remove_simple_command(s, r"\mathrm")

    # حذف \text{...}
    s = remove_simple_command(s, r"\text")

    # حذف \operatorname{...}
    s = remove_simple_command(s, r"\operatorname")

    # فاصله های LaTeX
    s = s.replace(r"\,", " ")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\:", " ")
    s = s.replace(r"\!", "")
    s = s.replace(r"\ ", " ")

    # پرانتزها
    s = s.replace(r"\left(", "(")
    s = s.replace(r"\right)", ")")

    s = s.replace(r"\left[", "[")
    s = s.replace(r"\right]", "]")

    s = s.replace(r"\left|", "|")
    s = s.replace(r"\right|", "|")

    # حذف backslash باقی مانده
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)

    # فاصله اضافی
    s = re.sub(r"\s+", " ", s)

    return s.strip()


def remove_simple_command(text, command):
    """
    \boxed{ABC}
    تبدیل می شود به:
    ABC
    """

    pattern = re.escape(command) + r"\{([^{}]*)\}"

    while re.search(pattern, text):
        text = re.sub(pattern, r"\1", text)

    return text


# ============================================================
# تبدیل یک عبارت ساده به OMML
# ============================================================

def create_m_run(text):
    """
    ایجاد یک m:r در OMML
    """

    r = ET.Element(M + "r")

    t = ET.SubElement(r, M + "t")
    t.text = text

    return r


def create_m_text(text):
    """
    ایجاد متن ریاضی OMML
    """

    r = ET.Element(M + "r")

    rpr = ET.SubElement(r, M + "rPr")

    sty = ET.SubElement(rpr, M + "sty")
    sty.set(M + "val", "p")

    t = ET.SubElement(r, M + "t")
    t.text = text

    return r


# ============================================================
# تشخیص زیروند و توان
# ============================================================

def parse_basic_math(expr):
    """
    تبدیل عبارت هایی مانند:

        D_{multi-source}
        D_{unified}^{13-class}
        B_p
        B_{gt}

    به OMML
    """

    result = []

    i = 0

    while i < len(expr):

        # ----------------------------------------------
        # حرف یا نماد اصلی
        # ----------------------------------------------

        if expr[i] in "_^":

            # اگر بدون base بود
            i += 1
            continue

        base = expr[i]

        # ----------------------------------------------
        # بررسی زیروند
        # ----------------------------------------------

        if i + 1 < len(expr) and expr[i + 1] == "_":

            i += 2

            sub = ""

            if i < len(expr) and expr[i] == "{":

                end = find_matching_brace(expr, i)

                if end != -1:
                    sub = expr[i + 1:end]
                    i = end + 1

            elif i < len(expr):
                sub = expr[i]
                i += 1

            # بررسی توان بعد از زیروند
            sup = None

            if i < len(expr) and expr[i] == "^":

                i += 1

                if i < len(expr) and expr[i] == "{":

                    end = find_matching_brace(expr, i)

                    if end != -1:
                        sup = expr[i + 1:end]
                        i = end + 1

                elif i < len(expr):
                    sup = expr[i]
                    i += 1

            # ساخت base
            e = ET.Element(M + "sSub" if sup is None else M + "sSubSup")

            base_el = ET.SubElement(e, M + "e")

            base_el.append(create_m_text(base))

            sub_el = ET.SubElement(e, M + "sub")
            sub_el.append(create_m_text(sub))

            if sup is not None:
                sup_el = ET.SubElement(e, M + "sup")
                sup_el.append(create_m_text(sup))

            result.append(e)

            continue

        # ----------------------------------------------
        # بررسی توان
        # ----------------------------------------------

        if i + 1 < len(expr) and expr[i + 1] == "^":

            i += 2

            sup = ""

            if i < len(expr) and expr[i] == "{":

                end = find_matching_brace(expr, i)

                if end != -1:
                    sup = expr[i + 1:end]
                    i = end + 1

            elif i < len(expr):
                sup = expr[i]
                i += 1

            e = ET.Element(M + "sSup")

            base_el = ET.SubElement(e, M + "e")
            base_el.append(create_m_text(base))

            sup_el = ET.SubElement(e, M + "sup")
            sup_el.append(create_m_text(sup))

            result.append(e)

            continue

        # ----------------------------------------------
        # حرف عادی
        # ----------------------------------------------

        result.append(create_m_text(base))

        i += 1

    return result


def find_matching_brace(text, start):
    """
    پیدا کردن } متناظر با {
    """

    depth = 0

    for i in range(start, len(text)):

        if text[i] == "{":
            depth += 1

        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                return i

    return -1


# ============================================================
# تبدیل کسر
# ============================================================

def create_fraction(numerator, denominator):

    frac = ET.Element(M + "f")

    num = ET.SubElement(frac, M + "num")
    num.append(create_m_text(numerator))

    den = ET.SubElement(frac, M + "den")
    den.append(create_m_text(denominator))

    return frac


# ============================================================
# تبدیل فرمول کامل
# ============================================================

def latex_to_omml(latex):

    latex = latex_cleanup(latex)

    omath = ET.Element(M + "oMath")

    # --------------------------------------------------------
    # اگر \frac داشته باشیم
    # --------------------------------------------------------

    frac_pattern = re.compile(
        r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}"
    )

    pos = 0

    while True:

        match = frac_pattern.search(latex, pos)

        if not match:
            break

        before = latex[pos:match.start()]

        if before:
            for element in parse_basic_math(before):
                omath.append(element)

        numerator = match.group(1)
        denominator = match.group(2)

        frac = create_fraction(
            numerator,
            denominator
        )

        omath.append(frac)

        pos = match.end()

    remaining = latex[pos:]

    if remaining:
        for element in parse_basic_math(remaining):
            omath.append(element)

    return omath


# ============================================================
# پیدا کردن فرمول های LaTeX داخل XML ورد
# ============================================================

LATEX_PATTERNS = [
    re.compile(r"\$\$(.*?)\$\$", re.S),
    re.compile(r"\\\[(.*?)\\\]", re.S),
    re.compile(r"\\\((.*?)\\\)", re.S),
]


def find_latex(text):

    matches = []

    for pattern in LATEX_PATTERNS:

        for match in pattern.finditer(text):

            matches.append(
                (
                    match.start(),
                    match.end(),
                    match.group(0)
                )
            )

    matches.sort(key=lambda x: x[0])

    return matches


# ============================================================
# جایگزینی Text داخل Word با OMML
# ============================================================

def process_document_xml(xml_data):

    root = ET.fromstring(xml_data)

    changed = False

    # همه paragraph ها
    for paragraph in root.iter(W + "p"):

        # جمع آوری run ها
        runs = list(paragraph.findall(".//" + W + "r"))

        for run in runs:

            text_elements = run.findall(".//" + W + "t")

            for text_element in text_elements:

                if text_element.text is None:
                    continue

                text = text_element.text

                matches = find_latex(text)

                if not matches:
                    continue

                changed = True

                parent = find_parent(
                    root,
                    text_element
                )

                if parent is None:
                    continue

                # در این نسخه هر فرمولی که داخل یک w:t باشد
                # به صورت OMML جایگزین می شود.

                new_elements = []

                last = 0

                for start, end, latex in matches:

                    before = text[last:start]

                    if before:
                        wr = ET.Element(W + "r")
                        wt = ET.SubElement(wr, W + "t")
                        wt.text = before
                        new_elements.append(wr)

                    omml = latex_to_omml(latex)

                    new_elements.append(omml)

                    last = end

                after = text[last:]

                if after:
                    wr = ET.Element(W + "r")
                    wt = ET.SubElement(wr, W + "t")
                    wt.text = after
                    new_elements.append(wr)

                # جایگزینی
                index = list(parent).index(run)

                parent.remove(run)

                for offset, element in enumerate(new_elements):
                    parent.insert(index + offset, element)

                break

    return ET.tostring(
        root,
        encoding="utf-8",
        xml_declaration=True
    ), changed


def find_parent(root, target):

    for parent in root.iter():

        for child in list(parent):

            if child is target:
                return parent

    return None


# ============================================================
# پردازش DOCX
# ============================================================

def process_docx(input_file, output_file):

    if not os.path.exists(input_file):

        print("خطا:")
        print("فایل پیدا نشد:")
        print(input_file)

        return

    temp_dir = tempfile.mkdtemp()

    try:

        # ----------------------------------------------
        # استخراج DOCX
        # ----------------------------------------------

        with zipfile.ZipFile(
            input_file,
            "r"
        ) as zip_ref:

            zip_ref.extractall(temp_dir)

        # ----------------------------------------------
        # فایل اصلی Word
        # ----------------------------------------------

        document_xml = os.path.join(
            temp_dir,
            "word",
            "document.xml"
        )

        if not os.path.exists(document_xml):

            print("document.xml پیدا نشد.")
            return

        # ----------------------------------------------
        # خواندن XML
        # ----------------------------------------------

        with open(
            document_xml,
            "rb"
        ) as f:

            xml_data = f.read()

        # ----------------------------------------------
        # تبدیل
        # ----------------------------------------------

        new_xml, changed = process_document_xml(
            xml_data
        )

        # ----------------------------------------------
        # ذخیره XML جدید
        # ----------------------------------------------

        if changed:

            with open(
                document_xml,
                "wb"
            ) as f:

                f.write(new_xml)

            print("فرمول‌ها تبدیل شدند.")

        else:

            print(
                "هیچ فرمول LaTeX قابل تشخیصی پیدا نشد."
            )

        # ----------------------------------------------
        # ساخت DOCX جدید
        # ----------------------------------------------

        with zipfile.ZipFile(
            output_file,
            "w",
            zipfile.ZIP_DEFLATED
        ) as new_docx:

            for root_dir, dirs, files in os.walk(temp_dir):

                for file in files:

                    full_path = os.path.join(
                        root_dir,
                        file
                    )

                    relative_path = os.path.relpath(
                        full_path,
                        temp_dir
                    )

                    new_docx.write(
                        full_path,
                        relative_path
                    )

        print()
        print("فایل جدید ساخته شد:")
        print(output_file)

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# اجرای برنامه
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("تبدیل فرمول‌های Word به Equation سازگار با Word 2010")
    print("=" * 70)

    print()
    print("فایل ورودی:")
    print(INPUT_FILE)

    print()
    print("فایل خروجی:")
    print(OUTPUT_FILE)

    print()

    process_docx(
        INPUT_FILE,
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("تمام شد.")
    print("=" * 70)