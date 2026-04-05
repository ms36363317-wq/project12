import os
import requests
import streamlit as st
import numpy as np
import cv2
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Assistant For Detection Of Retinal Diseases",
    initial_sidebar_state="collapsed"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: #060a10;
        color: #e8edf5;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 2rem 4rem; max-width: 1200px; }

    /* ── Hero ── */
    .hero {
        position: relative;
        text-align: center;
        padding: 3.5rem 2rem 2.5rem;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,180,255,0.12) 0%, transparent 70%),
            radial-gradient(ellipse 40% 30% at 20% 80%, rgba(0,80,200,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-eyebrow {
        font-size: 0.75rem; font-weight: 500; letter-spacing: 0.25em;
        text-transform: uppercase; color: #38bdf8; margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2.4rem, 5vw, 4rem);
        font-weight: 800; line-height: 1.05;
        letter-spacing: -0.02em; color: #f0f6ff; margin: 0 0 1rem;
    }
    .hero-title span {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-subtitle {
        font-size: 1rem; font-weight: 300; color: #8ba3bf;
        max-width: 520px; margin: 0 auto; line-height: 1.7;
    }

    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent);
        margin: 0 0 2.5rem;
    }

    /* ── Upload ── */
    .upload-section {
        background: rgba(255,255,255,0.03);
        border: 1.5px dashed rgba(56,189,248,0.25);
        border-radius: 20px; padding: 2.5rem 2rem;
        text-align: center; margin-bottom: 2rem;
    }
    .upload-label {
        font-family: 'Syne', sans-serif; font-size: 1.1rem;
        font-weight: 600; color: #c8d8ea; margin-bottom: 0.4rem;
    }
    .upload-hint { font-size: 0.82rem; color: #5a7a96; }

    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploader"] > div { border: none !important; background: transparent !important; padding: 0 !important; }
    [data-testid="stFileUploader"] label { color: #38bdf8 !important; font-size: 0.9rem; }

    /* ── Image Cards ── */
    .img-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 0.6rem 0.6rem 0.5rem;
        text-align: center;
        max-width: 220px;
        margin: 0 auto;
    }
    .img-card-label {
        font-size: 0.68rem; font-weight: 500;
        letter-spacing: 0.18em; text-transform: uppercase;
        color: #5a7a96; margin-top: 0.5rem;
    }

    /* ── Streamlit image ── */
    [data-testid="stImage"] img {
        border-radius: 10px;
        width: 100%;
        max-height: 200px;
        object-fit: cover;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0ea5e9, #6366f1) !important;
        border-radius: 999px !important;
    }
    .stProgress > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 999px !important; height: 8px !important;
    }

    /* ── Confidence ── */
    .confidence-label {
        font-size: 0.78rem; letter-spacing: 0.15em;
        text-transform: uppercase; color: #5a7a96; margin-bottom: 0.5rem;
    }
    .confidence-value {
        font-family: 'Syne', sans-serif; font-size: 2.4rem;
        font-weight: 800; color: #f0f6ff; line-height: 1;
    }
    .confidence-value span { font-size: 1rem; font-weight: 400; color: #5a7a96; }

    /* ── Disease Card ── */
    .disease-card {
        background: rgba(56,189,248,0.05);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 14px; padding: 1.2rem 1.4rem; margin-top: 1rem;
    }
    .disease-card-title {
        font-family: 'Syne', sans-serif; font-size: 0.95rem;
        font-weight: 700; color: #38bdf8; margin-bottom: 0.4rem;
    }
    .disease-card-text { font-size: 0.85rem; color: #8ba3bf; line-height: 1.65; }

    /* ── LLM Explanation Card ── */
    .llm-card {
        background: rgba(129,140,248,0.07);
        border: 1px solid rgba(129,140,248,0.22);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
    }
    .llm-card-title {
        font-family: 'Syne', sans-serif; font-size: 0.95rem;
        font-weight: 700; color: #818cf8; margin-bottom: 0.75rem;
        display: flex; align-items: center; gap: 0.4rem;
    }
    .llm-line {
        font-size: 0.86rem; color: #c8d8ea;
        line-height: 1.7; margin-bottom: 0.4rem;
        padding-left: 0.5rem;
        border-left: 2px solid rgba(129,140,248,0.3);
    }
    .llm-error {
        font-size: 0.82rem; color: #f59e0b;
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 8px; padding: 0.7rem 1rem;
        margin-top: 0.5rem;
    }

    /* ── Ollama Model Selector ── */
    .model-selector-label {
        font-size: 0.72rem; letter-spacing: 0.18em;
        text-transform: uppercase; color: #5a7a96;
        margin-bottom: 0.4rem;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: rgba(245,158,11,0.08);
        border: 1px solid rgba(245,158,11,0.2);
        border-radius: 12px; padding: 0.9rem 1.2rem;
        font-size: 0.78rem; color: #92762e;
        text-align: center; margin-top: 2.5rem; line-height: 1.6;
    }

    [data-testid="stSidebar"] {
        background: #080c14 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# Constants
# ==============================
MODEL_PATH = "best_efficientnetb3.h5"
FILE_ID = "1qnrKRAWa7UU5YbtT2UqGDbJij7uH6dIz"
OLLAMA_URL = "http://localhost:11434/api/generate"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

# ==============================
# Disease Info
# ==============================
disease_info = {
    "Diabetic Retinopathy": {
        "desc": "تلف في أوعية الدم الدقيقة بشبكية العين نتيجة مرض السكري. يُعدّ من الأسباب الرئيسية للعمى لدى البالغين.",
        "action": "يُنصح بفحص دوري كل 6 أشهر ومراقبة مستوى السكر في الدم.",
        "icon": "🩺"
    },
    "Disc Edema": {
        "desc": "تورم في القرص البصري قد يشير إلى ارتفاع ضغط الدم داخل الجمجمة أو اضطرابات عصبية.",
        "action": "يتطلب تقييمًا عصبيًا عاجلاً وصور أشعة للدماغ.",
        "icon": "🧠"
    },
    "Healthy": {
        "desc": "لم يُكتشف أي مؤشر مرضي. تبدو شبكية العين سليمة وبحالة جيدة.",
        "action": "حافظ على فحوصات دورية سنوية للعين للاطمئنان على صحتها.",
        "icon": "✅"
    },
    "Myopia": {
        "desc": "قِصَر النظر: صعوبة في رؤية الأشياء البعيدة بوضوح بسبب طول محور مقلة العين.",
        "action": "يمكن تصحيحه بالنظارات أو العدسات اللاصقة أو جراحة الليزر.",
        "icon": "👓"
    },
    "Pterygium": {
        "desc": "نسيج ليفي وعائي ينمو على سطح القرنية من الملتحمة، وقد يؤثر على الرؤية.",
        "action": "قد يحتاج إلى استئصال جراحي إذا تقدّم نحو مركز القرنية.",
        "icon": "🔬"
    },
    "Retinal Detachment": {
        "desc": "انفصال الشبكية عن طبقة الظهارة الصباغية، وهو طارئ طبي يستوجب تدخلاً فوريًا.",
        "action": "توجّه فورًا إلى أقرب طوارئ عيون — يمكن أن يؤدي التأخير إلى فقدان البصر نهائيًا.",
        "icon": "🚨"
    },
    "Retinitis Pigmentosa": {
        "desc": "مجموعة اضطرابات وراثية تُسبب تدهورًا تدريجيًا في خلايا الشبكية المستقبلة للضوء.",
        "action": "لا يوجد علاج شافٍ حتى الآن؛ التدبير يركز على إبطاء التقدم وتحسين جودة الحياة.",
        "icon": "🧬"
    },
}

severity_color = {
    "Healthy": "#22c55e",
    "Myopia": "#f59e0b",
    "Pterygium": "#f59e0b",
    "Diabetic Retinopathy": "#ef4444",
    "Disc Edema": "#ef4444",
    "Retinal Detachment": "#dc2626",
    "Retinitis Pigmentosa": "#ef4444",
}

# ==============================
# LLM Explanation — Ollama أو Claude API
# ==============================
PROMPT_TEMPLATE = """You are an ophthalmology AI assistant.

Write exactly 5 short medical lines about this eye disease prediction:

Prediction: {disease}
Confidence: {confidence:.1f}%

Structure (5 lines only, no headers, no repetition):
1. Prediction statement.
2. Short clinical definition.
3. Key symptoms the patient may notice.
4. Severity level (Mild / Moderate / Severe / Emergency).
5. Recommended next step."""


def _clean_lines(text: str) -> str:
    """خذ أول 5 أسطر غير فارغة."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return "\n".join(lines[:5])


def _explain_via_ollama(disease: str, confidence: float, ollama_model: str, ollama_url: str) -> str:
    prompt = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 200, "repeat_penalty": 1.2},
    }
    api_url = f"{ollama_url.rstrip('/')}/api/generate"
    response = requests.post(api_url, json=payload, timeout=60)
    response.raise_for_status()
    raw = response.json().get("response", "").strip()
    return _clean_lines(raw)


def _test_ollama_connection(ollama_url: str) -> tuple:
    """يختبر الاتصال بـ Ollama ويعيد (نجح، رسالة)."""
    try:
        r = requests.get(ollama_url.rstrip("/"), timeout=5)
        if r.status_code == 200:
            return True, "✅ Ollama يعمل بنجاح!"
        return False, f"⚠️ استجابة غير متوقعة: {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "❌ لا يمكن الاتصال — تأكد أن: ollama serve يعمل"
    except requests.exceptions.Timeout:
        return False, "❌ انتهت المهلة — الخادم لا يستجيب"
    except Exception as e:
        return False, f"❌ خطأ: {e}"


def _explain_via_claude(disease: str, confidence: float, api_key: str) -> str:
    prompt = PROMPT_TEMPLATE.format(disease=disease, confidence=confidence * 100)
    response = requests.post(
        ANTHROPIC_API_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()["content"][0]["text"].strip()
    return _clean_lines(raw)


def local_llm_explain(
    disease: str,
    confidence: float,
    ollama_model: str = "llama3",
    ollama_url: str = "http://localhost:11434",
    backend: str = "ollama",
    anthropic_api_key: str = "",
) -> str:
    try:
        if backend == "claude":
            if not anthropic_api_key.strip():
                return "ERROR: أدخل Anthropic API Key في إعدادات الشريط الجانبي."
            return _explain_via_claude(disease, confidence, anthropic_api_key.strip())
        else:
            return _explain_via_ollama(disease, confidence, ollama_model, ollama_url)

    except requests.exceptions.ConnectionError:
        if backend == "ollama":
            return f"ERROR: تعذّر الاتصال بـ Ollama على {ollama_url} — تأكد أن: ollama serve يعمل"
        return "ERROR: تعذّر الاتصال بـ Anthropic API — تحقق من اتصالك بالإنترنت."
    except requests.exceptions.Timeout:
        return "ERROR: انتهت مهلة الاستجابة — النموذج بطيء أو غير محمّل."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 401:
            return "ERROR: API Key غير صالح — تحقق من المفتاح."
        if status == 404 and backend == "ollama":
            return f"ERROR: النموذج «{ollama_model}» غير محمّل — نفّذ: ollama pull {ollama_model}"
        return f"ERROR: HTTP {status} — {e}"
    except Exception as e:
        return f"ERROR: خطأ غير متوقع: {e}"


# ==============================
# Load Vision Model
# ==============================
@st.cache_resource
def load_model_cached():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ جاري تحميل النموذج..."):
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_PATH,
                quiet=False
            )

    if not os.path.exists(MODEL_PATH):
        st.error("❌ النموذج غير موجود")
        st.stop()

    if os.path.getsize(MODEL_PATH) < 5_000_000:
        st.error("❌ ملف النموذج تالف")
        st.stop()

    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ فشل تحميل النموذج: {e}")
        st.stop()


# ==============================
# Classes
# ==============================
class_names = [
    'Diabetic Retinopathy', 'Disc Edema', 'Healthy',
    'Myopia', 'Pterygium', 'Retinal Detachment', 'Retinitis Pigmentosa'
]

# ==============================
# Helpers
# ==============================
def preprocess(img):
    img = img.resize((300, 300))
    arr = np.array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predict(img, model):
    preds = model.predict(preprocess(img))
    idx = np.argmax(preds[0])
    return class_names[idx], float(np.max(preds)), preds[0]


def overlay_heatmap(img, heatmap):
    arr = np.array(img.resize((300, 300)))
    return cv2.addWeighted(arr, 0.75, heatmap, 0.25, 0)


def gradcam(img, model):
    arr = np.array(img.resize((300, 300)))
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    target_layer = next(
        (l for l in reversed(model.layers) if isinstance(l, tf.keras.layers.Conv2D)),
        None
    )

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        outputs = grad_model(arr)
        conv_outputs = outputs[0]
        predictions = outputs[1]

        if isinstance(predictions, list):
            predictions = predictions[0]

        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            class_idx = tf.argmax(predictions[0]).numpy()
            loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    grads = grads / (tf.reduce_mean(tf.abs(grads)) + 1e-8)

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)[0].numpy()

    cam = np.maximum(cam, 0)
    if np.max(cam) > 0:
        cam /= np.max(cam)

    cam = np.power(cam, 0.3)
    cam = cv2.resize(cam, (300, 300))

    return cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)


# ==============================
# Hero
# ==============================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Ophthalmology</div>
    <h1 class="hero-title">Assistant For Detection Of Retinal Diseases <span>AI</span></h1>
    <p class="hero-subtitle">
        نظام ذكاء اصطناعي لتحليل صور قاع العين وكشف الأمراض بدقة عالية باستخدام EfficientNet و Grad-CAM و Ollama LLM
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ==============================
# Load Vision Model
# ==============================
model = load_model_cached()

# ==============================
# Sidebar — LLM Settings
# ==============================
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                color:#818cf8; margin-bottom:1rem; padding-bottom:0.5rem;
                border-bottom:1px solid rgba(129,140,248,0.2);">
        ⚙️ إعدادات النموذج اللغوي
    </div>
    """, unsafe_allow_html=True)

    enable_llm = st.toggle("تفعيل شرح LLM", value=True)

    llm_backend = st.radio(
        "مزوّد النموذج",
        options=["Ollama (محلي)", "Claude API (سحابي)"],
        index=0,
        help="اختر Ollama لو النموذج عندك محلياً، أو Claude API لو عندك مفتاح Anthropic"
    )
    backend_key = "ollama" if llm_backend.startswith("Ollama") else "claude"

    # ── Ollama settings ──
    if backend_key == "ollama":
        ollama_model = st.selectbox(
            "نموذج Ollama",
            options=["llama3", "mistral", "phi3", "gemma", "llama2", "neural-chat"],
            index=0,
            help="تأكد أن النموذج محمّل: ollama pull <model>"
        )
        ollama_url = st.text_input(
            "Ollama URL",
            value="http://localhost:11434",
            help="الرابط الافتراضي لـ Ollama"
        )
        anthropic_api_key = ""

        # زر اختبار الاتصال
        if st.button("🔌 اختبار الاتصال بـ Ollama", use_container_width=True):
            ok, msg = _test_ollama_connection(ollama_url)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

        st.markdown("""
        <div style="margin-top:1.2rem; font-size:0.75rem; color:#3a5a76; line-height:2;">
            <span style="color:#5a7a96; font-weight:500;">تشغيل Ollama:</span><br>
            <code style="background:rgba(56,189,248,0.08); color:#38bdf8;
                         padding:0.15rem 0.5rem; border-radius:4px;">ollama serve</code>
            <br><br>
            <span style="color:#5a7a96; font-weight:500;">تحميل نموذج:</span><br>
            <code style="background:rgba(56,189,248,0.08); color:#38bdf8;
                         padding:0.15rem 0.5rem; border-radius:4px;">ollama pull llama3</code>
        </div>
        """, unsafe_allow_html=True)

    # ── Claude API settings ──
    else:
        ollama_model = "llama3"
        ollama_url = "http://localhost:11434"
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-...",
            help="احصل على مفتاحك من: console.anthropic.com"
        )
        st.markdown("""
        <div style="margin-top:1rem; font-size:0.75rem; color:#3a5a76; line-height:1.8;">
            النموذج المستخدم:
            <code style="background:rgba(129,140,248,0.1); color:#818cf8;
                         padding:0.1rem 0.4rem; border-radius:4px;">claude-haiku</code>
            <br>
            <a href="https://console.anthropic.com" target="_blank"
               style="color:#38bdf8; text-decoration:none;">← احصل على API Key</a>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Layout
# ==============================
left_col, right_col = st.columns([1, 1.6], gap="large")

with left_col:
    st.markdown('<div class="upload-label">رفع صورة العين</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-hint">الصيغ المدعومة: JPG · PNG</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="اختر صورة",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        thumb = image.copy()
        thumb.thumbnail((210, 210))
        st.markdown('<div class="img-card">', unsafe_allow_html=True)
        st.image(thumb, use_container_width=False, width=220)
        st.markdown('<div class="img-card-label">الصورة الأصلية</div></div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="upload-section">
            <div style="font-size:2.5rem; margin-bottom:0.75rem; opacity:0.4">👁️</div>
            <div style="font-size:0.88rem; color:#3a5a76;">اسحب وأفلت الصورة هنا<br>أو انقر للاختيار</div>
        </div>
        """, unsafe_allow_html=True)

with right_col:
    if uploaded_file:
        with st.spinner("🔍 جاري التحليل..."):
            pred, conf, all_preds = predict(image, model)
            heatmap = gradcam(image, model)
            overlay = overlay_heatmap(image, heatmap)

        color = severity_color.get(pred, "#38bdf8")
        info = disease_info.get(pred, {})

        # ── Diagnosis ──
        st.markdown(f"""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
                        color:#5a7a96; margin-bottom:0.6rem;">نتيجة التشخيص</div>
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
                <span style="font-size:1.8rem;">{info.get('icon','🔬')}</span>
                <span style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
                             color:{color}; letter-spacing:-0.01em;">{pred}</span>
            </div>
            <div class="confidence-label">مستوى الثقة</div>
            <div class="confidence-value">{conf*100:.1f}<span>%</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(int(conf * 100))

        # ── Disease Card ──
        if info:
            st.markdown(f"""
            <div class="disease-card">
                <div class="disease-card-title">📋 عن هذه الحالة</div>
                <div class="disease-card-text">{info['desc']}</div>
                <div style="margin-top:0.7rem; padding-top:0.7rem;
                             border-top:1px solid rgba(56,189,248,0.1);">
                    <span style="font-size:0.75rem; color:#38bdf8; font-weight:500;">التوصية: </span>
                    <span class="disease-card-text">{info['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── LLM Explanation ──
        if enable_llm:
            backend_label = "Claude API (Haiku)" if backend_key == "claude" else f"Ollama — {ollama_model}"
            with st.spinner(f"🤖 جاري توليد الشرح الطبي عبر {backend_label}..."):
                llm_result = local_llm_explain(
                    pred, conf,
                    ollama_model=ollama_model,
                    ollama_url=ollama_url,
                    backend=backend_key,
                    anthropic_api_key=anthropic_api_key,
                )

            if llm_result.startswith("ERROR:"):
                error_msg = llm_result.replace("ERROR:", "").strip()
                st.markdown(f"""
                <div class="llm-card">
                    <div class="llm-card-title">🤖 شرح النموذج اللغوي — {backend_label}</div>
                    <div class="llm-error">⚠️ {error_msg}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                lines = [l.strip() for l in llm_result.split("\n") if l.strip()]
                lines_html = "".join(
                    f'<div class="llm-line">{line}</div>'
                    for line in lines
                )
                st.markdown(f"""
                <div class="llm-card">
                    <div class="llm-card-title">🤖 شرح النموذج اللغوي — {backend_label}</div>
                    {lines_html}
                </div>
                """, unsafe_allow_html=True)

        # ── Grad-CAM ──
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
                    color:#5a7a96; margin-bottom:0.75rem;">التحليل البصري — Grad-CAM</div>
        """, unsafe_allow_html=True)

        v1, v2 = st.columns(2)
        with v1:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(heatmap, width=200, channels="BGR")
            st.markdown('<div class="img-card-label">خريطة الحرارة</div></div>', unsafe_allow_html=True)
        with v2:
            st.markdown('<div class="img-card">', unsafe_allow_html=True)
            st.image(overlay, width=200, channels="BGR")
            st.markdown('<div class="img-card-label">الصورة المدمجة</div></div>', unsafe_allow_html=True)

        # ── All Probabilities ──
        with st.expander("📊 جميع الاحتمالات"):
            for i in np.argsort(all_preds)[::-1]:
                pct = float(all_preds[i]) * 100
                bar_color = color if class_names[i] == pred else "#1e3a4a"
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.75rem;
                             margin-bottom:0.5rem; font-size:0.82rem;">
                    <div style="width:160px; color:#8ba3bf; white-space:nowrap;
                                overflow:hidden; text-overflow:ellipsis;">{class_names[i]}</div>
                    <div style="flex:1; background:rgba(255,255,255,0.05); border-radius:999px; height:6px; overflow:hidden;">
                        <div style="width:{pct:.1f}%; height:100%;
                                    background:{bar_color}; border-radius:999px;"></div>
                    </div>
                    <div style="width:44px; text-align:right; color:#5a7a96;">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="display:flex; flex-direction:column; align-items:center;
                    justify-content:center; height:300px; opacity:0.3; text-align:center;">
            <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:600; color:#8ba3bf;">
                في انتظار صورة للتحليل
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==============================
# Disclaimer
# ==============================
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>تنبيه طبي:</strong> هذا النظام أداةٌ مساعدة للفحص الأولي ولا يُغني عن استشارة طبيب متخصص.
    يُرجى مراجعة طبيب عيون معتمد للتشخيص النهائي والعلاج المناسب.
</div>
""", unsafe_allow_html=True)
