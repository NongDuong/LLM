"""
Ứng dụng mô hình Transformer trong Phân tích Quan điểm và Cảm xúc Bình luận Khách hàng
Đồ án thay thế nghiệp - Streamlit Demo App
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

# ─── Page config — MUST be first Streamlit call ───────────────────────────────
st.set_page_config(
    page_title="Phân tích Cảm xúc Khách hàng",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports from src/ ────────────────────────────────────────────────────────
from src.models import load_roberta, load_vader, predict_roberta, predict_vader
from src.visualization import (
    score_bars, radar_chart, confusion_matrix as cm_fig,
    metric_bars, sentiment_distribution, COLORS, VI,
)

# ─── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
VADER_CSV  = ROOT / "vader.csv"
SAMPLE_CSV = ROOT / "sample.csv"

EMOJI = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}

EXAMPLES = [
    "This product is absolutely amazing! Best purchase I've ever made. Highly recommend to everyone!",
    "Terrible quality. It broke after just one day. Complete waste of money. Very disappointed.",
    "It's an okay product. Nothing special but it does what it's supposed to do.",
    "I love this tea! The flavor is rich and smooth. Will definitely order again and again!",
    "The quality is very poor. Expected much better based on the description. Not worth the price at all.",
    "Decent product for the price. Works as described and shipping was fast. Pretty happy overall.",
]

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .app-header {
    background: linear-gradient(135deg, #1f4e79 0%, #2874A6 100%);
    color: white; padding: 1.5rem 2rem; border-radius: 12px;
    margin-bottom: 1.5rem; text-align: center;
  }
  .app-header h1 { font-size: 1.75rem; margin: 0 0 0.3rem; }
  .app-header p  { font-size: 0.95rem; margin: 0; opacity: 0.85; }
  .badge-positive { background:#d4edda; color:#155724; padding:0.3rem 1rem;
    border-radius:20px; font-weight:700; font-size:1.05rem; display:inline-block; }
  .badge-negative { background:#f8d7da; color:#721c24; padding:0.3rem 1rem;
    border-radius:20px; font-weight:700; font-size:1.05rem; display:inline-block; }
  .badge-neutral  { background:#fff3cd; color:#856404; padding:0.3rem 1rem;
    border-radius:20px; font-weight:700; font-size:1.05rem; display:inline-block; }
  .section-title {
    font-size:1.25rem; font-weight:600; color:#1f4e79;
    border-bottom:3px solid #2874A6; padding-bottom:0.4rem; margin:1rem 0;
  }
  .result-box {
    text-align:center; padding:1rem; background:#f8f9fa;
    border-radius:10px; margin-bottom:0.5rem;
  }
  #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_vader_csv():
    df = pd.read_csv(VADER_CSV)
    df['ground_truth'] = pd.cut(
        df['Score'], bins=[0, 2, 3, 5],
        labels=['negative', 'neutral', 'positive']
    ).astype(str)
    rob = ['roberta_neg', 'roberta_neu', 'roberta_pos']
    df['roberta_pred'] = df[rob].idxmax(axis=1).str.replace('roberta_', '', regex=False)
    df['vader_pred'] = pd.cut(
        df['vader_compound'], bins=[-1.01, -0.05, 0.05, 1.01],
        labels=['negative', 'neutral', 'positive']
    ).astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_sample_csv():
    return pd.read_csv(SAMPLE_CSV)


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    labs  = ['negative', 'neutral', 'positive']
    valid = [(t, p) for t, p in zip(y_true, y_pred) if t in labs and p in labs]
    if not valid:
        return {}
    y_t, y_p = zip(*valid)
    return {
        'accuracy':  accuracy_score(y_t, y_p),
        'precision': precision_score(y_t, y_p, average='weighted', zero_division=0),
        'recall':    recall_score(y_t, y_p,    average='weighted', zero_division=0),
        'f1_score':  f1_score(y_t, y_p,        average='weighted', zero_division=0),
    }


def highlight_sentiment(series):
    mapping = {
        'positive': 'background-color:#d4edda; color:#155724',
        'negative': 'background-color:#f8d7da; color:#721c24',
        'neutral':  'background-color:#fff3cd; color:#856404',
    }
    return [mapping.get(v, '') for v in series]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Đồ án thay thế tốt nghiệp")
    st.markdown("""
**Ứng dụng mô hình Transformer trong Phân tích Quan điểm và Cảm xúc Bình luận Khách hàng**
""")
    st.divider()

    st.markdown("### ⚙️ Trạng thái Mô hình")
    with st.spinner("Đang tải mô hình..."):
        tokenizer, rob_model, rob_err = load_roberta()
        vader_analyzer, vader_err     = load_vader()

    if rob_err:
        st.error(f"❌ RoBERTa: {rob_err[:80]}")
        st.info("Kiểm tra kết nối internet — mô hình tải từ HuggingFace (~500 MB)")
    else:
        st.success("✅ RoBERTa (Transformer)")

    if vader_err:
        st.error(f"❌ VADER: {vader_err[:80]}")
    else:
        st.success("✅ VADER (Lexicon-based)")

    st.divider()
    st.markdown("""
### 📊 Dataset
- **Amazon Fine Food Reviews**
- Mô hình: RoBERTa + VADER
- Task: Phân tích cảm xúc 3 lớp
  (Tích cực / Trung lập / Tiêu cực)

### 📋 Hướng dẫn
1. **Tab 1** — Nhập văn bản phân tích
2. **Tab 2** — Tải CSV phân tích loạt
3. **Tab 3** — Khám phá dataset
4. **Tab 4** — So sánh mô hình
""")

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h1>🎭 Phân tích Cảm xúc Bình luận Khách hàng</h1>
  <p>Ứng dụng mô hình Transformer (RoBERTa) so sánh với VADER — Sentiment Analysis Demo</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Phân tích Văn bản",
    "📂 Phân tích Hàng loạt",
    "📊 Khám phá Dữ liệu",
    "🏆 Đánh giá Mô hình",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE TEXT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Phân tích cảm xúc văn bản</div>',
                unsafe_allow_html=True)

    # Dùng state riêng để tránh lỗi "cannot modify widget key after instantiation"
    if "_input_text" not in st.session_state:
        st.session_state["_input_text"] = ""

    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("**📌 Ví dụ nhanh:**")
        for i, ex in enumerate(EXAMPLES):
            if st.button(ex[:40] + "...", key=f"ex_{i}",
                         use_container_width=True, help=ex):
                st.session_state["_input_text"] = ex
                st.rerun()

    with col_left:
        user_text = st.text_area(
            "Nhập văn bản tiếng Anh cần phân tích:",
            value=st.session_state["_input_text"],
            height=130,
            placeholder="Ví dụ: This product is amazing! I love it so much...",
        )

    analyze_btn = st.button("🔍 Phân tích ngay", type="primary",
                            use_container_width=True, disabled=(rob_model is None))

    if analyze_btn:
        text_to_analyze = user_text.strip()
        if not text_to_analyze:
            st.warning("Vui lòng nhập văn bản trước khi phân tích.")
        else:
            with st.spinner("Đang chạy mô hình..."):
                rob  = predict_roberta(text_to_analyze, tokenizer, rob_model)
                vad  = predict_vader(text_to_analyze, vader_analyzer)

            st.markdown("---")
            st.markdown("### 📈 Kết quả phân tích")

            c1, c2 = st.columns(2)

            for col, result, model_name, extra in [
                (c1, rob, "🤖 RoBERTa (Transformer)",
                 f"Độ tin cậy: **{result['confidence']:.1%}**"),
                (c2, vad, "📊 VADER (Lexicon-based)",
                 f"Compound score: **{result['compound']:+.4f}**"),
            ] if False else []:
                pass  # placeholder — real loop below

            with c1:
                lbl = rob['label']
                st.markdown(f"#### 🤖 RoBERTa (Transformer)")
                st.markdown(f"""
<div class="result-box" style="border-left:5px solid {COLORS[lbl]}">
  <div style="font-size:3rem">{EMOJI[lbl]}</div>
  <div class="badge-{lbl}">{VI[lbl]}</div>
  <div style="margin-top:0.5rem;color:#555;font-size:0.9rem">
    Độ tin cậy: <strong>{rob['confidence']:.1%}</strong>
  </div>
</div>""", unsafe_allow_html=True)
                st.plotly_chart(
                    score_bars(rob['negative'], rob['neutral'], rob['positive'],
                               "Phân phối điểm RoBERTa"),
                    use_container_width=True
                )

            with c2:
                lbl = vad['label']
                st.markdown(f"#### 📊 VADER (Lexicon-based)")
                st.markdown(f"""
<div class="result-box" style="border-left:5px solid {COLORS[lbl]}">
  <div style="font-size:3rem">{EMOJI[lbl]}</div>
  <div class="badge-{lbl}">{VI[lbl]}</div>
  <div style="margin-top:0.5rem;color:#555;font-size:0.9rem">
    Compound: <strong>{vad['compound']:+.4f}</strong>
  </div>
</div>""", unsafe_allow_html=True)
                st.plotly_chart(
                    score_bars(vad['negative'], vad['neutral'], vad['positive'],
                               "Phân phối điểm VADER"),
                    use_container_width=True
                )

            # Radar comparison
            st.plotly_chart(radar_chart(rob, vad), use_container_width=True)

            # Agreement verdict
            if rob['label'] == vad['label']:
                st.success(
                    f"✅ Cả hai mô hình **đồng thuận**: văn bản mang cảm xúc "
                    f"**{VI[rob['label']]}** {EMOJI[rob['label']]}"
                )
            else:
                st.warning(
                    f"⚠️ Hai mô hình **không đồng thuận** — "
                    f"RoBERTa: **{VI[rob['label']]}** | VADER: **{VI[vad['label']]}**"
                )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Phân tích hàng loạt từ file CSV</div>',
                unsafe_allow_html=True)

    col_up, col_fmt = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader("Tải lên file CSV của bạn:", type=["csv"])

    with col_fmt:
        st.markdown("""
**📋 Định dạng yêu cầu:**
- Ít nhất 1 cột chứa văn bản
- Mã hóa UTF-8
""")
        demo_csv = pd.DataFrame({
            "Text": ["Great product!", "Terrible quality", "It is okay"],
            "Product": ["A", "B", "C"],
        }).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Tải file mẫu CSV", demo_csv,
                           "sample_input.csv", "text/csv")

    use_sample_data = st.checkbox(
        "Dùng dataset mẫu sẵn có (sample.csv)",
        value=(uploaded is None)
    )

    batch_df = None
    text_col = None

    if use_sample_data:
        batch_df = load_sample_csv()
        text_col = "Text"
        st.info(f"Đang dùng dataset mẫu: **{len(batch_df):,}** bình luận")
    elif uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            str_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
            if not str_cols:
                st.error("Không tìm thấy cột văn bản trong file.")
            else:
                # Tự động ưu tiên cột có tên gợi ý văn bản
                _text_hints = ['text', 'review', 'comment', 'content',
                                'body', 'message', 'summary', 'description']
                _default_col = next(
                    (c for c in str_cols
                     if c.lower() in _text_hints),
                    # fallback: cột string có độ dài trung bình lớn nhất
                    max(str_cols,
                        key=lambda c: batch_df[c].dropna().astype(str).str.len().mean())
                )
                text_col = st.selectbox(
                    "Chọn cột chứa văn bản:",
                    str_cols,
                    index=str_cols.index(_default_col),
                )
        except Exception as e:
            st.error(f"Lỗi đọc file: {e}")

    if batch_df is not None and text_col:
        st.markdown("**5 dòng đầu tiên:**")
        st.dataframe(batch_df[[text_col]].head(5), use_container_width=True)

        max_rows = min(500, len(batch_df))
        n = st.slider("Số dòng cần phân tích:", 10, max_rows, min(50, max_rows))

        run_btn = st.button("▶️ Bắt đầu phân tích", type="primary",
                            disabled=(rob_model is None))

        if run_btn:
            texts = batch_df[text_col].dropna().astype(str).head(n).tolist()
            rob_list, vad_list = [], []
            bar = st.progress(0, f"Đang xử lý 0/{len(texts)}...")
            for i, t in enumerate(texts):
                rob_list.append(predict_roberta(t, tokenizer, rob_model))
                vad_list.append(predict_vader(t, vader_analyzer))
                bar.progress((i + 1) / len(texts), f"Đang xử lý {i+1}/{len(texts)}...")
            bar.empty()

            res = batch_df[[text_col]].head(n).copy().reset_index(drop=True)
            res["RoBERTa"]      = [r["label"] for r in rob_list]
            res["Conf_RoBERTa"] = [f"{r['confidence']:.1%}" for r in rob_list]
            res["VADER"]        = [r["label"] for r in vad_list]
            res["Compound"]     = [f"{r['compound']:+.4f}" for r in vad_list]
            res["Đồng thuận"]   = ["✅" if a == b else "❌"
                                   for a, b in zip(res["RoBERTa"], res["VADER"])]

            st.success(f"Hoàn tất phân tích **{len(texts)}** bình luận!")

            styled = res.style.apply(
                highlight_sentiment, subset=["RoBERTa", "VADER"]
            )
            st.dataframe(styled, use_container_width=True, height=320)

            c_pie1, c_pie2 = st.columns(2)
            with c_pie1:
                counts = res["RoBERTa"].value_counts().to_dict()
                st.plotly_chart(
                    sentiment_distribution(counts, "Phân phối RoBERTa"),
                    use_container_width=True
                )
            with c_pie2:
                counts = res["VADER"].value_counts().to_dict()
                st.plotly_chart(
                    sentiment_distribution(counts, "Phân phối VADER"),
                    use_container_width=True
                )

            agree_rate = (res["Đồng thuận"] == "✅").mean()
            st.metric("Tỷ lệ đồng thuận giữa hai mô hình", f"{agree_rate:.1%}")

            csv_out = res.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Tải kết quả CSV", csv_out,
                               "sentiment_results.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Khám phá Dataset — Amazon Fine Food Reviews</div>',
                unsafe_allow_html=True)

    try:
        df = load_vader_csv()

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tổng số bình luận", f"{len(df):,}")
        k2.metric("Điểm sao trung bình", f"{df['Score'].mean():.2f} ⭐")
        k3.metric("Tích cực (RoBERTa)",
                  f"{(df['roberta_pred']=='positive').mean():.1%}")
        k4.metric("Tiêu cực (RoBERTa)",
                  f"{(df['roberta_pred']=='negative').mean():.1%}")

        st.divider()

        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            cnt = df["Score"].value_counts().sort_index()
            fig = px.bar(
                x=cnt.index, y=cnt.values,
                labels={"x": "Số sao ⭐", "y": "Số lượng bình luận"},
                title="Phân phối Đánh giá Sao",
                color=cnt.index.astype(str),
                color_discrete_sequence=["#dc3545", "#fd7e14",
                                          "#ffc107", "#20c997", "#28a745"],
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            cats    = ['negative', 'neutral', 'positive']
            cats_vi = ['Tiêu cực', 'Trung lập', 'Tích cực']
            rob_c   = df['roberta_pred'].value_counts()
            vad_c   = df['vader_pred'].value_counts()
            fig = go.Figure()
            for name, counts, color in [
                ('RoBERTa', rob_c, '#4472C4'),
                ('VADER',   vad_c, '#ED7D31'),
            ]:
                fig.add_trace(go.Bar(
                    name=name, x=cats_vi,
                    y=[counts.get(c, 0) for c in cats],
                    marker_color=color,
                ))
            fig.update_layout(
                barmode='group', title='Phân phối dự đoán của hai mô hình',
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Row 2
        c3, c4 = st.columns(2)
        with c3:
            avg = df.groupby('Score')[
                ['roberta_pos', 'roberta_neg', 'roberta_neu']
            ].mean()
            fig = go.Figure()
            for col, name, color in [
                ('roberta_pos', 'Tích cực', '#28a745'),
                ('roberta_neg', 'Tiêu cực', '#dc3545'),
                ('roberta_neu', 'Trung lập', '#ffc107'),
            ]:
                fig.add_trace(go.Scatter(
                    x=avg.index, y=avg[col],
                    name=name, mode='lines+markers',
                    line=dict(color=color, width=2),
                ))
            fig.update_layout(
                title='Điểm RoBERTa TB theo số sao',
                xaxis_title='Số sao', yaxis_title='Điểm',
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            avg_comp = df.groupby('Score')['vader_compound'].mean()
            fig = px.bar(
                x=avg_comp.index, y=avg_comp.values,
                labels={"x": "Số sao", "y": "Compound TB"},
                title='Điểm Compound VADER TB theo số sao',
                color=avg_comp.values,
                color_continuous_scale='RdYlGn',
                range_color=[-1, 1],
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter correlation
        st.markdown("#### Tương quan RoBERTa ↔ VADER")
        samp = df.sample(min(2000, len(df)), random_state=42)
        fig = px.scatter(
            samp, x='vader_compound', y='roberta_pos',
            color='Score', color_continuous_scale='RdYlGn',
            opacity=0.45,
            labels={
                'vader_compound': 'VADER Compound Score',
                'roberta_pos': 'RoBERTa Positive Score',
            },
            title=f'Tương quan giữa hai mô hình (n={len(samp):,} mẫu ngẫu nhiên)',
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Sample reviews
        st.markdown("#### Xem bình luận mẫu theo số sao")
        score_pick = st.select_slider(
            "Lọc theo số sao:", options=[1, 2, 3, 4, 5], value=(1, 5)
        )
        sub = df[(df['Score'] >= score_pick[0]) & (df['Score'] <= score_pick[1])]
        if len(sub) > 0:
            for _, row in sub.sample(min(4, len(sub)), random_state=7).iterrows():
                rl = str(row['roberta_pred'])
                vl = str(row['vader_pred'])
                txt = str(row['Text'])[:250]
                st.markdown(f"""
<div style="border-left:4px solid {COLORS.get(rl,'#999')};
  padding:0.8rem 1rem; margin:0.5rem 0;
  background:#f9f9f9; border-radius:0 8px 8px 0;">
  <strong>⭐ {int(row['Score'])} sao</strong> &nbsp;|&nbsp;
  RoBERTa: <span style="color:{COLORS.get(rl,'#333')};font-weight:600">
    {VI.get(rl, rl)}</span> &nbsp;|&nbsp;
  VADER: <span style="color:{COLORS.get(vl,'#333')};font-weight:600">
    {VI.get(vl, vl)}</span>
  <br><span style="color:#555"><em>"{txt}{"..." if len(str(row["Text"]))>250 else ""}"</em></span>
</div>""", unsafe_allow_html=True)

    except FileNotFoundError:
        st.error("Không tìm thấy `vader.csv`. Đảm bảo file nằm cùng thư mục với `app.py`.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Đánh giá & So sánh Mô hình</div>',
                unsafe_allow_html=True)

    try:
        df_e = load_vader_csv()
        y_true  = df_e['ground_truth'].tolist()
        y_rob   = df_e['roberta_pred'].tolist()
        y_vader = df_e['vader_pred'].tolist()

        rob_m   = compute_metrics(y_true, y_rob)
        vader_m = compute_metrics(y_true, y_vader)

        # KPI — RoBERTa row
        st.markdown("### Hiệu suất tổng thể")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Accuracy  · RoBERTa",  f"{rob_m.get('accuracy',0):.3f}",
                  f"{rob_m.get('accuracy',0)-vader_m.get('accuracy',0):+.3f} so với VADER")
        r2.metric("Precision · RoBERTa", f"{rob_m.get('precision',0):.3f}")
        r3.metric("Recall    · RoBERTa", f"{rob_m.get('recall',0):.3f}")
        r4.metric("F1-Score  · RoBERTa", f"{rob_m.get('f1_score',0):.3f}")

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Accuracy  · VADER",  f"{vader_m.get('accuracy',0):.3f}")
        v2.metric("Precision · VADER", f"{vader_m.get('precision',0):.3f}")
        v3.metric("Recall    · VADER", f"{vader_m.get('recall',0):.3f}")
        v4.metric("F1-Score  · VADER", f"{vader_m.get('f1_score',0):.3f}")

        # Bar chart
        st.plotly_chart(metric_bars(rob_m, vader_m), use_container_width=True)

        # Confusion matrices
        st.markdown("### Ma trận nhầm lẫn (Confusion Matrix)")
        cm1, cm2 = st.columns(2)
        with cm1:
            st.plotly_chart(
                cm_fig(y_true, y_rob, "RoBERTa (Transformer)"),
                use_container_width=True
            )
        with cm2:
            st.plotly_chart(
                cm_fig(y_true, y_vader, "VADER (Lexicon-based)"),
                use_container_width=True
            )

        # Model comparison table
        st.markdown("### So sánh đặc điểm hai mô hình")
        char_df = pd.DataFrame({
            "Đặc điểm": [
                "Loại mô hình", "Kiến trúc",
                "Dữ liệu huấn luyện", "Số tham số",
                "Tốc độ dự đoán", "Xử lý ngữ cảnh",
                "Xử lý từ không có trong từ điển", "Cần GPU",
            ],
            "RoBERTa": [
                "Deep Learning (Transformer)", "BERT-base (12 layers)",
                "~58M tweets", "~125 triệu tham số",
                "~0.05–0.2 giây/văn bản (CPU)", "✅ Xuất sắc",
                "✅ Tốt (subword tokenization)", "⚠️ Khuyến nghị",
            ],
            "VADER": [
                "Rule-based (Lexicon)", "Từ điển + Quy tắc ngữ pháp",
                "Social media corpus", "Không có (~7,500 từ)",
                "<1 ms/văn bản (rất nhanh)", "⚠️ Hạn chế",
                "❌ Bỏ qua từ lạ", "❌ Không cần",
            ],
        }).set_index("Đặc điểm")
        st.dataframe(char_df, use_container_width=True)

        st.info("""
**📌 Ghi chú về Ground Truth:** Nhãn thực tế được suy ra từ điểm sao Amazon:
- ⭐⭐ (1–2 sao) → **Tiêu cực**
- ⭐⭐⭐ (3 sao) → **Trung lập**
- ⭐⭐⭐⭐⭐ (4–5 sao) → **Tích cực**

Đây là xấp xỉ hợp lý nhưng không hoàn hảo — một số bình luận 5 sao vẫn có thể chứa
nội dung tiêu cực, và ngược lại.
""")

    except FileNotFoundError:
        st.error("Không tìm thấy `vader.csv`.")
    except ImportError:
        st.error("Thiếu `scikit-learn`. Chạy: `pip install scikit-learn`")
