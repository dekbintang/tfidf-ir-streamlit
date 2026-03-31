import streamlit as st
import pandas as pd
from index_builder import bangun_indeks
from preprocessing import preprocess
from tfidf import hitung_tf, hitung_tfidf, cosine_similarity
from stopwords import STOPWORDS

st.set_page_config(page_title="TF-IDF IR", page_icon="📚", layout="wide")
st.title("📚 Sistem TF-IDF — Information Retrieval")
st.caption("Tokenisasi → Stopword Removal → Stemming → TF → IDF → TF-IDF → Query")

docs, idf = bangun_indeks()

# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.header("Menu")
    menu = st.radio("Pilih halaman:", [
        "📄 Dokumen",
        "⚙️ Preprocessing",
        "📊 TF",
        "📈 IDF",
        "🔢 TF-IDF",
        "🔍 Query",
    ])
    st.divider()
    st.caption(f"Dokumen   : {len(docs)}")
    st.caption(f"Vocabulary : {len(idf)} term")
    st.caption(f"Stopword  : {len(STOPWORDS)} kata")


# ══════════════════════════════════════════════════════
#  1. DOKUMEN
# ══════════════════════════════════════════════════════
if menu == "📄 Dokumen":
    st.header("Koleksi Dokumen")
    st.write(f"Total **{len(docs)} dokumen** Bahasa Indonesia.")
    st.divider()

    for doc in docs:
        with st.expander(f"{doc['id']} — {doc['judul']}"):
            st.write(doc["teks"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Jumlah Token", len(doc["tokens"]))
            col2.metric("Setelah Stopword", len(doc["nostop"]))
            col3.metric("Term Unik (stem)", len(set(doc["stems"])))


# ══════════════════════════════════════════════════════
#  2. PREPROCESSING
# ══════════════════════════════════════════════════════
elif menu == "⚙️ Preprocessing":
    st.header("Preprocessing")
    st.write("Tiga tahap: **Tokenisasi → Hapus Stopword → Stemming**")
    st.divider()

    pilih = st.selectbox("Pilih dokumen:", [f"{d['id']} — {d['judul']}" for d in docs])
    doc   = next(d for d in docs if d["id"] == pilih.split(" — ")[0])

    st.subheader("Teks Asli")
    st.info(doc["teks"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"1. Tokenisasi ({len(doc['tokens'])} token)")
        st.caption("Huruf kecil, hapus tanda baca, minimal 3 karakter")
        st.write(doc["tokens"])

    with col2:
        dihapus = len(doc["tokens"]) - len(doc["nostop"])
        st.subheader(f"2. Hapus Stopword ({len(doc['nostop'])} kata)")
        st.caption(f"{dihapus} stopword dihapus")
        st.write(doc["nostop"])

    with col3:
        st.subheader(f"3. Stemming ({len(doc['stems'])} stem)")
        st.caption("Imbuhan dipotong ke bentuk dasar menggunakan PySastrawi")
        st.write(doc["stems"])

    st.divider()
    st.subheader("Perbandingan Sebelum & Sesudah Stemming")
    pasangan      = list(zip(doc["nostop"], doc["stems"]))
    berubah       = [(a, b) for a, b in pasangan if a != b]
    tidak_berubah = [(a, b) for a, b in pasangan if a == b]

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Kata berubah ({len(berubah)}):**")
        for asli, hasil in berubah:
            st.write(f"`{asli}` → **`{hasil}`**")
    with col2:
        st.write(f"**Tidak berubah ({len(tidak_berubah)}):**")
        for asli, _ in tidak_berubah:
            st.write(f"`{asli}`")


# ══════════════════════════════════════════════════════
#  3. TF — TERM FREQUENCY
# ══════════════════════════════════════════════════════
elif menu == "📊 TF":
    st.header("Term Frequency (TF)")
    st.latex(r"TF(t,d) = \frac{\text{jumlah kemunculan } t \text{ dalam } d}{\text{total kata dalam } d}")
    st.divider()

    pilih = st.selectbox("Pilih dokumen:", [f"{d['id']} — {d['judul']}" for d in docs])
    doc   = next(d for d in docs if d["id"] == pilih.split(" — ")[0])

    st.subheader(f"Nilai TF — {doc['id']}: {doc['judul']}")
    st.caption(f"Total kata (setelah stemming): {len(doc['stems'])} | Term unik: {len(doc['tf'])}")

    tf_sorted = sorted(doc["tf"].items(), key=lambda x: x[1], reverse=True)
    total     = len(doc["stems"])

    rows = []
    for term, val in tf_sorted:
        jumlah = doc["stems"].count(term)
        rows.append({
            "Term (stem)": term,
            "Muncul":      jumlah,
            "TF":          round(val, 4),
            "Perhitungan": f"{jumlah} / {total} = {val:.4f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
#  4. IDF — INVERSE DOCUMENT FREQUENCY
# ══════════════════════════════════════════════════════
elif menu == "📈 IDF":
    st.header("Inverse Document Frequency (IDF)")
    st.latex(r"IDF(t) = \log_{10}\left(\frac{N}{DF(t)}\right)")
    st.caption(f"N = {len(docs)} dokumen")
    st.divider()

    # Hitung Document Frequency tiap term
    df_count = {}
    for doc in docs:
        for term in set(doc["stems"]):
            df_count[term] = df_count.get(term, 0) + 1

    idf_sorted = sorted(idf.items(), key=lambda x: x[1], reverse=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("IDF Tertinggi (paling spesifik)")
        for term, val in idf_sorted[:15]:
            df = df_count.get(term, 1)
            st.write(f"`{term}` — DF={df} → log({len(docs)}/{df}) = **{val:.4f}**")
    with col2:
        st.subheader("IDF Terendah (paling umum)")
        for term, val in list(reversed(idf_sorted))[:15]:
            df = df_count.get(term, 1)
            st.write(f"`{term}` — DF={df} → log({len(docs)}/{df}) = **{val:.4f}**")

    st.divider()
    cari = st.text_input("Cari nilai IDF satu kata (bentuk stem):")
    if cari.strip():
        kata = cari.strip().lower()
        if kata in idf:
            df = df_count[kata]
            st.success(f"IDF(`{kata}`) = log({len(docs)}/{df}) = **{idf[kata]:.4f}**")
            st.write("Ditemukan di dokumen:")
            for doc in docs:
                if kata in doc["stems"]:
                    st.write(f"- {doc['id']}: {doc['judul']}")
        else:
            st.warning(f"Kata `{kata}` tidak ada dalam vocabulary.")


# ══════════════════════════════════════════════════════
#  5. TF-IDF
# ══════════════════════════════════════════════════════
elif menu == "🔢 TF-IDF":
    st.header("TF-IDF")
    st.latex(r"TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)")
    st.divider()

    pilih = st.selectbox("Pilih dokumen:", [f"{d['id']} — {d['judul']}" for d in docs])
    doc   = next(d for d in docs if d["id"] == pilih.split(" — ")[0])

    tfidf_sorted = sorted(doc["tfidf"].items(), key=lambda x: x[1], reverse=True)
    st.subheader(f"Bobot TF-IDF — {doc['id']}: {doc['judul']}")

    rows = []
    for term, val in tfidf_sorted:
        tf_val  = doc["tf"].get(term, 0)
        idf_val = idf.get(term, 0)
        rows.append({
            "Term (stem)": term,
            "TF":          round(tf_val, 4),
            "IDF":         round(idf_val, 4),
            "TF-IDF":      round(val, 6),
            "Perhitungan": f"{tf_val:.4f} × {idf_val:.4f} = {val:.6f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════
#  6. QUERY / PENCARIAN
# ══════════════════════════════════════════════════════
elif menu == "🔍 Query":
    st.header("Query / Pencarian Dokumen")
    st.write("Masukkan kata kunci. Query diproses dengan pipeline yang sama, lalu dihitung Cosine Similarity dengan setiap dokumen.")
    st.divider()

    query = st.text_input("Masukkan query:", "kesehatan tubuh olahraga")

    if query.strip():
        hasil_pre = preprocess(query)
        q_stems   = hasil_pre["stems"]

        with st.expander("Lihat proses preprocessing query"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Tokenisasi:** {hasil_pre['tokens']}")
            col2.write(f"**Setelah Stopword:** {hasil_pre['nostop']}")
            col3.write(f"**Setelah Stemming:** {q_stems}")

        if not q_stems:
            st.warning("Query tidak menghasilkan kata yang bisa dicari. Coba kata yang lebih spesifik.")
            st.stop()

        q_tf    = hitung_tf(q_stems)
        q_tfidf = hitung_tfidf(q_tf, idf)

        # Hitung Cosine Similarity untuk semua dokumen
        hasil = []
        for doc in docs:
            sim        = cosine_similarity(q_tfidf, doc["tfidf"])
            kata_cocok = [t for t in q_stems if t in doc["stems"]]
            hasil.append((doc, sim, kata_cocok))
        hasil.sort(key=lambda x: x[1], reverse=True)

        n_relevan = sum(1 for _, sim, _ in hasil if sim > 0)
        st.subheader(f"Hasil: {n_relevan} dokumen relevan ditemukan")

        for rank, (doc, sim, kata_cocok) in enumerate(hasil, 1):
            if sim == 0:
                continue
            label = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
            col1, col2 = st.columns([6, 1])
            with col1:
                st.write(f"{label} **{doc['id']} — {doc['judul']}**")
                st.caption(f"Kata cocok: {', '.join(kata_cocok) if kata_cocok else '-'}")
                st.write(doc["teks"])
            with col2:
                st.metric("Skor", f"{sim:.4f}")
            st.divider()