import sys
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import re
import seaborn as sns
import streamlit as st

try:
    from streamlit import runtime
    STREAMLIT_RUNNING = runtime.exists()
except Exception:
    STREAMLIT_RUNNING = False

if __name__ == "__main__" and not STREAMLIT_RUNNING:
    sys.exit(0)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "src"))

from pipeline import (
    limpiar_datos,
    entrenar_modelo,
    obtener_insights,
    cargar_modelo_entrenado,
    RUTA_MODELO,
)

st.set_page_config(page_title="Panel de Ventas", layout="wide")
st.title("📊 Panel integral de ventas")
st.caption("Carga tus datos, explora métricas y genera predicciones.")


@st.cache_data
def get_template_excel() -> bytes:
    cols = ["Fecha", "Region", "Categoria", "Subcategoria", "Cantidad", "Descuento", "Ventas", "Beneficio"]
    ejemplos = [
        {"Fecha": "2023-01-01", "Region": "Centro", "Categoria": "Oficina", "Subcategoria": "Sillas", "Cantidad": 2, "Descuento": 0.10, "Ventas": 250.0, "Beneficio": 50.0},
        {"Fecha": "2023-01-15", "Region": "Norte", "Categoria": "Tecnología", "Subcategoria": "Teléfonos", "Cantidad": 1, "Descuento": 0.05, "Ventas": 800.0, "Beneficio": 200.0},
        {"Fecha": "2023-02-03", "Region": "Sur", "Categoria": "Muebles", "Subcategoria": "Mesas", "Cantidad": 3, "Descuento": 0.15, "Ventas": 450.0, "Beneficio": 90.0},
        {"Fecha": "2023-02-10", "Region": "Este", "Categoria": "Oficina", "Subcategoria": "Escritorios", "Cantidad": 1, "Descuento": 0.0, "Ventas": 600.0, "Beneficio": 150.0},
        {"Fecha": "2023-03-05", "Region": "Oeste", "Categoria": "Tecnología", "Subcategoria": "Laptops", "Cantidad": 2, "Descuento": 0.20, "Ventas": 1500.0, "Beneficio": 300.0},
        {"Fecha": "2023-03-20", "Region": "Centro", "Categoria": "Muebles", "Subcategoria": "Sillas", "Cantidad": 5, "Descuento": 0.10, "Ventas": 625.0, "Beneficio": 125.0},
        {"Fecha": "2023-04-08", "Region": "Norte", "Categoria": "Oficina", "Subcategoria": "Archivadores", "Cantidad": 2, "Descuento": 0.05, "Ventas": 300.0, "Beneficio": 75.0},
        {"Fecha": "2023-04-22", "Region": "Sur", "Categoria": "Tecnología", "Subcategoria": "Monitores", "Cantidad": 3, "Descuento": 0.12, "Ventas": 900.0, "Beneficio": 180.0},
        {"Fecha": "2023-05-10", "Region": "Este", "Categoria": "Muebles", "Subcategoria": "Estantes", "Cantidad": 4, "Descuento": 0.08, "Ventas": 520.0, "Beneficio": 104.0},
        {"Fecha": "2023-05-25", "Region": "Oeste", "Categoria": "Oficina", "Subcategoria": "Sillas", "Cantidad": 6, "Descuento": 0.15, "Ventas": 750.0, "Beneficio": 150.0},
        {"Fecha": "2023-06-12", "Region": "Centro", "Categoria": "Tecnología", "Subcategoria": "Teclados", "Cantidad": 10, "Descuento": 0.10, "Ventas": 450.0, "Beneficio": 90.0},
        {"Fecha": "2023-06-28", "Region": "Norte", "Categoria": "Muebles", "Subcategoria": "Mesas", "Cantidad": 2, "Descuento": 0.05, "Ventas": 380.0, "Beneficio": 76.0},
        {"Fecha": "2023-07-15", "Region": "Sur", "Categoria": "Oficina", "Subcategoria": "Escritorios", "Cantidad": 1, "Descuento": 0.0, "Ventas": 550.0, "Beneficio": 137.5},
        {"Fecha": "2023-08-03", "Region": "Este", "Categoria": "Tecnología", "Subcategoria": "Mouse", "Cantidad": 15, "Descuento": 0.20, "Ventas": 225.0, "Beneficio": 45.0},
        {"Fecha": "2023-08-20", "Region": "Oeste", "Categoria": "Muebles", "Subcategoria": "Sillas", "Cantidad": 8, "Descuento": 0.10, "Ventas": 1000.0, "Beneficio": 200.0},
    ]
    df = pd.DataFrame(ejemplos, columns=cols)
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Ventas')
    buffer.seek(0)
    return buffer.getvalue()


st.markdown("### 📥 Paso 1: Descarga la plantilla")
st.download_button(
    label="📊 Descargar plantilla Excel",
    data=get_template_excel(),
    file_name="ventas_plantilla.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Plantilla con 15 ejemplos de datos"
)

st.markdown("### 📤 Paso 2: Sube tu archivo de ventas")
st.info("👆 Descarga la plantilla, llénala con tus datos y súbela aquí.")

archivo_subido = st.file_uploader(
    "Sube tu archivo CSV o Excel con datos de ventas",
    type=["csv", "xlsx", "xls"],
    help="Debe contener: Fecha, Region, Categoria, Subcategoria, Cantidad, Descuento, Ventas, Beneficio"
)

expected_columns = ["Fecha", "Region", "Categoria", "Subcategoria", "Cantidad", "Descuento", "Ventas", "Beneficio"]
df = None

if archivo_subido:
    def normalize(name: str) -> str:
        name = str(name).strip()
        name = unicodedata.normalize("NFKD", name)
        name = "".join(ch for ch in name if not unicodedata.combining(ch))
        name = name.lower().strip()
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
        return name

    canonical = {
        "fecha": "Fecha", "order_date": "Fecha", "orderdate": "Fecha", "date": "Fecha",
        "region": "Region",
        "categoria": "Categoria", "category": "Categoria",
        "subcategoria": "Subcategoria", "sub_category": "Subcategoria", "subcategory": "Subcategoria",
        "cantidad": "Cantidad", "quantity": "Cantidad",
        "descuento": "Descuento", "discount": "Descuento",
        "ventas": "Ventas", "sales": "Ventas", "venta": "Ventas",
        "beneficio": "Beneficio", "profit": "Beneficio", "beneficios": "Beneficio",
    }

    nombre_archivo = archivo_subido.name.lower()
    es_excel = nombre_archivo.endswith(('.xlsx', '.xls'))

    try:
        if es_excel:
            archivo_subido.seek(0)
            df_raw = pd.read_excel(archivo_subido, engine='openpyxl')
        else:
            delims = [",", ";", "\t"]
            sep = ","
            for d in delims:
                try:
                    archivo_subido.seek(0)
                    preview = pd.read_csv(archivo_subido, sep=d, nrows=5, encoding="utf-8-sig")
                    sep = d
                    break
                except Exception:
                    continue

            archivo_subido.seek(0)
            df_raw = pd.read_csv(archivo_subido, sep=sep, encoding="utf-8-sig")

        rename_map = {}
        for col in df_raw.columns:
            n = normalize(col)
            if n in canonical:
                rename_map[col] = canonical[n]
        df_raw = df_raw.rename(columns=rename_map)

        faltantes = [c for c in expected_columns if c not in df_raw.columns]

        if faltantes:
            st.error(f"❌ Faltan columnas: **{', '.join(faltantes)}**")
            st.write("📋 Columnas encontradas:", list(df_raw.columns))
            st.dataframe(df_raw.head(), use_container_width=True)
            st.warning("Descarga la plantilla y úsala como referencia.")
            df = None
        else:
            with st.spinner("🔄 Procesando datos..."):
                df = limpiar_datos(df_raw)
            st.success(f"✅ {len(df)} registros cargados correctamente")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.warning("Verifica el formato del archivo.")
        df = None
else:
    st.warning("⚠️ Sube un archivo para comenzar el análisis.")
    st.stop()

if df is None or df.empty:
    st.error("No hay datos válidos.")
    st.stop()

st.markdown("---")
st.markdown("### 📊 Paso 3: Análisis de datos")

insights = obtener_insights(df)

col1, col2, col3 = st.columns(3)
col1.metric("Ventas totales", f"${insights['ventas_totales']:,.0f}")
col2.metric("Beneficio total", f"${insights['beneficio_total']:,.0f}")
col3.metric("Filas analizadas", len(df))

tab_graficos, tab_modelo = st.tabs(["Visualizaciones", "Modelo predictivo"])

with tab_graficos:
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=insights["ventas_por_region"].values, y=insights["ventas_por_region"].index, ax=ax, palette="Blues_r")
        ax.set_title("Ventas por región")
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=insights["ventas_por_categoria"].values, y=insights["ventas_por_categoria"].index, ax=ax, palette="Greens_r")
        ax.set_title("Ventas por categoría")
        st.pyplot(fig)

    col_c, col_d = st.columns(2)
    col_c.write("Top 5 subcategorías rentables")
    col_c.dataframe(insights["top_productos"].reset_index().rename(columns={"Beneficio": "Beneficio acumulado"}), use_container_width=True)
    col_d.write("Bottom 5 subcategorías rentables")
    col_d.dataframe(insights["bottom_productos"].reset_index().rename(columns={"Beneficio": "Beneficio acumulado"}), use_container_width=True)

with tab_modelo:
    if "metricas_modelo" not in st.session_state:
        st.session_state.metricas_modelo = None

    n_samples = len(df)
    if n_samples < 10:
        st.warning(f"⚠️ **Dataset muy pequeño**: {n_samples} registros. Se necesitan al menos 10.")
    else:
        if n_samples < 20:
            st.info(f"ℹ️ Tienes {n_samples} registros. Se recomienda 50-100 para mejores resultados.")

        if st.button("Entrenar / actualizar modelo"):
            try:
                with st.spinner("🔄 Entrenando modelo..."):
                    resultado = entrenar_modelo(df)
                    st.session_state.metricas_modelo = {
                        "r2": resultado["r2"],
                        "rmse": resultado["rmse"],
                        "model": resultado.get("model_name", "Desconocido"),
                        "n_samples": resultado.get("n_samples", n_samples),
                        "n_train": resultado.get("n_train", 0),
                        "n_test": resultado.get("n_test", 0),
                        "used_cv": resultado.get("used_cv", False),
                        "n_splits": resultado.get("n_splits", None),
                    }

                    msg = f"✅ Modelo: **{st.session_state.metricas_modelo['model']}**\n\n"
                    msg += f"- Entrenamiento: {resultado['n_train']} registros\n"
                    msg += f"- Prueba: {resultado['n_test']} registros\n"
                    if resultado.get('used_cv'):
                        msg += f"- Validación cruzada: {resultado['n_splits']} splits\n"
                    else:
                        msg += f"- Sin validación cruzada (dataset pequeño)\n"

                    st.success(msg)
            except ValueError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    modelo = cargar_modelo_entrenado()
    if modelo is None:
        st.info("💡 Entrena un modelo usando el botón de arriba.")
    else:
        metricas = st.session_state.metricas_modelo
        if metricas:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Modelo", metricas.get('model', 'Desconocido'))
            col_m2.metric("R² Score", f"{metricas['r2']:.3f}")
            col_m3.metric("RMSE", f"${metricas['rmse']:.2f}")

            if metricas.get('used_cv'):
                st.caption(f"✅ Validación cruzada: {metricas.get('n_splits', 5)} splits")
            else:
                st.caption(f"⚠️ Entrenamiento simple ({metricas.get('n_samples', 0)} muestras)")

        with st.form("form_prediccion"):
            categoria = st.selectbox("Categoría", sorted(df["Categoria"].unique()))
            region = st.selectbox("Región", sorted(df["Region"].unique()))
            descuento = st.slider("Descuento", 0.0, 0.9, 0.1, 0.01)
            cantidad = st.slider("Cantidad", 1, 20, 1)
            año = st.number_input("Año", min_value=2020, max_value=2030, value=2024)
            mes = st.slider("Mes", 1, 12, 6)

            enviado = st.form_submit_button("Predecir ventas")
            if enviado:
                entrada = pd.DataFrame([{
                    "Descuento": descuento,
                    "Cantidad": cantidad,
                    "Categoria": categoria,
                    "Region": region,
                    "Año": año,
                    "Mes": mes,
                }])
                prediccion = modelo.predict(entrada)[0]
                st.success(f"Venta estimada: ${prediccion:,.2f}")

