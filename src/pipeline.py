import joblib
import pandas as pd
import unicodedata
import re
from pathlib import Path
from io import IOBase
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

RUTA_BASE = Path(__file__).resolve().parents[1]
RUTA_DATOS = RUTA_BASE / "data" / "ventas.csv"
RUTA_MODELO = RUTA_BASE / "models" / "modelo_regresion.pkl"


def cargar_datos(ruta: str | Path | IOBase = RUTA_DATOS) -> pd.DataFrame:
    def _peek_first_line(buff_or_path) -> str | None:
        try:
            if isinstance(buff_or_path, (str, Path)):
                with open(buff_or_path, "rb") as f:
                    data = f.read(256)
            else:
                pos = None
                try: pos = buff_or_path.tell()
                except Exception: pos = None
                data = buff_or_path.read(256)
                try: buff_or_path.seek(pos or 0)
                except Exception: pass
            if not data:
                return None
            for enc in ("utf-8-sig", "cp1252", "latin-1"):
                try: return data.decode(enc, errors="ignore").splitlines()[0]
                except Exception: continue
        except Exception:
            return None
        return None

    def _normalize(name: str) -> str:
        s = str(name)
        s = s.replace("\ufeff", "").replace("ï»¿", "")
        s = s.strip().strip('"\'')
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        s = re.sub(r"[^0-9a-zA-Z]+", "_", s.lower()).strip("_")
        s = re.sub(r"_+", "_", s)
        return s

    canonical = {
        "fecha": "Fecha", "order_date": "Fecha", "orderdate": "Fecha", "date": "Fecha",
        "region": "Region",
        "categoria": "Categoria", "category": "Categoria",
        "subcategoria": "Subcategoria", "sub_category": "Subcategoria", "subcategory": "Subcategoria",
        "cantidad": "Cantidad", "quantity": "Cantidad",
        "descuento": "Descuento", "discount": "Descuento",
        "ventas": "Ventas", "sales": "Ventas", "venta": "Ventas",
        "beneficio": "Beneficio", "profit": "Beneficio",
    }

    expected = {"Fecha", "Region", "Categoria", "Subcategoria", "Cantidad", "Descuento", "Ventas", "Beneficio"}

    def _rename(df: pd.DataFrame) -> pd.DataFrame:
        rename = {}
        for c in df.columns:
            n = _normalize(c)
            if n in canonical:
                rename[c] = canonical[n]
        if rename:
            df = df.rename(columns=rename)
        return df

    if hasattr(ruta, "seek"):
        try: ruta.seek(0)
        except Exception: pass

    # Excel directo
    if isinstance(ruta, (str, Path)) and str(ruta).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(ruta, engine="openpyxl")
        df = _rename(df)
    else:
        magic = _peek_first_line(ruta)
        magic_sep, skiprows = (None, 0)
        if magic:
            m = re.match(r"^\s*sep\s*=\s*(.)\s*$", magic.strip(), flags=re.IGNORECASE)
            if m:
                magic_sep, skiprows = m.group(1), 1
        encodings = ["utf-8-sig", "cp1252", "latin-1"]
        seps = [magic_sep] if magic_sep else []
        seps += [",", ";", "\t", "|"]
        df = None
        for sep in seps:
            if sep is None:
                continue
            for enc in encodings:
                try:
                    if hasattr(ruta, "seek"):
                        try: ruta.seek(0)
                        except Exception: pass
                    kw = {"sep": sep, "encoding": enc, "skiprows": skiprows}
                    if sep == ";": kw["decimal"] = ","
                    tmp = pd.read_csv(ruta, **kw)
                    tmp = _rename(tmp)
                    # Reintento si quedó una sola columna con separadores en el header
                    if tmp.shape[1] == 1 and any(s in str(tmp.columns[0]) for s in [";", "\t", "|", ","]):
                        if hasattr(ruta, "seek"):
                            try: ruta.seek(0)
                            except Exception: pass
                        kw2 = {"sep": sep, "encoding": enc, "engine": "python", "skiprows": skiprows}
                        if sep == ";": kw2["decimal"] = ","
                        tmp = pd.read_csv(ruta, **kw2)
                        tmp = _rename(tmp)
                    if expected.issubset(set(tmp.columns)):
                        df = tmp
                        break
                except Exception:
                    continue
            if df is not None:
                break
        if df is None:
            # Sniffer
            try:
                if hasattr(ruta, "seek"):
                    try: ruta.seek(0)
                    except Exception: pass
                tmp = pd.read_csv(ruta, sep=None, engine="python", encoding="utf-8-sig", skiprows=skiprows)
                df = _rename(tmp)
            except Exception:
                pass

    if df is None:
        raise ValueError("No se pudo leer el archivo. Prueba CSV con ',', ';', tab o '|' o un Excel.")
    if df.empty:
        raise ValueError("El archivo está vacío.")

    faltantes = expected.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}. Detectadas: {list(df.columns)}")

    return df


def normalizar_fecha(df: pd.DataFrame, col: str = "Fecha") -> pd.DataFrame:
    if col not in df.columns or pd.api.types.is_datetime64_any_dtype(df[col]):
        return df
    formatos = ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for f in formatos:
        try:
            df[col] = pd.to_datetime(df[col], format=f, errors="raise")
            return df
        except Exception:
            continue
    try:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    except Exception:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    df = normalizar_fecha(df, "Fecha")
    for c in ["Beneficio", "Descuento", "Cantidad", "Ventas"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Beneficio"] = df["Beneficio"].fillna(df["Beneficio"].median())
    df["Descuento"] = df["Descuento"].clip(0, 0.9)
    df = df.dropna()
    return df


def preparar_datos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.assign(Año=df["Fecha"].dt.year, Mes=df["Fecha"].dt.month)
    X = df[["Descuento", "Cantidad", "Categoria", "Region", "Año", "Mes"]]
    y = df["Ventas"]
    return X, y


def entrenar_modelo(df: pd.DataFrame, ruta_modelo: Path = RUTA_MODELO) -> dict:
    X, y = preparar_datos(df)
    if len(X) < 10:
        raise ValueError("Se requieren al menos 10 registros para entrenar.")

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["Categoria", "Region"]),
        ("num", "passthrough", ["Descuento", "Cantidad", "Año", "Mes"]),
    ])

    pipe = Pipeline([("prep", pre), ("lr", LinearRegression())])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    rmse = float(mean_squared_error(y_te, y_pred, squared=False))

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, ruta_modelo)

    return {"modelo": pipe, "model_name": "LinearRegression", "r2": float(r2), "rmse": rmse,
            "n_train": len(X_tr), "n_test": len(X_te), "n_samples": len(X)}


def obtener_insights(df: pd.DataFrame) -> dict:
    return {
        "ventas_totales": float(df["Ventas"].sum()),
        "beneficio_total": float(df["Beneficio"].sum()),
        "ventas_por_region": df.groupby("Region")["Ventas"].sum().sort_values(),
        "ventas_por_categoria": df.groupby("Categoria")["Ventas"].sum().sort_values(),
        "top_productos": df.groupby("Subcategoria")["Beneficio"].sum().sort_values(ascending=False).head(5),
        "bottom_productos": df.groupby("Subcategoria")["Beneficio"].sum().sort_values().head(5),
    }


def cargar_modelo_entrenado(ruta_modelo: Path = RUTA_MODELO):
    return joblib.load(ruta_modelo) if ruta_modelo.exists() else None


if __name__ == "__main__":
    if not RUTA_DATOS.exists():
        print(f"No se encontró {RUTA_DATOS}. Ejecuta: streamlit run app/app.py")
    else:
        df_datos = limpiar_datos(cargar_datos())
        res = entrenar_modelo(df_datos)
        print(f"Modelo: {res['model_name']} | R²: {res['r2']:.3f} | RMSE: {res['rmse']:.2f}")
