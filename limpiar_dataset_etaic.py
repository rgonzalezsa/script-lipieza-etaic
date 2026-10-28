#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Limpieza, Validación Estricta, Detección de Datos No Fiables y Auditoría.
Dataset: «Encuesta sobre vida universitaria 2026» (ETAIC 2026-27).

Principios Arquitectónicos y Estándares:
  - Deep Modules: API pública mínima y concisa (`DatasetCleaner.clean`),
    implementación interna rica y robusta.
  - Locality & Code Judo: contratos inmutables, vectorización en Pandas/NumPy,
    eliminación de estado global mutable y bucles celda a celda.
  - Clasificación en dos niveles:
      * Nivel de campo: VALIDO, AUSENTE, INVALIDO, SOSPECHOSO, NO_APLICABLE.
      * Nivel de registro: VALIDO, INCOMPLETO, EN_REVISION, NO_FIABLE, DUPLICADO.
  - Política de exclusión objetiva: infracciones críticas provocan el estado
    NO_FIABLE y la exclusión de los datos admitidos para el análisis principal,
    sin eliminar físicamente el registro original de la auditoría.
  - Gestión de preguntas condicionales y obligatorias respetando la semántica
    de la encuesta (no toda celda vacía indica ausencia punible o falsedad).
"""

from __future__ import annotations

import argparse
from enum import Enum
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import pandas as pd


# ============================================================================
# ENUMERACIONES Y ESTADOS FORMALES
# ============================================================================

class FieldStatus(str, Enum):
    """Estado de validación de un campo o celda individual."""
    VALIDO = "VALIDO"
    AUSENTE = "AUSENTE"
    INVALIDO = "INVALIDO"
    SOSPECHOSO = "SOSPECHOSO"
    NO_APLICABLE = "NO_APLICABLE"


class RecordStatus(str, Enum):
    """
    Estado global de fiabilidad y completitud de un registro/encuesta.
    Prioridad determinista: DUPLICADO > NO_FIABLE > EN_REVISION > INCOMPLETO > VALIDO.
    """
    VALIDO = "VALIDO"
    INCOMPLETO = "INCOMPLETO"
    EN_REVISION = "EN_REVISION"
    NO_FIABLE = "NO_FIABLE"
    DUPLICADO = "DUPLICADO"


class Severity(str, Enum):
    """Nivel de severidad de una infracción o regla de validación."""
    CRITICA = "CRITICA"          # Imposibilidad física/lógica -> Exclusión (NO_FIABLE)
    ADVERTENCIA = "ADVERTENCIA"  # Anomalía o incoherencia sospechosa -> EN_REVISION
    INFORMATIVA = "INFORMATIVA"  # Inconsistencia plausible o ausencia de campo


# ============================================================================
# CONTRATOS DE DATOS INMUTABLES
# ============================================================================

@dataclass(frozen=True)
class ValidationRule:
    """Definición canónica de una regla de validación o coherencia."""
    regla_id: str
    nombre: str
    variables: tuple[str, ...]
    severidad: Severity
    condicion: str
    descripcion: str
    accion: str
    excluye_registro: bool = False


@dataclass(frozen=True)
class AuditEntry:
    """Registro inmutable de una transformación o anulación sobre una celda."""
    id: str
    variable: str
    valor_original: Any
    valor_nuevo: Any
    regla: str
    severidad: str = "INFORMATIVA"


@dataclass(frozen=True)
class InconsistencyAlert:
    """Registro estructurado de una alerta inter-variable detectada."""
    regla_id: str
    comprobacion: str
    severidad: str
    n_casos: int
    ids: str
    afecta_estado_global: bool
    excluye_analisis_principal: bool


@dataclass(frozen=True)
class RecordClassification:
    """Clasificación global e inventario de anomalías para un registro individual."""
    id: str
    estado_global: RecordStatus
    motivos: str
    n_invalidos: int
    n_sospechosos: int
    n_ausentes: int
    n_no_aplicables: int
    excluido: bool
    recomendacion: str


@dataclass(frozen=True)
class CleaningConfig:
    """
    Configuración inmutable de límites, rangos y reglas de calidad.
    """
    recuperar_nota_x10: bool = False
    excluir_en_revision: bool = False
    fecha_min: pd.Timestamp = pd.Timestamp("2026-10-05")
    fecha_max: pd.Timestamp = pd.Timestamp("2026-10-20")

    # Columnas obligatorias generales que todo estudiante debe responder
    columnas_obligatorias: tuple[str, ...] = (
        "id", "fecha_respuesta", "titulacion", "curso", "edad",
        "genero", "vivienda", "beca", "trabaja_horas", "desplaz_min",
        "asistencia_pct", "nota_media", "mov_modo"
    )

    # Valores canónicos admitidos para variables categóricas
    validos: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "titulacion": ("INF", "ADET"),
            "genero": ("Hombre", "Mujer", "Otro / No responde"),
            "vivienda": ("Casa familiar", "Piso compartido", "Residencia", "Otra"),
            "beca": ("Sí", "No"),
            "sue_hora_acostarse": (
                "Antes de las 23:00", "23:00-00:59", "01:00-02:59", "A partir de las 03:00"
            ),
            "sue_siesta": ("Sí", "No"),
            "tra_tipo": (
                "Hostelería", "Comercio", "Clases particulares",
                "Informática / tecnología", "Otro"
            ),
            "tra_relacion_carrera": ("Sí", "Parcial", "No"),
            "tra_turno": ("Mañana", "Tarde", "Noche", "Fines de semana", "Variable"),
            "tra_motivo": ("Económico", "Experiencia", "Ambos"),
            "ia_herramienta": ("ChatGPT", "Copilot", "Gemini", "Claude", "Otra", "Ninguna"),
            "ia_tarea": (
                "Programar", "Redactar", "Resumir", "Estudiar / explicar", "Traducir", "Ninguna"
            ),
            "ia_declara_uso": ("Siempre", "A veces", "Nunca", "No aplica"),
            "ia_conoce_politica": ("Sí", "No"),
            "dig_movil_clase": ("Nunca", "A veces", "Casi siempre"),
            "dig_notificaciones": ("Ninguna", "Algunas", "Todas"),
            "dig_plataforma": (
                "Instagram", "TikTok", "YouTube", "WhatsApp", "X (Twitter)", "Otra"
            ),
            "fis_tipo_ejercicio": (
                "Gimnasio", "Deporte de equipo", "Correr / caminar", "Otro", "Ninguno"
            ),
            "fis_desayuna": ("Siempre", "A veces", "Nunca"),
            "eco_ahorra": ("Sí", "No"),
            "mov_modo": (
                "A pie", "Bici / patinete", "Bus urbano", "Coche", "Tren / autobús interurbano"
            ),
            "mov_horario_preferido": ("Mañana", "Tarde", "Indiferente"),
            "soc_voluntariado": ("Sí", "No"),
            "fut_practicas": ("Sí", "Lo he planeado", "No"),
            "fut_ingles": ("A2", "B1", "B2", "C1"),
            "fut_postgrado": ("Sí", "No", "No sé"),
            "fut_perfil": ("Técnico", "Gestión", "Investigación", "Emprendedor", "Sin definir"),
        }
    )

    # 1. Límites Físicos e Imposibilidades Matemáticas (Severidad CRÍTICA -> NO_FIABLE)
    limites_fisicos: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "edad": (16.0, 90.0),
            "curso": (1.0, 4.0),
            "asistencia_pct": (0.0, 100.0),
            "nota_media": (0.0, 10.0),
            "sue_horas_semana": (0.01, 24.0),
            "sue_horas_finde": (0.01, 24.0),
            "sue_pantalla_antes_min": (0.0, 1440.0),
            "dig_pantalla_h": (0.0, 24.0),
            "dig_redes_h": (0.0, 24.0),
            "dig_uso_nocturno_min": (0.0, 1440.0),
            "fis_dias_ejercicio": (0.0, 7.0),
            "mov_tiempo_campus_h": (0.0, 24.0),
            "mov_horas_libres_dia": (0.0, 24.0),
            "tra_faltas_mes": (0.0, 31.0),
            "mov_retrasos_mes": (0.0, 31.0),
            "trabaja_horas": (0.0, 112.0),
            "tra_horas_estudio_sem": (0.0, 112.0),
            "fis_comidas_fuera_sem": (0.0, 35.0),
            "fis_comida_rapida_sem": (0.0, 35.0),
        }
    )

    # 2. Rangos de Aceptación Operativa (Severidad ADVERTENCIA -> SOSPECHOSO / EN_REVISION)
    rangos_operativos: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "edad": (17.0, 50.0),
            "trabaja_horas": (0.0, 60.0),
            "desplaz_min": (0.0, 240.0),
            "sue_horas_semana": (3.0, 12.0),
            "sue_horas_finde": (3.0, 14.0),
            "sue_pantalla_antes_min": (0.0, 300.0),
            "sue_cafeina_tazas": (0.0, 15.0),
            "sue_despertares": (0.0, 10.0),
            "tra_horas_estudio_sem": (0.0, 80.0),
            "est_entregas_simult": (0.0, 20.0),
            "est_semanas_pico": (0.0, 20.0),
            "est_horas_estudio_examen": (0.0, 100.0),
            "fis_bebidas_energeticas_sem": (0.0, 30.0),
            "fis_hidratacion_l": (0.0, 10.0),
            "eco_ingresos_mes": (0.0, 5000.0),
            "eco_gasto_mes": (0.0, 5000.0),
            "eco_coste_vivienda": (0.0, 3000.0),
            "eco_gasto_ocio": (0.0, 3000.0),
            "eco_ayuda_familiar": (0.0, 3000.0),
            "eco_gasto_transporte": (0.0, 1000.0),
            "mov_biblioteca_sem": (0.0, 21.0),
            "soc_asociaciones": (0.0, 20.0),
            "soc_amigos_clase": (0.0, 100.0),
            "soc_actividades_campus": (0.0, 50.0),
            "soc_eventos": (0.0, 50.0),
            "fut_salario_esperado": (10000.0, 100000.0),
            "fut_certificaciones": (0.0, 20.0),
        }
    )


@dataclass(frozen=True)
class CleaningResult:
    """Contenedor integral de artefactos producidos por el pipeline."""
    datos_limpios: pd.DataFrame
    datos_excluidos: pd.DataFrame
    clasificacion_registros: pd.DataFrame
    registro_cambios: pd.DataFrame
    avisos_coherencia: pd.DataFrame
    valores_vacios: pd.DataFrame
    resumen: pd.DataFrame
    reglas_aplicadas: pd.DataFrame


# ============================================================================
# LEDGER DE AUDITORÍA VECTORIZADO
# ============================================================================

class AuditLedger:
    """Acumulador atómico y tipado de transformaciones y eventos de calidad."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self, id_: str, variable: str, before: Any, after: Any, rule: str, severity: str = "INFORMATIVA"
    ) -> None:
        """Registra un evento individual."""
        self._entries.append(
            AuditEntry(
                id=str(id_),
                variable=variable,
                valor_original=before,
                valor_nuevo=after,
                regla=rule,
                severidad=severity,
            )
        )

    def record_batch(
        self,
        ids: Sequence[str] | pd.Series,
        variable: str,
        before: Sequence[Any] | pd.Series,
        after: Sequence[Any] | pd.Series | Any,
        rule: str,
        severity: str = "INFORMATIVA",
    ) -> None:
        """Registra un lote vectorizado de eventos."""
        id_list = list(ids)
        before_list = list(before)
        if isinstance(after, (pd.Series, list, tuple)):
            after_list = list(after)
        else:
            after_list = [after] * len(id_list)

        for i_val, b_val, a_val in zip(id_list, before_list, after_list):
            self._entries.append(
                AuditEntry(
                    id=str(i_val),
                    variable=variable,
                    valor_original=b_val,
                    valor_nuevo=a_val,
                    regla=rule,
                    severidad=severity,
                )
            )

    def to_dataframe(self) -> pd.DataFrame:
        """Exporta los eventos acumulados como DataFrame."""
        if not self._entries:
            return pd.DataFrame(
                columns=["id", "variable", "valor_original", "valor_nuevo", "regla", "severidad"]
            )
        return pd.DataFrame([e.__dict__ for e in self._entries])


# ============================================================================
# HELPERS FUNCIONALES
# ============================================================================

def normalizar_clave(texto: Any) -> str:
    """Normaliza texto eliminando diacríticos, espacios extra y capitalización."""
    if pd.isna(texto):
        return ""
    cadena = unicodedata.normalize("NFKD", str(texto))
    cadena = "".join(c for c in cadena if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", cadena).strip().casefold()


def extraer_rangos_diccionario(
    df_dic: pd.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Extrae dinámicamente escalas (ej. Escala 1-5, Escala 0-10) del Diccionario."""
    escalas: dict[str, tuple[float, float]] = {}
    if "Tipo" not in df_dic.columns or "Variable" not in df_dic.columns:
        return escalas

    patron = re.compile(r"Escala\s+(\d+)-(\d+)", re.IGNORECASE)
    for _, fila in df_dic.iterrows():
        tipo_str = str(fila.get("Tipo", ""))
        match = patron.search(tipo_str)
        if match:
            var_name = str(fila.get("Variable", "")).strip()
            if var_name:
                escalas[var_name] = (float(match.group(1)), float(match.group(2)))
    return escalas


def parsear_fechas_hibridas(series: pd.Series) -> pd.Series:
    """Parsea fechas seriales de Excel (base 1899-12-30) y cadenas ISO/texto."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    resultado = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    numerico = pd.to_numeric(series, errors="coerce")
    mascara_serie_excel = numerico.notna() & (numerico >= 1000) & (numerico <= 100000)

    if mascara_serie_excel.any():
        resultado.loc[mascara_serie_excel] = pd.to_datetime(
            numerico.loc[mascara_serie_excel], unit="D", origin="1899-12-30"
        )

    mascara_texto = (~mascara_serie_excel) & series.notna()
    if mascara_texto.any():
        resultado.loc[mascara_texto] = pd.to_datetime(
            series.loc[mascara_texto], errors="coerce", dayfirst=True
        )

    return resultado


# ============================================================================
# MOTOR CENTRAL DE CALIDAD, VALIDACIÓN Y EXCLUSIÓN (DEEP MODULE)
# ============================================================================

class DatasetCleaner:
    """
    Módulo profundo para la limpieza, auditoría, detección de datos no fiables
    y exclusión de registros. Interfaz externa limpia y desacoplada.
    """

    def __init__(self, config: CleaningConfig | None = None) -> None:
        self.config: Final[CleaningConfig] = config or CleaningConfig()

    def clean(
        self, df_raw: pd.DataFrame, df_dict: pd.DataFrame | None = None
    ) -> CleaningResult:
        """
        Ejecuta el pipeline completo garantizando inmutabilidad del DataFrame original.
        """
        # Validación de estructura de entrada
        if "id" not in df_raw.columns:
            raise KeyError("La columna obligatoria 'id' no está presente en el dataset.")

        # Manejo de dataset vacío
        if df_raw.empty:
            empty_df = pd.DataFrame(columns=df_raw.columns)
            empty_clasif = pd.DataFrame(
                columns=[
                    "id", "estado_global", "motivos", "n_invalidos",
                    "n_sospechosos", "n_ausentes", "n_no_aplicables", "excluido", "recomendacion"
                ]
            )
            empty_resumen = pd.DataFrame(columns=["concepto", "valor"])
            empty_vacios = pd.DataFrame(columns=["variable", "n_vacios", "pct_vacios", "es_obligatoria"])
            empty_alertas = pd.DataFrame(
                columns=["regla_id", "comprobacion", "severidad", "n_casos", "ids", "afecta_estado_global", "excluye_analisis_principal"]
            )
            empty_reglas = pd.DataFrame(
                columns=["regla_id", "nombre", "variables", "severidad", "condicion", "descripcion", "accion", "excluye_registro"]
            )
            return CleaningResult(
                datos_limpios=empty_df.copy(),
                datos_excluidos=empty_df.copy(),
                clasificacion_registros=empty_clasif,
                registro_cambios=AuditLedger().to_dataframe(),
                avisos_coherencia=empty_alertas,
                valores_vacios=empty_vacios,
                resumen=empty_resumen,
                reglas_aplicadas=empty_reglas,
            )

        df = df_raw.copy().reset_index(drop=True)
        df_original = df.copy()
        df["id"] = df["id"].astype(str).str.strip()

        ledger = AuditLedger()

        # Estructura de evaluación indexada por fila física única (int idx)
        row_eval: dict[int, dict[str, Any]] = {
            idx: {
                "id": str(df.at[idx, "id"]).strip(),
                "motivos": set(),
                "criticos": set(),
                "sospechosos": set(),
                "ausentes": set(),
                "no_aplicables": set(),
                "es_duplicado": False,
            }
            for idx in df.index
        }

        # 1. Normalización de texto y categorías canónicas
        df = self._limpiar_texto(df, ledger, row_eval)

        # 2. Detección y clasificación determinista de duplicados
        df, idx_duplicados = self._gestionar_duplicados(df, ledger, row_eval)

        # 3. Validación de límites físicos, rangos operativos y fechas
        df = self._validar_limites_y_rangos(df, df_dict, ledger, row_eval)

        # 4. Evaluación de obligatoriedad y aplicabilidad condicional
        self._evaluar_condicionales_y_ausencias(df, row_eval)

        # 5. Auditoría de coherencia inter-variable
        avisos_coherencia, catalogo_reglas = self._evaluar_coherencia(df, row_eval)

        # 6. Clasificación final y decisión de exclusión determinista por fila
        df_clasificacion, idx_excluidos = self._clasificar_registros(
            df_original=df_original, row_eval=row_eval
        )

        # 7. Partición en Datos limpios (admitidos) y Datos excluidos
        # Una fila es admitida si NO está excluida y NO es duplicada
        mascara_admitidos = ~df_original.index.isin(idx_excluidos)
        df_limpios = df.loc[mascara_admitidos].copy().reset_index(drop=True)

        # Datos excluidos: se conservan con datos originales y motivo registrado
        df_excluidos = df_original.loc[~mascara_admitidos].copy()
        # Agregar columnas de metadatos de exclusión al principio
        meta_motivos = df_clasificacion.set_index(df_original.index)["motivos"].to_dict()
        meta_estado = df_clasificacion.set_index(df_original.index)["estado_global"].to_dict()
        df_excluidos.insert(0, "estado_global", df_excluidos.index.map(meta_estado))
        df_excluidos.insert(1, "motivo_exclusion", df_excluidos.index.map(meta_motivos))
        df_excluidos = df_excluidos.reset_index(drop=True)

        # 8. Construcción de reportes y catálogo
        resumen, valores_vacios = self._construir_reportes(
            df_original=df_original,
            df_limpios=df_limpios,
            df_excluidos=df_excluidos,
            df_clasificacion=df_clasificacion,
            ledger=ledger,
        )

        return CleaningResult(
            datos_limpios=df_limpios,
            datos_excluidos=df_excluidos,
            clasificacion_registros=df_clasificacion,
            registro_cambios=ledger.to_dataframe(),
            avisos_coherencia=avisos_coherencia,
            valores_vacios=valores_vacios,
            resumen=resumen,
            reglas_aplicadas=catalogo_reglas,
        )

    def _limpiar_texto(
        self, df: pd.DataFrame, ledger: AuditLedger, row_eval: dict[int, dict[str, Any]]
    ) -> pd.DataFrame:
        """Normalización vectorizada de categorías a formas canónicas."""
        for col, validos in self.config.validos.items():
            if col not in df.columns:
                continue

            serie_original = df[col]
            mapa_canonico = {normalizar_clave(v): v for v in validos}

            claves = serie_original.map(normalizar_clave)
            canonicos = claves.map(mapa_canonico)

            mascara_valido = serie_original.notna()
            mascara_invalido = mascara_valido & canonicos.isna()
            mascara_modificado = mascara_valido & canonicos.notna() & (serie_original != canonicos)

            if mascara_invalido.any():
                ids_inv = df.loc[mascara_invalido, "id"]
                ledger.record_batch(
                    ids=ids_inv,
                    variable=col,
                    before=serie_original.loc[mascara_invalido],
                    after=np.nan,
                    rule="Categoría no válida -> vacío",
                    severity="ADVERTENCIA",
                )
                for idx in df.index[mascara_invalido]:
                    row_eval[idx]["sospechosos"].add(f"Categoría no válida en '{col}'")
                    row_eval[idx]["motivos"].add(f"Categoría no válida en '{col}'")
                df.loc[mascara_invalido, col] = np.nan

            if mascara_modificado.any():
                ledger.record_batch(
                    ids=df.loc[mascara_modificado, "id"],
                    variable=col,
                    before=serie_original.loc[mascara_modificado],
                    after=canonicos.loc[mascara_modificado],
                    rule="Normalizar mayúsculas/acentos/espacios",
                    severity="INFORMATIVA",
                )
                df.loc[mascara_modificado, col] = canonicos.loc[mascara_modificado]

        return df

    def _gestionar_duplicados(
        self, df: pd.DataFrame, ledger: AuditLedger, row_eval: dict[int, dict[str, Any]]
    ) -> tuple[pd.DataFrame, list[int]]:
        """
        Detección y clasificación determinista de duplicados.
        No descarta arbitrariamente: la primera fila se conserva; las repeticiones
        se marcan como DUPLICADO. En ID colisionados con datos distintos, se evalúa
        completitud informativa.
        """
        idx_duplicados: list[int] = []

        # 1. Duplicados exactos de toda la fila
        mascara_exactos = df.duplicated(keep="first")
        if mascara_exactos.any():
            for idx in df.index[mascara_exactos]:
                r_id = str(df.at[idx, "id"])
                row_eval[idx]["es_duplicado"] = True
                row_eval[idx]["motivos"].add("Duplicado exacto de fila completa")
                ledger.record(
                    id_=r_id,
                    variable="(fila completa)",
                    before="fila duplicada idéntica",
                    after="marcada como duplicada",
                    rule="Duplicado exacto",
                    severity="CRITICA",
                )
                idx_duplicados.append(idx)

        # 2. Identificadores repetidos con datos distintos (solo entre filas NO idénticas)
        df_sin_exactos = df.loc[~mascara_exactos]
        mascara_id_discrepante = df_sin_exactos["id"].duplicated(keep=False)
        if mascara_id_discrepante.any():
            ids_repetidos = df_sin_exactos.loc[mascara_id_discrepante, "id"].unique()
            for r_id in ids_repetidos:
                sub_df = df_sin_exactos[df_sin_exactos["id"] == r_id]
                completitud = sub_df.notna().sum(axis=1)
                mejor_idx = completitud.idxmax()
                for idx in sub_df.index:
                    if idx != mejor_idx:
                        row_eval[idx]["es_duplicado"] = True
                        row_eval[idx]["motivos"].add(
                            f"Colisión de ID con datos discrepantes (descartada por menor completitud: {completitud[idx]} vs {completitud[mejor_idx]})"
                        )
                        ledger.record(
                            id_=str(r_id),
                            variable="id",
                            before="id repetido con datos discrepantes",
                            after="descartado por menor completitud",
                            rule="Id duplicado discrepante",
                            severity="CRITICA",
                        )
                        idx_duplicados.append(idx)

        return df, idx_duplicados

    def _validar_limites_y_rangos(
        self,
        df: pd.DataFrame,
        df_dict: pd.DataFrame | None,
        ledger: AuditLedger,
        row_eval: dict[int, dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Distingue rigurosamente entre:
          1. Imposibilidad física/lógica (CRITICA -> NO_FIABLE).
          2. Fuera de rango operativo / atípico (ADVERTENCIA -> EN_REVISION).
          3. Escalas psicométricas del diccionario.
        """
        # Extraer escalas psicométricas del diccionario
        escalas_dic = extraer_rangos_diccionario(df_dict) if df_dict is not None else {}

        # Validar escalas psicométricas (imposibilidad fuera de escala)
        for col, (min_e, max_e) in escalas_dic.items():
            if col not in df.columns:
                continue
            serie = pd.to_numeric(df[col], errors="coerce")
            fuera_escala = serie.notna() & ((serie < min_e) | (serie > max_e))
            if fuera_escala.any():
                ids_bad = df.loc[fuera_escala, "id"]
                ledger.record_batch(
                    ids=ids_bad,
                    variable=col,
                    before=serie.loc[fuera_escala],
                    after=np.nan,
                    rule=f"Valor fuera de escala psicométrica [{min_e}, {max_e}]",
                    severity="CRITICA",
                )
                for idx in df.index[fuera_escala]:
                    row_eval[idx]["criticos"].add(f"Escala inválida en '{col}' [{min_e}, {max_e}]")
                    row_eval[idx]["motivos"].add(f"Escala psicométrica imposible en '{col}'")
                df.loc[fuera_escala, col] = np.nan

        # 1. Validación de Límites Físicos (CRÍTICA -> exclusión global)
        for col, (min_f, max_f) in self.config.limites_fisicos.items():
            if col not in df.columns:
                continue

            num_serie = pd.to_numeric(df[col], errors="coerce")
            # Texto no numérico en columna cuantitativa
            texto_invalido = df[col].notna() & num_serie.isna()
            if texto_invalido.any():
                ids_bad = df.loc[texto_invalido, "id"]
                ledger.record_batch(
                    ids=ids_bad,
                    variable=col,
                    before=df.loc[texto_invalido, col],
                    after=np.nan,
                    rule="Texto no numérico en campo cuantitativo",
                    severity="CRITICA",
                )
                for idx in df.index[texto_invalido]:
                    row_eval[idx]["criticos"].add(f"Formato no numérico en '{col}'")
                    row_eval[idx]["motivos"].add(f"Error de tipo/formato en '{col}'")
                df.loc[texto_invalido, col] = np.nan

            # Tratamiento especial nota_media > 10
            if col == "nota_media" and self.config.recuperar_nota_x10:
                mascara_div_10 = num_serie.notna() & (num_serie > 10.0) & (num_serie <= 100.0)
                if mascara_div_10.any():
                    recuperados = (num_serie.loc[mascara_div_10] / 10.0).round(2)
                    ledger.record_batch(
                        ids=df.loc[mascara_div_10, "id"],
                        variable=col,
                        before=num_serie.loc[mascara_div_10],
                        after=recuperados,
                        rule="Nota >10 recuperada por error de coma decimal (÷10)",
                        severity="INFORMATIVA",
                    )
                    num_serie.loc[mascara_div_10] = recuperados

            # Infracción de límite físico inequívoco
            mascara_imposible = num_serie.notna() & ((num_serie < min_f) | (num_serie > max_f))
            if mascara_imposible.any():
                ids_imp = df.loc[mascara_imposible, "id"]
                etiqueta = f"[{int(min_f) if min_f.is_integer() else min_f}, {int(max_f) if max_f.is_integer() else max_f}]"
                ledger.record_batch(
                    ids=ids_imp,
                    variable=col,
                    before=num_serie.loc[mascara_imposible],
                    after=np.nan,
                    rule=f"Valor físicamente imposible fuera de {etiqueta}",
                    severity="CRITICA",
                )
                for idx in df.index[mascara_imposible]:
                    row_eval[idx]["criticos"].add(f"Imposibilidad física en '{col}' (límite {etiqueta})")
                    row_eval[idx]["motivos"].add(f"Valor físicamente imposible en '{col}'")
                num_serie.loc[mascara_imposible] = np.nan

            df[col] = num_serie

        # 2. Validación de Rangos Operativos (ADVERTENCIA -> SOSPECHOSO, no crítico)
        for col, (min_o, max_o) in self.config.rangos_operativos.items():
            if col not in df.columns:
                continue

            num_serie = pd.to_numeric(df[col], errors="coerce")
            mascara_atipico = num_serie.notna() & ((num_serie < min_o) | (num_serie > max_o))
            if mascara_atipico.any():
                ids_at = df.loc[mascara_atipico, "id"]
                etiqueta_op = f"[{int(min_o) if min_o.is_integer() else min_o}, {int(max_o) if max_o.is_integer() else max_o}]"
                ledger.record_batch(
                    ids=ids_at,
                    variable=col,
                    before=num_serie.loc[mascara_atipico],
                    after=num_serie.loc[mascara_atipico],
                    rule=f"Valor atípico fuera de rango operativo {etiqueta_op}",
                    severity="ADVERTENCIA",
                )
                for idx in df.index[mascara_atipico]:
                    row_eval[idx]["sospechosos"].add(f"Valor atípico en '{col}' {etiqueta_op}")
                    row_eval[idx]["motivos"].add(f"Valor atípico en '{col}'")

        # 3. Validación de Fechas
        if "fecha_respuesta" in df.columns:
            fechas_parseadas = parsear_fechas_hibridas(df["fecha_respuesta"])
            mascara_fecha_invalida = fechas_parseadas.isna() | (
                (fechas_parseadas < self.config.fecha_min)
                | (fechas_parseadas > self.config.fecha_max)
            )

            if mascara_fecha_invalida.any():
                ids_bad_fecha = df.loc[mascara_fecha_invalida, "id"]
                ledger.record_batch(
                    ids=ids_bad_fecha,
                    variable="fecha_respuesta",
                    before=df.loc[mascara_fecha_invalida, "fecha_respuesta"],
                    after=pd.NaT,
                    rule="Fecha fuera del periodo lectivo oficial (5-20 oct 2026)",
                    severity="CRITICA",
                )
                for idx in df.index[mascara_fecha_invalida]:
                    row_eval[idx]["criticos"].add("Fecha de respuesta inválida o fuera de periodo")
                    row_eval[idx]["motivos"].add("Fecha fuera de periodo oficial")
                fechas_parseadas.loc[mascara_fecha_invalida] = pd.NaT

            df["fecha_respuesta"] = fechas_parseadas

        return df

    def _evaluar_condicionales_y_ausencias(
        self, df: pd.DataFrame, row_eval: dict[int, dict[str, Any]]
    ) -> None:
        """
        Evalúa campos obligatorios y aplicabilidad condicional.
        Distingue claramente entre AUSENTE, NO_APLICABLE y SOSPECHOSO.
        """
        # A) Campos obligatorios generales
        for col in self.config.columnas_obligatorias:
            if col in df.columns:
                nulos = df[col].isna()
                for idx in df.index[nulos]:
                    row_eval[idx]["ausentes"].add(f"Campo obligatorio ausente: '{col}'")
                    row_eval[idx]["motivos"].add(f"Falta campo obligatorio '{col}'")

        # B) Reglas Condicionales
        # 1. Trabajo
        if "trabaja_horas" in df.columns:
            trabaja_horas = pd.to_numeric(df["trabaja_horas"], errors="coerce")
            cols_laborales = (
                "tra_tipo", "tra_relacion_carrera", "tra_turno",
                "tra_motivo", "tra_flexibilidad", "tra_faltas_mes", "tra_conciliacion"
            )
            no_trabaja = trabaja_horas == 0.0
            for col in cols_laborales:
                if col in df.columns:
                    vacios_legitimos = no_trabaja & df[col].isna()
                    for idx in df.index[vacios_legitimos]:
                        row_eval[idx]["no_aplicables"].add(col)

                    llenos_incoherentes = no_trabaja & df[col].notna()
                    for idx in df.index[llenos_incoherentes]:
                        row_eval[idx]["sospechosos"].add(f"Declara '{col}' sin trabajar (0 horas)")
                        row_eval[idx]["motivos"].add(f"Datos en '{col}' incongruentes con 0 h de trabajo")

            si_trabaja = trabaja_horas > 0.0
            for col in cols_laborales:
                if col in df.columns:
                    vacios_laborales = si_trabaja & df[col].isna()
                    for idx in df.index[vacios_laborales]:
                        row_eval[idx]["ausentes"].add(f"Campo laboral ausente: '{col}'")
                        row_eval[idx]["motivos"].add(f"Falta información laboral '{col}'")

        # 2. Movilidad y aparcamiento
        if "mov_modo" in df.columns and "mov_aparcar_facil" in df.columns:
            mov_modo = df["mov_modo"]
            mov_aparcar = pd.to_numeric(df["mov_aparcar_facil"], errors="coerce")

            no_coche = mov_modo != "Coche"
            vacios_ok = no_coche & mov_aparcar.isna()
            for idx in df.index[vacios_ok]:
                row_eval[idx]["no_aplicables"].add("mov_aparcar_facil")

            aparcamiento_incoherente = no_coche & mov_aparcar.notna()
            for idx in df.index[aparcamiento_incoherente]:
                row_eval[idx]["sospechosos"].add("Responde facilidad de aparcar sin desplazarse en coche")
                row_eval[idx]["motivos"].add("Aparcamiento valorado sin usar coche")

            va_coche = mov_modo == "Coche"
            coche_vacio = va_coche & mov_aparcar.isna()
            for idx in df.index[coche_vacio]:
                row_eval[idx]["ausentes"].add("Falta valorar aparcamiento usando coche")
                row_eval[idx]["motivos"].add("Aparcamiento no valorado por conductor")

        # 3. Ejercicio y modalidad
        if "fis_dias_ejercicio" in df.columns and "fis_tipo_ejercicio" in df.columns:
            dias_ej = pd.to_numeric(df["fis_dias_ejercicio"], errors="coerce")
            tipo_ej = df["fis_tipo_ejercicio"]

            cero_dias = dias_ej == 0.0
            ej_incoherente = cero_dias & tipo_ej.notna() & (~tipo_ej.isin(["Ninguno", ""]))
            for idx in df.index[ej_incoherente]:
                row_eval[idx]["sospechosos"].add("Declara modalidad de ejercicio con 0 días semanales")
                row_eval[idx]["motivos"].add("Modalidad deportiva con 0 días de ejercicio")

            ej_contradictorio = (dias_ej > 0.0) & (tipo_ej == "Ninguno")
            for idx in df.index[ej_contradictorio]:
                row_eval[idx]["sospechosos"].add("Declara 'Ninguno' pero practica ejercicio >0 días")
                row_eval[idx]["motivos"].add("Tipo ejercicio 'Ninguno' con días activos")

    def _evaluar_coherencia(
        self, df: pd.DataFrame, row_eval: dict[int, dict[str, Any]]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Auditoría exhaustiva de contradicciones inter-variables mediante
        catálogo centralizado con asignación explícita de severidad y acción.
        """
        alertas: list[InconsistencyAlert] = []
        reglas_cat: list[ValidationRule] = []

        if df.empty:
            return (
                pd.DataFrame(columns=["regla_id", "comprobacion", "severidad", "n_casos", "ids", "afecta_estado_global", "excluye_analisis_principal"]),
                pd.DataFrame(columns=["regla_id", "nombre", "variables", "severidad", "condicion", "descripcion", "accion", "excluye_registro"])
            )

        def get_series(col_name: str) -> pd.Series | None:
            if col_name in df.columns:
                return df[col_name]
            return None

        def get_numeric(col_name: str) -> pd.Series | None:
            if col_name in df.columns:
                return pd.to_numeric(df[col_name], errors="coerce")
            return None

        def registrar_regla(
            rule_id: str,
            nombre: str,
            variables: tuple[str, ...],
            severidad: Severity,
            condicion_str: str,
            descripcion: str,
            accion: str,
            mascara: pd.Series,
            excluye: bool = False,
        ) -> None:
            reglas_cat.append(
                ValidationRule(
                    regla_id=rule_id,
                    nombre=nombre,
                    variables=variables,
                    severidad=severidad,
                    condicion=condicion_str,
                    descripcion=descripcion,
                    accion=accion,
                    excluye_registro=excluye,
                )
            )
            casos_idx = df.index[mascara].tolist()
            if casos_idx:
                casos_ids = [str(df.at[i, "id"]) for i in casos_idx]
                ids_trunc = ", ".join(casos_ids[:15]) + (" ..." if len(casos_ids) > 15 else "")
                alertas.append(
                    InconsistencyAlert(
                        regla_id=rule_id,
                        comprobacion=descripcion,
                        severidad=severidad.value,
                        n_casos=len(casos_idx),
                        ids=ids_trunc,
                        afecta_estado_global=True,
                        excluye_analisis_principal=excluye,
                    )
                )
                for idx in casos_idx:
                    if severidad == Severity.CRITICA:
                        row_eval[idx]["criticos"].add(descripcion)
                    elif severidad == Severity.ADVERTENCIA:
                        row_eval[idx]["sospechosos"].add(descripcion)
                    row_eval[idx]["motivos"].add(descripcion)

        # Regla COH_DIG_01: Redes sociales superan tiempo de pantalla total
        redes = get_numeric("dig_redes_h")
        pantalla = get_numeric("dig_pantalla_h")
        if redes is not None and pantalla is not None:
            m_dig1 = redes.notna() & pantalla.notna() & (redes > pantalla)
            registrar_regla(
                rule_id="COH_DIG_01",
                nombre="Redes superan pantalla total",
                variables=("dig_redes_h", "dig_pantalla_h"),
                severidad=Severity.CRITICA,
                condicion_str="dig_redes_h > dig_pantalla_h",
                descripcion="Horas en redes sociales exceden el tiempo total de pantalla diaria",
                accion="Exclusión global del registro (NO_FIABLE)",
                mascara=m_dig1,
                excluye=True,
            )

        # Regla COH_DIG_02: Suma de pantalla y sueño supera 24 horas del día
        sue_sem = get_numeric("sue_horas_semana")
        if pantalla is not None and sue_sem is not None:
            m_dig2 = pantalla.notna() & sue_sem.notna() & ((pantalla + sue_sem) > 24.0)
            registrar_regla(
                rule_id="COH_DIG_02",
                nombre="Sueño + pantalla excede día",
                variables=("dig_pantalla_h", "sue_horas_semana"),
                severidad=Severity.CRITICA,
                condicion_str="(dig_pantalla_h + sue_horas_semana) > 24",
                descripcion="La suma diaria de sueño y pantalla supera las 24 horas del día",
                accion="Exclusión global del registro (NO_FIABLE)",
                mascara=m_dig2,
                excluye=True,
            )

        # Regla COH_ECO_01: Desglose de gastos supera gasto total declarado
        g_ocio = get_numeric("eco_gasto_ocio")
        g_viv = get_numeric("eco_coste_vivienda")
        g_tra = get_numeric("eco_gasto_transporte")
        g_mes = get_numeric("eco_gasto_mes")
        if g_mes is not None and g_ocio is not None and g_viv is not None and g_tra is not None:
            m_eco1 = (
                g_ocio.notna()
                & g_viv.notna()
                & g_tra.notna()
                & g_mes.notna()
                & ((g_ocio + g_viv + g_tra) > g_mes)
            )
            registrar_regla(
                rule_id="COH_ECO_01",
                nombre="Desglose excede gasto total",
                variables=("eco_gasto_ocio", "eco_coste_vivienda", "eco_gasto_transporte", "eco_gasto_mes"),
                severidad=Severity.CRITICA,
                condicion_str="(eco_gasto_ocio + eco_coste_vivienda + eco_gasto_transporte) > eco_gasto_mes",
                descripcion="La suma de vivienda, ocio y transporte supera el gasto mensual total",
                accion="Exclusión global del registro (NO_FIABLE)",
                mascara=m_eco1,
                excluye=True,
            )

        # Regla COH_TRA_01: Horas de trabajo > 0 pero tra_tipo vacío (ausencia de información)
        th = get_numeric("trabaja_horas")
        tt = get_series("tra_tipo")
        if th is not None and tt is not None:
            m_tr1 = (th > 0.0) & tt.isna()
            registrar_regla(
                rule_id="COH_TRA_01",
                nombre="Trabaja con tipo de empleo ausente",
                variables=("trabaja_horas", "tra_tipo"),
                severidad=Severity.INFORMATIVA,
                condicion_str="(trabaja_horas > 0) & tra_tipo.isna()",
                descripcion="Declara horas de trabajo pero no especifica el sector o tipo de empleo",
                accion="Marcar como INCOMPLETO",
                mascara=m_tr1,
                excluye=False,
            )

            m_tr2 = (th == 0.0) & tt.notna()
            registrar_regla(
                rule_id="COH_TRA_02",
                nombre="No trabaja pero declara tipo de empleo",
                variables=("trabaja_horas", "tra_tipo"),
                severidad=Severity.ADVERTENCIA,
                condicion_str="(trabaja_horas == 0) & tra_tipo.notna()",
                descripcion="Declara no trabajar (0 h) pero cumplimenta detalles del sector laboral",
                accion="Marcar como SOSPECHOSO (EN_REVISION)",
                mascara=m_tr2,
                excluye=False,
            )

            m_tr3 = th.isna() & tt.notna()
            registrar_regla(
                rule_id="COH_TRA_03",
                nombre="Datos laborales sin horas de trabajo",
                variables=("trabaja_horas", "tra_tipo"),
                severidad=Severity.INFORMATIVA,
                condicion_str="trabaja_horas.isna() & tra_tipo.notna()",
                descripcion="Cumplimenta detalles laborales pero omite las horas semanales trabajadas",
                accion="Marcar como INCOMPLETO",
                mascara=m_tr3,
                excluye=False,
            )

        # Regla COH_MOV_01: En coche pero sin valorar aparcamiento (ausencia de información)
        mm = get_series("mov_modo")
        ma = get_numeric("mov_aparcar_facil")
        if mm is not None and ma is not None:
            m_mov1 = (mm == "Coche") & ma.isna()
            registrar_regla(
                rule_id="COH_MOV_01",
                nombre="Conductor sin valorar aparcamiento",
                variables=("mov_modo", "mov_aparcar_facil"),
                severidad=Severity.INFORMATIVA,
                condicion_str="(mov_modo == 'Coche') & mov_aparcar_facil.isna()",
                descripcion="Se desplaza en coche pero omite la valoración de facilidad para aparcar",
                accion="Marcar como INCOMPLETO",
                mascara=m_mov1,
                excluye=False,
            )

            m_mov2 = (mm != "Coche") & ma.notna()
            registrar_regla(
                rule_id="COH_MOV_02",
                nombre="Valora aparcamiento sin usar coche",
                variables=("mov_modo", "mov_aparcar_facil"),
                severidad=Severity.ADVERTENCIA,
                condicion_str="(mov_modo != 'Coche') & mov_aparcar_facil.notna()",
                descripcion="Responde sobre la facilidad de aparcar sin acudir en coche al campus",
                accion="Marcar como SOSPECHOSO (EN_REVISION)",
                mascara=m_mov2,
                excluye=False,
            )

        # Regla COH_IA_01: Herramienta Ninguna con frecuencia > 1
        ia_herramienta = get_series("ia_herramienta")
        ia_frecuencia = get_numeric("ia_frecuencia")
        if ia_herramienta is not None and ia_frecuencia is not None:
            m_ia1 = (ia_herramienta == "Ninguna") & (ia_frecuencia > 1.0)
            registrar_regla(
                rule_id="COH_IA_01",
                nombre="IA Ninguna con uso activo",
                variables=("ia_herramienta", "ia_frecuencia"),
                severidad=Severity.ADVERTENCIA,
                condicion_str="(ia_herramienta == 'Ninguna') & (ia_frecuencia > 1)",
                descripcion="Declara no usar herramientas de IA pero reporta frecuencia de uso activa",
                accion="Marcar como SOSPECHOSO (EN_REVISION)",
                mascara=m_ia1,
                excluye=False,
            )

        # Regla COH_IA_02: Frecuencia 1 (nunca) pero declara uso Siempre
        ia_declara = get_series("ia_declara_uso")
        if ia_frecuencia is not None and ia_declara is not None:
            m_ia2 = (ia_frecuencia == 1.0) & (ia_declara == "Siempre")
            registrar_regla(
                rule_id="COH_IA_02",
                nombre="Contradicción frecuencia y declaración IA",
                variables=("ia_frecuencia", "ia_declara_uso"),
                severidad=Severity.ADVERTENCIA,
                condicion_str="(ia_frecuencia == 1) & (ia_declara_uso == 'Siempre')",
                descripcion="Declara no usar nunca IA pero afirma declarar su uso siempre en trabajos",
                accion="Marcar como SOSPECHOSO (EN_REVISION)",
                mascara=m_ia2,
                excluye=False,
            )

        # Regla COH_TRA_04: Carga semanal combinada de trabajo y estudio > 112 h
        estudio_h = get_numeric("tra_horas_estudio_sem")
        if th is not None and estudio_h is not None:
            m_tra4 = th.notna() & estudio_h.notna() & ((th + estudio_h) > 112.0)
            registrar_regla(
                rule_id="COH_TRA_04",
                nombre="Carga extrema trabajo + estudio",
                variables=("trabaja_horas", "tra_horas_estudio_sem"),
                severidad=Severity.ADVERTENCIA,
                condicion_str="(trabaja_horas + tra_horas_estudio_sem) > 112",
                descripcion="Carga semanal combinada supera 16 horas diarias de dedicación",
                accion="Marcar como SOSPECHOSO (EN_REVISION)",
                mascara=m_tra4,
                excluye=False,
            )

        # Regla COH_ECO_02: Gasto mensual > ingresos + ayuda familiar (Plausible con ahorros/deuda)
        ingresos = get_numeric("eco_ingresos_mes")
        ayuda = get_numeric("eco_ayuda_familiar")
        if g_mes is not None and ingresos is not None and ayuda is not None:
            m_eco2 = (
                g_mes.notna()
                & ingresos.notna()
                & ayuda.notna()
                & (g_mes > (ingresos + ayuda + 1.0))
            )
            registrar_regla(
                rule_id="COH_ECO_02",
                nombre="Gasto supera ingresos declarados",
                variables=("eco_gasto_mes", "eco_ingresos_mes", "eco_ayuda_familiar"),
                severidad=Severity.INFORMATIVA,
                condicion_str="eco_gasto_mes > (eco_ingresos_mes + eco_ayuda_familiar + 1)",
                descripcion="Gasto mensual superior a ingresos y ayudas familiares (financiamiento por ahorro o crédito)",
                accion="Aviso analítico (No excluye registro)",
                mascara=m_eco2,
                excluye=False,
            )

        # Regla COH_SUE_01: Sueño fin de semana < entre semana - 1h (Plausible)
        sue_fin = get_numeric("sue_horas_finde")
        if sue_sem is not None and sue_fin is not None:
            m_sue1 = sue_sem.notna() & sue_fin.notna() & (sue_fin < (sue_sem - 1.0))
            registrar_regla(
                rule_id="COH_SUE_01",
                nombre="Menor descanso en fin de semana",
                variables=("sue_horas_finde", "sue_horas_semana"),
                severidad=Severity.INFORMATIVA,
                condicion_str="sue_horas_finde < (sue_horas_semana - 1)",
                descripcion="Duerme más de 1 hora menos en fin de semana (plausible por empleo nocturno u ocio)",
                accion="Aviso analítico (No excluye registro)",
                mascara=m_sue1,
                excluye=False,
            )

        df_alertas = (
            pd.DataFrame([a.__dict__ for a in alertas])
            if alertas
            else pd.DataFrame(
                columns=[
                    "regla_id", "comprobacion", "severidad", "n_casos",
                    "ids", "afecta_estado_global", "excluye_analisis_principal"
                ]
            )
        )
        df_catalogo = pd.DataFrame([r.__dict__ for r in reglas_cat])

        return df_alertas, df_catalogo

    def _clasificar_registros(
        self, df_original: pd.DataFrame, row_eval: dict[int, dict[str, Any]]
    ) -> tuple[pd.DataFrame, set[int]]:
        """
        Aplica la jerarquía determinista de clasificación y exclusión por fila:
          1. DUPLICADO
          2. NO_FIABLE (si tiene >=1 infracción crítica)
          3. EN_REVISION (si tiene advertencias o valores sospechosos)
          4. INCOMPLETO (si tiene campos obligatorios ausentes)
          5. VALIDO
        """
        clasificaciones: list[RecordClassification] = []
        idx_excluidos: set[int] = set()

        for idx in df_original.index:
            eval_data = row_eval[idx]
            r_id = eval_data["id"]

            criticos = eval_data["criticos"]
            sospechosos = eval_data["sospechosos"]
            ausentes = eval_data["ausentes"]
            no_aplicables = eval_data["no_aplicables"]
            es_dup = eval_data["es_duplicado"]
            motivos_todos = sorted(list(eval_data["motivos"]))

            # Prioridad determinista
            if es_dup:
                estado = RecordStatus.DUPLICADO
                excluido = True
                recom = "Descartado: fila duplicada"
            elif len(criticos) > 0:
                estado = RecordStatus.NO_FIABLE
                excluido = True
                recom = "Excluido: contiene anomalías físicas o lógicas críticas"
            elif len(sospechosos) > 0:
                estado = RecordStatus.EN_REVISION
                excluido = self.config.excluir_en_revision
                recom = "Revisión manual requerida (anomalías atípicas no críticas)"
            elif len(ausentes) > 0:
                estado = RecordStatus.INCOMPLETO
                excluido = False
                recom = "Admitido con reserva: faltan datos obligatorios para ciertos análisis"
            else:
                estado = RecordStatus.VALIDO
                excluido = False
                recom = "Admitido plenamente para análisis estadístico principal"

            if excluido:
                idx_excluidos.add(idx)

            clasificaciones.append(
                RecordClassification(
                    id=r_id,
                    estado_global=estado,
                    motivos="; ".join(motivos_todos) if motivos_todos else "Conforme a reglas",
                    n_invalidos=len(criticos),
                    n_sospechosos=len(sospechosos),
                    n_ausentes=len(ausentes),
                    n_no_aplicables=len(no_aplicables),
                    excluido=excluido,
                    recomendacion=recom,
                )
            )

        df_clasif = pd.DataFrame([c.__dict__ for c in clasificaciones])
        return df_clasif, idx_excluidos

    def _construir_reportes(
        self,
        df_original: pd.DataFrame,
        df_limpios: pd.DataFrame,
        df_excluidos: pd.DataFrame,
        df_clasificacion: pd.DataFrame,
        ledger: AuditLedger,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calcula métricas estadísticas dinámicas (sin N=1200 codificado)."""
        n_total = len(df_original)
        n_admitidos = len(df_limpios)
        n_excluidos = len(df_excluidos)

        conteo_estados = df_clasificacion["estado_global"].value_counts().to_dict()
        n_validos = conteo_estados.get(RecordStatus.VALIDO, 0)
        n_incompletos = conteo_estados.get(RecordStatus.INCOMPLETO, 0)
        n_revision = conteo_estados.get(RecordStatus.EN_REVISION, 0)
        n_no_fiables = conteo_estados.get(RecordStatus.NO_FIABLE, 0)
        n_duplicados = conteo_estados.get(RecordStatus.DUPLICADO, 0)

        tasa_fiabilidad = (
            round(((n_validos + n_incompletos + n_revision) / n_total) * 100.0, 2)
            if n_total > 0
            else 0.0
        )

        nulos_iniciales = int(df_original.isna().sum().sum())
        nulos_admitidos = int(df_limpios.isna().sum().sum())

        resumen = pd.DataFrame(
            {
                "concepto": [
                    "Total registros evaluados",
                    "Registros VÁLIDOS (completos y conformes)",
                    "Registros INCOMPLETOS (ausencias no críticas)",
                    "Registros EN REVISIÓN (atípicos / advertencias)",
                    "Registros NO FIABLES (excluidos por anomalía crítica)",
                    "Registros DUPLICADOS (excluidos por redundancia)",
                    "Total registros admitidos (Datos limpios)",
                    "Total registros excluidos (Datos excluidos)",
                    "Tasa de fiabilidad global del dataset (%)",
                    "Celdas vacías en dataset de entrada",
                    "Celdas vacías en conjunto de datos limpios",
                    "Total transformaciones registradas en auditoría",
                    "Política de recuperación de nota >10",
                    "Política de exclusión de registros en revisión",
                ],
                "valor": [
                    str(n_total),
                    str(n_validos),
                    str(n_incompletos),
                    str(n_revision),
                    str(n_no_fiables),
                    str(n_duplicados),
                    str(n_admitidos),
                    str(n_excluidos),
                    f"{tasa_fiabilidad}%",
                    str(nulos_iniciales),
                    str(nulos_admitidos),
                    str(len(ledger.to_dataframe())),
                    "Activa (÷10)" if self.config.recuperar_nota_x10 else "Inactiva (anulada a NaN)",
                    "Excluidos" if self.config.excluir_en_revision else "Admitidos con marca",
                ],
            }
        )

        # Cálculo de valores vacíos por variable
        faltantes = (
            (df_limpios.isna().mean() * 100.0)
            .round(1)
            .rename("pct_vacios")
            .reset_index()
            .rename(columns={"index": "variable"})
        )
        faltantes["n_vacios"] = df_limpios.isna().sum().values
        faltantes["es_obligatoria"] = faltantes["variable"].isin(self.config.columnas_obligatorias)
        faltantes = faltantes.sort_values("pct_vacios", ascending=False).reset_index(drop=True)

        return resumen, faltantes


# ============================================================================
# EXPORTADOR Y CAPA DE INFRAESTRUCTURA (EXCEL / CSV)
# ============================================================================

class ExcelReportWriter:
    """Exportador desacoplado a Excel multisheet y CSVs con formato formal."""

    @staticmethod
    def export(result: CleaningResult, output_dir: Path) -> None:
        """Escribe los 8 libros y artefactos CSV en disco."""
        output_dir.mkdir(parents=True, exist_ok=True)
        xlsx_path = output_dir / "ETAIC_Dataset_LIMPIO.xlsx"

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            result.datos_limpios.to_excel(writer, sheet_name="Datos_limpios", index=False)
            result.datos_excluidos.to_excel(writer, sheet_name="Datos_excluidos", index=False)
            result.clasificacion_registros.to_excel(writer, sheet_name="Clasificacion_registros", index=False)
            result.resumen.to_excel(writer, sheet_name="Resumen", index=False)
            result.reglas_aplicadas.to_excel(writer, sheet_name="Reglas_aplicadas", index=False)
            result.registro_cambios.to_excel(writer, sheet_name="Registro_cambios", index=False)
            result.avisos_coherencia.to_excel(writer, sheet_name="Avisos_coherencia", index=False)
            result.valores_vacios.to_excel(writer, sheet_name="Valores_vacios", index=False)

            # Auto-dimensionar columnas y congelar encabezados
            for ws in writer.book.worksheets:
                ws.freeze_panes = "B2"
                for col_cells in ws.columns:
                    first_cell = col_cells[0]
                    max_len = max(
                        len(str(c.value)) if c.value is not None else 0
                        for c in list(col_cells)[:60]
                    )
                    col_letter = first_cell.column_letter
                    ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 60)

        # Exportaciones CSV
        result.datos_limpios.to_csv(
            output_dir / "ETAIC_Dataset_LIMPIO.csv", index=False, encoding="utf-8-sig"
        )
        result.datos_excluidos.to_csv(
            output_dir / "ETAIC_Dataset_EXCLUIDOS.csv", index=False, encoding="utf-8-sig"
        )
        result.clasificacion_registros.to_csv(
            output_dir / "clasificacion_registros.csv", index=False, encoding="utf-8-sig"
        )
        result.registro_cambios.to_csv(
            output_dir / "registro_cambios.csv", index=False, encoding="utf-8-sig"
        )


# ============================================================================
# INTERFAZ DE LÍNEA DE COMANDOS (CLI)
# ============================================================================

def resolver_ruta_entrada(ruta_proporcionada: Path) -> Path | None:
    """Busca el archivo de origen en rutas relativas comunes."""
    if ruta_proporcionada.exists():
        return ruta_proporcionada

    candidatos = [
        Path("excel") / ruta_proporcionada.name,
        Path("excel") / "ETAIC_Dataset_VidaUniversitaria2026.xlsx",
        Path("ETAIC_Dataset_VidaUniversitaria2026.xlsx"),
    ]
    for c in candidatos:
        if c.exists():
            return c
    return None


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Pipeline de validación estricta, detección de no fiables y limpieza ETAIC 2026.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path("ETAIC_Dataset_VidaUniversitaria2026.xlsx"),
        help="Ruta al archivo Excel del dataset original.",
    )
    parser.add_argument(
        "--salida-dir",
        type=Path,
        default=Path("salida_limpieza"),
        help="Directorio destino para los libros limpios, excluidos y auditorías.",
    )
    parser.add_argument(
        "--recuperar-nota-x10",
        action="store_true",
        default=False,
        help="Recupera notas medias entre 10 y 100 dividiendo por 10.",
    )
    parser.add_argument(
        "--excluir-en-revision",
        action="store_true",
        default=False,
        help="Aplica política de exclusión estricta a registros EN_REVISION además de NO_FIABLE.",
    )
    return parser


def main() -> int:
    """Punto de entrada de ejecución del script."""
    parser = build_parser()
    args = parser.parse_args()

    input_path = resolver_ruta_entrada(args.entrada)
    if input_path is None:
        sys.stderr.write(
            f"Error crítico: Archivo de entrada no encontrado en '{args.entrada}'.\n"
        )
        return 1

    output_dir: Path = args.salida_dir

    print(f"Iniciando pipeline de calidad y validación sobre: {input_path.resolve()}")

    try:
        df_raw = pd.read_excel(input_path, sheet_name="Datos")
        try:
            df_dict = pd.read_excel(input_path, sheet_name="Diccionario")
        except Exception:
            df_dict = None
    except Exception as exc:
        sys.stderr.write(f"Error crítico de lectura Excel: {exc}\n")
        return 1

    config = CleaningConfig(
        recuperar_nota_x10=args.recuperar_nota_x10,
        excluir_en_revision=args.excluir_en_revision,
    )
    cleaner = DatasetCleaner(config=config)

    print("Ejecutando validación de límites físicos, rangos operativos, condicionales y coherencia...")
    resultado = cleaner.clean(df_raw=df_raw, df_dict=df_dict)

    print(f"Exportando resultados a: {output_dir.resolve()} ...")
    ExcelReportWriter.export(resultado, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("RESUMEN DE EJECUCIÓN Y CALIDAD DE DATOS")
    print("=" * 70)
    print(resultado.resumen.to_string(index=False))
    print("\nAVISOS DE INCOHERENCIA DETECTADOS:")
    if not resultado.avisos_coherencia.empty:
        print(resultado.avisos_coherencia[["regla_id", "comprobacion", "severidad", "n_casos"]].to_string(index=False))
    else:
        print("  Ninguna incoherencia detectada.")
    print(f"\n[✓] Proceso completado exitosamente en: {output_dir.resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
