# 📑 Informe Ejecutivo – Fuga de Clientes **TelecomX LATAM**

> **Fecha:** Mayo 2025 • **Autor:** Juan José Ramos

---

## 1 · Resumen ejecutivo 📊

| KPI | Valor | Observación |
|-----|------:|------------|
| Clientes analizados | **7 043** | Base limpia (`df_clean`) |
| Tasa global de bajas | **26 %** | 1 810 clientes se dieron de baja |
| ROC‑AUC modelo LightGBM | **0,81** | Buen poder discriminativo |
| Segmento crítico (SC) | **9,6 %** de la base | 63 % de prob. de baja, ARPU alto |

> **Conclusión rápida:** La fuga se concentra en clientes _month‑to‑month_ 🌙, con Internet fibra óptica y pago por _electronic check_ (SC). Retener ese 9,6 % impacta ≈12,4 % de los ingresos anuales.

---

## 2 · Hallazgos clave 🔍

1. **Contrato & Tenure**  
   * Churn del **43 %** en _month‑to‑month_ (≤ 12 meses de antigüedad).  
   * Contratos de 1‑2 años presentan solo 11 % de bajas.
2. **Método de pago**  
   * _Electronic check_ duplica la fuga (36 %) frente a tarjetas automáticas (18 %).
3. **Servicio de Internet**  
   * Fibra óptica muestra 8 p.p. más de bajas que DSL.  
   * La combinación fibra + _EC_ eleva el riesgo a **52 %**.
4. **Valor (ARPU)**  
   * Clientes que se fugan pagan **US$ 80** (mediana) versus US$ 65 de clientes activos.

---

## 3 · Medidas preventivas 🚨🛠️

| # | Acción | Segmento objetivo | KPI esperado |
|:-:|--------|-------------------|--------------|
| 1 | **Migrar a contrato anual** con descuento de 12 % | SC<br>(MTM ≤ 12m) | Reducir bajas ↘ 15 p.p. |
| 2 | Incentivar **pago por tarjeta** (debito/crédito) | EC users | ↘ 9 p.p. de bajas en EC |
| 3 | Paquete **“Fibra + Streaming”** con upgrade gratuito 3 m | Fibra, Tenure < 6m | Incrementar fidelidad 6 m |
| 4 | **Campaña proactiva** de soporte (call‑out) | Predicción `p_baja > 0,60` | 25 % contacto efectivo |
| 5 | **Feedback loop** mensual al modelo | Toda la base | Mantener ROC‑AUC > 0,80 |

> 🎯 **Prioridad:** Acciones 1 y 2 entregan el mayor _ROI_ inmediato sobre ingresos recurrentes.
>
> ## 4 · Roadmap de seguimiento 🔄

1. **Despliegue cloud** del dashboard en Azure Web Apps.  
2. **Automatizar scoring semanal** (GitHub Actions + Streamlit Cache).  
3. **Integración CRM** para triggers de marketing.  
4. **A/B Testing** de ofertas (MTM → Anual) medido por retención a 90 días.

## 5 · Dashboard 🔄

[📊 Dashboard](https://challenge2-jjramoss.streamlit.app/)

## 6 · Contacto

juan.ramos.s@usach.cl
