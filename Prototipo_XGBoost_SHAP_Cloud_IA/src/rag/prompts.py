# TODO Paso 11: plantillas por rol (Familia/Docente/Terapeuta)
def plantilla_por_rol(rol, perfil, shap_top, chunks, trigger):
    return f"""[ROL: {rol}] Trigger: {trigger}
Perfil: {perfil}
Drivers: {shap_top}
Citas: {len(chunks)}
(Reemplazar con prompt real)"""
