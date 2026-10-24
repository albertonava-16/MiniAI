
/**
 * MiniAI - Cliente HTTP
 * Ejercicio 20.2
 */

async function solicitarJSON(ruta, opciones = {}) {
    const respuesta = await fetch(ruta, opciones);

    const datos = await respuesta.json();

    if (!respuesta.ok) {
        let mensaje = `Error HTTP ${respuesta.status}`;

        if (typeof datos.detail === "string") {
            mensaje = datos.detail;
        } else if (datos.detail) {
            mensaje = JSON.stringify(datos.detail);
        }

        throw new Error(mensaje);
    }

    return datos;
}


export async function consultarEstado() {
    return solicitarJSON("/health");
}


export async function enviarPrompt(parametros) {
    return solicitarJSON("/chat", {
        method: "POST",

        headers: {
            "Content-Type": "application/json"
        },

        body: JSON.stringify(parametros)
    });
}
