
/**
 * MiniAI - Aplicación principal
 * Ejercicio 20.2
 */

import {
    consultarEstado,
    enviarPrompt
} from "./api.js";

import {
    agregarMensaje,
    actualizarEstado,
    actualizarLatencia,
    establecerCargando,
    limpiarHistorial
} from "./ui.js";


const formulario = document.getElementById("prompt-form");
const entrada = document.getElementById("prompt-input");

const temperature = document.getElementById("temperature");
const topP = document.getElementById("top-p");
const topK = document.getElementById("top-k");
const maxTokens = document.getElementById("max-tokens");

const temperatureValor = document.getElementById("temperature-value");
const topPValor = document.getElementById("top-p-value");

const botonLimpiar = document.getElementById("clear-history");

let generando = false;


// ============================================================
// CONTROLES
// ============================================================

temperature.addEventListener("input", () => {
    temperatureValor.textContent = temperature.value;
});

topP.addEventListener("input", () => {
    topPValor.textContent = topP.value;
});


// ============================================================
// ESTADO DEL SERVIDOR
// ============================================================

async function verificarServidor() {
    try {
        const datos = await consultarEstado();
        actualizarEstado(true, datos);

        return true;
    } catch (error) {
        console.error("Error consultando /health:", error);
        actualizarEstado(false);

        return false;
    }
}


// ============================================================
// GENERACIÓN
// ============================================================

formulario.addEventListener("submit", async (evento) => {
    evento.preventDefault();

    if (generando) {
        return;
    }

    const prompt = entrada.value.trim();

    if (!prompt) {
        return;
    }

    const parametros = {
        prompt: prompt,
        max_tokens: Number(maxTokens.value),
        temperature: Number(temperature.value),
        top_k: Number(topK.value),
        top_p: Number(topP.value)
    };

    generando = true;
    establecerCargando(true);

    agregarMensaje(prompt, "user");

    try {
        const resultado = await enviarPrompt(parametros);

        agregarMensaje(resultado.respuesta, "ai");

        actualizarLatencia(resultado.latencia_ms);

        actualizarEstado(true, {
            dispositivo: resultado.dispositivo
        });

        entrada.value = "";

    } catch (error) {
        agregarMensaje(error.message, "error");

        // Un 422 no significa que el servidor esté desconectado.
        // Comprobamos /health antes de actualizar su estado.
        await verificarServidor();

    } finally {
        generando = false;
        establecerCargando(false);
        entrada.focus();
    }
});


// ============================================================
// HISTORIAL
// ============================================================

botonLimpiar.addEventListener("click", () => {
    limpiarHistorial();
});


// ============================================================
// INICIALIZACIÓN
// ============================================================

verificarServidor();
