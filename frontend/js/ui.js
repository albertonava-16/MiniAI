
/**
 * MiniAI - Control de interfaz
 * Ejercicio 20.2
 */

const mensajes = document.getElementById("messages");
const estadoServidor = document.getElementById("server-status");
const textoEstado = document.getElementById("status-text");
const estadoFooter = document.getElementById("footer-status");
const dispositivo = document.getElementById("device");
const latencia = document.getElementById("latency");
const botonEnviar = document.getElementById("send-button");


export function agregarMensaje(texto, tipo) {
    const bienvenida = document.getElementById("empty-state");

    if (bienvenida) {
        bienvenida.remove();
    }

    const elemento = document.createElement("div");

    elemento.className = `message message-${tipo}`;
    elemento.textContent = texto;

    mensajes.appendChild(elemento);
    mensajes.scrollTo({
    top: mensajes.scrollHeight,
    behavior: "smooth"
});
}


export function actualizarEstado(conectado, datos = null) {
    const mensaje = conectado ? "Online" : "Offline";

    textoEstado.textContent = mensaje;
    estadoFooter.textContent = mensaje;

    estadoServidor.classList.toggle(
        "online",
        conectado
    );

    estadoServidor.classList.toggle(
        "offline",
        !conectado
    );

    dispositivo.textContent = (
        conectado && datos
            ? datos.dispositivo
            : "—"
    );
}


export function actualizarLatencia(valor) {
    latencia.textContent = `${valor} ms`;
}


export function establecerCargando(cargando) {
    botonEnviar.disabled = cargando;

    botonEnviar.textContent = (
        cargando ? "Generando..." : "Generar →"
    );
}


export function limpiarHistorial() {
    mensajes.replaceChildren();

    latencia.textContent = "—";

    const bienvenida = document.createElement("div");
    bienvenida.className = "empty-state";
    bienvenida.id = "empty-state";

    const titulo = document.createElement("h3");
    titulo.textContent = "Historial limpio";

    const descripcion = document.createElement("p");
    descripcion.textContent = (
        "Escribe un nuevo prompt para generar texto."
    );

    bienvenida.append(titulo, descripcion);
    mensajes.appendChild(bienvenida);
}
