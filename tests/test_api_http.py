
import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE_URL = "http://127.0.0.1:8000"


def solicitar(ruta, metodo="GET", datos=None):
    contenido = None
    headers = {}

    if datos is not None:
        contenido = json.dumps(datos).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        BASE_URL + ruta,
        data=contenido,
        headers=headers,
        method=metodo
    )

    try:
        with urlopen(request, timeout=30) as respuesta:
            return respuesta.status, json.load(respuesta)

    except HTTPError as error:
        return error.code, json.load(error)


def main():
    # 1. Comprobar disponibilidad
    status, datos = solicitar("/health")

    assert status == 200
    assert datos["status"] == "ok"
    print("GET /health: OK")

    # 2. Generación válida
    status, datos = solicitar(
        "/chat",
        metodo="POST",
        datos={
            "prompt": "el perro",
            "max_tokens": 12,
            "temperature": 0.7,
            "top_k": 3,
            "top_p": 0.9
        }
    )

    assert status == 200
    assert datos["respuesta"].startswith("el perro")
    assert datos["latencia_ms"] >= 0
    print("POST /chat: OK")
    print("Generación:", datos["respuesta"])

    # 3. Token desconocido
    status, datos = solicitar(
        "/chat",
        metodo="POST",
        datos={"prompt": "el dinosaurio"}
    )

    assert status == 422
    assert "detail" in datos
    print("Token desconocido: OK")

    # 4. Parámetros inválidos
    status, datos = solicitar(
        "/chat",
        metodo="POST",
        datos={
            "prompt": "el perro",
            "temperature": -1
        }
    )

    assert status == 422
    print("Validación de parámetros: OK")

    print("\nPRUEBAS HTTP COMPLETADAS")


if __name__ == "__main__":
    main()
