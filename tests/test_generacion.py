
from miniai.inferencia import cargar_modelo
from miniai.generacion import generar


def main():

    modelo, checkpoint, device = cargar_modelo()

    print("Dispositivo:", device)
    print("Modelo cargado correctamente.\n")

    # Experimento A
    respuesta = generar(
        modelo,
        checkpoint,
        device,
        prompt="el perro",
        temperature=0.7,
        top_k=3,
        top_p=0.9
    )

    print("Experimento A")
    print("Generación:", respuesta)

    assert respuesta.startswith("el perro")

    # Experimento B
    respuesta = generar(
        modelo,
        checkpoint,
        device,
        prompt="la niña",
        temperature=1.0,
        top_p=0.9
    )

    print("\nExperimento B")
    print("Generación:", respuesta)

    assert respuesta.startswith("la niña")

    # Experimento C: token desconocido
    print("\nExperimento C")

    try:
        generar(
            modelo,
            checkpoint,
            device,
            prompt="el dinosaurio"
        )
    except ValueError as error:
        assert "Tokens desconocidos" in str(error)
        print("Error controlado:", error)
    else:
        raise AssertionError(
            "Se esperaba rechazar el token desconocido"
        )

    print("\nPRUEBA COMPLETADA")


if __name__ == "__main__":
    main()
