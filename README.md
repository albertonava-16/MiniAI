# MiniAI

MiniAI es un laboratorio educativo para construir una inteligencia artificial pequeña desde cero y entender qué ocurre detrás de una red neuronal, un Transformer y un sistema de herramientas.

El proyecto avanza desde la matemática fundamental hasta una arquitectura capaz de ofrecer inferencia, RAG, tools y MCP. La prioridad es comprender cada componente antes de ocultarlo detrás de abstracciones o frameworks.

## Objetivos

- Entender cómo una neurona aprende ajustando pesos y bias.
- Construir una red neuronal y un Transformer pequeño.
- Comparar implementaciones manuales con PyTorch.
- Entrenar e inferir localmente usando la GPU cuando sea apropiado.
- Convertir el modelo en un servicio mediante una API.
- Separar claramente entrenamiento, inferencia, RAG, tool calling y MCP.
- Documentar experimentos, métricas, decisiones y errores relevantes.

## Estado actual

- **Fase 0 - Laboratorio:** completada.
- **Fase 1 - Neurona artificial:** en progreso.
- **Implementado:** una neurona con función sigmoide y un ciclo de entrenamiento para aprender la función lógica AND.
- **Siguiente paso:** ampliar el experimento hacia una red neuronal con varias capas.

## Arquitectura objetivo

```text
Usuario
  |
Chat / interfaz
  |
FastAPI
  |
MiniAI
  |
Transformer
  |
PyTorch / CUDA
  |
GPU local
```

En fases posteriores se añadiran RAG, tools y MCP:

```text
MiniAI
  |-- RAG ------> documentos
  |-- Tools ----> funciones y APIs
  `-- MCP ------> servicios externos
```

## Estructura del repositorio

```text
MiniAI/
|-- datasets/                 # Datos y corpus para los experimentos
|-- documentation/            # Plan de trabajo y memoria tecnica
|-- models/                   # Modelos y checkpoints generados
|-- notebooks/                # Experimentos interactivos
|-- src/                      # Codigo fuente organizado por fases
|   `-- fase1/
|       |-- 01_neurona_basica.py
|       `-- 02_neurona_entrenamiento.py
|-- .venv/                    # Entorno virtual local, no versionar
|-- .gitignore
`-- README.md
```

## Requisitos

- Windows 11
- Python 3.13 o compatible
- Git
- Visual Studio Code
- GPU NVIDIA compatible con CUDA para los experimentos que usen aceleracion

La configuracion de hardware y las comprobaciones del entorno estan documentadas en [documentation/MiniAI_Memoria_Tecnica.txt](documentation/MiniAI_Memoria_Tecnica.txt).

## Instalacion

Desde Git Bash:

```bash
git clone <URL_DEL_REPOSITORIO>
cd MiniAI
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install numpy matplotlib
```

Para instalar PyTorch con CUDA, usa el indice correspondiente a tu hardware y a la version disponible en la [pagina oficial de PyTorch](https://pytorch.org/get-started/locally/). La instalacion utilizada durante la fase 0 fue:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

Comprueba que PyTorch detecta CUDA:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No detectada')"
```

## Ejecutar la fase 1

Con el entorno virtual activo:

```bash
python src/fase1/01_neurona_basica.py
python src/fase1/02_neurona_entrenamiento.py
```

El primer script muestra las predicciones y el error de una neurona con pesos definidos. El segundo entrena los pesos durante varias iteraciones y vuelve a evaluar las entradas de la función AND.

## Roadmap

| Fase | Tema | Estado |
| --- | --- | --- |
| 0 | Preparar el laboratorio | Completada |
| 1 | Neurona artificial desde cero | En progreso |
| 2 | Red neuronal desde cero | Pendiente |
| 3 | PyTorch y entrenamiento con GPU | Pendiente |
| 4 | Tokenización y embeddings | Pendiente |
| 5 | Self-attention | Pendiente |
| 6 | Construir un Transformer pequeño | Pendiente |
| 7 | Entrenar MiniAI | Pendiente |
| 8 | Convertir MiniAI en servicio | Pendiente |
| 9 | RAG y tools | Pendiente |
| 10 | MCP y arquitectura distribuida | Pendiente |
| 11 | Consolidación y documentación | Pendiente |

El detalle de tareas, checkpoints y conceptos de cada fase está en [documentation/MiniAI_Plan_de_Trabajo.txt](documentation/MiniAI_Plan_de_Trabajo.txt).

## Principios del proyecto

1. Entender antes de abstraer.
2. Medir loss, tiempo, uso de VRAM y resultados.
3. Mantener los modelos pequeños y experimentales.
4. Vigilar el consumo de memoria y la temperatura del hardware.
5. Separar entrenamiento, inferencia, recuperación y ejecución de herramientas.
6. Registrar los errores y aprendizajes que ayuden a retomar el proyecto.

## Licencia

Este proyecto todavía no define una licencia publica.