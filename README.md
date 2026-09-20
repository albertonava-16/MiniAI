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

Avance registrado al 19 de septiembre de 2026, según la memoria técnica:

- **Fase 0 - Laboratorio:** completada según las pruebas previas del entorno.
- **Fase 1 - Neurona artificial:** implementados los ejercicios de neurona básica, entrenamiento de AND y frontera de decisión.
- **Fase 2 - Red neuronal:** trabajada con una red XOR en NumPy y un ejercicio de backpropagation.
- **Fase 3 - PyTorch y GPU:** completada con tensores en CPU/GPU, XOR en PyTorch, `autograd`, gradientes reales, `optimizer.step()` y benchmarks CPU vs GPU.
- **Siguiente paso:** pasar a la Fase 4 para tokenización y embeddings.

## Documentación

Los documentos siguen un formato común en Markdown, con secciones, ejemplos y enlaces de consulta:

- [Memoria técnica](documentation/code%20docs/02_Memoria_Tecnica.md): entorno, comandos, diagnóstico y punto para retomar.
- [Plan de trabajo](documentation/code%20docs/01_Plan_de_Trabajo.md): fases, tareas, criterios de salida y estimaciones.
- [Breviario de conceptos](documentation/code%20docs/00_breviario.md): definiciones y ejemplos de los conceptos aprendidos.

La memoria técnica y el plan se convirtieron de texto plano a Markdown. Las próximas actualizaciones se registrarán en estos archivos `.md`.

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
|-- documentation/            # Plan, memoria técnica y breviario
|   `-- code docs/
|       |-- 00_breviario.md
|       |-- 01_Plan_de_Trabajo.md
|       `-- 02_Memoria_Tecnica.md
|-- src/                      # Código fuente organizado por fases
|   |-- fase1/
|   |   |-- 01_neurona_basica.py
|   |   |-- 02_neurona_entrenamiento.py
|   |   `-- 03_frontera_decision.py
|   |-- fase2/
|   |   |-- 01_red_xor.py
|   |   `-- 02_backprop_xor.py
|   `-- fase3/
|       |-- 01_tensores_gpu.py
|       |-- 02_xor_pytorch.py
|       |-- 03_autograd_basico.py
|       |-- 04_gradientes_red.py
|       |-- 05_peso_antes_despues.py
|       |-- 06_cpu_vs_gpu.py
|       `-- 07_matrices_cpu_vs_gpu.py
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

La configuración de hardware y las comprobaciones previas del entorno están documentadas en la [memoria técnica](documentation/code%20docs/02_Memoria_Tecnica.md).

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

## Retomar la fase 2

Con el entorno virtual activo, ejecutar primero el ejercicio de backpropagation y después la red completa:

```bash
python src/fase2/02_backprop_xor.py
python src/fase2/01_red_xor.py
```

Ambos ejercicios usan NumPy en CPU y no requieren CUDA. La gráfica de la red todavía muestra datos de prueba de Matplotlib; conectarla con `loss_history` es uno de los pendientes. Cerrar la ventana de la gráfica permite continuar con las impresiones posteriores a `plt.show()`.

## Ejecutar la fase 3

Con el entorno virtual activo y CUDA disponible:

```bash
python src/fase3/01_tensores_gpu.py
python src/fase3/02_xor_pytorch.py
python src/fase3/03_autograd_basico.py
python src/fase3/04_gradientes_red.py
python src/fase3/05_peso_antes_despues.py
python src/fase3/06_cpu_vs_gpu.py
python src/fase3/07_matrices_cpu_vs_gpu.py
```

Esta fase reimplementa XOR con PyTorch, usa `nn.Module`, `nn.Linear`, `MSELoss`, `loss.backward()` y `optimizer.step()`. También compara CPU y GPU; en matrices de `3000x3000`, la RTX 5050 fue aproximadamente **11-14x** más rápida que la CPU.

## Roadmap

| Fase | Tema | Estado |
| --- | --- | --- |
| 0 | Preparar el laboratorio | Completada |
| 1 | Neurona artificial desde cero | Ejercicios implementados |
| 2 | Red neuronal desde cero | Trabajada |
| 3 | PyTorch y entrenamiento con GPU | Completada |
| 4 | Tokenización y embeddings | Pendiente |
| 5 | Self-attention | Pendiente |
| 6 | Construir un Transformer pequeño | Pendiente |
| 7 | Entrenar MiniAI | Pendiente |
| 8 | Convertir MiniAI en servicio | Pendiente |
| 9 | RAG y tools | Pendiente |
| 10 | MCP y arquitectura distribuida | Pendiente |
| 11 | Consolidación y documentación | Pendiente |

El detalle de tareas, criterios de salida y conceptos de cada fase está en el [plan de trabajo](documentation/code%20docs/01_Plan_de_Trabajo.md).

## Principios del proyecto

1. Entender antes de abstraer.
2. Medir loss, tiempo, uso de VRAM y resultados.
3. Mantener los modelos pequeños y experimentales.
4. Vigilar el consumo de memoria y la temperatura del hardware.
5. Separar entrenamiento, inferencia, recuperación y ejecución de herramientas.
6. Registrar los errores y aprendizajes que ayuden a retomar el proyecto.

## Licencia

Este proyecto todavía no define una licencia publica.
