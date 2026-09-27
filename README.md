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

Avance registrado al 27 de septiembre de 2026, según la memoria técnica y la confirmación de ejecución del autor:

- **Fase 0 - Laboratorio:** completada según las pruebas previas del entorno.
- **Fase 1 - Neurona artificial:** implementados los ejercicios de neurona básica, entrenamiento de AND y frontera de decisión.
- **Fase 2 - Red neuronal:** trabajada con una red XOR en NumPy y un ejercicio de backpropagation.
- **Fase 3 - PyTorch y GPU:** completada con tensores en CPU/GPU, XOR en PyTorch, `autograd`, gradientes reales, `optimizer.step()` y benchmarks CPU vs GPU.
- **Fase 4 - Tokenización, embeddings y predicción:** completada con siete ejercicios, desde texto e IDs hasta modelos de siguiente token con contextos de uno y dos tokens.
- **Fase 5 - Attention:** completada con siete ejercicios, desde atención para un token hasta Multi-Head Causal Self-Attention con dos cabezas y proyección de salida.
- **Fase 6 - Transformer:** completada con diez ejercicios: bloques con atención causal, embeddings posicionales, entrenamiento de MiniGPT, generación autoregresiva y recuperación desde un checkpoint.
- **Siguiente paso:** comenzar la Fase 7 con un corpus más amplio, separación de entrenamiento y validación, y seguimiento de pérdida y generaciones.

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
|   |-- fase3/
|   |   |-- 01_tensores_gpu.py
|   |   |-- 02_xor_pytorch.py
|   |   |-- 03_autograd_basico.py
|   |   |-- 04_gradientes_red.py
|   |   |-- 05_peso_antes_despues.py
|   |   |-- 06_cpu_vs_gpu.py
|   |   `-- 07_matrices_cpu_vs_gpu.py
|   |-- fase4/
|   |   |-- 01_tokenizacion_basica.py
|   |   |-- 02_encode_decode.py
|   |   |-- 03_embeddings.py
|   |   |-- 04_similitud_embeddings.py
|   |   |-- 05_contexto_siguiente_token.py
|   |   |-- 06_modelo_lenguaje_basico.py
|   |   `-- 07_contexto_dos_tokens.py
|   |-- fase5/
|   |   |-- 01_attention_intuicion.py
|   |   |-- 02_query_key_value.py
|   |   |-- 03_self_attention_todos_tokens.py
|   |   |-- 04_scaled_dot_product_attention.py
|   |   |-- 05_causal_attention.py
|   |   |-- 06_attention_aprendible.py
|   |   `-- 07_multi_head_attention.py
|   `-- fase6/
|       |-- 01_bloque_transformer.py
|       |-- 02_positional_embeddings.py
|       |-- 03_transformer_con_posicion.py
|       |-- 04_transformer_apilado.py
|       |-- 05_salida_vocabulario.py
|       |-- 06_entrenar_minigpt.py
|       |-- 07_generar_texto.py
|       |-- 08_cargar_modelo.py
|       |-- 09_checkpoint_completo.py
|       `-- 10_cargar_checkpoint.py
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

## Ejecutar la fase 4

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
python src/fase4/01_tokenizacion_basica.py
python src/fase4/02_encode_decode.py
python src/fase4/03_embeddings.py
python src/fase4/04_similitud_embeddings.py
python src/fase4/05_contexto_siguiente_token.py
python src/fase4/06_modelo_lenguaje_basico.py
python src/fase4/07_contexto_dos_tokens.py
```

Los ejercicios 01, 02 y 05 usan Python sin PyTorch. Los ejercicios 03 y 04 muestran embeddings aleatorios de cuatro dimensiones en CPU; 06 y 07 entrenan embeddings de ocho dimensiones y una capa lineal, seleccionando CUDA si está disponible o CPU en caso contrario.

El recorrido aprendido es `texto → tokens → IDs → embeddings → logits → probabilidades → siguiente token`. Durante el entrenamiento se pasan los logits directamente a `CrossEntropyLoss` y se actualizan tanto los embeddings como la capa de salida con Adam.

Un contexto como `el perro` admite `come` y `duerme` en el corpus: la red aprende una distribución sobre las continuaciones. La pérdida no tiene por qué llegar a cero. Estos modelos aún usan contexto fijo y no incorporan atención.

El detalle de los ejercicios, sus límites y el bloqueo de `shm.dll` resuelto durante la fase están en la [memoria técnica](documentation/code%20docs/02_Memoria_Tecnica.md#fase-4-tokenización-embeddings-y-predicción-de-siguiente-token).

## Ejecutar la fase 5

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
python src/fase5/01_attention_intuicion.py
python src/fase5/02_query_key_value.py
python src/fase5/03_self_attention_todos_tokens.py
python src/fase5/04_scaled_dot_product_attention.py
python src/fase5/05_causal_attention.py
python src/fase5/06_attention_aprendible.py
python src/fase5/07_multi_head_attention.py
```

Los siete ejercicios usan PyTorch en CPU y embeddings definidos manualmente. El recorrido es `Q/K/V → QKᵀ → escalado → máscara causal → softmax → combinación de V`. El último ejercicio combina dos cabezas, concatena sus salidas y aplica una proyección final.

Con tres tokens, `embedding_dim=4` y `num_heads=2`, cada cabeza trabaja con `head_dim=2`. La salida final conserva la forma `[3, 4]` y los tokens futuros reciben peso cero.

Los módulos de los ejercicios 06 y 07 tienen parámetros entrenables, pero estos scripts solo ejecutan el forward: no incluyen pérdida, backpropagation ni optimizador. Las distribuciones distintas de las cabezas muestran el funcionamiento del mecanismo con su inicialización aleatoria, no una especialización aprendida.

Los archivos, dimensiones y resultados reportados se detallan en la [memoria técnica](documentation/code%20docs/02_Memoria_Tecnica.md#fase-5-attention).

## Ejecutar la fase 6

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
python src/fase6/01_bloque_transformer.py
python src/fase6/02_positional_embeddings.py
python src/fase6/03_transformer_con_posicion.py
python src/fase6/04_transformer_apilado.py
python src/fase6/05_salida_vocabulario.py
python src/fase6/06_entrenar_minigpt.py
python src/fase6/07_generar_texto.py
python src/fase6/08_cargar_modelo.py
python src/fase6/09_checkpoint_completo.py
python src/fase6/10_cargar_checkpoint.py
```

Los ejercicios 01–05 muestran la arquitectura en CPU sin entrenarla. Los scripts 06, 07 y 09 entrenan cada uno un modelo nuevo durante 2000 épocas; 06–10 seleccionan CUDA si está disponible o CPU en caso contrario.

MiniGPT combina embeddings de tokens y posiciones, dos bloques Transformer con dos cabezas y una capa de salida hacia cinco palabras. Usa dimensión de embedding 8, contexto máximo de 20 posiciones, `CrossEntropyLoss` y Adam con `lr=0.01`.

El script 07 guarda `minigpt.pth` y 08 lo carga; 09 guarda `minigpt_checkpoint.pth` con pesos, configuración y vocabulario, y 10 lo recupera. Los archivos se escriben en el directorio desde el que se ejecutan los comandos y se sobrescriben al repetir el guardado. Ejecutar cada cargador después de su script de guardado, desde el mismo directorio. Los cargadores generan sin volver a entrenar.

El resultado reportado por el autor, tanto después de entrenar como al recuperar el modelo, fue:

```text
el perro come el gato duerme
```

La generación comienza con `el` y añade cinco tokens mediante `argmax`. El experimento demuestra aprendizaje y persistencia sobre un corpus de seis tokens; todavía no demuestra comprensión del español ni generalización a textos nuevos. Los resultados compartidos, las dimensiones y los límites están en la [memoria técnica](documentation/code%20docs/02_Memoria_Tecnica.md#fase-6-transformer).

## Roadmap

| Fase | Tema | Estado |
| --- | --- | --- |
| 0 | Preparar el laboratorio | Completada |
| 1 | Neurona artificial desde cero | Ejercicios implementados |
| 2 | Red neuronal desde cero | Trabajada |
| 3 | PyTorch y entrenamiento con GPU | Completada |
| 4 | Tokenización, embeddings y predicción de siguiente token | Completada |
| 5 | Self-attention causal y multi-head attention | Completada |
| 6 | Construir un Transformer pequeño | Completada |
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
