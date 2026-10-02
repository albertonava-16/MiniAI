# Plan de trabajo de MiniAI

Este documento organiza el recorrido de MiniAI: desde una neurona artificial hasta un sistema que genere texto, ofrezca una API y utilice información y herramientas externas.

La intención es avanzar por fases, entender cada componente y registrar resultados antes de pasar al siguiente nivel.

**Duración estimada total:** 44–60 horas, según la suma de las fases.  
**Último avance registrado:** 1 de octubre de 2026.
**Reorganización de la documentación:** 19 de septiembre de 2026.  
**Punto para retomar:** Fase 8, separar la inferencia y exponer MiniAI como servicio.

Documentos relacionados: [memoria técnica](02_Memoria_Tecnica.md), [breviario de conceptos](00_breviario.md) y [README](../../README.md).

Documento convertido a partir de `MiniAI_Plan_de_Trabajo.txt`. Las próximas actualizaciones se registrarán en esta versión Markdown.

---

## Contenido

- [Objetivo general](#objetivo-general)
- [Metodología](#metodología)
- [Estado actual y siguiente sesión](#estado-actual-y-siguiente-sesión)
- [Resumen de fases y duración](#resumen-de-fases-y-duración)
- [Arquitectura objetivo](#arquitectura-objetivo)
- [Fase 0: Preparar el laboratorio](#fase-0-preparar-el-laboratorio)
- [Fase 1: Crear una neurona artificial desde cero](#fase-1-crear-una-neurona-artificial-desde-cero)
- [Fase 2: Crear una red neuronal desde cero](#fase-2-crear-una-red-neuronal-desde-cero)
- [Fase 3: PyTorch y entrenamiento con GPU](#fase-3-pytorch-y-entrenamiento-con-gpu)
- [Fase 4: Tokenización y embeddings](#fase-4-tokenización-y-embeddings)
- [Fase 5: Self-attention](#fase-5-self-attention)
- [Fase 6: Construir nuestro Transformer](#fase-6-construir-nuestro-transformer)
- [Fase 7: Entrenar MiniAI](#fase-7-entrenar-miniai)
- [Fase 8: Convertir MiniAI en servicio](#fase-8-convertir-miniai-en-servicio)
- [Fase 9: RAG y herramientas](#fase-9-rag-y-herramientas)
- [Fase 10: MCP y arquitectura distribuida](#fase-10-mcp-y-arquitectura-distribuida)
- [Fase 11: Consolidación y documentación](#fase-11-consolidación-y-documentación)
- [Reglas del proyecto](#reglas-del-proyecto)
- [Notas de la reorganización](#notas-de-la-reorganización)

---

## Objetivo general

Construir una IA pequeña desde cero para entender técnicamente cómo funcionan las redes neuronales, el entrenamiento, los Transformers, la generación de texto, la inferencia, RAG, las herramientas y MCP.

El proyecto cubre tanto la capa matemática del modelo como la capa de software e integración.

---

## Metodología

- Avanzar por fases.
- Entender qué hacen las abstracciones antes de utilizarlas.
- Construir primero versiones simples y educativas.
- Medir resultados en cada fase.
- Mantener todo versionado en Git.
- Documentar decisiones y aprendizajes.
- Priorizar la seguridad del hardware y el uso razonable de la GPU.

Cada fase incluye un objetivo, tareas, conceptos clave, un criterio de salida y un resultado esperado. Los criterios de salida sirven para comprobar el aprendizaje antes de avanzar.

---

## Estado actual y siguiente sesión

El avance se toma de la [memoria técnica](02_Memoria_Tecnica.md#avance-y-punto-para-retomar), cuyo último registro corresponde al **1 de octubre de 2026**.

| Fase | Avance registrado |
| --- | --- |
| 0. Laboratorio | Completada según las pruebas previas del entorno. |
| 1. Neurona artificial | Implementados los ejercicios de neurona básica, entrenamiento de AND y frontera de decisión. |
| 2. Red neuronal | Trabajada: red XOR con NumPy y ejercicio de backpropagation. |
| 3. PyTorch y GPU | Completada: tensores, CUDA, XOR en PyTorch, gradientes reales, optimizador y benchmark CPU vs GPU. |
| 4. Tokenización y embeddings | Completada: siete ejercicios ejecutados por el autor, incluyendo predicción con uno y dos tokens de contexto. |
| 5. Attention | Completada: siete ejercicios, máscara causal, proyecciones entrenables y dos cabezas con proyección final. |
| 6. Transformer | Completada: diez ejercicios, MiniGPT entrenable, generación autoregresiva y persistencia con pesos, configuración y vocabulario. |
| 7. Entrenar MiniAI | Completada: diez ejercicios con train/validation, batches, regularización, early stopping, BOS/EOS, sampling y checkpoint final. |
| 8–11 | Pendientes en el registro de avance. |

Los estados describen el avance documentado. La Fase 3 conserva el resultado observado del benchmark de matrices. La Fase 4 se registra como completada a partir de la confirmación del autor y la revisión de los siete archivos de `src/fase4`; no se aportaron valores exactos de pérdida ni tiempos de ejecución. La Fase 5 se cierra con la confirmación del autor, sus resultados de ejemplo y la revisión de los siete scripts de `src/fase5`; sus módulos de atención se ejecutan sin un ciclo de entrenamiento. La Fase 6 se cierra con el contexto y los resultados compartidos por el autor y la revisión de los diez scripts de `src/fase6`. La Fase 7 se cierra con la revisión de sus diez scripts y la última ejecución reportada por el autor; no se volvió a entrenar el modelo durante esta actualización documental.

### Primera tarea al retomar

Comenzar la [Fase 8](#fase-8-convertir-miniai-en-servicio) a partir del checkpoint final de Fase 7.

- [ ] Extraer las clases del modelo y la función de generación a módulos reutilizables.
- [ ] Crear un cargador para `miniai_fase7_checkpoint.pth` que reconstruya el modelo sin entrenarlo.
- [ ] Definir un endpoint de inferencia con FastAPI y un esquema de entrada/salida.
- [ ] Validar prompts, parámetros de sampling y tokens desconocidos.
- [ ] Medir latencia y comprobar una petición local de extremo a extremo.

La ubicación del checkpoint final y el comando para reproducirlo están en [Inicio rápido de la memoria técnica](02_Memoria_Tecnica.md#inicio-rápido). La carga independiente sin reentrenar se implementará al comenzar la Fase 8.

---

## Resumen de fases y duración

Los tiempos son estimaciones de dedicación por fase, no horas medidas ni fechas de entrega.

| Fase | Tema | Estimación | Estado registrado |
| ---: | --- | ---: | --- |
| 0 | Preparar el laboratorio | 2–3 h | Completada |
| 1 | Neurona artificial | 3–4 h | Ejercicios implementados |
| 2 | Red neuronal | 4–5 h | Trabajada |
| 3 | PyTorch y GPU | 3–4 h | Completada |
| 4 | Tokenización, embeddings y predicción de siguiente token | 4–5 h | Completada |
| 5 | Self-attention causal y multi-head attention | 5–6 h | Completada |
| 6 | Transformer | 6–8 h | Completada |
| 7 | Entrenamiento de MiniAI | 4–6 h | Completada |
| 8 | API e interfaz | 3–4 h | Pendiente |
| 9 | RAG y herramientas | 4–5 h | Pendiente |
| 10 | MCP y arquitectura distribuida | 4–6 h | Pendiente |
| 11 | Consolidación y documentación | 2–4 h | Pendiente |
| **Total** | | **44–60 h** | |

---

## Arquitectura objetivo

El modelo se integrará progresivamente con una interfaz y una API:

```text
Usuario
  ↓
Interfaz / Chat
  ↓
FastAPI
  ↓
MiniAI
  ↓
Transformer
  ↓
PyTorch / CUDA
  ↓
RTX 5050
```

En las fases posteriores se incorporarán recuperación de información, herramientas y MCP:

```text
MiniAI
  ├── RAG → documentos y contexto recuperado
  ├── Tools → funciones y APIs
  └── Cliente MCP → servidor MCP → archivos, bases de datos y servicios
```

Estos diagramas representan la arquitectura prevista, no componentes que ya estén implementados.

---

## Fase 0: Preparar el laboratorio

**Estado:** completada según el registro previo.  
**Tiempo estimado:** 2–3 horas.

### Objetivo

Dejar listo un entorno reproducible para desarrollar, entrenar y probar MiniAI.

### Tareas

- Crear el repositorio MiniAI.
- Crear un entorno virtual de Python.
- Verificar Python y pip.
- Instalar PyTorch, NumPy y Matplotlib.
- Verificar el soporte CUDA y la arquitectura de la GPU.
- Confirmar que PyTorch detecta la RTX 5050.
- Ejecutar una operación tensorial en GPU.
- Abrir el proyecto en VS Code.
- Preparar la estructura básica del repositorio.

### Estructura inicial sugerida

Esta es la estructura propuesta en el plan original. El detalle de los archivos de trabajo está en la [memoria técnica](02_Memoria_Tecnica.md#estructura-del-proyecto).

```text
MiniAI/
├── datasets/
├── models/
├── notebooks/
├── src/
├── .venv/
├── .gitignore
└── README.md
```

### Conceptos clave

Entorno virtual, dependencias, CPU y GPU, CUDA, tensores y VRAM.

### Criterio de salida

- Python operativo y entorno virtual activo.
- Git funcional.
- PyTorch instalado y CUDA disponible.
- RTX 5050 visible desde PyTorch.
- Primera operación ejecutada en `cuda:0`.

### Resultado esperado

Laboratorio listo para trabajar con IA localmente.

### Registro del hito completado

El hito original confirmó Python `3.13.15`, pip `26.2.1`, el entorno `.venv`, PyTorch `2.14.0+cu132`, CUDA `13.2`, la arquitectura `sm_120`, la detección de la NVIDIA GeForce RTX 5050 Laptop GPU y operaciones en GPU funcionando.

Las versiones, los comandos y las salidas de esas pruebas se conservan en la [memoria técnica](02_Memoria_Tecnica.md#software-y-versiones-registradas). Son comprobaciones previas, no resultados nuevos de esta reorganización.

---

## Fase 1: Crear una neurona artificial desde cero

**Estado:** ejercicios implementados según la memoria técnica.  
**Tiempo estimado:** 3–4 horas.

### Objetivo

Entender qué es una neurona artificial y cómo aprende modificando pesos.

### Tareas

- Crear entradas numéricas simples.
- Definir pesos y bias.
- Calcular la suma ponderada.
- Aplicar una función de activación.
- Generar una predicción.
- Definir una función de pérdida simple.
- Calcular cómo cambia el error.
- Ajustar pesos manualmente.
- Introducir derivadas y gradientes de manera sencilla.
- Programar el ciclo de aprendizaje sin PyTorch.

### Ejemplo conceptual

```text
Entradas
  ↓
Suma ponderada con pesos y bias
  ↓
Activación → salida → predicción mediante un umbral
                ↓
       Pérdida frente al valor esperado
                ↓
             Gradiente
                ↓
     Actualización de pesos y bias
```

### Experimento sugerido

Aprender una función lógica sencilla, como AND:

| x1 | x2 | AND |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Los ejercicios implementados están en [src/fase1](../../src/fase1). Las definiciones se pueden consultar en el [breviario](00_breviario.md).

### Conceptos clave

Neurona artificial, peso, bias, activación, error, gradiente y tasa de aprendizaje (`learning_rate`).

### Criterio de salida

La neurona produce mejores respuestas después del entrenamiento.

### Resultado esperado

Primera IA que aprende a partir de ejemplos, ajustando sus parámetros en lugar de recibir la respuesta como una regla programada explícitamente.

---

## Fase 2: Crear una red neuronal desde cero

**Estado:** en curso.  
**Tiempo estimado:** 4–5 horas.

### Objetivo

Pasar de una neurona a varias capas conectadas y entender el flujo completo del aprendizaje.

### Tareas

- Crear las capas de entrada, oculta y salida.
- Implementar la propagación hacia adelante (`forward propagation`).
- Implementar una función de pérdida.
- Introducir la retropropagación (`backpropagation`).
- Ajustar pesos de varias neuronas.
- Entrenar durante múltiples épocas (`epochs`).
- Graficar la pérdida.
- Comparar distintas tasas de aprendizaje.

### Arquitectura conceptual

```text
Entrada
  ↓
Capa oculta
  ↓
Salida
```

El ejercicio de Fase 2 utiliza XOR, con 2 entradas, 2 neuronas ocultas y 1 neurona de salida. Su avance y los pendientes conservados están en la [memoria técnica](02_Memoria_Tecnica.md#avance-y-punto-para-retomar).

### Conceptos clave

Capa (`layer`), forward pass, backward pass, época, batch, overfitting y convergencia.

### Criterio de salida

La red resuelve un problema no trivial y muestra una reducción consistente de la pérdida (`loss`).

### Resultado esperado

Comprender el flujo completo de aprendizaje de una red neuronal.

---

## Fase 3: PyTorch y entrenamiento con GPU

**Estado:** completada.
**Tiempo estimado:** 3–4 horas.

### Objetivo

Traducir lo aprendido manualmente a herramientas de aprendizaje automático.

### Tareas

- [x] Introducir `torch.Tensor`.
- [x] Comparar NumPy con los tensores de PyTorch.
- [x] Ejecutar operaciones en CPU.
- [x] Mover tensores a GPU.
- [x] Crear un modelo con `torch.nn`.
- [x] Usar `autograd`.
- [x] Ejecutar `loss.backward()`.
- [x] Usar un optimizador.
- [x] Entrenar en GPU.
- [ ] Medir el uso de VRAM.
- [x] Comparar tiempos en CPU y GPU.

### Ejercicios implementados

| Archivo | Propósito |
| --- | --- |
| [01_tensores_gpu.py](../../src/fase3/01_tensores_gpu.py) | Crea tensores, detecta CUDA y mueve datos al dispositivo elegido. |
| [02_xor_pytorch.py](../../src/fase3/02_xor_pytorch.py) | Reimplementa XOR con PyTorch usando `nn.Module`, `nn.Linear`, `MSELoss` y `torch.optim.SGD`. |
| [03_autograd_basico.py](../../src/fase3/03_autograd_basico.py) | Muestra `requires_grad=True`, `backward()` y el gradiente de una expresión simple. |
| [04_gradientes_red.py](../../src/fase3/04_gradientes_red.py) | Imprime gradientes reales de pesos y bias después de `loss.backward()`. |
| [05_peso_antes_despues.py](../../src/fase3/05_peso_antes_despues.py) | Muestra el cambio visible de un peso tras `optimizer.step()`. |
| [06_cpu_vs_gpu.py](../../src/fase3/06_cpu_vs_gpu.py) | Compara el entrenamiento de XOR en CPU y GPU. |
| [07_matrices_cpu_vs_gpu.py](../../src/fase3/07_matrices_cpu_vs_gpu.py) | Compara multiplicación de matrices grandes en CPU y GPU. |

### Registro del hito completado

La RTX 5050 quedó disponible para PyTorch mediante CUDA. Los ejercicios confirmaron que los tensores pueden moverse de CPU a GPU y que la red XOR puede entrenarse con PyTorch usando el mismo flujo conceptual aprendido a mano:

```text
Forward
  ↓
MSELoss
  ↓
optimizer.zero_grad()
  ↓
loss.backward()
  ↓
optimizer.step()
```

En el benchmark de matrices `3000x3000`, la GPU fue aproximadamente **11-14x** más rápida que la CPU. Este resultado aparece en operaciones grandes de álgebra lineal; en redes diminutas como XOR, la GPU no necesariamente muestra ventaja por el costo de coordinar operaciones pequeñas.

### Conceptos clave

Tensor, autograd, optimizador (`optimizer`), CUDA, dispositivo (`device`), VRAM y tamaño de lote (`batch size`).

### Criterio de salida

Una red neuronal se entrena correctamente en la RTX 5050.

### Resultado esperado

Entender qué automatiza PyTorch y qué ocurre detrás de `loss.backward()` y `optimizer.step()`.

---

## Fase 4: Tokenización y embeddings

**Estado:** completada el 26 de septiembre de 2026, según la confirmación del autor.

**Tiempo estimado:** 4–5 horas; no se midió la duración real.

### Objetivo

Pasar de entradas numéricas simples a procesar texto y entrenar un modelo básico que prediga el siguiente token.

### Tareas

- [x] Crear un corpus pequeño de texto y un vocabulario.
- [x] Tokenizar por espacios en blanco con `split()`.
- [x] Convertir texto a identificadores (`IDs`) y reconstruir texto desde IDs.
- [x] Crear embeddings y comparar vectores mediante similitud coseno.
- [x] Preparar pares de entrada y objetivo (`input/target`).
- [x] Entrenar un modelo `Embedding → Linear` con `CrossEntropyLoss` y Adam.
- [x] Convertir logits en probabilidades y seleccionar un token con `argmax`.
- [x] Ampliar el contexto de uno a dos tokens y concatenar sus embeddings.
- [x] Comprender que un contexto puede tener varias continuaciones válidas.

No se introdujeron tokens especiales: el corpus se procesa como una secuencia continua. Incorporar límites de frase o un token desconocido queda como ampliación futura, no como requisito pendiente del cierre de esta fase.

### Ejercicios implementados y ejecutados

| Archivo | Propósito |
| --- | --- |
| [01_tokenizacion_basica.py](../../src/fase4/01_tokenizacion_basica.py) | Separar texto en tokens, construir el vocabulario ordenado y asignar IDs. |
| [02_encode_decode.py](../../src/fase4/02_encode_decode.py) | Recorrer texto → IDs → texto mediante dos diccionarios. |
| [03_embeddings.py](../../src/fase4/03_embeddings.py) | Consultar vectores de cuatro dimensiones con `nn.Embedding`. |
| [04_similitud_embeddings.py](../../src/fase4/04_similitud_embeddings.py) | Comparar embeddings aleatorios con similitud coseno. |
| [05_contexto_siguiente_token.py](../../src/fase4/05_contexto_siguiente_token.py) | Construir los 11 pares consecutivos de un corpus de 12 tokens. |
| [06_modelo_lenguaje_basico.py](../../src/fase4/06_modelo_lenguaje_basico.py) | Entrenar y consultar un modelo con contexto de un token. |
| [07_contexto_dos_tokens.py](../../src/fase4/07_contexto_dos_tokens.py) | Entrenar con 10 ventanas de dos tokens y probar `el perro` y `el gato`. |

### Registro del hito completado

Los modelos 06 y 07 usan embeddings de dimensión 8, Adam con `lr=0.05` y 3000 épocas sobre todos los ejemplos juntos. Los embeddings y la capa lineal se ajustan mediante backpropagation. Se imprime la pérdida final y se muestran predicciones, sin guardar checkpoints ni separar validación.

El contexto `el perro` aparece con los objetivos `come` y `duerme`. Ampliar la ventana aporta información, pero no elimina esta ambigüedad: el modelo debe distribuir probabilidad entre las continuaciones. Las predicciones pueden variar entre ejecuciones porque los scripts no fijan una semilla.

La ejecución quedó desbloqueada después de diagnosticar Smart App Control de Windows. El incidente y los límites del modelo están descritos en la [memoria técnica](02_Memoria_Tecnica.md#fase-4-tokenización-embeddings-y-predicción-de-siguiente-token).

### Conceptos clave

Token, vocabulario, encode/decode, embedding, similitud coseno, ventana de contexto, target, logits, softmax, `argmax`, `CrossEntropyLoss`, Adam y ambigüedad de las continuaciones.

### Criterio de salida

Podemos convertir texto a tokens, embeddings y objetivos de entrenamiento, entrenar los modelos con uno y dos tokens de contexto y explicar por qué una entrada puede admitir varias respuestas.

### Resultado alcanzado y siguiente fase

Primer flujo de procesamiento y predicción de lenguaje completado. La Fase 5 incorpora atención para calcular pesos sobre el contexto según su contenido; el modelo de Fase 4 concatena embeddings y usa una capa lineal.

---

## Fase 5: Self-attention

**Estado:** completada el 26 de septiembre de 2026, según la confirmación del autor.

**Tiempo estimado:** 5–6 horas; no se midió la duración real.

### Objetivo

Entender y construir el mecanismo de atención para combinar información de una secuencia y producir representaciones contextualizadas.

### Tareas completadas

- [x] Calcular atención para un token mediante producto punto y suma ponderada.
- [x] Introducir Query, Key y Value con matrices manuales.
- [x] Implementar `Q @ K.T` para todos los tokens.
- [x] Escalar los scores por `sqrt(d_k)` y aplicar softmax por fila.
- [x] Combinar los values mediante `pesos @ V`.
- [x] Inspeccionar los pesos de atención impresos en la terminal.
- [x] Bloquear posiciones futuras con una máscara causal antes del softmax.
- [x] Crear proyecciones entrenables con `nn.Linear(..., bias=False)`.
- [x] Implementar múltiples cabezas, concatenar sus salidas y proyectar el resultado.

La inspección se hizo con matrices impresas; no hay un mapa de calor implementado. Los módulos tienen parámetros entrenables, pero estos ejercicios no incluyen entrenamiento.

### Ejercicios implementados y ejecutados

| Archivo | Propósito |
| --- | --- |
| [01_attention_intuicion.py](../../src/fase5/01_attention_intuicion.py) | Usar `come` como query, calcular scores y combinar embeddings. |
| [02_query_key_value.py](../../src/fase5/02_query_key_value.py) | Separar Q, K y V usando matrices identidad. |
| [03_self_attention_todos_tokens.py](../../src/fase5/03_self_attention_todos_tokens.py) | Calcular atención de todos los tokens en una sola operación matricial. |
| [04_scaled_dot_product_attention.py](../../src/fase5/04_scaled_dot_product_attention.py) | Escalar por la raíz de la dimensión de las keys. |
| [05_causal_attention.py](../../src/fase5/05_causal_attention.py) | Aplicar una máscara triangular superior con `-inf` antes del softmax. |
| [06_attention_aprendible.py](../../src/fase5/06_attention_aprendible.py) | Encapsular atención causal en `CausalSelfAttention` con parámetros entrenables. |
| [07_multi_head_attention.py](../../src/fase5/07_multi_head_attention.py) | Construir dos `AttentionHead`, registrarlas con `ModuleList`, concatenar y proyectar. |

### Ecuación principal

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k) + M) V

M[i, j] = 0     si j <= i
M[i, j] = -inf  si j > i
```

La máscara `M` representa la variante causal. En los ejercicios se aplica mediante `masked_fill`.

### Registro del hito completado

La implementación final recibe tres embeddings de dimensión 4. Usa dos cabezas de dimensión 2, produce una matriz de pesos `[3, 3]` por cabeza y conserva una salida final `[3, 4]` tras concatenación y proyección.

El autor reportó, para la fila de `perro`, pesos `[0.5621, 0.4379, 0.0000]` y `[0.4425, 0.5575, 0.0000]` en las dos cabezas. Son resultados de ejemplo de una ejecución: los pesos dependen de la inicialización aleatoria y no demuestran relaciones semánticas aprendidas. El cero en la posición de `come` corresponde a la máscara causal.

### Conceptos clave

Attention, self-attention, Query, Key, Value, producto punto, scores, softmax, escalado, máscara causal, parámetros entrenables, representación contextualizada, múltiples cabezas y proyección de salida.

### Criterio de salida alcanzado

Podemos inspeccionar cómo cada posición combina información de las posiciones permitidas y seguir las dimensiones desde los embeddings hasta la salida de múltiples cabezas.

### Resultado alcanzado y siguiente fase

Multi-Head Causal Self-Attention implementada y ejecutada. La [Fase 6](#fase-6-construir-nuestro-transformer) la integró con conexiones residuales, LayerNorm, una red feed-forward e información posicional para formar un bloque Transformer.

---

## Fase 6: Construir nuestro Transformer

**Estado:** completada; cierre registrado el 27 de septiembre de 2026, según la confirmación del autor.

**Tiempo estimado:** 6–8 horas; no se midió la duración real.

### Objetivo

Ensamblar las piezas anteriores en un Transformer autoregresivo pequeño, entrenarlo para predecir el siguiente token y recuperar su aprendizaje desde un archivo.

### Tareas completadas

- [x] Integrar Multi-Head Causal Self-Attention en un bloque Transformer.
- [x] Añadir conexiones residuales y LayerNorm después de cada suma.
- [x] Implementar una red feed-forward por posición con expansión a cuatro veces la dimensión y ReLU.
- [x] Sumar embeddings de tokens y embeddings posicionales entrenables.
- [x] Recibir IDs directamente y apilar bloques con `nn.ModuleList`.
- [x] Proyectar las representaciones hacia logits del vocabulario.
- [x] Preparar entrada y objetivo desplazados un token.
- [x] Entrenar con `CrossEntropyLoss`, backpropagation y Adam.
- [x] Generar cinco tokens a partir de `el` con selección por `argmax`.
- [x] Guardar y recuperar un `state_dict` sin volver a entrenar.
- [x] Guardar pesos, configuración y vocabulario en un checkpoint y reconstruir MiniGPT.

### Recorrido implementado

Los [diez ejercicios y sus comandos](02_Memoria_Tecnica.md#fase-6-transformer) avanzan desde el bloque básico (01), posiciones (02), integración con IDs (03) y apilado (04), hasta salida al vocabulario (05), entrenamiento (06), generación y guardado (07), carga de pesos (08), checkpoint completo (09) y recuperación del checkpoint (10).

```text
Texto → split() → IDs
  → token embeddings + position embeddings
  → bloque Transformer × 2
      atención causal → residual + LayerNorm
      → feed-forward → residual + LayerNorm
  → Linear → logits
  → softmax → argmax → siguiente token → ampliar contexto
```

### Configuración utilizada

| Elemento | Valor en el modelo entrenado |
| --- | --- |
| Corpus | `el perro come el gato duerme` |
| Vocabulario | 5 palabras, ordenadas alfabéticamente |
| Longitud de entrada de entrenamiento | 5 tokens |
| Posiciones disponibles | 20 (`max_seq_len`) |
| Dimensión del embedding | 8 |
| Cabezas de atención | 2; dimensión 4 por cabeza |
| Bloques Transformer | 2 |
| Feed-forward | `8 → 32 → 8`, con ReLU |
| Optimizador y pérdida | Adam, `lr=0.01`; `CrossEntropyLoss` |
| Épocas | 2000 en cada script de entrenamiento (06, 07 y 09) |
| Dispositivo | CUDA si está disponible; CPU en caso contrario |

Los primeros ejercicios usan dimensión 4; el ejemplo de apilado usa tres bloques. La propuesta original de contexto 128, embeddings 128–256 y cuatro bloques queda como referencia para una ampliación futura, no como configuración implementada.

### Registro del hito completado

El autor reportó una pérdida inicial aproximada de `2.004` y final de `0.000035`, con las cinco predicciones de la secuencia correctas. El modelo generó `el perro come el gato duerme` y volvió a producirla después de cargar los pesos y después de recuperar configuración, vocabulario y parámetros desde el checkpoint. Son resultados reportados, no mediciones nuevas de esta actualización.

El checkpoint contiene `model_state_dict`, `config` y `vocabulario`. Permite reconstruir el modelo para inferencia con el código de la arquitectura; todavía no incluye estado del optimizador ni época para reanudar exactamente el entrenamiento.

### Conceptos clave

Bloque Transformer, conexiones residuales, LayerNorm, feed-forward, embeddings posicionales, apilado, logits, objetivos desplazados, entrenamiento causal, generación autoregresiva, `state_dict` y checkpoint.

### Criterio de salida alcanzado

MiniGPT recibe IDs, produce logits por posición, se entrena con la tarea de siguiente token, genera la secuencia de ejemplo y recupera lo aprendido sin reentrenar.

### Resultado alcanzado y siguiente fase

Primer Transformer autoregresivo entrenable y persistente construido con componentes de PyTorch. El corpus diminuto permite demostrar memorización y persistencia, pero no comprensión del español ni generalización. La Fase 7 amplió los datos y añadió validación, lotes, seguimiento de métricas y experimentos de generación.

---

## Fase 7: Entrenar MiniAI

**Estado:** completada.
**Tiempo estimado:** 4–6 horas.

### Objetivo

Ampliar el entrenamiento del MiniGPT de Fase 6 a un corpus más variado y evaluar su comportamiento con datos de validación. La fase estudia aprendizaje y generalización más allá de una secuencia memorizada, además de incorporar estrategias de decodificación y un checkpoint final.

### Tareas completadas

- Ampliar el corpus a 30 oraciones y separar seis para validación.
- Convertir las frases en ventanas de contexto y objetivos de siguiente token.
- Entrenar con `DataLoader` y mini-batches de ocho ejemplos.
- Adaptar attention y el Transformer a tensores con dimensión batch.
- Comparar train loss y validation loss para detectar overfitting.
- Añadir dropout, AdamW, weight decay y early stopping.
- Restaurar los pesos correspondientes al mejor validation loss.
- Añadir `<BOS>` y `<EOS>` para modelar inicio y fin de secuencia.
- Comparar argmax con sampling y experimentar con temperature, top-k y top-p.
- Guardar pesos, configuración, vocabulario, tokens especiales y parámetros de generación en `miniai_fase7_checkpoint.pth`.

### Ejercicios implementados

| Archivo | Progreso principal |
| --- | --- |
| `01_preparar_dataset.py` | Tokenización, vocabulario y primera separación train/validation. |
| `02_crear_batches.py` | Ventanas de contexto y tensores de entradas/objetivos. |
| `03_entrenar_con_batches.py` | DataLoader, mini-batches y soporte batch en el modelo. |
| `04_mas_datos.py` | Corpus ampliado y observación de overfitting. |
| `05_entrenamiento_regularizado.py` | Dropout, AdamW, weight decay y early stopping. |
| `06_probar_generacion.py` | Generación desde varios prompts. |
| `07_tokens_especiales.py` | Tokens BOS/EOS y finalización aprendida. |
| `08_sampling_temperature_topk.py` | Sampling, temperature y top-k. |
| `09_top_p_sampling.py` | Nucleus sampling con distintos valores de top-p. |
| `10_checkpoint_fase7.py` | Checkpoint final y configuración reproducible para inferencia. |

### Configuración final

| Elemento | Valor |
| --- | --- |
| Corpus | 30 oraciones; 24 train y 6 validation |
| Ventana de contexto | 6 tokens |
| Batch size | 8 |
| Embedding | 16 |
| Cabezas / bloques | 4 / 2 |
| Dropout | 0.20 |
| Optimizador | AdamW |
| Learning rate / weight decay | 0.003 / 0.01 |
| Early stopping | patience 8; evaluación cada 20 épocas |
| Decodificación guardada | temperature 1.0, top-k desactivado, top-p 0.90 |

### Registro del hito completado

El experimento inicial llegó aproximadamente a train loss `0.205` y validation loss `6.39`, señal clara de overfitting. Con regularización y early stopping, la última ejecución reportada alcanzó:

```text
Mejor Validation Loss: 1.0344474554061889
Checkpoint: miniai_fase7_checkpoint.pth
```

Entre las generaciones reportadas para `el perro` aparecen:

```text
el perro mira por la ventana
el perro juega con el niño
el perro duerme en la casa
el perro come su comida
```

También aparecieron combinaciones incorrectas como `el perro duerme en la pelota`. El resultado muestra recombinación de patrones y finalización mediante EOS dentro del vocabulario entrenado; no demuestra comprensión general del español.

### Conceptos clave

Dataset, train/validation, batch, generalización, overfitting, dropout, AdamW, weight decay, early stopping, BOS/EOS, sampling, temperature, top-k, top-p, decodificación y checkpoint.

### Criterio de salida alcanzado

El modelo se entrena con múltiples ejemplos, se evalúa en datos separados, conserva los mejores pesos y genera texto con estructura reconocible usando distintas estrategias de decodificación.

### Resultado alcanzado y siguiente fase

MiniAI pasó de memorizar una secuencia a aprender distribuciones sobre un corpus pequeño, medir generalización y generar continuaciones variadas. El checkpoint final conserva lo necesario para reconstruir inferencia junto con el código, pero no guarda `optimizer.state_dict()`, época ni estados aleatorios para reanudar exactamente el entrenamiento.

La [Fase 8](#fase-8-convertir-miniai-en-servicio) separará la carga y la inferencia del script de entrenamiento y expondrá el modelo mediante una API.

---

## Fase 8: Convertir MiniAI en servicio

**Estado:** pendiente.  
**Tiempo estimado:** 3–4 horas.

### Objetivo

Exponer el modelo como servicio para usarlo desde otras aplicaciones.

### Tareas

- Crear una API con FastAPI.
- Crear el endpoint `POST /chat`.
- Cargar el modelo al iniciar.
- Recibir un prompt y generar una respuesta.
- Devolver JSON.
- Añadir logs.
- Manejar errores.
- Crear una interfaz web sencilla.
- Probar el acceso desde otro dispositivo en la red local (`LAN`).

### Arquitectura conceptual

```text
Cliente
  ↓
POST /chat
  ↓
FastAPI
  ↓
MiniAI
  ↓
Respuesta
```

### Conceptos clave

API, servidor de inferencia (`inference server`), solicitud y respuesta (`request/response`), serialización, latencia y concurrencia.

### Criterio de salida

Podemos hablar con MiniAI desde un navegador o un cliente HTTP.

### Resultado esperado

MiniAI deja de ser un script y se convierte en un servicio.

---

## Fase 9: RAG y herramientas

**Estado:** pendiente.  
**Tiempo estimado:** 4–5 horas.

### Objetivo

Separar el conocimiento del modelo, la recuperación de información y la capacidad de actuar.

### Tareas

- Crear embeddings para documentos.
- Indexar documentos.
- Buscar contexto relevante.
- Inyectar contexto en los prompts.
- Implementar RAG.
- Crear una herramienta sencilla.
- Permitir que el sistema invoque una función.
- Registrar las llamadas a herramientas.
- Validar argumentos.
- Aplicar límites de seguridad.

### Arquitectura conceptual

```text
MiniAI
  ├── RAG → documentos
  └── Tools → funciones / APIs
```

### Conceptos clave

Recuperación (`retrieval`), búsqueda vectorial (`vector search`), embeddings, incorporación de contexto (`context injection`), llamadas a herramientas (`tool calling`) y salida estructurada (`structured output`).

### Criterio de salida

MiniAI responde usando información externa y ejecuta una herramienta controlada.

### Resultado esperado

Distinguir claramente lo que el modelo aprendió, lo que recupera y lo que puede ejecutar.

---

## Fase 10: MCP y arquitectura distribuida

**Estado:** pendiente.  
**Tiempo estimado:** 4–6 horas.

### Objetivo

Integrar MCP y distribuir componentes entre las dos laptops.

### Distribución prevista

| Equipo | Componentes y responsabilidad |
| --- | --- |
| Lenovo LOQ | MiniAI, PyTorch, CUDA, RTX 5050 e inferencia. |
| IdeaPad | FastAPI, servidor MCP, archivos, bases de datos, APIs y servicios auxiliares. |

### Tareas

- Crear un servidor MCP.
- Definir herramientas (`tools`).
- Definir recursos (`resources`).
- Conectar un cliente MCP.
- Probar la invocación de herramientas.
- Añadir autenticación y autorización.
- Añadir tiempos de espera (`timeouts`).
- Añadir logs y límites de acceso.
- Probar la comunicación entre laptops por LAN.
- Documentar la arquitectura final.

### Conceptos clave

MCP, cliente y servidor, tools, resources, permisos, sandboxing, observabilidad y participación humana (`human-in-the-loop`).

### Criterio de salida

MiniAI consume herramientas externas mediante MCP de forma controlada.

### Resultado esperado

Un sistema de laboratorio que reúna los componentes desarrollados:

```text
Usuario
  ↓
Chat / interfaz
  ↓
API
  ↓
MiniAI
  ├── Transformer → GPU
  ├── RAG → documentos
  └── Herramientas / cliente MCP → servidor MCP → servicios externos
```

---

## Fase 11: Consolidación y documentación

**Estado:** pendiente.  
**Tiempo estimado:** 2–4 horas.

### Objetivo

Cerrar el proyecto como un trabajo técnico presentable y reutilizable.

### Tareas

- Limpiar el repositorio.
- Crear un README completo.
- Añadir un diagrama de arquitectura.
- Documentar la instalación.
- Documentar el entrenamiento.
- Documentar las decisiones técnicas.
- Guardar métricas.
- Guardar capturas o ejemplos de generación.
- Añadir conclusiones.
- Crear un plan de mejoras futuras.

### Conceptos clave

Documentación técnica, reproducibilidad, métricas, decisiones de diseño y comunicación de resultados.

### Criterio de salida

El repositorio reúne los entregables finales y explica cómo instalar, entrenar y utilizar el sistema, incluyendo sus resultados y decisiones técnicas.

### Entregables finales

- Código fuente.
- Modelo entrenado.
- Checkpoints.
- Dataset utilizado.
- API funcional.
- RAG.
- Herramientas.
- Servidor MCP.
- Documentación.
- Diagrama de arquitectura.
- Notas de aprendizaje.

### Resultado esperado

Repositorio que demuestre comprensión de IA desde la neurona hasta la integración con servicios.

---

## Reglas del proyecto

1. **Entender antes de abstraer.** No usar frameworks complejos para esconder conceptos que todavía no entendemos.
2. **Medir.** Registrar pérdida, tiempo, uso de VRAM y resultados.
3. **Versionar.** Usar Git desde el principio.
4. **Mantener modelos pequeños.** El objetivo inicial es aprender, sin intentar competir con modelos comerciales.
5. **Proteger el hardware.** Usar tamaños de lote razonables y vigilar temperatura y VRAM.
6. **Separar conceptos.** Distinguir entrenamiento, inferencia, RAG, tool calling, MCP y APIs.
7. **Experimentar.** Cambiar hiperparámetros y observar sus efectos.
8. **Documentar errores.** Registrar los fallos cuando sus causas y soluciones aporten al aprendizaje.

---

## Notas de la reorganización

- **Duración:** el original indicaba 42–56 horas en la introducción y 44–60 en el resumen. Se unifica en **44–60 horas**, que corresponde a sumar los rangos de las doce fases.
- **Avance:** el hito original señalaba la Fase 0 completada y proponía comenzar la Fase 1. Se conserva ese hito en su fase y se actualiza el punto para retomar a la **Fase 4**, conforme a la memoria técnica del 19 de septiembre de 2026.
- **Cierre de fases:** los ejercicios de la Fase 1 se describen como implementados; esta conversión no declara una nueva validación de sus resultados ni da por concluida la Fase 2.
- **Formato:** se unifican títulos, tareas, conceptos, criterios de salida, ejemplos y resultados. En la Fase 11 se explicitan conceptos y un criterio de salida a partir de sus tareas y entregables originales.
