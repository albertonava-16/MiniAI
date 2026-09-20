# Plan de trabajo de MiniAI

Este documento organiza el recorrido de MiniAI: desde una neurona artificial hasta un sistema que genere texto, ofrezca una API y utilice información y herramientas externas.

La intención es avanzar por fases, entender cada componente y registrar resultados antes de pasar al siguiente nivel.

**Duración estimada total:** 44–60 horas, según la suma de las fases.  
**Último avance registrado:** 19 de septiembre de 2026.
**Reorganización de la documentación:** 19 de septiembre de 2026.  
**Punto para retomar:** Fase 4, tokenización y embeddings.

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

El avance se toma de la [memoria técnica](02_Memoria_Tecnica.md#avance-y-punto-para-retomar), cuyo último registro corresponde al **19 de septiembre de 2026**.

| Fase | Avance registrado |
| --- | --- |
| 0. Laboratorio | Completada según las pruebas previas del entorno. |
| 1. Neurona artificial | Implementados los ejercicios de neurona básica, entrenamiento de AND y frontera de decisión. |
| 2. Red neuronal | Trabajada: red XOR con NumPy y ejercicio de backpropagation. |
| 3. PyTorch y GPU | Completada: tensores, CUDA, XOR en PyTorch, gradientes reales, optimizador y benchmark CPU vs GPU. |
| 4–11 | Pendientes en el registro de avance. |

Los estados describen el avance documentado. La Fase 3 registra los ejercicios implementados en `src/fase3` y el resultado observado del benchmark de matrices.

### Primera tarea al retomar

Comenzar la [Fase 4](#fase-4-tokenización-y-embeddings), creando un corpus pequeño, un vocabulario y un tokenizador sencillo.

Después:

- [ ] Definir el corpus mínimo de prueba.
- [ ] Convertir texto a IDs y de IDs a texto.
- [ ] Crear embeddings pequeños.
- [ ] Preparar pares `input/target` para predicción del siguiente token.

Los comandos para comenzar están en [Inicio rápido de la memoria técnica](02_Memoria_Tecnica.md#inicio-rápido).

---

## Resumen de fases y duración

Los tiempos son estimaciones de dedicación por fase, no horas medidas ni fechas de entrega.

| Fase | Tema | Estimación | Estado registrado |
| ---: | --- | ---: | --- |
| 0 | Preparar el laboratorio | 2–3 h | Completada |
| 1 | Neurona artificial | 3–4 h | Ejercicios implementados |
| 2 | Red neuronal | 4–5 h | Trabajada |
| 3 | PyTorch y GPU | 3–4 h | Completada |
| 4 | Tokenización y embeddings | 4–5 h | Pendiente |
| 5 | Self-attention | 5–6 h | Pendiente |
| 6 | Transformer | 6–8 h | Pendiente |
| 7 | Entrenamiento de MiniAI | 4–6 h | Pendiente |
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

**Estado:** pendiente.  
**Tiempo estimado:** 4–5 horas.

### Objetivo

Pasar de números genéricos a lenguaje.

### Tareas

- Crear un corpus pequeño de texto.
- Construir un vocabulario.
- Crear un tokenizador sencillo.
- Convertir texto a identificadores (`IDs`).
- Convertir IDs a texto.
- Introducir un token especial si es necesario.
- Crear embeddings y representar tokens como vectores.
- Preparar secuencias de contexto.
- Crear pares de entrada y objetivo (`input/target`).

### Ejemplo

```text
Texto: "hola mundo"

hola  → 17
mundo → 42

17
 ↓
Embedding
 ↓
[0.28, -0.91, 0.37, ...]
```

Los IDs y los valores del vector son ilustrativos.

### Conceptos clave

Token, vocabulario (`vocabulary`), token ID, embedding, ventana de contexto (`context window`), secuencia y predicción del siguiente token (`next-token prediction`).

### Criterio de salida

Podemos convertir texto a tokens, embeddings y objetivos de entrenamiento.

### Resultado esperado

Primer flujo de procesamiento de lenguaje.

---

## Fase 5: Self-attention

**Estado:** pendiente.  
**Tiempo estimado:** 5–6 horas.

### Objetivo

Construir el mecanismo central de los Transformers.

### Tareas

- Introducir Query, Key y Value.
- Implementar el producto `QK^T`.
- Aplicar escalamiento.
- Aplicar softmax.
- Multiplicar por `V`.
- Visualizar los pesos de atención.
- Introducir una máscara causal (`causal masking`).
- Implementar una cabeza de atención.
- Evolucionar a atención con múltiples cabezas (`multi-head attention`).

### Ecuación principal

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

### Conceptos clave

Query, Key, Value, puntuación de atención (`attention score`), softmax, máscara causal, cabeza de atención y multi-head attention.

### Criterio de salida

Podemos inspeccionar cómo un token asigna importancia a otros tokens.

### Resultado esperado

Comprender qué hace la atención y por qué es relevante para el modelo.

---

## Fase 6: Construir nuestro Transformer

**Estado:** pendiente.  
**Tiempo estimado:** 6–8 horas.

### Objetivo

Ensamblar las piezas anteriores en un modelo generativo pequeño de tipo GPT.

### Tareas

- Crear embeddings de tokens.
- Añadir información posicional.
- Crear un bloque Transformer.
- Implementar self-attention.
- Implementar un MLP.
- Añadir conexiones residuales.
- Añadir normalización.
- Apilar múltiples bloques.
- Crear la capa de salida.
- Generar logits.
- Convertir logits en probabilidades.
- Implementar la generación autorregresiva.

### Arquitectura conceptual

```text
Tokens
  ↓
Embeddings
  ↓
Información posicional
  ↓
Bloque Transformer
  ↓
Bloque Transformer
  ↓
Capa lineal
  ↓
Logits
  ↓
Selección del siguiente token
```

### Configuración inicial sugerida

| Elemento | Propuesta |
| --- | --- |
| Contexto | 128 tokens |
| Dimensión del embedding | 128–256 |
| Cabezas de atención | 4 |
| Capas | 4 |
| Parámetros | Unos pocos millones; por confirmar al definir el modelo y el vocabulario. |

### Conceptos clave

Bloque Transformer, conexión residual, normalización por capa (`layer normalization`), MLP, logits y generación autorregresiva.

### Criterio de salida

El modelo acepta tokens y produce una distribución sobre el siguiente token.

### Resultado esperado

Primer MiniGPT construido por nosotros.

---

## Fase 7: Entrenar MiniAI

**Estado:** pendiente.  
**Tiempo estimado:** 4–6 horas.

### Objetivo

Entrenar el Transformer y observar cómo pasa de ruido a patrones de lenguaje.

### Tareas

- Elegir un corpus.
- Preparar el dataset.
- Separar los datos de entrenamiento y validación (`train/validation`).
- Configurar hiperparámetros.
- Entrenar por pasos (`steps`).
- Registrar la pérdida.
- Guardar checkpoints.
- Probar generaciones periódicas.
- Detectar overfitting.
- Ajustar el tamaño de lote, la tasa de aprendizaje y la longitud de contexto.
- Medir temperatura del hardware y uso de VRAM.

### Ejemplo de evolución

Ejemplo ilustrativo del cambio que se busca observar; no son resultados ya obtenidos ni una garantía para un número concreto de pasos:

```text
Paso 0:
"casa perro java función la de"

Paso 5000:
"Java es un lenguaje de..."
```

### Conceptos clave

Ciclo de entrenamiento (`training loop`), pérdida de validación, checkpoint, hiperparámetro, overfitting, muestreo (`sampling`) y temperatura de generación (`temperature`).

### Criterio de salida

El modelo genera texto con estructura reconocible.

### Resultado esperado

Nuestro propio modelo generativo entrenado localmente.

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
