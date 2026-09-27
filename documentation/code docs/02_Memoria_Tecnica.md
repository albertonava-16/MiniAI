# Memoria técnica de MiniAI

Este documento reúne la configuración del entorno, los comandos de trabajo y los aprendizajes técnicos de MiniAI.

La intención es tener una referencia rápida para retomar el proyecto después de reiniciar la computadora o al comenzar una nueva fase.

**Último avance registrado:** 27 de septiembre de 2026.
**Reorganización de la documentación:** 19 de septiembre de 2026.  
**Punto para retomar:** Fase 7, ampliar el corpus y evaluar el entrenamiento de MiniAI.

Las versiones y los resultados de GPU se conservan como registro del entorno anterior. Las actualizaciones de Fases 4, 5 y 6 recogen la ejecución y el cierre confirmados por el autor, su contexto de aprendizaje y la revisión del código. En Fase 5 se documenta el forward de módulos con parámetros entrenables; no se ejecuta un ciclo de entrenamiento. En Fase 6 se documentan el entrenamiento, la generación y la recuperación desde archivos; las pérdidas y salidas numéricas proceden del contexto compartido por el autor. No se volvieron a entrenar los modelos al editar esta documentación; tampoco se midieron nuevas pérdidas, tiempos ni resultados de CUDA.

Documentos relacionados: [plan de trabajo](01_Plan_de_Trabajo.md), [breviario de conceptos](00_breviario.md) y [README](../../README.md).

Documento convertido a partir de `MiniAI_Memoria_Tecnica.txt`. Las próximas actualizaciones se registrarán en esta versión Markdown.

---

## Contenido

- [Inicio rápido](#inicio-rápido)
- [Avance y punto para retomar](#avance-y-punto-para-retomar)
- [Equipo principal](#equipo-principal)
- [Software y versiones registradas](#software-y-versiones-registradas)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Entorno virtual](#entorno-virtual)
- [GPU y Lenovo Legion](#gpu-y-lenovo-legion)
- [PyTorch y CUDA](#pytorch-y-cuda)
- [Avance de la Fase 3](#avance-de-la-fase-3)
- [Fase 4: Tokenización, embeddings y predicción de siguiente token](#fase-4-tokenización-embeddings-y-predicción-de-siguiente-token)
- [Fase 5: Attention](#fase-5-attention)
- [Fase 6: Transformer](#fase-6-transformer)
- [Primera prueba de GPU](#primera-prueba-de-gpu)
- [Rutina para retomar el proyecto](#rutina-para-retomar-el-proyecto)
- [Comandos de diagnóstico](#comandos-de-diagnóstico)
- [Errores y soluciones](#errores-y-soluciones)
- [Recomendaciones de trabajo](#recomendaciones-de-trabajo)

---

## Inicio rápido

Abrir Git Bash y ejecutar:

```bash
cd ~/desktop/MiniAI
source .venv/Scripts/activate
code .
```

Confirmar que aparezca `(.venv)` al inicio de la terminal.

La Fase 6 está completada y el siguiente trabajo es ampliar el corpus y evaluar el entrenamiento en Fase 7. Para recuperar MiniGPT sin reentrenarlo, ejecutar desde la raíz donde se guardó `minigpt_checkpoint.pth`:

```bash
python src/fase6/10_cargar_checkpoint.py
```

Si aún no existe ese archivo, crearlo primero con:

```bash
python src/fase6/09_checkpoint_completo.py
```

El script 09 entrena un modelo nuevo durante 2000 épocas y guarda el checkpoint; repetirlo sobrescribe ese archivo. El script 10 recupera configuración, vocabulario y pesos, y genera cinco tokens a partir de `el` sin entrenar. Ambos eligen CUDA si está disponible o CPU en caso contrario. Los comandos y dependencias de los diez ejercicios están en [Fase 6](#fase-6-transformer).

Para repasar el ejercicio de backpropagation de la Fase 2:

```bash
python src/fase2/02_backprop_xor.py
```

Después, para entrenar la red completa:

```bash
python src/fase2/01_red_xor.py
```

Estos ejercicios usan CPU con NumPy y no requieren CUDA. Si la gráfica abre una ventana, cerrarla para continuar con las impresiones que aparecen después de `plt.show()`.

Cuando la fase utilice GPU, comprobar su disponibilidad antes de entrenar:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Si devuelve `True`, continuar. Si devuelve `False`, revisar el [problema de GPU registrado](#gpu-y-lenovo-legion).

Para ejecutar los ejercicios de la Fase 3:

```bash
python src/fase3/01_tensores_gpu.py
python src/fase3/02_xor_pytorch.py
python src/fase3/03_autograd_basico.py
python src/fase3/04_gradientes_red.py
python src/fase3/05_peso_antes_despues.py
python src/fase3/06_cpu_vs_gpu.py
python src/fase3/07_matrices_cpu_vs_gpu.py
```

---

## Avance y punto para retomar

La fecha de corte del avance es el **27 de septiembre de 2026**.

| Fase | Estado registrado | Alcance |
| --- | --- | --- |
| 0. Laboratorio | Completada | Configuración y pruebas de GPU según el registro previo. |
| 1. Neurona artificial | Ejercicios implementados | Neurona básica, entrenamiento de AND y frontera de decisión. |
| 2. Red neuronal | Trabajada | Red XOR con NumPy y ejercicio de backpropagation. |
| 3. PyTorch y GPU | Completada | Tensores en CPU/GPU, CUDA, XOR en PyTorch, autograd, gradientes, optimizador y benchmark. |
| 4. Tokenización y embeddings | Completada | Texto e IDs, embeddings, similitud coseno y modelos de siguiente token con contextos de uno y dos tokens. |
| 5. Attention | Completada | Q/K/V, escalado, máscara causal, proyecciones entrenables y multi-head attention. |
| 6. Transformer | Completada | Diez ejercicios: bloque Transformer, posiciones, apilado, entrenamiento, generación y persistencia de MiniGPT. |
| 7. Entrenar MiniAI | Pendiente; siguiente fase | Corpus más amplio, entrenamiento y validación, lotes, métricas y seguimiento de generaciones. |

### Avance de la Fase 1

| Archivo | Qué permite estudiar |
| --- | --- |
| [01_neurona_basica.py](../../src/fase1/01_neurona_basica.py) | Calcula suma ponderada, sigmoide, predicción y pérdida para AND con pesos fijos. No ajusta los pesos. |
| [02_neurona_entrenamiento.py](../../src/fase1/02_neurona_entrenamiento.py) | Entrena durante 10,000 épocas, ajustando dos pesos y un bias. Al terminar, calcula predicciones con los parámetros aprendidos. |
| [03_frontera_decision.py](../../src/fase1/03_frontera_decision.py) | Dibuja los puntos de AND y la recta de separación con pesos ya escritos en el archivo. |

Las definiciones y los ejemplos de estos conceptos están en el [breviario](00_breviario.md).

### Avance de la Fase 2

El archivo principal es [01_red_xor.py](../../src/fase2/01_red_xor.py).

| Elemento | Configuración |
| --- | --- |
| Arquitectura | 2 entradas, 2 neuronas ocultas y 1 neurona de salida. |
| Cálculos | NumPy en CPU. |
| Gráficas | Matplotlib. |
| Semilla | `42`. |
| Tasa de aprendizaje | `learning_rate = 0.5`. |
| Épocas | `epochs = 10000`. |
| Datos por época | Los cuatro ejemplos de XOR juntos. |
| Pérdida registrada | `loss = np.mean(error ** 2)`, almacenada en `loss_history`. |
| Umbral de clasificación | `0.5`. |

El ciclo de entrenamiento sigue este recorrido:

```text
Forward propagation
        ↓
Cálculo del error y de la pérdida
        ↓
Backpropagation
        ↓
Ajuste de pesos y bias
        ↓
Siguiente época
```

El entrenamiento empieza en `for epoch in range(epochs):` y termina tras la actualización de `b1`. El comentario `AQUÍ YA TERMINÓ EL ENTRENAMIENTO` marca el paso a las gráficas y a los cálculos posteriores.

El programa incluye comentarios explicativos, muestra el recorrido de la entrada `[0, 1]` por la red entrenada e imprime las activaciones ocultas de cada entrada. Al final recalcula los resultados con los pesos actualizados y obtiene las clases `0` y `1`.

El segundo archivo es [02_backprop_xor.py](../../src/fase2/02_backprop_xor.py). Usa pesos y bias fijados en el código para estudiar la entrada `[0, 1]`, cuya respuesta esperada es `1`. Imprime el forward, el error, `output_delta`, `hidden_error` y `hidden_delta`. No contiene un ciclo de entrenamiento ni actualiza parámetros; permite observar los cálculos intermedios.

### Conceptos repasados

- AND produce `1` cuando ambas entradas son `1`; XOR produce `1` cuando son distintas.
- Aprender consiste en ajustar pesos y bias a partir del error.
- Las neuronas ocultas son cálculos intermedios entre la entrada y la salida.
- `sigmoid` transforma la suma ponderada en una activación entre `0` y `1`.
- `sigmoid_derivative` recibe esa activación y calcula la pendiente de la sigmoide.
- Una época recorre todo el conjunto de entrenamiento.

### Pendientes conservados de la Fase 2

- [ ] Explicar paso a paso `output_delta`, `hidden_error` y `hidden_delta` en `02_backprop_xor.py`.
- [ ] Conectar la gráfica de `01_red_xor.py` con `loss_history`. Actualmente dibuja `[10, 8, 6, 4, 2, 1]` y guarda `prueba_matplotlib.png` como prueba de Matplotlib.
- [ ] Ejecutar la red y comprobar las cuatro predicciones de XOR.
- [ ] Registrar la pérdida inicial y final, y comparar distintas tasas de aprendizaje.
- [ ] Cerrar la Fase 2 después de verificar la reducción de la pérdida y repasar el flujo completo, antes de pasar a PyTorch.

Estos pendientes se identificaron leyendo el código. No representan métricas nuevas ni resultados de una ejecución durante esta reorganización.

El plan en Markdown refleja este mismo avance. El hito de Fase 0 del plan original era anterior a este registro.

### Avance de la Fase 3

La Fase 3 traslada el aprendizaje manual de NumPy a PyTorch y confirma que la RTX 5050 puede ejecutar operaciones con CUDA.

| Archivo | Qué permite estudiar |
| --- | --- |
| [01_tensores_gpu.py](../../src/fase3/01_tensores_gpu.py) | `torch.Tensor`, detección de CUDA, selección de `device` y movimiento de tensores con `.to(device)`. |
| [02_xor_pytorch.py](../../src/fase3/02_xor_pytorch.py) | Red XOR en PyTorch con `nn.Module`, dos capas `nn.Linear`, `torch.sigmoid`, `MSELoss` y `SGD`. |
| [03_autograd_basico.py](../../src/fase3/03_autograd_basico.py) | Cálculo automático de gradientes con `requires_grad=True` y `backward()`. |
| [04_gradientes_red.py](../../src/fase3/04_gradientes_red.py) | Gradientes reales de `weight` y `bias` después de ejecutar `loss.backward()`. |
| [05_peso_antes_despues.py](../../src/fase3/05_peso_antes_despues.py) | Cambio de un peso específico antes y después de `optimizer.step()`. |
| [06_cpu_vs_gpu.py](../../src/fase3/06_cpu_vs_gpu.py) | Comparación de entrenamiento XOR en CPU y GPU. |
| [07_matrices_cpu_vs_gpu.py](../../src/fase3/07_matrices_cpu_vs_gpu.py) | Benchmark de multiplicación de matrices `3000x3000` en CPU y GPU. |

El flujo de entrenamiento quedó expresado así:

```text
output = modelo(X)
loss = criterio(output, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`loss.backward()` calcula los gradientes de los parámetros que participaron en el forward. Después, `optimizer.step()` modifica esos parámetros usando los gradientes acumulados y la tasa de aprendizaje.

En `05_peso_antes_despues.py` se observó el cambio visible de un peso de `modelo.capa1.weight[0, 0]` después de una actualización. Este ejercicio conecta el concepto manual de "ajustar pesos" con la forma en que PyTorch lo automatiza.

### Benchmark CPU vs GPU

La prueba de matrices usa multiplicaciones `a @ b` con matrices de `3000x3000` y varias repeticiones.

Resultado registrado:

```text
GPU aproximadamente 11-14x más rápida que la CPU
```

La diferencia se nota en matrices grandes porque la GPU puede ejecutar muchas multiplicaciones y sumas en paralelo. En ejercicios pequeños, como XOR, el costo de preparar y sincronizar operaciones puede ocultar la ventaja de la GPU.

### Punto para retomar

La siguiente fase es la [Fase 7 del plan](01_Plan_de_Trabajo.md#fase-7-entrenar-miniai): ampliar el entrenamiento y evaluar MiniAI.

Partir del MiniGPT y el checkpoint de `src/fase6/09_checkpoint_completo.py` y `src/fase6/10_cargar_checkpoint.py`. Preparar un corpus más amplio, separar entrenamiento y validación, registrar pérdidas y generaciones, y conservar el estado del optimizador para reanudar el entrenamiento.

El modelo actual procesa una sola secuencia de forma `[seq_len, embedding_dim]` dentro de los bloques. Al añadir lotes habrá que adaptar las operaciones y transposiciones; todavía no existe una dimensión de batch en estos módulos.

---

## Fase 4: Tokenización, embeddings y predicción de siguiente token

**Cierre registrado:** 26 de septiembre de 2026. El autor confirmó que resolvió el bloqueo de ejecución y terminó la fase. Los detalles de implementación siguientes se contrastaron con los siete scripts de `src/fase4`.

### Objetivo y recorrido

Pasar de entradas numéricas simples a texto y construir un modelo pequeño que aprenda a predecir el siguiente token.

```text
Texto → tokens → vocabulario → IDs → embeddings → capa lineal → logits
                                                                  |
                         Entrenamiento: CrossEntropyLoss ← target |
                                    ↓                             |
                              backpropagation                     |
                                    ↓                             |
                         actualizar embeddings y pesos            |
                                                                  ↓
                            Inferencia: softmax → argmax → token
```

### Ejercicios y alcance

| Archivo | Implementación y aprendizaje |
| --- | --- |
| [01_tokenizacion_basica.py](../../src/fase4/01_tokenizacion_basica.py) | `split()`, `sorted(set(tokens))` y diccionario token → ID para `hola mundo hola ia`. |
| [02_encode_decode.py](../../src/fase4/02_encode_decode.py) | Diccionarios token → ID e ID → token; reconstrucción con `" ".join(...)`. |
| [03_embeddings.py](../../src/fase4/03_embeddings.py) | Tabla `nn.Embedding(3, 4)`; consulta de los IDs de `hola`, `mundo` e `ia`. No entrena. |
| [04_similitud_embeddings.py](../../src/fase4/04_similitud_embeddings.py) | `F.cosine_similarity` entre vectores aleatorios de dimensión 4. No entrena ni demuestra similitud semántica. |
| [05_contexto_siguiente_token.py](../../src/fase4/05_contexto_siguiente_token.py) | Construcción e impresión de 11 pares de tokens consecutivos. |
| [06_modelo_lenguaje_basico.py](../../src/fase4/06_modelo_lenguaje_basico.py) | `MiniModeloLenguaje`: embedding y capa lineal; entrenamiento con contexto de un token y predicciones para todo el vocabulario. |
| [07_contexto_dos_tokens.py](../../src/fase4/07_contexto_dos_tokens.py) | `MiniModeloContexto`: embeddings, `flatten(start_dim=1)` y capa lineal; ventanas de dos tokens y función `predecir(token1, token2)`. |

El vocabulario de 01 y 02 es `['hola', 'ia', 'mundo']`; el texto se codifica como `[0, 2, 0, 1]`. El ID es una etiqueta, no una medida de significado. La decodificación recupera el texto de ejemplo, pero no preserva espacios repetidos ni saltos de línea del texto original.

### Corpus y objetivos de entrenamiento

Los ejercicios 05 y 06 usan:

```text
el perro come
el gato come
el perro duerme
el gato duerme
```

El ejercicio 07 usa las mismas cuatro frases en otro orden:

```text
el perro come
el gato duerme
el perro duerme
el gato come
```

En ambos casos hay 12 tokens y 5 elementos de vocabulario:

```text
come → 0
duerme → 1
el → 2
gato → 3
perro → 4
```

`split()` trata los saltos de línea como espacios en blanco. No se agregan marcadores de inicio o fin de frase. Por eso aparecen transiciones entre líneas como `come → el` y `duerme → el`. En el ejercicio 07 también aparecen ventanas como `[perro, come] → el`; el cambio de orden de las frases cambia algunas ventanas que cruzan sus límites.

Con un token de contexto se obtienen `12 - 1 = 11` ejemplos; con dos tokens, `12 - 2 = 10`. El target siempre es el ID del token inmediatamente posterior al contexto.

### Configuración de los modelos

| Elemento | Ejercicio 06 | Ejercicio 07 |
| --- | --- | --- |
| Contexto | 1 token | 2 tokens |
| Vocabulario | 5 tokens | 5 tokens |
| Embedding | `nn.Embedding(5, 8)` | `nn.Embedding(5, 8)` |
| Capa de salida | `nn.Linear(8, 5)` | `nn.Linear(16, 5)` |
| Entrada `X` | `[11]`, IDs enteros | `[10, 2]`, IDs enteros |
| Embeddings del lote | `[11, 8]` | `[10, 2, 8]` |
| Entrada a la capa lineal | `[11, 8]` | `[10, 16]` tras aplanar |
| Logits | `[11, 5]` | `[10, 5]` |
| Objetivos `y` | `[11]`, tipo `torch.long` | `[10]`, tipo `torch.long` |
| Pérdida | `nn.CrossEntropyLoss()` | `nn.CrossEntropyLoss()` |
| Optimizador | Adam, `lr=0.05` | Adam, `lr=0.05` |
| Épocas | 3000 | 3000 |
| Lote | Todos los ejemplos juntos | Todos los ejemplos juntos |
| Dispositivo | CUDA si está disponible; CPU en caso contrario | CUDA si está disponible; CPU en caso contrario |

Los ejercicios 03 y 04 se ejecutan en CPU tal como están escritos. En 06 y 07, el modelo y los tensores se trasladan al mismo `device`.

### Entrenamiento e inferencia

El ciclo utilizado es:

```python
logits = modelo(X)
loss = criterio(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

`CrossEntropyLoss` recibe los logits sin aplicar softmax previamente y los IDs de los targets. Cada token del vocabulario es una clase. `modelo.parameters()` incluye tanto la tabla de embeddings como los pesos y el bias de la capa lineal; todos participan en el aprendizaje.

Para consultar el modelo se usa `torch.no_grad()`, se convierten logits en probabilidades con `torch.softmax(logits, dim=1)` y se selecciona el ID de mayor probabilidad con `torch.argmax(..., dim=1)`. Los programas imprimen el token seleccionado, no la distribución completa. La selección con argmax no realiza muestreo.

### Resultados y ambigüedad

El autor reportó predicciones como `gato → come`, `perro → come`, `come → el` y `duerme → el`. Son ejemplos observados, no salidas fijas para cada ejecución. No se compartieron valores numéricos de pérdida final ni mediciones de tiempo.

El corpus incluye tanto `el perro come` como `el perro duerme`. El mismo contexto `[el, perro]` tiene dos targets distintos; ocurre lo mismo con `[el, gato]`. Un modelo que ve entradas idénticas no puede asignar probabilidad 100 % a ambos objetivos a la vez. Para esas continuaciones igualmente frecuentes, una distribución equilibrada es coherente con los datos y la pérdida conserva una contribución positiva.

Por eso, una pérdida que deja de bajar no implica por sí sola que el entrenamiento esté roto. La tarea contiene ambigüedad. Pequeñas diferencias de probabilidad pueden cambiar la palabra elegida con `argmax`, aunque la distribución sea parecida.

Los scripts imprimen la pérdida calculada en el último forward de entrenamiento, antes del último `optimizer.step()`; las consultas posteriores sí usan los parámetros ya actualizados.

### Aleatoriedad y límites actuales

- Los scripts no fijan `torch.manual_seed`; pesos, embeddings iniciales y algunas predicciones pueden variar.
- Como experimento de reproducibilidad se puede añadir `torch.manual_seed(42)` antes de crear el modelo. No está incorporado al código actual y no garantiza resultados idénticos entre todo hardware y toda configuración.
- El corpus es pequeño y cerrado. No existe tratamiento de tokens desconocidos: consultar una palabra fuera del vocabulario produce un `KeyError`.
- No hay tokens especiales ni separación explícita de frases.
- No hay conjunto de validación, métricas de generalización, checkpoints ni generación de secuencias completas.
- Los embeddings iniciales de 03 y 04 no tienen semántica aprendida. Los entrenados se ajustan a esta tarea diminuta; no demuestran comprensión general del lenguaje.
- La ventana es fija. La concatenación conserva posiciones y la capa lineal aprende pesos por posición, pero no hay un mecanismo de atención que calcule relevancia según el contenido.
- Aumentar el contexto no garantiza una respuesta única ni una mejora de pérdida en cualquier corpus.

### Comandos para repetir la fase

Desde la raíz del repositorio, con `.venv` activo:

```bash
python src/fase4/01_tokenizacion_basica.py
python src/fase4/02_encode_decode.py
python src/fase4/03_embeddings.py
python src/fase4/04_similitud_embeddings.py
python src/fase4/05_contexto_siguiente_token.py
python src/fase4/06_modelo_lenguaje_basico.py
python src/fase4/07_contexto_dos_tokens.py
```

Cada script es independiente. Los modelos de 06 y 07 se inicializan y entrenan desde cero; no reutilizan los embeddings de 03 o 04.

El incidente de Windows ocurrido en esta fase se conserva en [WinError 4551 al importar PyTorch](#winerror-4551-al-importar-pytorch).

---

## Fase 5: Attention

**Cierre registrado:** 26 de septiembre de 2026. El autor confirmó la fase completada y compartió resultados de ejemplo. Esta sección contrasta ese contexto con los siete scripts de `src/fase5`.

### Objetivo y alcance

Construir atención desde operaciones elementales hasta Multi-Head Causal Self-Attention. Los tokens de ejemplo son `el`, `perro` y `come`.

Los siete scripts usan embeddings definidos manualmente y se ejecutan en CPU tal como están escritos. No cargan la tabla de embeddings entrenada en Fase 4. Los ejercicios 01–05 usan tensores fijos; 06 y 07 crean capas con parámetros entrenables, pero solo calculan e imprimen el forward. No hay targets, loss, optimizador ni actualizaciones de pesos.

### Progresión de los ejercicios

| Archivo | Implementación | Resultado que se inspecciona |
| --- | --- | --- |
| [01_attention_intuicion.py](../../src/fase5/01_attention_intuicion.py) | Usa el embedding de `come` como query; compara con todos los embeddings y calcula una suma ponderada. | Scores, pesos y vector de salida para un token. |
| [02_query_key_value.py](../../src/fase5/02_query_key_value.py) | Define `W_Q`, `W_K` y `W_V` como matrices identidad y calcula Q, K y V. | Separación conceptual de las tres funciones aunque sus valores coincidan. |
| [03_self_attention_todos_tokens.py](../../src/fase5/03_self_attention_todos_tokens.py) | `Q @ K.T`, softmax por fila y `pesos_attention @ V`. | Una salida contextualizada para cada posición, sin máscara. |
| [04_scaled_dot_product_attention.py](../../src/fase5/04_scaled_dot_product_attention.py) | Divide los scores por `sqrt(d_k)` antes del softmax. | Scores escalados y distribución de atención. |
| [05_causal_attention.py](../../src/fase5/05_causal_attention.py) | Máscara triangular superior y `masked_fill(mask, -inf)`. | Peso cero en todas las posiciones futuras. |
| [06_attention_aprendible.py](../../src/fase5/06_attention_aprendible.py) | Clase `CausalSelfAttention` con tres capas `nn.Linear(2, 2, bias=False)`. | Salida `[3, 2]` y pesos `[3, 3]` con parámetros aleatorios entrenables. |
| [07_multi_head_attention.py](../../src/fase5/07_multi_head_attention.py) | Clases `AttentionHead` y `MultiHeadAttention`, dos cabezas y proyección final. | Salida `[3, 4]` y dos matrices de pesos `[3, 3]`. |

### Atención para un token y para toda la secuencia

En 01–05, los embeddings manuales son:

```text
el    → [1, 0]
perro → [0, 1]
come  → [1, 1]
```

En el ejercicio 01, la query de `come` produce scores `[1, 1, 2]` y softmax aproximadamente `[0.2119, 0.2119, 0.5761]`. La suma ponderada produce un vector cercano a `[0.7881, 0.7881]`. Estos números se derivan de los tensores fijos; no son relaciones semánticas aprendidas. En particular, `el` y `perro` reciben el mismo peso en este ejemplo.

Con Q, K y V de toda la secuencia, `Q @ K.T` crea una matriz `[3, 3]`. La fila indica la posición que consulta y la columna indica la posición consultada. Los scores son puntuaciones sin normalizar; se convierten en pesos al aplicar softmax por fila, sobre las keys.

En las matrices manuales se escribe `Q = X @ W_Q`. Con `nn.Linear`, PyTorch calcula la entrada por la transpuesta del atributo `weight` y añade bias si existe; en las proyecciones Q/K/V de estos ejercicios el bias está desactivado.

### Escalado y máscara causal

La variante escalada divide por la raíz de la dimensión de las keys, no por la cantidad de tokens:

```text
scores = QKᵀ / sqrt(d_k)
pesos = softmax(scores con máscara, por fila)
salida = pesos @ V
```

El escalado reduce la tendencia de los productos punto a crecer con la dimensión y producir distribuciones excesivamente concentradas.

La máscara usa `torch.triu(..., diagonal=1).bool()`:

```text
          el     perro  come
el        False  True   True
perro     False  False  True
come      False  False  False
```

`True` marca una posición bloqueada. Antes del softmax se sustituye su score por `-inf`; su exponencial es cero. Como cada fila conserva al menos la posición propia, puede normalizarse sin que toda la fila quede bloqueada.

Así, `el` solo consulta `el`; `perro` consulta `el` y `perro`; `come` consulta las tres posiciones. La diagonal permanece visible. Los pesos de cada fila suman aproximadamente 1 y los futuros tienen peso cero.

### Parámetros entrenables y múltiples cabezas

En 06, `W_Q`, `W_K` y `W_V` son capas de un `nn.Module`. Sus pesos están registrados para que autograd pueda calcular gradientes si se conecta una pérdida y se llama a `backward()`. El script no realiza ese paso.

En 07, cada cabeza recibe los cuatro componentes del embedding y tiene sus propias proyecciones `Linear(4, 2, bias=False)`. No se limita a tomar una mitad fija del embedding. `nn.ModuleList` registra las cabezas como submódulos, incluyendo sus parámetros.

| Etapa | Forma en el ejemplo 07 |
| --- | --- |
| Embeddings de entrada | `[3, 4]` |
| Número de cabezas | 2 |
| Dimensión por cabeza | `4 // 2 = 2` |
| Q, K y V por cabeza | `[3, 2]` |
| Scores y pesos por cabeza | `[3, 3]` |
| Salida por cabeza | `[3, 2]` |
| `torch.cat(salidas, dim=1)` | `[3, 4]` |
| Proyección `nn.Linear(4, 4)` | `[3, 4]` |

El constructor comprueba `embedding_dim % num_heads == 0`. Cada cabeza escala por `sqrt(head_dim)`. La proyección final mezcla las características concatenadas y sí incluye bias, porque usa el valor predeterminado de `nn.Linear`.

```text
Embeddings [3, 4]
  ├─ Cabeza 1: Q/K/V → scores / sqrt(2) → máscara → softmax → pesos @ V [3, 2]
  └─ Cabeza 2: Q/K/V → scores / sqrt(2) → máscara → softmax → pesos @ V [3, 2]
          ↓
Concatenación [3, 4] → proyección Linear(4, 4) → salida [3, 4]
```

### Resultado reportado y cómo interpretarlo

El autor reportó una salida de tres tokens por cuatro dimensiones y estas filas de pesos para `perro`:

| Cabeza | Atención a el | Atención a perro | Atención a come |
| --- | ---: | ---: | ---: |
| 1 | 0.5621 | 0.4379 | 0.0000 |
| 2 | 0.4425 | 0.5575 | 0.0000 |

Los valores proceden del contexto compartido por el autor, no de una nueva ejecución al editar la documentación. Ambas filas suman 1 con el redondeo mostrado y respetan el bloqueo del futuro. Las cabezas pueden producir distribuciones distintas por tener parámetros independientes; aquí esos parámetros se inicializan aleatoriamente y no se han optimizado.

La representación es contextualizada porque combina values de las posiciones permitidas. Esto no demuestra por sí solo comprensión del lenguaje ni especialización semántica de las cabezas.

### Límites y siguiente fase

- No se fija una semilla en 06 ni 07; sus números pueden cambiar entre ejecuciones.
- No hay entrenamiento, validación ni métricas de pérdida en esta fase.
- Los embeddings son manuales y no hay tokenizador integrado.
- Se procesa una sola secuencia de forma `[seq_len, embedding_dim]`, sin dimensión de batch.
- No hay embeddings posicionales, conexiones residuales, LayerNorm ni red feed-forward.
- No existe una salida sobre el vocabulario ni generación de texto integrada con atención.

La Fase 6 integró esos componentes para construir un bloque Transformer y avanzar hacia MiniGPT. La Fase 5 queda completada como construcción y comprensión del mecanismo de atención.

### Comandos para repetir la fase

Desde la raíz del proyecto, con `.venv` activo:

```bash
python src/fase5/01_attention_intuicion.py
python src/fase5/02_query_key_value.py
python src/fase5/03_self_attention_todos_tokens.py
python src/fase5/04_scaled_dot_product_attention.py
python src/fase5/05_causal_attention.py
python src/fase5/06_attention_aprendible.py
python src/fase5/07_multi_head_attention.py
```

Cada script es independiente. La máscara de 06 y 07 se crea en `x.device`, pero los ejemplos no trasladan el modelo ni los datos a GPU.

---

## Fase 6: Transformer

**Estado:** completada; cierre registrado el 27 de septiembre de 2026 a partir de la confirmación y el contexto del autor.

Se conectó el recorrido completo: `tokens → embeddings + posición → bloques Transformer → logits → entrenamiento → generación → persistencia`. La implementación construye atención y bloques con operaciones y capas de PyTorch; no utiliza un modelo preentrenado ni un bloque Transformer ya ensamblado.

### Ejercicios implementados

| Archivo | Propósito |
| --- | --- |
| [01_bloque_transformer.py](../../src/fase6/01_bloque_transformer.py) | Construye atención causal, dos residuales, LayerNorm y feed-forward; conserva la forma `[3, 4]`. |
| [02_positional_embeddings.py](../../src/fase6/02_positional_embeddings.py) | Suma embeddings manuales de tokens y embeddings posicionales entrenables para posiciones 0, 1 y 2. |
| [03_transformer_con_posicion.py](../../src/fase6/03_transformer_con_posicion.py) | Integra `nn.Embedding` de tokens y posiciones con un bloque; `MiniTransformer` recibe IDs. |
| [04_transformer_apilado.py](../../src/fase6/04_transformer_apilado.py) | Apila tres bloques independientes en `nn.ModuleList` y conserva `[3, 4]`. |
| [05_salida_vocabulario.py](../../src/fase6/05_salida_vocabulario.py) | Añade `Linear(4, 3)` a MiniGPT; muestra logits, softmax y predicciones sin entrenamiento. |
| [06_entrenar_minigpt.py](../../src/fase6/06_entrenar_minigpt.py) | Entrena MiniGPT con objetivos desplazados y muestra las cinco predicciones; no guarda archivos. |
| [07_generar_texto.py](../../src/fase6/07_generar_texto.py) | Entrena un modelo nuevo, guarda `minigpt.pth` y genera cinco tokens desde `el`. |
| [08_cargar_modelo.py](../../src/fase6/08_cargar_modelo.py) | Reconstruye arquitectura y vocabulario definidos en el código, carga `minigpt.pth` y genera sin entrenar. |
| [09_checkpoint_completo.py](../../src/fase6/09_checkpoint_completo.py) | Entrena un modelo nuevo, guarda pesos, configuración y vocabulario en `minigpt_checkpoint.pth`, y genera. |
| [10_cargar_checkpoint.py](../../src/fase6/10_cargar_checkpoint.py) | Lee el checkpoint, reconstruye MiniGPT y los diccionarios del vocabulario, carga pesos y genera sin entrenar. |

01–05 ejecutan ejemplos en CPU con parámetros iniciales, sin pérdida ni optimizador. 06–10 seleccionan CUDA si está disponible o CPU en caso contrario. Los scripts repiten las clases de la arquitectura para poder estudiarlas por separado; los cargadores 08 y 10 sí dependen del archivo guardado correspondiente.

### Bloque Transformer y posiciones

El bloque sigue el orden post-normalización: primero suma la transformación a la entrada y después aplica LayerNorm.

```text
x [T, D]
  → Multi-Head Causal Self-Attention
  → LayerNorm(x + attention(x))
  → FeedForward: Linear(D, 4D) → ReLU → Linear(4D, D)
  → LayerNorm(x + feed_forward(x))
  → salida [T, D]
```

La segunda suma usa la representación resultante de la primera normalización. Attention comunica información entre posiciones permitidas; la red feed-forward aplica la misma transformación a cada posición por separado. Las conexiones residuales conservan un camino para la entrada y los gradientes; LayerNorm normaliza las características de cada token y tiene parámetros aprendibles.

Cada cabeza calcula `softmax(QKᵀ / sqrt(head_dim) + máscara) @ V`. Las posiciones futuras se bloquean con `-inf` antes de softmax. Las salidas de las cabezas se concatenan y se proyectan de nuevo a dimensión `D`.

Los embeddings posicionales son aprendidos, no sinusoidales. Para una secuencia de longitud `T`, se consultan posiciones `0…T-1` y se suman a los embeddings de los tokens. El modelo puede así representar el mismo token de forma distinta según su posición y contexto.

El ejercicio 01 conserva `[3, 4] → [3, 4]`; 03 recibe IDs y produce esa misma forma. En 04 se apilan tres bloques con parámetros independientes mediante `nn.ModuleList`. En 05, `nn.Linear(4, 3)` convierte cada vector en tres logits, uno por palabra del vocabulario de ese ejemplo.

### Arquitectura y dimensiones del modelo entrenado

```text
Texto → split() → IDs [T]
  → token embeddings [T, 8] + position embeddings [T, 8]
  → bloque Transformer 1 [T, 8]
  → bloque Transformer 2 [T, 8]
  → Linear(8, 5) → logits [T, 5]
      ├─ entrenamiento: CrossEntropyLoss(logits, objetivos)
      └─ generación: logits[-1] → softmax → argmax → añadir token
```

| Elemento | Valor en 06, 07 y 09; recuperado o reconstruido en 08 y 10 |
| --- | --- |
| Corpus | `el perro come el gato duerme` |
| Vocabulario / IDs | `come: 0, duerme: 1, el: 2, gato: 3, perro: 4` |
| Embedding / cabezas / bloques | `8 / 2 / 2` |
| Dimensión por cabeza | `8 // 2 = 4` |
| Feed-forward | `8 → 32 → 8`, con ReLU |
| Máximo de posiciones | `max_seq_len=20` |
| Longitud usada para entrenar | 5 tokens, una sola secuencia sin dimensión de batch |
| Épocas y optimizador | 2000; Adam con `lr=0.01` |
| Pérdida | `nn.CrossEntropyLoss()` sobre las cinco posiciones |

### Entrenamiento autoregresivo

El corpus tiene seis tokens. Se desplaza un token para crear cinco objetivos:

```text
Entrada X:  el     perro  come  el    gato
Objetivo y: perro  come   el    gato  duerme

IDs X: [2, 4, 0, 2, 3]
IDs y: [4, 0, 2, 3, 1]
```

Cada forward calcula las cinco predicciones a la vez, pero la máscara causal impide consultar tokens posteriores a cada posición. Los logits tienen forma `[5, 5]` y los targets `[5]`, de tipo `torch.long`. Se pasan los logits directamente a `CrossEntropyLoss`, sin softmax previo.

El ciclo es `forward → loss → zero_grad → backward → optimizer.step`. Se ajustan los embeddings de tokens y posiciones usadas, las proyecciones de atención, las redes feed-forward, las normalizaciones y la capa de salida.

El autor compartió estos resultados del entrenamiento:

| Medida | Valor reportado aproximado |
| --- | ---: |
| Pérdida en época 0 | 2.004 |
| Pérdida final | 0.000035 |

```text
el       → perro
perro    → come
come     → el
el       → gato
gato     → duerme
```

Las métricas y predicciones provienen del contexto del autor, no de una nueva ejecución. El valor que los scripts imprimen como pérdida final es el calculado en el último forward del ciclo, antes de su última actualización de pesos.

Los dos usos de `el` reciben objetivos diferentes porque sus representaciones disponen de distinta posición y contexto. Este ejemplo muestra que el modelo puede distinguir ambas apariciones; no permite aislar cuánto depende de posición frente a contexto, ni demuestra generalización.

### Generación autoregresiva

La inferencia parte únicamente de `el`. En cada paso se procesa el prefijo completo, se toma `logits[-1]`, se aplica softmax sobre el vocabulario y se elige `argmax`. El token elegido se añade a la entrada siguiente:

```text
el
el perro
el perro come
el perro come el
el perro come el gato
el perro come el gato duerme
```

La llamada usa `cantidad_tokens=5`: cinco tokens nuevos y seis palabras en total. Es selección greedy, sin muestreo ni temperatura. `modelo.eval()` activa el modo de evaluación y `torch.no_grad()` evita construir el grafo de gradientes durante el forward. La generación no modifica los pesos.

En entrenamiento se conocen todos los tokens del ejemplo y se predicen todas las posiciones con máscara; en generación cada nuevo token depende del anterior y se obtiene en una iteración separada.

### Persistencia: pesos y checkpoint

El primer formato, escrito por 07, es:

```python
torch.save(modelo.state_dict(), "minigpt.pth")
```

08 necesita reconstruir la misma arquitectura y el mismo vocabulario, incluido su orden. Lee el archivo con `torch.load(..., map_location=device)`, aplica `load_state_dict` y genera sin optimizador ni entrenamiento. Esto permite recuperar los parámetros aprendidos en otro proceso.

09 amplía el formato:

```text
minigpt_checkpoint.pth
├── model_state_dict
│   └── parámetros del modelo
├── config
│   ├── embedding_dim: 8
│   ├── num_heads: 2
│   ├── num_blocks: 2
│   └── max_seq_len: 20
└── vocabulario: ['come', 'duerme', 'el', 'gato', 'perro']
```

10 lee `config` y `vocabulario`, reconstruye los diccionarios por el orden guardado, calcula `vocab_size=len(vocabulario)`, instancia MiniGPT y carga `model_state_dict`. El autor confirmó la configuración y el vocabulario recuperados, el mensaje de pesos cargados y la salida:

```text
el perro come el gato duerme
```

El checkpoint conserva la información necesaria para reconstruir esta instancia **junto con el código de la arquitectura**. Aunque el ejercicio lo llama «completo», su alcance es la recuperación para inferencia: no guarda el estado de Adam, la época/paso ni los estados aleatorios para reanudar exactamente el entrenamiento.

### Comandos y archivos necesarios

Desde la raíz del repositorio, con `.venv` activo, los primeros seis ejercicios pueden ejecutarse por separado:

```bash
python src/fase6/01_bloque_transformer.py
python src/fase6/02_positional_embeddings.py
python src/fase6/03_transformer_con_posicion.py
python src/fase6/04_transformer_apilado.py
python src/fase6/05_salida_vocabulario.py
python src/fase6/06_entrenar_minigpt.py
```

Para probar el guardado y la carga de pesos, ejecutar en este orden:

```bash
python src/fase6/07_generar_texto.py
python src/fase6/08_cargar_modelo.py
```

Para probar el checkpoint con configuración y vocabulario:

```bash
python src/fase6/09_checkpoint_completo.py
python src/fase6/10_cargar_checkpoint.py
```

07 y 09 entrenan desde cero de forma independiente; no continúan el entrenamiento de 06 ni cargan un archivo previo. Cada uno sobrescribe su archivo de salida si ya existe. Si solo se quiere recuperar un modelo guardado, basta ejecutar 08 o 10 según el formato.

Las rutas de los `.pth` son relativas al **directorio de ejecución**, no al de los scripts. Con estos comandos quedan en la raíz del repositorio. Un cargador fallará si su archivo no existe allí; ejecutar primero el guardado correspondiente o situarse en el directorio que contiene el archivo.

### Límites y transición a Fase 7

- Corpus de seis tokens y cinco palabras únicas, sin separación de entrenamiento y validación.
- La pérdida pequeña y la secuencia reconstruida muestran ajuste al ejemplo; no prueban comprensión del español ni capacidad de generalizar.
- No se fija una semilla; las pérdidas y generaciones pueden variar al volver a entrenar.
- El vocabulario es cerrado, sin token desconocido ni token de fin de secuencia; la generación termina por la cantidad solicitada.
- El modelo acepta una secuencia `[T]`, sin procesamiento por lotes.
- `max_seq_len=20` limita cada entrada a posiciones 0–19. No hay recorte de contexto; pasar una entrada mayor excede la tabla posicional. Tener veinte posiciones disponibles no significa que se hayan entrenado todas: aquí solo se usan las primeras cinco.
- No se registraron nuevas mediciones de tiempo, VRAM ni rendimiento comparado entre CPU y GPU.
- No hay temperatura de muestreo, caché de atención ni estado de entrenamiento completo en el checkpoint.

La Fase 6 queda cerrada: MiniGPT recibe tokens, se entrena, genera, guarda su aprendizaje y lo recupera. El siguiente paso es la [Fase 7](01_Plan_de_Trabajo.md#fase-7-entrenar-miniai), ampliando el corpus y evaluando el modelo con datos de validación.

---

## Equipo principal

La laptop principal es una **Lenovo LOQ**.

| Componente | Hardware registrado |
| --- | --- |
| CPU | Intel Core i5-13450HX |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |

Se utiliza para el desarrollo principal, el entrenamiento de modelos y la inferencia de MiniAI, incluyendo el trabajo con PyTorch, CUDA y la RTX 5050.

---

## Software y versiones registradas

Estas versiones corresponden al registro previo del entorno; no se volvieron a comprobar al convertir este documento a Markdown.

| Componente | Versión o configuración registrada |
| --- | --- |
| Python | `3.13.15` |
| pip | `26.2.1` |
| PyTorch | `2.14.0+cu132` |
| torchvision | `0.29.0+cu132` |
| CUDA usada por PyTorch | `13.2` |
| Arquitectura GPU registrada | `sm_120` |
| NumPy | `2.5.2` |
| Sistema operativo | Windows 11 Home Single Language |
| Editor | Visual Studio Code |
| Terminal principal | Git Bash |

---

## Estructura del proyecto

Ruta principal en Windows:

```text
C:\Users\mephi\Desktop\MiniAI
```

Ruta de trabajo desde Git Bash:

```bash
cd ~/desktop/MiniAI
```

Archivos de trabajo y documentación:

```text
MiniAI/
├── documentation/
│   └── code docs/
│       ├── 00_breviario.md
│       ├── 01_Plan_de_Trabajo.md
│       └── 02_Memoria_Tecnica.md
├── src/
│   ├── fase1/
│   │   ├── 01_neurona_basica.py
│   │   ├── 02_neurona_entrenamiento.py
│   │   └── 03_frontera_decision.py
│   ├── fase2/
│   │   ├── 01_red_xor.py
│   │   └── 02_backprop_xor.py
│   ├── fase3/
│   │   ├── 01_tensores_gpu.py
│   │   ├── 02_xor_pytorch.py
│   │   ├── 03_autograd_basico.py
│   │   ├── 04_gradientes_red.py
│   │   ├── 05_peso_antes_despues.py
│   │   ├── 06_cpu_vs_gpu.py
│   │   └── 07_matrices_cpu_vs_gpu.py
│   ├── fase4/
│   │   ├── 01_tokenizacion_basica.py
│   │   ├── 02_encode_decode.py
│   │   ├── 03_embeddings.py
│   │   ├── 04_similitud_embeddings.py
│   │   ├── 05_contexto_siguiente_token.py
│   │   ├── 06_modelo_lenguaje_basico.py
│   │   └── 07_contexto_dos_tokens.py
│   ├── fase5/
│   │   ├── 01_attention_intuicion.py
│   │   ├── 02_query_key_value.py
│   │   ├── 03_self_attention_todos_tokens.py
│   │   ├── 04_scaled_dot_product_attention.py
│   │   ├── 05_causal_attention.py
│   │   ├── 06_attention_aprendible.py
│   │   └── 07_multi_head_attention.py
│   └── fase6/
│       ├── 01_bloque_transformer.py
│       ├── 02_positional_embeddings.py
│       ├── 03_transformer_con_posicion.py
│       ├── 04_transformer_apilado.py
│       ├── 05_salida_vocabulario.py
│       ├── 06_entrenar_minigpt.py
│       ├── 07_generar_texto.py
│       ├── 08_cargar_modelo.py
│       ├── 09_checkpoint_completo.py
│       └── 10_cargar_checkpoint.py
├── .venv/
├── .gitignore
└── README.md
```

El entorno `.venv/` es local. El registro previo indica que Git está inicializado y que la rama observada fue `develop`.

---

## Entorno virtual

El entorno se llama `.venv` y su ubicación registrada es:

```text
C:\Users\mephi\Desktop\MiniAI\.venv
```

Las librerías del proyecto deben instalarse dentro de este entorno virtual.

### Activar el entorno desde Git Bash

```bash
cd ~/desktop/MiniAI
source .venv/Scripts/activate
```

Al activarlo, la terminal debe mostrar un prefijo similar a este:

```text
(.venv)
mephi@Baticomputadora MINGW64 ~/desktop/MiniAI (develop)
$
```

### Comprobar el intérprete y pip

```bash
which python
python --version
pip --version
```

La ruta de Python debe apuntar al entorno, por ejemplo:

```text
/c/Users/mephi/desktop/MiniAI/.venv/Scripts/python
```

La salida de `pip --version` debe indicar una ruta dentro de `MiniAI\.venv\Lib\site-packages`.

### Desactivar el entorno

```bash
deactivate
```

El prefijo `(.venv)` desaparecerá.

---

## GPU y Lenovo Legion

### Problema registrado

PyTorch estaba instalado con CUDA, pero `torch.cuda.is_available()` devolvía `False`. Además, `nvidia-smi` no podía comunicarse correctamente con el driver NVIDIA.

### Solución aplicada

1. Abrir Lenovo Legion / Lenovo Legion Space.
2. Entrar en **Configuración conveniente → Modalidad de funcionamiento de la GPU**.
3. Cambiar temporalmente de **Modalidad híbrida** a **Modalidad dGPU**.
4. Reiniciar la computadora.

Después del reinicio, `nvidia-smi` volvió a detectar la **NVIDIA GeForce RTX 5050 Laptop GPU** y PyTorch pudo utilizarla.

### Si vuelve a ocurrir

Revisar primero que la GPU NVIDIA esté disponible en Lenovo Legion. Si aparece el mismo problema, la solución que funcionó en este equipo fue probar la modalidad dGPU y reiniciar.

Después, ejecutar:

```bash
nvidia-smi
```

Si la RTX 5050 aparece correctamente, continuar con las comprobaciones de PyTorch y CUDA de la siguiente sección.

---

## PyTorch y CUDA

### Instalación registrada

El registro del entorno indica que PyTorch se instaló con soporte para CUDA 13.2 mediante:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

También se instalaron NumPy y Matplotlib:

```bash
pip install numpy matplotlib
```

Estos son los comandos usados en la instalación original; no es necesario ejecutarlos cada vez que se retoma el proyecto.

### Comprobar PyTorch

Con el entorno virtual activo:

```bash
python -c "import torch; print(torch.__version__)"
```

Versión del registro previo:

```text
2.14.0+cu132
```

### Comprobar CUDA y la GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No detectada')"
```

Salida esperada según el registro:

```text
CUDA: True
GPU: NVIDIA GeForce RTX 5050 Laptop GPU
```

### Comprobar las arquitecturas soportadas

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

Salida observada en la comprobación original:

```text
['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

La memoria original registra `sm_120` como la arquitectura utilizada por la RTX 5050.

### Comprobar la versión de CUDA usada por PyTorch

```bash
python -c "import torch; print(torch.version.cuda)"
```

Versión del registro previo:

```text
13.2
```

---

## Primera prueba de GPU

Con CUDA disponible y el entorno activo, la prueba registrada fue:

```bash
python -c "import torch; a=torch.tensor([1.,2.,3.],device='cuda'); b=torch.tensor([4.,5.,6.],device='cuda'); print(a*b); print('Dispositivo:', (a*b).device)"
```

Salida esperada:

```text
tensor([ 4., 10., 18.], device='cuda:0')
Dispositivo: cuda:0
```

El dispositivo `cuda:0` permite comprobar que la operación se ejecutó en la GPU. Esta prueba forma parte del registro de la Fase 0 y no se repitió al reorganizar la documentación.

---

## Rutina para retomar el proyecto

1. Si se va a utilizar GPU, verificar que la RTX 5050 esté disponible. Consultar Lenovo Legion si CUDA presenta el problema registrado.
2. Abrir Git Bash y entrar al proyecto con `cd ~/desktop/MiniAI`.
3. Activar el entorno con `source .venv/Scripts/activate` y confirmar el prefijo `(.venv)`.
4. Abrir Visual Studio Code con `code .`.
5. Verificar que VS Code use `MiniAI\.venv\Scripts\python.exe`.
6. Si la fase requiere GPU, realizar una comprobación rápida de CUDA antes de entrenar.
7. Ejecutar el ejercicio correspondiente desde Git Bash o VS Code.

Si VS Code usa otro intérprete, presionar `Ctrl + Shift + P`, buscar **Python: Select Interpreter** y seleccionar el Python dentro de `MiniAI\.venv`.

Comprobación rápida opcional para fases con GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

Salida esperada según el registro del entorno:

```text
True
NVIDIA GeForce RTX 5050 Laptop GPU
```

Forma general de ejecutar un script, sustituyendo el nombre por un archivo real:

```bash
python src/nombre_script.py
```

---

## Comandos de diagnóstico

Ejecutar desde Git Bash con `.venv` activo:

| Qué revisar | Comando |
| --- | --- |
| Versión de Python | `python --version` |
| Versión y ruta de pip | `pip --version` |
| Ruta de Python | `which python` |
| GPU NVIDIA | `nvidia-smi` |
| Versión de PyTorch | `python -c "import torch; print(torch.__version__)"` |
| Disponibilidad de CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| GPU detectada | `python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"` |
| CUDA usada por PyTorch | `python -c "import torch; print(torch.version.cuda)"` |
| Arquitecturas compatibles | `python -c "import torch; print(torch.cuda.get_arch_list())"` |
| Información completa del entorno | `python -m torch.utils.collect_env` |

---

## Errores y soluciones

### WinError 4551 al importar PyTorch

**Fecha:** 26 de septiembre de 2026. **Estado:** resuelto según confirmación del autor.

Al ejecutar `03_embeddings.py` o la comprobación `python -c "import torch; print(torch.__version__)"`, Windows bloqueó `.venv/Lib/site-packages/torch/lib/shm.dll`:

```text
OSError: [WinError 4551] Una directiva de Control de aplicaciones bloqueó este archivo.
```

El fallo aparecía en `import torch`, antes de ejecutar la lógica de embeddings. Usar `py` en lugar de `python` produjo el mismo error.

La consulta del registro `Microsoft-Windows-CodeIntegrity/Operational` mostró eventos 3077 para `shm.dll` y eventos 3118 de Smart App Control a las 11:32:06, 11:32:30 y 11:37:30. En la última comprobación previa a la resolución, `VerifiedAndReputablePolicyState` todavía valía `1` (activado).

Se orientó a revisar **Seguridad de Windows → Control de aplicaciones y navegador → Configuración de Control inteligente de aplicaciones**, seleccionar **Desactivado** y confirmar; si ya figuraba desactivado, guardar el trabajo y reiniciar para volver a comprobar. Después, el autor confirmó que funcionó y completó la fase. No quedó registrado si el paso decisivo fue confirmar el cambio, reiniciar o ambos.

El diagnóstico corresponde a una política de confianza/firma de Windows; el mensaje por sí solo no demuestra que la DLL sea malware. Smart App Control es una protección adicional al antivirus y desactivarlo afecta al equipo completo. La posibilidad de reactivarlo depende de las actualizaciones de Windows. Consultar las [preguntas frecuentes de Microsoft](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions) y la [referencia de estados y eventos](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/test-your-app-with-smart-app-control).

Si se repite, comprobar el estado real de esa protección y los eventos recientes antes de atribuirlo al código o reinstalar PyTorch. La prueba mínima de importación es:

```bash
python -c "import torch; print(torch.__version__)"
```

### Un carácter extra en el comando

Se escribió `which python~` por accidente. El comando correcto es:

```bash
which python
```

### El intérprete de Python muestra `...`

En el caso registrado, Python esperaba que terminara un bloque multilínea. Una línea vacía permite cerrar el bloque.

Para pruebas rápidas también se puede usar `python -c "..."`, sustituyendo `...` por las instrucciones de Python que se quieren ejecutar.

### CUDA no disponible aunque PyTorch incluye soporte CUDA

Se observó `torch.cuda.is_available()` con resultado `False`, aunque PyTorch mostraba `2.14.0+cu132` y `torch.version.cuda` mostraba `13.2`.

La causa encontrada en ese momento fue que la GPU NVIDIA no estaba correctamente expuesta al sistema bajo la configuración previa. La solución aplicada fue cambiar a **Modalidad dGPU** en Lenovo Legion y reiniciar, como se describe en [GPU y Lenovo Legion](#gpu-y-lenovo-legion).

---

## Recomendaciones de trabajo

Estas recomendaciones se conservan de la memoria original:

- No instalar CUDA Toolkit adicional salvo que una fase futura lo requiera.
- Revisar la disponibilidad de la GPU antes de reinstalar PyTorch por un fallo de CUDA.
- Mantener el entorno virtual separado e instalar dependencias con `.venv` activo.
- Guardar un registro de las dependencias importantes.
- Incorporar frameworks cuando sean necesarios para la fase de aprendizaje.
- Comprobar CUDA antes de entrenamientos largos que utilicen GPU.
- Mantener la laptop conectada a corriente durante el entrenamiento.
- Vigilar temperatura y VRAM en las fases de entrenamiento.
