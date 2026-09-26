# Memoria técnica de MiniAI

Este documento reúne la configuración del entorno, los comandos de trabajo y los aprendizajes técnicos de MiniAI.

La intención es tener una referencia rápida para retomar el proyecto después de reiniciar la computadora o al comenzar una nueva fase.

**Último avance registrado:** 26 de septiembre de 2026.
**Reorganización de la documentación:** 19 de septiembre de 2026.  
**Punto para retomar:** Fase 5, self-attention.

Las versiones y los resultados de GPU se conservan como registro del entorno anterior. La actualización de Fase 4 recoge la ejecución y el cierre confirmados por el autor, su contexto de aprendizaje y la revisión del código. No se volvieron a entrenar los modelos al editar esta documentación; tampoco se midieron nuevas pérdidas, tiempos ni resultados de CUDA.

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

La Fase 4 está completada y el siguiente trabajo es self-attention. Para repasar el último modelo antes de comenzar la Fase 5:

```bash
python src/fase4/07_contexto_dos_tokens.py
```

Este script vuelve a entrenar desde cero en cada ejecución. Usa CUDA si está disponible y, en caso contrario, CPU. La lista de los siete ejercicios está en [Fase 4](#fase-4-tokenización-embeddings-y-predicción-de-siguiente-token).

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

La fecha de corte del avance es el **26 de septiembre de 2026**.

| Fase | Estado registrado | Alcance |
| --- | --- | --- |
| 0. Laboratorio | Completada | Configuración y pruebas de GPU según el registro previo. |
| 1. Neurona artificial | Ejercicios implementados | Neurona básica, entrenamiento de AND y frontera de decisión. |
| 2. Red neuronal | Trabajada | Red XOR con NumPy y ejercicio de backpropagation. |
| 3. PyTorch y GPU | Completada | Tensores en CPU/GPU, CUDA, XOR en PyTorch, autograd, gradientes, optimizador y benchmark. |
| 4. Tokenización y embeddings | Completada | Texto e IDs, embeddings, similitud coseno y modelos de siguiente token con contextos de uno y dos tokens. |
| 5. Self-attention | Pendiente; siguiente fase | Query, Key, Value, puntuaciones y pesos de atención. |

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

La siguiente fase es la [Fase 5 del plan](01_Plan_de_Trabajo.md#fase-5-self-attention): self-attention.

Partir de los embeddings del contexto y construir Query, Key y Value. Después, calcular puntuaciones con `QK^T`, escalarlas, aplicar softmax y usar los pesos obtenidos para combinar los values. Más adelante se incorporarán máscara causal y múltiples cabezas.

El modelo actual conserva el orden al concatenar embeddings y puede aprender pesos distintos para cada posición. Todavía no calcula pesos de atención que dependan del contenido del contexto.

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
│   └── fase4/
│       ├── 01_tokenizacion_basica.py
│       ├── 02_encode_decode.py
│       ├── 03_embeddings.py
│       ├── 04_similitud_embeddings.py
│       ├── 05_contexto_siguiente_token.py
│       ├── 06_modelo_lenguaje_basico.py
│       └── 07_contexto_dos_tokens.py
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
