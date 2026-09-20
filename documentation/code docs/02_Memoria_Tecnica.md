# Memoria técnica de MiniAI

Este documento reúne la configuración del entorno, los comandos de trabajo y los aprendizajes técnicos de MiniAI.

La intención es tener una referencia rápida para retomar el proyecto después de reiniciar la computadora o al comenzar una nueva fase.

**Último avance registrado:** 19 de septiembre de 2026.
**Reorganización de la documentación:** 19 de septiembre de 2026.  
**Punto para retomar:** Fase 4, tokenización y embeddings.

Las versiones y los resultados de GPU se conservan como registro del entorno anterior. Esta reorganización no incluye nuevas ejecuciones de los ejercicios ni una comprobación del entorno o de CUDA.

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
- [Fase 3: PyTorch y entrenamiento con GPU](#fase-3-pytorch-y-entrenamiento-con-gpu)
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

La fecha de corte del avance es el **19 de septiembre de 2026**.

| Fase | Estado registrado | Alcance |
| --- | --- | --- |
| 0. Laboratorio | Completada | Configuración y pruebas de GPU según el registro previo. |
| 1. Neurona artificial | Ejercicios implementados | Neurona básica, entrenamiento de AND y frontera de decisión. |
| 2. Red neuronal | Trabajada | Red XOR con NumPy y ejercicio de backpropagation. |
| 3. PyTorch y GPU | Completada | Tensores en CPU/GPU, CUDA, XOR en PyTorch, autograd, gradientes, optimizador y benchmark. |

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

La siguiente fase es la [Fase 4 del plan](01_Plan_de_Trabajo.md#fase-4-tokenización-y-embeddings): tokenización y embeddings.

Primera tarea recomendada:

- Crear un corpus pequeño.
- Construir un vocabulario.
- Convertir texto a IDs.
- Convertir IDs de vuelta a texto.
- Crear embeddings para representar tokens como vectores.

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
│   └── fase3/
│       ├── 01_tensores_gpu.py
│       ├── 02_xor_pytorch.py
│       ├── 03_autograd_basico.py
│       ├── 04_gradientes_red.py
│       ├── 05_peso_antes_despues.py
│       ├── 06_cpu_vs_gpu.py
│       └── 07_matrices_cpu_vs_gpu.py
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
