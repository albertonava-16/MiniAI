# Breviario de conceptos de MiniAI

Este documento reúne los conceptos aprendidos durante la construcción de MiniAI.

La intención no es memorizar todas las definiciones, sino tener una referencia rápida que podamos ampliar conforme avance el proyecto.

**Última ampliación:** 26 de septiembre de 2026, [conceptos de la Fase 5](#fase-5-attention-y-representaciones-contextualizadas). También se conservan los [conceptos de la Fase 4](#fase-4-del-texto-a-la-predicción). Los ejercicios y resultados de atención están en la [memoria técnica](02_Memoria_Tecnica.md#fase-5-attention).

---

## Inteligencia artificial

La inteligencia artificial es un campo que busca construir sistemas capaces de realizar tareas que normalmente relacionamos con la inteligencia, como:

- Reconocer imágenes.
- Comprender texto.
- Generar respuestas.
- Clasificar información.
- Encontrar patrones.
- Tomar decisiones.

---

## Machine Learning

Machine Learning o aprendizaje automático es una rama de la inteligencia artificial.

En lugar de programar directamente todas las reglas, proporcionamos ejemplos para que el sistema encuentre valores que le permitan resolver un problema.

En nuestro ejercicio no escribimos:

```python
if x1 == 1 and x2 == 1:
    return 1
```

Le proporcionamos ejemplos:

```text
(0, 0) → 0
(0, 1) → 0
(1, 0) → 0
(1, 1) → 1
```

La neurona ajustó sus parámetros hasta encontrar una forma de responder correctamente.

---

## Dataset

Un `dataset` es el conjunto de datos que utilizamos para entrenar o evaluar un modelo.

Nuestro primer dataset es:

```python
datos = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]
```

Cada elemento contiene:

```text
entrada 1, entrada 2, resultado esperado
```

---

## AND

`AND` es una operación lógica que produce `1` únicamente cuando sus dos entradas son `1`.

| x1 | x2 | AND |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

AND es solamente el problema que utilizamos para enseñar a nuestra primera neurona.

No es una parte obligatoria de todas las redes neuronales.

---

## `and` en Python

`and`, escrito en minúsculas, es un operador lógico de Python.

Permite comprobar que dos condiciones sean verdaderas:

```python
if x1 == 1 and x2 == 1:
    print("Las dos condiciones son verdaderas")
```

No debe confundirse con el dataset de AND que estamos usando para entrenar la neurona.

---

## Entrada o input

Las entradas son los valores que recibe una neurona.

En nuestro ejemplo:

```python
x1 = 1
x2 = 0
```

`x1` y `x2` son las entradas.

En un modelo real, las entradas podrían representar palabras, píxeles, sonidos, temperaturas o cualquier dato convertido en números.

---

## Neurona artificial

Una neurona artificial es una operación matemática que:

1. Recibe entradas.
2. Multiplica cada entrada por un peso.
3. Suma un bias.
4. aplica una función de activación.
5. Produce una salida.

En nuestro código:

```python
z = x1 * w1 + x2 * w2 + bias
output = sigmoid(z)
```

Una neurona puede estar formada por muy pocas líneas. Lo importante es la operación matemática que realiza.

---

## Weight o peso

Un `weight` es un número que determina cuánto influye una entrada en el resultado.

En nuestro código:

```python
w1 = 0.8
w2 = 0.4
```

Cada entrada se multiplica por su peso:

```python
x1 * w1
x2 * w2
```

Si un peso es grande, esa entrada puede tener mayor influencia.

Durante el entrenamiento, la neurona modifica sus pesos para reducir sus errores.

---

## Bias o sesgo

El `bias` es un ajuste adicional que no depende directamente de las entradas.

```python
bias = -0.5
```

Se incluye en el cálculo:

```python
z = x1 * w1 + x2 * w2 + bias
```

Podemos imaginarlo como la dificultad inicial que tiene la neurona para activarse.

Un bias muy negativo exige que las entradas aporten más para obtener una salida alta.

---

## Parámetro

Un parámetro es un valor que el modelo aprende durante el entrenamiento.

Nuestra neurona tiene tres parámetros:

```text
w1
w2
bias
```

Después del entrenamiento obtuvo aproximadamente:

```text
w1   = 7.285
w2   = 7.285
bias = -11.015
```

Un modelo de `7B parámetros` contiene aproximadamente siete mil millones de valores aprendidos.

---

## z o suma ponderada

`z` es el resultado de combinar las entradas, sus pesos y el bias:

```python
z = x1 * w1 + x2 * w2 + bias
```

Todavía no es la salida final. Primero debe pasar por la función de activación:

```python
output = sigmoid(z)
```

---

## Función de activación

Una función de activación transforma el resultado `z` para producir la salida de una neurona.

En nuestra primera neurona usamos la función sigmoide.

Las redes neuronales modernas también utilizan otras funciones, como `ReLU` y `GELU`.

---

## Sigmoid o sigmoide

La sigmoide convierte cualquier número en un valor entre `0` y `1`.

```python
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
```

Algunos ejemplos aproximados:

| Entrada | Salida |
|---:|---:|
| -10 | 0.0000 |
| -1 | 0.2689 |
| 0 | 0.5000 |
| 1 | 0.7311 |
| 10 | 1.0000 |

Una salida cercana a `0` representa poca activación.

Una salida cercana a `1` representa mucha activación.

---

## Output o salida

El `output` es el resultado continuo producido por la neurona:

```python
output = sigmoid(z)
```

Por ejemplo:

```text
output = 0.9722
```

Este valor todavía no necesariamente es una decisión final de `0` o `1`.

---

## Umbral

El umbral es el punto a partir del cual convertimos la salida continua en una clase.

Nosotros usamos `0.5`:

```python
prediccion = 1 if output >= 0.5 else 0
```

Por lo tanto:

```text
output menor que 0.5 → predicción 0
output igual o mayor que 0.5 → predicción 1
```

---

## Predicción

La predicción es la decisión final del modelo.

```python
prediccion = 1 if output >= 0.5 else 0
```

Por ejemplo:

```text
Salida:     0.9722
Predicción: 1
```

La salida conserva más información que la predicción, porque muestra qué tan cerca está el resultado de `0` o de `1`.

---

## Valor esperado o label

El valor esperado es la respuesta correcta que proporcionamos durante el entrenamiento.

```text
Entrada:  (1, 1)
Esperado: 1
```

También puede llamarse:

- Etiqueta.
- `label`.
- Objetivo.
- `target`.

El modelo compara su salida con este valor para medir cuánto se equivocó.

---

## Error

El error representa la diferencia entre el resultado esperado y la salida obtenida:

```python
error = esperado - output
```

Ejemplo:

```text
Esperado = 0
Salida   = 0.5744
Error    = -0.5744
```

El signo negativo indica que la salida obtenida fue demasiado alta.

---

## Loss o pérdida

La pérdida convierte el error en un número que indica qué tan mal está funcionando el modelo.

En nuestro primer ejemplo usamos el error cuadrático:

```python
loss = error ** 2
```

Una pérdida pequeña indica que el resultado está cerca del esperado.

Una pérdida grande indica que el modelo necesita mejorar.

El objetivo del entrenamiento es reducir la pérdida.

---

## Gradiente

Un gradiente indica cómo cambiarían los errores si modificáramos los parámetros.

Podemos imaginarlo como una señal que nos dice:

```text
Qué parámetro modificar.
En qué dirección moverlo.
Qué tan grande debe ser el cambio.
```

Los gradientes permiten corregir los pesos y el bias de forma sistemática.

---

## Delta

En nuestra neurona usamos:

```python
delta = error * output * (1 - output)
```

`delta` combina:

- El error cometido.
- La sensibilidad de la función sigmoide.

Lo utilizamos para calcular cuánto debemos ajustar cada parámetro.

---

## Learning rate o tasa de aprendizaje

El `learning_rate` controla el tamaño de cada ajuste:

```python
learning_rate = 0.5
```

Si es demasiado pequeño, el modelo aprende lentamente.

Si es demasiado grande, puede pasarse continuamente de la solución y no estabilizarse.

Podemos imaginarlo como el tamaño de cada paso mientras buscamos una solución.

---

## Epoch o época

Una `epoch` ocurre cuando el modelo recorre una vez todo el dataset.

Nuestro dataset contiene cuatro ejemplos:

```text
(0, 0)
(0, 1)
(1, 0)
(1, 1)
```

Este código realiza 10,000 épocas:

```python
for epoch in range(10000):
```

Eso significa que la neurona estudia los cuatro ejemplos 10,000 veces.

---

## Entrenamiento

El entrenamiento es el proceso mediante el cual el modelo ajusta sus parámetros.

El ciclo básico es:

```text
Calcular una salida
        ↓
Compararla con el resultado esperado
        ↓
Medir el error
        ↓
Calcular la corrección
        ↓
Actualizar pesos y bias
        ↓
Repetir
```

En nuestro ejercicio, la neurona aprendió valores adecuados para resolver AND.

---

## Inferencia

La inferencia sucede cuando utilizamos un modelo ya entrenado para producir una respuesta.

Durante la inferencia no necesariamente modificamos sus parámetros.

En nuestro programa, el segundo ciclo utiliza los pesos finales para obtener las predicciones:

```python
for x1, x2, esperado in datos:
    z = x1 * w1 + x2 * w2 + bias
    output = sigmoid(z)
```

---

## Frontera de decisión

La frontera de decisión divide las regiones donde el modelo predice clases diferentes.

En nuestra neurona aparece cuando:

```text
w1*x1 + w2*x2 + bias = 0
```

En un problema con dos entradas, esta ecuación representa una línea.

Para AND, la neurona encontró una línea que separa:

```text
(0, 0)
(0, 1)
(1, 0)
```

del punto:

```text
(1, 1)
```

---

## CPU

La CPU tiene pocos núcleos relativamente potentes y puede ejecutar diferentes tipos de instrucciones.

Nuestra primera neurona es tan pequeña que puede entrenarse perfectamente utilizando solamente la CPU.

---

## GPU

La GPU contiene muchos núcleos especializados en ejecutar operaciones similares en paralelo.

Las redes neuronales realizan enormes cantidades de multiplicaciones y sumas, por lo que una GPU puede acelerar considerablemente el entrenamiento.

La GPU no vuelve más inteligente al modelo. Solamente permite realizar una gran cantidad de cálculos más rápido.

---

## CUDA

CUDA es la plataforma que permite usar una GPU NVIDIA para cálculos generales, como operaciones de PyTorch.

En la Fase 3, PyTorch pudo usar la RTX 5050 cuando `torch.cuda.is_available()` devolvió `True`.

---

## Tensor

Un tensor es una estructura numérica parecida a un arreglo de NumPy, pero preparada para trabajar con PyTorch y moverse entre CPU y GPU.

```python
x = torch.tensor([[0.0, 1.0]])
x = x.to(device)
```

El `device` indica dónde vive el tensor: CPU o GPU.

---

## `nn.Module`

`nn.Module` es la clase base que usa PyTorch para definir modelos.

En la Fase 3, la red XOR se definió como una clase:

```python
class RedXOR(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa1 = nn.Linear(2, 2)
        self.capa2 = nn.Linear(2, 1)
```

Esto permite que PyTorch registre los parámetros entrenables del modelo.

---

## `nn.Linear`

`nn.Linear` representa una capa lineal con pesos y bias.

```text
salida = entrada * pesos + bias
```

En nuestra red XOR se usaron dos capas lineales: una de entrada a capa oculta y otra de capa oculta a salida.

---

## `MSELoss`

`MSELoss` calcula el error cuadrático medio.

Es una forma de medir la diferencia entre la salida del modelo y el valor esperado:

```python
criterio = nn.MSELoss()
loss = criterio(output, y)
```

---

## Autograd

`autograd` es el sistema de PyTorch que calcula gradientes automáticamente.

Cuando un tensor o parámetro participa en operaciones, PyTorch puede construir el historial necesario para calcular derivadas.

```python
loss.backward()
```

Esa llamada calcula los gradientes de los parámetros que participaron en la pérdida.

---

## Optimizer

Un optimizador aplica los cambios a los parámetros usando los gradientes calculados.

En la Fase 3 usamos descenso de gradiente estocástico:

```python
optimizer = torch.optim.SGD(modelo.parameters(), lr=0.5)
optimizer.step()
```

La llamada `optimizer.step()` es el momento en que los pesos y bias cambian.

---

## Red neuronal

Una red neuronal se forma conectando varias neuronas.

La salida de unas neuronas se convierte en la entrada de otras.

Esto permite aprender relaciones más complejas que una sola neurona no puede representar.

---

## Backpropagation

`Backpropagation` o retropropagación es el proceso utilizado para calcular cómo contribuyó cada parámetro al error.

Después, esos valores se utilizan para ajustar los parámetros desde la salida de la red hacia las capas anteriores.

En nuestra neurona estamos realizando manualmente una versión muy sencilla de este proceso.

---

## Gradient descent

`Gradient descent` o descenso de gradiente es el método utilizado para mover los parámetros en una dirección que reduzca la pérdida.

En nuestro código, las actualizaciones tienen esta forma:

```python
w1 = w1 + learning_rate * delta * x1
w2 = w2 + learning_rate * delta * x2
bias = bias + learning_rate * delta
```

---

## Resumen de nuestra primera neurona

Nuestra neurona realiza el siguiente recorrido:

```text
Entradas: x1, x2
        ↓
Pesos: w1, w2
        ↓
Suma ponderada + bias
        ↓
z
        ↓
Función sigmoide
        ↓
output
        ↓
Umbral de 0.5
        ↓
predicción
```

Durante el entrenamiento:

```text
predicción
    ↓
comparación con el resultado esperado
    ↓
error y loss
    ↓
gradientes
    ↓
actualización de parámetros
    ↓
nueva predicción
```

---

## Fase 4: Del texto a la predicción

La Fase 4 conecta el entrenamiento neuronal con el lenguaje:

```text
Texto → tokens → IDs → embeddings → contexto → logits
                                                |
                       Entrenar: pérdida → gradientes → actualizar parámetros
                                                |
                       Predecir: softmax → argmax → siguiente token
```

## Corpus

Un corpus es el conjunto de textos con el que trabajamos. En esta fase usamos frases pequeñas como `el perro come` y `el gato duerme`. El modelo aprende patrones de ese corpus; aprender estos ejemplos no demuestra que pueda generalizar a cualquier texto.

---

## Token y tokenización

Un token es una unidad en la que dividimos el texto. En nuestros ejercicios:

```python
texto = "hola mundo hola ia"
tokens = texto.split()
# ["hola", "mundo", "hola", "ia"]
```

`split()` sin argumentos separa por espacios en blanco, incluidos saltos de línea y tabulaciones. No separa por sí solo la puntuación: `hola,` y `hola` serían tokens distintos.

En otros tokenizadores, un token puede ser una palabra, una subpalabra, un símbolo o parte de un número. Aquí cada fragmento separado por espacios en blanco se trata como un token.

---

## Vocabulario e ID de token

El vocabulario contiene los tokens únicos. En los ejercicios 01 y 02 se construye con `sorted(set(tokens))`, para asignar IDs en un orden estable:

| Token | ID |
| --- | ---: |
| hola | 0 |
| ia | 1 |
| mundo | 2 |

El ID identifica un token; que dos IDs sean cercanos no significa que sus palabras sean parecidas. El orden del vocabulario debe coincidir con el de la tabla de embeddings y el de las clases de salida.

---

## Encode y decode

**Encode** convierte tokens a IDs mediante `token_a_id`. **Decode** hace el recorrido inverso mediante `id_a_token`:

```text
"hola mundo hola ia" → [0, 2, 0, 1] → "hola mundo hola ia"
```

El ejercicio 02 implementa estas operaciones con diccionarios y listas, sin definir funciones llamadas `encode()` o `decode()`. Al unir con `" ".join(...)`, los espacios repetidos y saltos de línea originales no se recuperan.

---

## Embedding y dimensión del embedding

Un embedding es un vector de números asociado a un token. `nn.Embedding` contiene una tabla de parámetros aprendibles y consulta la fila correspondiente a cada ID:

```python
embedding = nn.Embedding(num_embeddings=3, embedding_dim=4)
```

Esta tabla tiene 3 filas y 4 números por fila. `embedding_dim` determina la longitud del vector. En 03 y 04 se usan 4 dimensiones; en los modelos 06 y 07, 8.

Los embeddings empiezan con valores aleatorios. Solo cambian mediante entrenamiento si participan en la pérdida y el optimizador actualiza sus parámetros. Consultar la tabla por sí solo no la entrena.

---

## Similitud coseno

La similitud coseno compara la dirección de dos vectores no nulos:

```text
coseno(a, b) = producto_punto(a, b) / (norma(a) * norma(b))
```

| Valor | Interpretación geométrica |
| ---: | --- |
| Cercano a 1 | Direcciones similares. |
| Cercano a 0 | Direcciones aproximadamente perpendiculares. |
| Cercano a -1 | Direcciones opuestas. |

En el ejercicio 04 se usa `F.cosine_similarity`. Como los embeddings son aleatorios y no se entrenan, esos valores no demuestran relación semántica entre las palabras. Incluso después de entrenar, su utilidad depende de la tarea y los datos.

---

## Contexto, target y predicción del siguiente token

El contexto es la entrada utilizada para predecir una continuación; el target es el token correcto del ejemplo:

```text
Contexto: perro       → target: come
Contexto: [el, perro] → target: come
```

La tarea se llama predicción del siguiente token. El target se extrae del propio texto desplazándose una posición más allá del contexto.

Con `N` tokens y una ventana de tamaño `C`, estos ejercicios producen `N - C` ejemplos. Con 12 tokens se obtienen 11 ejemplos de un token y 10 ejemplos de dos tokens.

---

## Ventana de contexto

La ventana de contexto es la cantidad de tokens previos que el modelo usa para una predicción. En el ejercicio 06 es 1 y en el 07 es 2.

Una ventana mayor permite observar más información, pero no garantiza una única respuesta. `[el, perro]` puede continuar con `come` o `duerme` en nuestro corpus.

---

## Concatenación de embeddings y flatten

En el ejercicio 07, cada ejemplo contiene dos embeddings de ocho números:

```text
IDs:                  [batch, 2]
Embeddings:           [batch, 2, 8]
flatten(start_dim=1):  [batch, 16]
Linear(16, 5):        [batch, 5]
```

`flatten(start_dim=1)` conserva la dimensión del lote y une las dimensiones del contexto y del embedding. No suma ni promedia los vectores.

La concatenación conserva el orden: la primera y la segunda posición ocupan lugares distintos. La capa lineal puede aprender pesos distintos para cada posición, pero todavía no calcula atención según el contenido.

---

## Logits

Los logits son las puntuaciones que produce la capa final, una por cada token del vocabulario. Pueden ser positivos o negativos y no necesitan sumar 1.

Con cinco tokens de vocabulario, cada ejemplo produce cinco logits. La posición de cada logit corresponde al ID del token que podría venir después.

---

## Softmax y argmax

Softmax convierte logits en una distribución de probabilidades:

```text
p(i) = exp(logit_i) / suma_j(exp(logit_j))
```

Las probabilidades suman 1, salvo pequeñas diferencias numéricas. En estos modelos, `torch.softmax(logits, dim=1)` opera sobre los cinco tokens posibles de cada ejemplo.

`argmax` devuelve el índice del valor mayor. Selecciona una sola continuación y no muestra la incertidumbre restante ni realiza muestreo. Puede aplicarse directamente a los logits para obtener el mismo máximo; el softmax del ejercicio permite interpretar las puntuaciones como probabilidades.

---

## CrossEntropyLoss

`nn.CrossEntropyLoss()` mide el error de una clasificación entre varias clases; en esta fase cada clase es un token del vocabulario.

```python
criterio = nn.CrossEntropyLoss()
loss = criterio(logits, y)
```

Para el caso de estos ejercicios, recibe logits de forma `[batch, vocab_size]` y targets enteros de forma `[batch]`, con tipo `torch.long`. No se aplica softmax antes de pasar los logits a esta función: ya incorpora el cálculo equivalente a log-softmax y la pérdida correspondiente.

Para un ejemplo, la pérdida equivale a `-log(probabilidad_del_target)`. Cuanta menos probabilidad se asigna al target, mayor es la penalización.

---

## Adam y aprendizaje de embeddings

Adam es el optimizador usado en los modelos de la Fase 4:

```python
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.05)
```

Ajusta los parámetros usando los gradientes y estadísticas acumuladas de esos gradientes. La tasa de aprendizaje indicada pertenece al experimento; no es una recomendación universal.

El ciclo es `forward → pérdida → zero_grad → backward → step`. `backward()` calcula gradientes y `step()` actualiza los parámetros. Como el embedding forma parte del modelo, también recibe actualizaciones.

---

## Ambigüedad y distribución de continuaciones

El corpus contiene:

```text
[el, perro] → come
[el, perro] → duerme
```

El modelo recibe la misma entrada en ambos casos. No puede asignar simultáneamente probabilidad 1 a dos tokens diferentes. Aprender a repartir la probabilidad entre continuaciones es parte de la tarea.

En este ejemplo equilibrado, asignar aproximadamente la mitad a cada continuación es coherente con los datos. La pérdida conserva una contribución positiva; no siempre debe llegar a cero. `argmax` elige una palabra aunque haya otra casi igual de probable.

---

## Límites de frase y tokens especiales

En esta fase, `split()` elimina los saltos de línea como separadores de texto y no crea tokens de fin de frase. El modelo ve una secuencia continua, por lo que aprende transiciones como `come → el` al pasar de una línea a la siguiente.

Un token de fin de secuencia o uno para palabras desconocidas podría añadirse en una ampliación. Actualmente no existen; consultar un token fuera del vocabulario produce `KeyError`.

---

## Aleatoriedad y reproducibilidad

Los pesos iniciales aleatorios pueden producir diferencias entre ejecuciones. Para un experimento más reproducible se puede fijar una semilla antes de construir el modelo:

```python
torch.manual_seed(42)
```

Los scripts actuales de Fase 4 no incluyen esa línea. La semilla ayuda a repetir experimentos bajo las mismas condiciones, pero no garantiza igualdad absoluta entre distintos dispositivos, versiones o algoritmos.

---

## Puente hacia attention

El modelo de dos tokens de Fase 4 concatena embeddings y aplica una capa lineal. Self-attention, construida en Fase 5, calcula pesos que dependen del contenido para combinar información del contexto.

La [Fase 5](01_Plan_de_Trabajo.md#fase-5-self-attention) introdujo Query, Key, Value, puntuaciones de atención, softmax, máscara causal y múltiples cabezas en ejercicios independientes. La integración en un Transformer corresponde a la Fase 6.

---

## Fase 5: Attention y representaciones contextualizadas

La atención produce una representación para cada posición combinando información de otras posiciones permitidas. El recorrido aprendido es:

```text
Embeddings → Q/K/V → scores → escalado → máscara causal
          → softmax → pesos @ V → salida por cabeza
          → concatenación de cabezas → proyección final
```

## Attention y representación contextualizada

Attention calcula pesos para combinar vectores de información. Una representación contextualizada depende tanto del token que consulta como de los tokens a los que puede atender.

En el ejemplo `el perro come`, la posición de `come` combina información de las tres posiciones. Cuánto aporta cada una depende de sus queries, keys y values; no está fijado por el significado que una persona atribuye a las palabras.

En estos ejercicios los embeddings son manuales y las proyecciones finales no se entrenan. La contextualización describe el cálculo realizado, no una comprensión semántica demostrada.

---

## Query, Key y Value

| Representación | Intuición | Función matemática |
| --- | --- | --- |
| Query (Q) | Qué busco. | Se compara con las keys. |
| Key (K) | Cómo puedo ser encontrado. | Determina el score frente a cada query. |
| Value (V) | Qué información aporto. | Se combina usando los pesos de atención. |

Se obtienen mediante proyecciones:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

Las matrices de 02–05 son identidades fijas, por eso Q, K y V coinciden numéricamente con los embeddings aunque sus funciones sean distintas. En 06 y 07 se usan capas `nn.Linear(..., bias=False)` con parámetros entrenables.

---

## Producto punto y scores de atención

El producto punto suma los productos de las componentes correspondientes. Para una query y una key devuelve un score:

```text
score(q, k) = suma_i(q_i * k_i)
```

Depende tanto de la dirección como de la magnitud; no equivale a la similitud coseno normalizada.

Para toda la secuencia, `Q @ K.T` produce `[seq_len, seq_len]`. La fila corresponde a la query y la columna a la key. Un score mayor recibe más peso tras softmax dentro de esa fila, pero los scores originales todavía no son probabilidades.

---

## Softmax en atención

Softmax convierte los scores de una query en pesos sobre las posiciones consultadas. En los ejercicios de una query se aplica sobre `dim=0`; en la matriz de queries se aplica por fila con `dim=1`.

```text
scores [1, 1, 2] → pesos [0.2119, 0.2119, 0.5761]
```

Los pesos suman aproximadamente 1. Después se usan para sumar los values: `salida = pesos @ V`.

En Fase 4, softmax distribuía probabilidad entre tokens del vocabulario. Aquí distribuye peso entre posiciones del contexto. Los pesos de atención no son probabilidades del siguiente token.

---

## Self-attention

Self-attention significa que Q, K y V proceden de la misma secuencia. Todas sus posiciones generan queries y consultan keys de esa secuencia.

Sin máscara, cada posición puede consultar todas las posiciones. Con máscara causal, cada una consulta solamente su posición y las anteriores.

---

## Scaled dot-product attention

La atención de producto punto escalado usa:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
```

`d_k` es la dimensión de las keys de una cabeza. Dividir por su raíz reduce el crecimiento de los scores asociado a la dimensión y ayuda a evitar un softmax demasiado concentrado.

En la implementación de dos cabezas, `embedding_dim=4` y `head_dim=2`: cada cabeza divide por `sqrt(2)`, no por `sqrt(4)` ni por la raíz del número de tokens.

---

## Máscara causal

La máscara causal impide que una posición consulte tokens posteriores:

```text
el    → el
perro → el, perro
come  → el, perro, come
```

El código construye una máscara triangular superior con `diagonal=1`. Los valores `True` indican posiciones bloqueadas:

```text
False  True   True
False  False  True
False  False  False
```

Se sustituyen esos scores por `-inf` antes del softmax. Como `exp(-inf)=0`, reciben peso cero al normalizar junto con las posiciones válidas. La diagonal queda disponible, de modo que ninguna fila del ejemplo queda completamente bloqueada.

---

## Parámetro entrenable frente a parámetro entrenado

Un parámetro entrenable está preparado para recibir gradientes y actualizaciones. Un parámetro entrenado ya ha pasado por ese proceso.

Las capas Q/K/V de 06 y 07 son entrenables, pero los scripts solo ejecutan el forward. Para ajustarlas haría falta conectar la salida con una tarea, calcular una pérdida, llamar a `backward()` y actualizar con un optimizador.

Que aparezca un `grad_fn` en un tensor indica que autograd registra operaciones; no significa que se haya entrenado el modelo.

---

## Multi-head attention

Multi-head attention calcula varias atenciones con proyecciones independientes. Cada cabeza puede aprender relaciones diferentes durante un entrenamiento posterior; la arquitectura no garantiza que esas relaciones sean distintas o tengan una interpretación lingüística específica.

En el ejercicio 07:

```text
embedding_dim = 4
num_heads = 2
head_dim = 2
```

Cada cabeza recibe el embedding completo de cuatro dimensiones y lo proyecta a dos dimensiones. No se divide el vector original en dos mitades fijas. Se exige que `embedding_dim` sea divisible entre `num_heads`.

---

## ModuleList, concatenación y proyección de salida

`nn.ModuleList` registra las cabezas como submódulos, permitiendo que PyTorch incluya sus parámetros al consultar `modelo.parameters()`.

`torch.cat(salidas, dim=1)` junta las características de las cabezas para cada token:

```text
Cabeza 1 [3, 2] + cabeza 2 [3, 2]
           → concatenación [3, 4]
           → Linear(4, 4)
           → salida final [3, 4]
```

La concatenación por sí sola no mezcla las características; la proyección lineal posterior aprende a combinarlas cuando se entrena. Conservar la forma facilita integrar una conexión residual en la siguiente fase.

---

## Leer una matriz de atención

Cada fila responde a «para esta posición, cuánto peso recibe cada posición del contexto». Por ejemplo, los pesos reportados para `perro` fueron:

```text
         el      perro   come
Head 1   0.5621  0.4379  0.0000
Head 2   0.4425  0.5575  0.0000
```

El último cero demuestra el bloqueo de la posición futura en esas filas. Las diferencias entre cabezas son compatibles con sus parámetros independientes y aleatorios. No demuestran que una cabeza haya aprendido una relación semántica concreta.

La salida contextualizada contiene la combinación de los values; no debe confundirse con la matriz de pesos usada para calcularla.

---

## Puente hacia el Transformer

La [Fase 6](01_Plan_de_Trabajo.md#fase-6-construir-nuestro-transformer) integrará la atención con conexiones residuales, Layer Normalization, una red feed-forward e información posicional.

La Fase 5 deja listo el mecanismo de atención para una secuencia. Todavía faltan esos componentes, una salida sobre el vocabulario y la integración del entrenamiento para construir el modelo de lenguaje con Transformer.
