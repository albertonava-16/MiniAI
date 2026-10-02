# Breviario de conceptos de MiniAI

Este documento reúne los conceptos aprendidos durante la construcción de MiniAI.

La intención no es memorizar todas las definiciones, sino tener una referencia rápida que podamos ampliar conforme avance el proyecto.

**Última ampliación:** 1 de octubre de 2026, [conceptos de la Fase 7](#fase-7-generalización-regularización-y-decodificación). También se conservan los conceptos de [Fase 4](#fase-4-del-texto-a-la-predicción), [Fase 5](#fase-5-attention-y-representaciones-contextualizadas) y [Fase 6](#fase-6-transformer-entrenamiento-y-persistencia). Los ejercicios, resultados y límites actuales están en la [memoria técnica](02_Memoria_Tecnica.md#fase-7-entrenamiento-generalización-y-generación).

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

La [Fase 6](01_Plan_de_Trabajo.md#fase-6-construir-nuestro-transformer) integró la atención con conexiones residuales, Layer Normalization, una red feed-forward e información posicional.

La Fase 5 dejó listo el mecanismo de atención para una secuencia. La Fase 6 añadió esos componentes, la salida sobre el vocabulario y el entrenamiento para construir MiniGPT.


---

## Fase 6: Transformer, entrenamiento y persistencia

En esta fase se conectaron los conceptos anteriores en un modelo de lenguaje pequeño:

```text
Texto → tokens → IDs → embeddings de token + posición
  → bloques Transformer → logits
  → entrenamiento → generación → guardado → recuperación
```

## Bloque Transformer

Un bloque combina atención causal, conexiones residuales, normalización y una red feed-forward. Nuestra variante aplica LayerNorm después de cada suma residual:

```python
x = norm1(x + attention(x))
x = norm2(x + feed_forward(x))
```

Conserva la forma `[tokens, embedding_dim]`. El primer ejemplo transforma `[3, 4]` en `[3, 4]`, lo que permite conectar otro bloque a continuación. Conservar la forma no significa conservar los mismos valores.

---

## Conexión residual

Una conexión residual suma la entrada y una transformación de ella: `x + f(x)`.

La suma mantiene un camino directo para la información y los gradientes, mientras la rama `f(x)` aporta una transformación aprendida. Ambas ramas deben tener formas compatibles. No es una concatenación: la dimensión de la salida sigue siendo la misma.

---

## LayerNorm y post-normalización

`nn.LayerNorm(embedding_dim)` normaliza las características del vector de cada token. También aprende una escala y un desplazamiento por característica.

En nuestro bloque, la normalización ocurre **después** de sumar la conexión residual; por eso se habla de post-normalización o *post-norm*. No normaliza mezclando todos los tokens ni usa estadísticas de otros ejemplos del batch.

---

## Feed-forward por posición

La red feed-forward procesa cada posición con las mismas capas:

```text
Linear(D, 4D) → ReLU → Linear(4D, D)
```

Con `D=8`, expande el vector a 32 componentes y lo devuelve a 8. Attention intercambia información entre tokens; la red feed-forward transforma las características de cada token sin consultar directamente otras posiciones. Puede procesar información de contexto porque recibe la salida de atención.

---

## Embeddings posicionales aprendidos

El embedding del token representa su identidad; el embedding posicional aporta su lugar en la secuencia:

```text
entrada[i] = token_embedding(token_id[i]) + position_embedding(i)
```

Ambos tienen la misma dimensión y se suman, no se concatenan. Las posiciones empiezan en cero. `nn.Embedding(max_seq_len, D)` almacena una tabla entrenable; en este proyecto no se usa una fórmula sinusoidal.

Un mismo `el` en posiciones distintas puede comenzar con representaciones diferentes. Una tabla de veinte posiciones admite índices 0–19, pero el entrenamiento con cinco tokens solo utiliza las primeras cinco.

---

## Apilado de bloques

Cada bloque recibe la representación producida por el anterior y la transforma de nuevo. `nn.ModuleList` registra los bloques y sus parámetros; el forward los ejecuta mediante un bucle.

En el ejercicio 04 hay tres bloques con dimensión 4; en el MiniGPT entrenado hay dos con dimensión 8. Los bloques tienen parámetros independientes. Apilar bloques similares aumenta la profundidad, pero no garantiza por sí solo capacidad lingüística.

---

## Proyección hacia el vocabulario

El Transformer produce un vector por posición. La capa `nn.Linear(embedding_dim, vocab_size)` lo transforma en un logit por token posible:

```text
Representaciones [T, 8] → Linear(8, 5) → logits [T, 5]
```

Los logits son puntuaciones sin normalizar. Softmax sobre el vocabulario produce probabilidades del siguiente token; estos valores son distintos de los pesos de atención sobre posiciones del contexto.

---

## Objetivos desplazados y entrenamiento causal

Para predecir el siguiente token, se desplaza la secuencia una posición:

```text
Entrada:  el     perro  come  el    gato
Objetivo: perro  come   el    gato  duerme
```

El modelo calcula todas esas predicciones en un forward. La máscara causal asegura que la predicción de una posición solo dependa de esa posición y de las anteriores, aunque se haya entregado toda la entrada al modelo.

`CrossEntropyLoss` recibe directamente logits `[T, vocab_size]` y objetivos enteros `[T]`. En estos scripts no hay dimensión de batch: cada fila corresponde a una posición de la única secuencia. El ciclo `forward → loss → zero_grad → backward → step` ajusta los parámetros, incluidos los embeddings utilizados.

---

## Generación autoregresiva y selección greedy

Generar autoregresivamente significa añadir cada predicción al contexto para obtener la siguiente:

```text
el → el perro → el perro come → el perro come el
   → el perro come el gato → el perro come el gato duerme
```

Se utiliza la última fila de logits, `logits[-1]`, porque representa la continuación del prefijo actual. `argmax` elige el token con mayor puntuación o probabilidad: es selección greedy. No es muestreo aleatorio.

Los scripts aplican softmax antes de `argmax`; seleccionar el máximo directamente en los logits daría el mismo orden. No hay temperatura de generación ni token de fin: se solicitan cinco tokens nuevos.

---

## Modo de evaluación y ausencia de gradientes

`modelo.eval()` cambia el comportamiento de capas que distinguen entrenamiento y evaluación, como dropout. No desactiva por sí mismo autograd ni carga los pesos aprendidos.

`torch.no_grad()` evita registrar el grafo de gradientes durante la inferencia. En nuestro modelo no hay dropout ni BatchNorm, pero se utiliza el patrón de evaluación y ausencia de gradientes para expresar que se está generando, sin actualizar parámetros.

---

## state_dict y persistencia del aprendizaje

`modelo.state_dict()` contiene los parámetros y buffers persistentes registrados por el modelo. Guardarlo conserva los valores aprendidos:

```python
torch.save(modelo.state_dict(), "minigpt.pth")
```

Para usarlos en otro proceso se necesita construir una arquitectura compatible y llamar a `load_state_dict`. El archivo de pesos por sí solo no contiene la definición de las clases ni nuestro vocabulario.

El vocabulario debe conservar también el orden: cambiar qué palabra representa cada ID cambia la interpretación de las entradas y las salidas, aunque el número de palabras sea el mismo.

---

## Checkpoint para reconstruir el modelo

El checkpoint de Fase 6 reúne:

| Clave | Qué conserva |
| --- | --- |
| `model_state_dict` | Parámetros aprendidos de MiniGPT. |
| `config` | Dimensión del embedding, cabezas, bloques y máximo de posiciones. |
| `vocabulario` | Palabras en el orden que define sus IDs. |

El cargador obtiene la configuración y el vocabulario, reconstruye MiniGPT, carga los pesos y genera. `map_location=device` sitúa los tensores cargados en el dispositivo seleccionado.

Este checkpoint sirve para recuperar la inferencia junto con el código de la arquitectura. Para reanudar exactamente el entrenamiento haría falta conservar también el estado del optimizador, el paso o época y los estados aleatorios pertinentes.

---

## Memorización y generalización en MiniGPT

Reproducir `el perro come el gato duerme` después de entrenar y después de cargar el checkpoint demuestra que se ajustaron y recuperaron parámetros útiles para ese ejemplo.

La pérdida reportada, aproximadamente `0.000035`, se mide sobre el mismo corpus diminuto usado para entrenar. No mide el desempeño con frases nuevas. Distinguir los dos usos de `el` es compatible con aprovechar posición y contexto; no demuestra comprensión del español.

La [Fase 7](01_Plan_de_Trabajo.md#fase-7-entrenar-miniai) continuó con más datos, validación y seguimiento del entrenamiento para estudiar ese límite.

---

## Fase 7: Generalización, regularización y decodificación

Esta fase separa dos preguntas que antes estaban mezcladas:

```text
Entrenamiento: ¿qué probabilidades aprende el modelo?
Decodificación: ¿cómo elegimos texto a partir de esas probabilidades?
```

## Train y validation

El conjunto de train se usa para calcular gradientes y actualizar parámetros. El conjunto de validation se consulta sin `backward()` ni `optimizer.step()`; permite estimar cómo se comporta el modelo en ejemplos que no usó para ajustar sus pesos.

Comparar ambas pérdidas ayuda a distinguir aprendizaje y memorización. Una pérdida baja en train no basta para afirmar que el modelo generaliza.

---

## Batch y mini-batch

Un batch agrupa varios ejemplos para procesarlos en una misma operación. Un mini-batch es un grupo menor que el dataset completo. En Fase 7, `batch_size=8` produce entradas con forma:

```text
[batch, seq_len] → embeddings → [batch, seq_len, embedding_dim]
```

`DataLoader` organiza esos grupos y puede mezclar el orden de train entre épocas.

---

## Generalización y overfitting

Generalizar significa rendir razonablemente en datos no usados para actualizar los pesos. Overfitting ocurre cuando el modelo se ajusta demasiado a train y empeora fuera de él.

Una señal típica es:

```text
train loss baja
validation loss sube
```

El experimento de Fase 7 observó esta divergencia antes de añadir regularización. El validation loss mide el conjunto reservado del experimento; no demuestra comprensión general del idioma.

---

## Dropout

`nn.Dropout(p)` pone aleatoriamente parte de las activaciones en cero durante entrenamiento. Esto reduce la dependencia de rutas concretas. Con `modelo.eval()`, dropout deja de aplicar ese ruido y se usan todas las activaciones.

`p=0.20` significa una probabilidad del 20% de anular cada activación afectada durante entrenamiento; no elimina permanentemente neuronas ni parámetros.

---

## AdamW y weight decay

AdamW es un optimizador adaptativo que separa el weight decay de la actualización basada en el gradiente. En esta fase se utilizó:

```python
torch.optim.AdamW(
    modelo.parameters(),
    lr=0.003,
    weight_decay=0.01
)
```

Weight decay penaliza parámetros grandes y actúa como regularización. Su valor no representa un porcentaje directo de neuronas eliminadas.

---

## Early stopping y mejor modelo

Early stopping detiene el entrenamiento cuando validation deja de mejorar durante un número definido de evaluaciones, llamado `patience`.

El script copia el `state_dict` cada vez que obtiene un validation loss menor. Al terminar restaura esa copia. Así, «mejor modelo» significa el estado con mejor métrica de validation observada, no los pesos de la última época.

---

## BOS y EOS

`<BOS>` (*Beginning Of Sequence*) marca el inicio de una secuencia. En los ejercicios también rellena por la izquierda los contextos cortos.

`<EOS>` (*End Of Sequence*) representa el final. Durante generación, si el modelo selecciona EOS, el bucle se detiene. Ambos son tokens del vocabulario y tienen IDs y embeddings aprendidos.

---

## Sampling

Sampling elige aleatoriamente un token de acuerdo con su distribución de probabilidad:

```python
siguiente_id = torch.multinomial(probabilidades, num_samples=1)
```

A diferencia de argmax, puede producir distintas continuaciones para el mismo prompt. Los tokens más probables siguen teniendo mayor oportunidad de ser elegidos.

---

## Temperature

Temperature divide los logits antes de softmax:

```python
logits_ajustados = logits / temperature
```

Una temperature menor que 1 concentra la distribución y suele volver la generación más conservadora. Una mayor que 1 la aplana, aumenta la diversidad y también el riesgo de seleccionar opciones débiles. Debe ser mayor que cero.

---

## Top-k

Top-k conserva una cantidad fija de los tokens con logits más altos y descarta los demás antes del muestreo. Con `top_k=5`, solo cinco candidatos pueden ser elegidos, aunque sus probabilidades sean muy diferentes.

---

## Top-p o nucleus sampling

Top-p ordena candidatos por probabilidad y, en su forma habitual, conserva el conjunto más pequeño que cubre aproximadamente una masa acumulada elegida. La cantidad de candidatos cambia según la distribución.

```text
top-k → número fijo de candidatos
top-p → masa de probabilidad y número variable
```

Un top-p alto suele admitir más variedad. En todos los casos se conserva al menos el candidato principal para evitar una distribución vacía. La implementación de Fase 7 elimina directamente las posiciones cuya suma ya supera el umbral; como no conserva el primer token que lo cruza, puede retener menos masa que la variante estándar.

---

## Decodificación

Decodificar es convertir los logits del modelo en una secuencia concreta. Argmax, sampling, temperature, top-k y top-p son decisiones de decodificación.

Cambiar estas opciones no vuelve a entrenar el modelo ni añade conocimiento. Modifica qué continuaciones se permiten y cómo se seleccionan a partir de las probabilidades ya aprendidas.

---

## Checkpoint de Fase 7

El checkpoint final reúne pesos, arquitectura, vocabulario, tokens especiales, métricas e hiperparámetros de generación. Esto permite reconstruir el modelo para inferencia junto con el código.

Guardar el nombre `AdamW`, el learning rate y el weight decay no equivale a guardar `optimizer.state_dict()`. Para reanudar exactamente el entrenamiento también harían falta el estado del optimizador, la época o paso y los estados aleatorios pertinentes.
