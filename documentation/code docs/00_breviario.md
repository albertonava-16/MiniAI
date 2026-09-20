# Breviario de conceptos de MiniAI

Este documento reúne los conceptos aprendidos durante la construcción de MiniAI.

La intención no es memorizar todas las definiciones, sino tener una referencia rápida que podamos ampliar conforme avance el proyecto.

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
