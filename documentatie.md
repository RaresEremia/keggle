# Documentație — Model de Clasificare MLP (Perceptron Multi-Strat)

---

## 1. Descrierea Modelului

### 1.1 Tipul modelului

Modelul implementat este un **Perceptron Multi-Strat** (*Multi-Layer Perceptron — MLP*), o rețea neuronală artificială de tip *feedforward* (propagare înainte), utilizată pentru **clasificare multi-clasă**.

Rețeaua primește la intrare vectori de caracteristici de dimensiune **784** (imagini aplatizate de 28×28 pixeli) și prezice una dintre cele **10 clase** posibile (etichetele 1–10).

Modelul este implementat **de la zero**, folosind exclusiv biblioteca **NumPy**, fără framework-uri externe de tip TensorFlow sau PyTorch.

---

### 1.2 Arhitectura rețelei

Rețeaua este compusă din **4 straturi** (1 strat de intrare, 2 straturi ascunse, 1 strat de ieșire):

| Strat   | Tip       | Număr neuroni | Funcție de activare  |
|---------|-----------|---------------|----------------------|
| Strat 0 | Intrare   | **784**       | — (date normalizate) |
| Strat 1 | Ascuns    | **256**       | ReLU                 |
| Strat 2 | Ascuns    | **128**       | ReLU                 |
| Strat 3 | Ieșire    | **10**        | Softmax              |

**Diagrama arhitecturii:**

```
[Intrare: 784] → [Ascuns 1: 256, ReLU] → [Ascuns 2: 128, ReLU] → [Ieșire: 10, Softmax]
```

**Număr total de parametri antrenabili:**

| Conexiune  | Ponderi (W)          | Bias-uri (b) | Total       |
|------------|----------------------|--------------|-------------|
| 784 → 256  | 784 × 256 = 200.704  | 256          | 200.960     |
| 256 → 128  | 256 × 128 = 32.768   | 128          | 32.896      |
| 128 → 10   | 128 × 10  = 1.280    | 10           | 1.290       |
| **Total**  |                      |              | **235.146** |

---

### 1.3 Funcțiile de activare

#### ReLU (*Rectified Linear Unit*) — straturi ascunse

```
ReLU(z) = max(0, z)
```

Funcția ReLU introduce **neliniaritate** în rețea, permițând modelului să învețe reprezentări complexe. Setează la zero toate valorile negative și menține valorile pozitive nemodificate. Este preferată pentru straturile ascunse datorită gradienților stabili și vitezei de convergență ridicate.

#### Softmax — stratul de ieșire

```
Softmax(z_i) = exp(z_i) / sum_j( exp(z_j) )
```

Funcția Softmax transformă valorile brute de ieșire (*logits*) într-o **distribuție de probabilitate** peste cele 10 clase. Suma tuturor ieșirilor este întotdeauna 1, iar clasa prezisă este cea cu probabilitatea maximă.

---

### 1.4 Inițializarea ponderilor

Ponderile sunt inițializate prin metoda **He (Kaiming) initialization**, concepută special pentru rețele cu activare ReLU:

```
W ~ N(0, sqrt(2 / n_intrare))
```

Aceasta previne problema **dispariției sau exploziei gradienților**, asigurând că varianța activărilor rămâne stabilă pe parcursul straturilor. Bias-urile sunt inițializate cu **zero**.

---

### 1.5 Funcția de pierdere

Modelul minimizează **entropia încrucișată categorială** (*Categorical Cross-Entropy*):

```
L = -(1/N) * sum_i sum_k [ y_ik * log(p_ik + eps) ]
```

unde:
- `N` = numărul de exemple din batch
- `y_ik` = eticheta reală în format *one-hot*
- `p_ik` = probabilitatea prezisă de model pentru clasa `k`
- `eps = 1e-9` = termen mic pentru stabilitate numerică (evită `log(0)`)

---

### 1.6 Algoritmul de antrenament

**Metoda:** Mini-batch Stochastic Gradient Descent (SGD) cu Backpropagation

#### Parametrii algoritmului:

| Parametru                  | Valoare  | Descriere                                  |
|----------------------------|----------|--------------------------------------------|
| **Rata de învățare** (lr)  | **0.05** | Pasul de actualizare al ponderilor         |
| **Dimensiunea batch-ului** | **64**   | Numărul de exemple procesate simultan      |
| **Numărul de epoci**       | **200**  | Numărul de treceri complete prin date      |
| **Optimizator**            | **SGD**  | Gradient descent standard, fără momentum   |
| **Inițializare ponderi**   | **He**   | `W ~ N(0, sqrt(2/fan_in))`                 |

#### Procesul de antrenament per epocă:

1. **Amestecare** aleatorie a datelor de antrenament
2. Împărțire în **mini-batch-uri** de 64 de exemple
3. Pentru fiecare mini-batch:
   - **Forward pass**: calculul predicțiilor prin rețea
   - **Calculul pierderii**: entropie încrucișată
   - **Backward pass**: calculul gradienților prin backpropagation
   - **Actualizarea ponderilor**: `W ← W - lr * dL/dW`

#### Formulele de backpropagation:

Gradient la stratul de ieșire (Softmax + Cross-Entropy combinate):
```
delta_L = (p_prezis - y_onehot) / N
```

Propagare înapoi prin straturi ascunse (prin ReLU):
```
delta_l = (delta_{l+1} @ W_{l+1}^T) * (z_l > 0)
```

Actualizarea ponderilor și bias-urilor:
```
W_l ← W_l - lr * (a_{l-1}^T @ delta_l)
b_l ← b_l - lr * sum(delta_l)
```

---

### 1.7 Preprocesarea datelor

Datele de intrare sunt **normalizate** prin împărțirea la valoarea maximă:

```
X_norm = X / (X_max + eps)
```

Aceasta asigură că toate caracteristicile se află în intervalul **[0, 1]**, facilitând convergența mai rapidă. Etichetele sunt convertite din valori întregi (1–10) în format **one-hot encoding** (vectori binari de dimensiune 10).

---

### 1.8 Rezumat parametri model

| Caracteristică              | Valoare                          |
|-----------------------------|----------------------------------|
| Tip model                   | MLP (Perceptron Multi-Strat)     |
| Nr. straturi totale         | 4                                |
| Dimensiuni straturi         | 784 → 256 → 128 → 10             |
| Activare straturi ascunse   | ReLU                             |
| Activare strat ieșire       | Softmax                          |
| Funcție de pierdere         | Entropie încrucișată categorială |
| Optimizator                 | Mini-batch SGD                   |
| Rată de învățare            | 0.05                             |
| Dimensiune batch            | 64                               |
| Nr. epoci                   | 200                              |
| Inițializare ponderi        | He (Kaiming)                     |
| Normalizare intrări         | Scalare la [0, 1]                |
| Total parametri antrenabili | 235.146                          |
| Implementare                | NumPy pur (fără framework)       |

---

## 2. Evaluare — Validare Încrucișată 10-Fold

### 2.1 Metodologie

Pentru evaluarea performanței modelului s-a utilizat **validarea încrucișată 10-fold** (*10-fold cross-validation*).

**Procedura:**
1. Setul de antrenament (1000 de exemple) a fost împărțit în **10 subseturi egale** de câte 100 de exemple
2. La fiecare iterație (*fold*), 9 subseturi (900 exemple) au fost folosite pentru antrenament, iar 1 subset (100 exemple) pentru validare
3. Procesul s-a repetat de **10 ori**, fiecare subset servind o dată ca set de validare
4. S-au calculat acuratețea medie și intervalul de încredere de 90%

---

### 2.2 Rezultatele validării încrucișate

| Fold    | Acuratețe |
|---------|-----------|
| Fold 1  | 0.9300    |
| Fold 2  | 0.9100    |
| Fold 3  | 0.9600    |
| Fold 4  | 0.9000    |
| Fold 5  | 0.9000    |
| Fold 6  | 0.9400    |
| Fold 7  | 0.9600    |
| Fold 8  | 0.9200    |
| Fold 9  | 0.9200    |
| Fold 10 | 0.9300    |

---

### 2.3 Acuratețea medie și intervalul de încredere 90%

| Metrică                       | Valoare              |
|-------------------------------|----------------------|
| **Acuratețe medie**           | **0.9270 (92.70%)**  |
| **Deviație standard**         | 0.0216               |
| **Interval de încredere 90%** | **[0.9145, 0.9395]** |

**Formula intervalului de încredere** (distribuția t-Student, df=9, t₀.₉₀ = 1.833):

```
IC_90% = medie ± 1.833 × (std / sqrt(10))
       = 0.9270 ± 1.833 × (0.0216 / sqrt(10))
       = 0.9270 ± 0.0125
       = [0.9145, 0.9395]
```

---

### 2.4 Matricile de confuzie — câte una pentru fiecare fold

> Liniile reprezintă **clasa reală** (T1–T10), coloanele reprezintă **clasa prezisă** (P1–P10).
> Valorile de pe diagonală (bold) reprezintă predicțiile corecte.

**Fold 1** — Acuratețe: 93.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1     7    0    0    0    1    0    0    0    0    0
T2     0   12    0    0    0    0    0    0    0    0
T3     0    1    6    1    0    0    1    1    0    0
T4     0    0    0   16    0    1    0    0    0    0
T5     0    0    0    0    7    0    0    0    0    0
T6     0    0    0    0    0    4    0    0    0    0
T7     0    0    0    0    0    0   16    0    0    0
T8     0    0    0    0    0    0    0    9    0    0
T9     0    0    0    0    0    0    0    0    5    0
T10    1    0    0    0    0    0    0    0    0   11
```

**Fold 2** — Acuratețe: 91.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    11    0    0    0    0    0    0    0    0    0
T2     0   14    1    0    0    0    0    0    0    0
T3     0    0    5    1    0    0    0    1    0    0
T4     0    0    0    8    0    0    0    2    0    0
T5     0    0    0    0    9    0    0    0    0    0
T6     1    0    0    1    1   12    0    0    0    0
T7     0    0    0    0    0    0    9    0    0    0
T8     0    0    0    0    0    0    0    8    0    1
T9     0    0    0    0    0    0    0    0    7    0
T10    0    0    0    0    0    0    0    0    0    8
```

**Fold 3** — Acuratețe: 96.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    13    0    0    0    0    0    0    0    0    0
T2     0    9    0    0    0    0    0    0    0    0
T3     0    0    7    1    0    0    0    0    0    0
T4     0    0    0    5    0    0    0    0    2    0
T5     0    0    0    0    8    0    0    0    0    1
T6     0    0    0    0    0    8    0    0    0    0
T7     0    0    0    0    0    0   13    0    0    0
T8     0    0    0    0    0    0    0    9    0    0
T9     0    0    0    0    0    0    0    0    8    0
T10    0    0    0    0    0    0    0    0    0   16
```

**Fold 4** — Acuratețe: 90.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    11    0    0    0    0    0    0    0    0    0
T2     0   10    0    0    0    0    0    0    1    0
T3     0    0   10    0    0    0    0    0    0    0
T4     0    0    0    8    0    1    0    0    0    0
T5     0    0    0    0   12    0    0    0    0    0
T6     0    0    0    1    0    7    0    0    0    0
T7     0    0    0    0    0    0    7    0    0    0
T8     1    0    0    0    1    0    0    8    0    0
T9     0    0    0    1    1    0    1    0   11    0
T10    0    0    0    0    1    0    0    1    0    6
```

**Fold 5** — Acuratețe: 90.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    10    0    0    0    0    0    0    0    0    0
T2     0    8    0    0    0    0    0    0    0    0
T3     0    0   11    0    0    0    0    0    0    0
T4     0    0    0   12    0    0    0    0    0    0
T5     0    0    0    0    8    0    0    0    0    1
T6     0    0    0    1    0    9    0    0    0    0
T7     0    0    0    0    0    1    7    0    0    0
T8     0    0    3    0    0    0    0   10    0    0
T9     0    0    0    0    0    0    0    0    7    0
T10    0    1    0    0    2    0    0    1    0    8
```

**Fold 6** — Acuratețe: 94.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    11    0    0    0    0    0    0    0    0    1
T2     0   10    0    0    0    0    0    0    0    0
T3     0    0    8    0    0    0    0    0    0    0
T4     0    0    0   14    0    0    0    0    0    0
T5     0    0    0    0    4    0    0    0    0    0
T6     0    0    0    0    0    5    0    0    0    0
T7     0    0    0    0    0    0   13    0    0    0
T8     0    0    0    1    0    0    0   12    0    0
T9     0    0    0    2    0    0    0    0    6    0
T10    0    0    0    0    1    0    0    1    0   11
```

**Fold 7** — Acuratețe: 96.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    12    0    0    0    0    0    0    0    0    0
T2     0    8    0    0    0    0    0    0    0    0
T3     0    0   13    0    0    0    0    0    1    0
T4     0    0    0   10    0    0    0    0    0    0
T5     0    0    0    0    8    0    0    0    0    0
T6     0    0    0    0    0    7    0    0    0    0
T7     0    0    0    0    0    0   11    0    0    0
T8     0    0    0    0    0    0    0   12    0    1
T9     0    0    0    0    0    0    1    0    8    0
T10    1    0    0    0    0    0    0    0    0    7
```

**Fold 8** — Acuratețe: 92.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    10    0    0    0    0    0    0    0    0    0
T2     0   14    1    0    0    0    0    0    0    0
T3     0    0   10    0    0    1    0    0    0    0
T4     0    1    0    8    0    0    0    0    0    0
T5     0    0    1    0   12    0    1    0    0    0
T6     0    0    0    0    0    7    0    0    0    0
T7     0    0    0    0    0    0    6    0    0    0
T8     0    0    0    0    0    0    0    9    0    0
T9     0    0    1    0    0    0    1    0    5    0
T10    0    0    0    1    0    0    0    0    0   11
```

**Fold 9** — Acuratețe: 92.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    11    0    0    0    0    0    0    0    0    0
T2     0    9    0    0    0    0    0    0    0    0
T3     0    0    7    0    1    0    0    0    0    0
T4     0    0    1   17    0    0    0    0    0    1
T5     0    0    0    0    6    0    0    0    1    0
T6     0    0    0    0    0    5    0    0    2    0
T7     0    0    0    0    0    0   10    0    0    0
T8     0    0    0    0    0    0    0    9    0    0
T9     0    0    0    0    0    0    0    0    8    1
T10    1    0    0    0    0    0    0    0    0   10
```

**Fold 10** — Acuratețe: 93.00%
```
      P1   P2   P3   P4   P5   P6   P7   P8   P9  P10
T1    13    0    0    0    0    0    1    0    1    0
T2     0   10    0    0    0    0    0    1    0    0
T3     0    0    6    0    0    0    0    0    0    0
T4     0    0    0    8    0    0    0    0    0    0
T5     0    1    0    0    8    0    0    0    0    0
T6     0    0    0    0    0    9    0    0    0    0
T7     0    0    0    0    0    0   13    1    0    0
T8     0    0    0    0    0    0    0    7    0    0
T9     0    0    1    0    0    0    0    1   13    0
T10    0    0    0    0    0    0    0    0    0    6
```

---

## 3. Codul Modelului cu Comentarii

```python
"""
MLP (Perceptron Multi-Strat) implementat de la zero folosind doar NumPy.
Arhitectura: 784 -> 256 -> 128 -> 10
Antrenament: Mini-batch SGD + Cross-Entropy Loss
"""

import numpy as np


# ─────────────────────────────────────────────
# FUNCȚII DE ACTIVARE
# ─────────────────────────────────────────────

def relu(x):
    """
    Funcția de activare ReLU (Rectified Linear Unit).
    Returnează max(0, x) element-wise.
    Folosită în straturile ascunse pentru a introduce neliniaritate.
    """
    return np.maximum(0, x)


def relu_grad(x):
    """
    Derivata funcției ReLU față de intrarea sa z (pre-activare).
    Returnează 1 unde z > 0, și 0 în rest.
    Folosită în backpropagation pentru a propaga gradienții înapoi.
    """
    return (x > 0).astype(float)


def softmax(x):
    """
    Funcția Softmax — transformă logits în probabilități.
    Scade maximul din fiecare rând pentru stabilitate numerică (evită overflow).
    Suma probabilităților pe fiecare rând este întotdeauna 1.
    Folosită exclusiv la stratul de ieșire.
    """
    e = np.exp(x - x.max(axis=1, keepdims=True))  # stabilitate numerică
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────
# FUNCȚIA DE PIERDERE
# ─────────────────────────────────────────────

def cross_entropy(probs, y_onehot):
    """
    Entropia încrucișată categorială (Categorical Cross-Entropy Loss).
    Măsoară diferența dintre distribuția prezisă și cea reală.
    eps=1e-9 previne log(0) care ar produce valori -inf.

    Args:
        probs:    probabilitățile prezise,          shape (N, 10)
        y_onehot: etichetele reale în format one-hot, shape (N, 10)
    Returns:
        pierderea medie pe batch (scalar)
    """
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-9), axis=1))


# ────────────��────────────────────────────────
# CLASA MLP
# ─────────────────────────────────────────────

class MLP:
    def __init__(self, layer_sizes, lr=0.01, seed=42):
        """
        Inițializează rețeaua neuronală MLP.

        Args:
            layer_sizes: lista cu dimensiunile fiecărui strat, ex: [784, 256, 128, 10]
            lr:          rata de învățare (learning rate)
            seed:        sămânța pentru reproducibilitate
        """
        rng = np.random.default_rng(seed)  # generator aleator reproductibil
        self.lr = lr
        self.weights = []  # lista de matrice de ponderi W pentru fiecare strat
        self.biases  = []  # lista de vectori de bias b pentru fiecare strat

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]      # numărul de neuroni din stratul anterior
            fan_out = layer_sizes[i + 1]  # numărul de neuroni din stratul curent

            # Inițializare He (Kaiming): W ~ N(0, sqrt(2/fan_in))
            # Concepută pentru ReLU — menține varianța activărilor stabilă
            W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
            b = np.zeros(fan_out)  # bias-urile se inițializează cu zero

            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Propagare înainte (Forward Pass).
        Calculează activările fiecărui strat și returnează predicțiile finale.

        Args:
            X: datele de intrare, shape (N, 784)
        Returns:
            a:     probabilitățile Softmax de ieșire, shape (N, 10)
            cache: lista de tupluri (a_prev, z) pentru fiecare strat,
                   necesare în backpropagation
        """
        cache = []
        a = X  # activarea curentă — la început este chiar intrarea

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b  # combinație liniară: z = a * W + b

            if i < len(self.weights) - 1:
                a_next = relu(z)     # straturi ascunse: activare ReLU
            else:
                a_next = softmax(z)  # strat de ieșire: activare Softmax

            cache.append((a, z))  # salvăm (intrarea stratului, pre-activarea)
            a = a_next            # activarea devine intrarea pentru stratul următor

        return a, cache

    def backward(self, probs, y_onehot, cache):
        """
        Propagare înapoi (Backward Pass — Backpropagation).
        Calculează gradienții și actualizează ponderile și bias-urile.

        Args:
            probs:    probabilitățile prezise de forward pass, shape (N, 10)
            y_onehot: etichetele reale one-hot, shape (N, 10)
            cache:    activările și pre-activările salvate din forward pass
        """
        n = probs.shape[0]  # numărul de exemple din batch

        # Gradientul la stratul de ieșire:
        # Combinând derivata Cross-Entropy și Softmax rezultă formula simplă:
        # delta = (p_prezis - y_real) / N
        delta = (probs - y_onehot) / n

        # Parcurgem straturile în ordine inversă (de la ieșire spre intrare)
        for i in reversed(range(len(self.weights))):
            a_prev, z = cache[i]  # recuperăm activarea anterioară și pre-activarea

            dW = a_prev.T @ delta   # gradientul față de ponderi W
            db = delta.sum(axis=0)  # gradientul față de bias-uri b

            if i > 0:
                # Propagăm gradientul înapoi prin ReLU:
                # delta_nou = (delta @ W^T) * relu_grad(z_anterior)
                _, z_prev = cache[i - 1]
                delta = (delta @ self.weights[i].T) * relu_grad(z_prev)

            # Actualizăm ponderile și bias-urile prin gradient descent
            self.weights[i] -= self.lr * dW
            self.biases[i]  -= self.lr * db

    def predict(self, X):
        """
        Prezice clasa pentru datele de intrare X.
        Returnează indicele clasei cu probabilitatea maximă (argmax).

        Args:
            X: datele de intrare, shape (N, 784)
        Returns:
            array de indici de clasă (0-indexed), shape (N,)
        """
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=1)

    def train(self, X, y_onehot, epochs=100, batch_size=64):
        """
        Antrenează modelul folosind Mini-batch SGD.

        Args:
            X:          datele de antrenament, shape (N, 784)
            y_onehot:   etichetele one-hot, shape (N, 10)
            epochs:     numărul de epoci de antrenament
            batch_size: dimensiunea unui mini-batch
        """
        n = X.shape[0]  # numărul total de exemple de antrenament

        for epoch in range(1, epochs + 1):
            # Amestecăm datele la fiecare epocă pentru a evita ordinea fixă
            idx = np.random.permutation(n)
            X, y_onehot = X[idx], y_onehot[idx]

            total_loss = 0.0

            # Parcurgem toate mini-batch-urile
            for start in range(0, n, batch_size):
                Xb = X[start:start + batch_size]        # mini-batch intrări
                yb = y_onehot[start:start + batch_size] # mini-batch etichete

                probs, cache = self.forward(Xb)                    # forward pass
                total_loss += cross_entropy(probs, yb) * len(Xb)  # acumulăm pierderea
                self.backward(probs, yb, cache)                    # backward + update

            # Afișăm progresul la fiecare 10 epoci
            if epoch % 10 == 0 or epoch == 1:
                avg_loss = total_loss / n
                preds = self.predict(X)
                acc   = np.mean(preds == np.argmax(y_onehot, axis=1))
                print(f"Epoca {epoch:3d}/{epochs}  pierdere={avg_loss:.4f}  acuratete={acc:.4f}")


# ─────────────────────────────────────────────
# FUNCȚIA PRINCIPALĂ
# ─────────────────────────────────────────────

def main():
    # Încărcăm datele
    # Formatul original este (784 caracteristici × N exemple) → transpunem la (N × 784)
    X_train = np.loadtxt("train_samples.csv", delimiter=",").T   # (1000, 784)
    y_raw   = np.loadtxt("train_labels.csv")                     # (1000,)  valori 1-10
    X_test  = np.loadtxt("test_samples.csv",  delimiter=",").T   # (5000, 784)

    # Normalizăm datele la intervalul [0, 1]
    X_train = X_train / (X_train.max() + 1e-8)
    X_test  = X_test  / (X_test.max()  + 1e-8)

    # Convertim etichetele: 1-10 → 0-9 (indexare de la zero) + format one-hot
    n_classes = 10
    y = y_raw.astype(int) - 1        # 0..9
    y_onehot = np.eye(n_classes)[y]  # (1000, 10) — matrice one-hot

    # Construim și antrenăm modelul
    model = MLP(layer_sizes=[784, 256, 128, n_classes], lr=0.05)
    model.train(X_train, y_onehot, epochs=200, batch_size=64)

    # Generăm predicții pentru setul de test
    test_preds = model.predict(X_test) + 1  # +1 pentru a reveni la etichetele 1-10

    # Salvăm predicțiile în fișierul de submisie
    with open("submission.csv", "w") as f:
        f.write("Id,Prediction\n")
        for i, p in enumerate(test_preds, start=1):
            f.write(f"{i},{p}\n")

    print(f"\nSubmisie salvata in submission.csv ({len(test_preds)} predictii)")


if __name__ == "__main__":
    main()
```

---

*Documentație generată pentru proiectul de clasificare — MLP implementat cu NumPy*  
*Data: 1 Iunie 2026*

