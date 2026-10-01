# EP1 — Árvores e florestas de decisão oblíquas

O [oDT.py](oDT.py) implementa uma **árvore de decisão oblíqua** (`oDT`, *oblique Decision Tree*) e uma **floresta de árvores oblíquas** (`oRF`). O experimento compara a floresta proposta com a `RandomForestClassifier` do scikit-learn, medindo acurácia e tempo de execução, e depois treina a oRF com todos os dados rotulados para gerar uma submissão.

[Voltar ao README principal](../README.md)

## A ideia: cortes que combinam atributos

Uma árvore tradicional divide os dados usando um atributo por vez, com uma regra como `x1 <= 2`. Sua fronteira fica alinhada aos eixos do espaço de atributos.

Na árvore oblíqua, cada nó combina vários atributos em uma projeção:

```text
z = x @ w = w1*x1 + w2*x2 + ... + wm*xm

z <= th_star  → ramo esquerdo
z >  th_star  → ramo direito
```

O vetor `w` determina a direção da projeção e `th_star` determina o limiar. Por exemplo, `x1 + x2 <= 3` produz um corte diagonal em duas dimensões. Esse tipo de regra pode representar separações que exigiriam vários cortes alinhados aos eixos.

A implementação busca boas direções e bons limiares em cada nó; ela não resolve uma otimização global da árvore.

## Como uma árvore é construída

### 1. Propor direções de corte

`generate_directions` reúne candidatos de três fontes:

| Fonte | Como o código obtém a direção | Papel na busca |
| --- | --- | --- |
| SVM linear | Treina `LinearSVC` no nó e usa os vetores de `coef_`. | Propõe direções supervisionadas de separação entre classes. |
| LDA | Treina `LinearDiscriminantAnalysis` e usa os vetores de `coef_`. | Acrescenta direções obtidas por análise discriminante. |
| Aleatória | Sorteia vetores com `np.random.randn`. | Explora outras orientações e diversifica as árvores. |

Todos os vetores aceitos são normalizados para norma 1. A SVM usa `class_weight="balanced"`; esse balanceamento se aplica à geração de suas direções, não ao cálculo da entropia da árvore.

**SVM e LDA sugerem a orientação, mas não decidem o corte final.** Seus interceptos não são utilizados: o limiar é escolhido separadamente pelo critério da árvore. Se o ajuste de SVM ou LDA falhar, a exceção é ignorada e a busca continua com os candidatos disponíveis.

### 2. Encontrar o melhor limiar de cada projeção

Para cada vetor `w`, `best_threshold_for_projection`:

1. Calcula `z = X @ w` e ordena as amostras por essa projeção.
2. Considera posições que deixem pelo menos `min_samples_leaf` amostras em cada lado.
3. Descarta posições entre valores iguais, pois não permitem separar as amostras por um limiar.
4. Se houver muitos candidatos, seleciona até `n_thresholds` posições distribuídas ao longo da lista de posições válidas.
5. Calcula o ganho de informação de cada candidato e escolhe o maior.

O limiar é a média dos dois valores de projeção que delimitam a posição escolhida. Com `n_thresholds=None`, todas as posições válidas são avaliadas.

### 3. Comparar os cortes pelo ganho de informação

O código mede a mistura de classes pela entropia, em bits:

```text
H(Y) = -Σ p(c) * log2(p(c))

Ganho = H(pai)
        - (n_esquerda/n) * H(esquerda)
        - (n_direita/n) * H(direita)
```

Quanto maior o ganho, maior a redução da incerteza sobre a classe após a divisão. `get_best_split` escolhe o par `(w_star, th_star)` com maior ganho entre as direções propostas.

Para evitar recontar as classes a cada limiar, o código codifica os rótulos em índices, monta uma matriz *one-hot* e calcula contagens acumuladas. Assim, obtém as contagens à esquerda e à direita para vários candidatos em operações vetorizadas do NumPy.

### 4. Construir os filhos ou criar uma folha

`build_tree` repete o processo recursivamente. Um nó vira folha quando:

- Todas as amostras pertencem à mesma classe.
- A profundidade atinge `max_depth`.
- Há menos de `2 * min_samples_leaf` amostras.
- Nenhum corte válido é encontrado, ou um dos filhos ficaria pequeno demais.

Uma folha guarda a classe majoritária. Em empate, vence a primeira classe na ordem produzida por `np.unique`. O código não exige um ganho estritamente positivo para aceitar um corte.

Na predição, cada amostra percorre a árvore pelas regras de projeção até alcançar uma folha.

## Da árvore à floresta

A classe `oRF` treina várias árvores usando **bootstrap**: para cada árvore, sorteia com reposição tantas amostras quanto existem no conjunto de treino. Algumas amostras aparecem várias vezes e outras ficam de fora daquele sorteio.

Cada árvore faz sua própria busca de cortes. Para prever uma classe, a floresta coleta as previsões de todas as árvores e escolhe a mais votada. Empates seguem a ordem de `np.unique`.

Essa implementação usa todos os atributos na geração das direções; não há seleção aleatória de um subconjunto de atributos por nó. A diversidade vem do bootstrap e das direções aleatórias. O treinamento das árvores ocorre sequencialmente.

## Fluxo do experimento

A função `main` executa o seguinte fluxo:

1. Fixa a semente aleatória em `42` e carrega o arquivo `data.npz`.
2. Separa os dados rotulados em **80% para treino e 20% para validação**, com permutação aleatória sem estratificação.
3. Padroniza os atributos da oRF usando apenas média e desvio padrão do treino, aplicando os mesmos valores à validação.
4. Treina e avalia a oRF e a Random Forest tradicional na mesma divisão.
5. Salva a comparação de acurácia, tempo de treino e tempo de predição.
6. Recalcula a padronização com todos os dados rotulados, treina uma nova oRF e prevê as classes do teste.
7. Salva as previsões em `submission.csv`.

A padronização faz diferença para as projeções e para o ajuste de SVM/LDA. Colunas com desvio padrão zero recebem divisor 1. A Random Forest tradicional recebe os atributos sem padronização.

O treinamento final usa a **oRF**, independentemente de qual modelo obtenha a maior acurácia na validação.

## Parâmetros do experimento

| Parâmetro da oRF | Valor | Efeito |
| --- | --- | --- |
| `n_trees` | `100` | Quantidade de árvores e de votos por previsão. |
| `max_depth` | `10` | Profundidade máxima, com a raiz no nível 0. |
| `min_samples_leaf` | `8` | Mínimo de amostras em cada filho de um corte. |
| `n_random` | `5` | Quantidade de direções aleatórias propostas por nó. |
| `n_thresholds` | `60` | Limite de posições de corte avaliadas por direção. |

A Random Forest de comparação também usa 100 árvores, profundidade máxima 10 e mínimo de 8 amostras por folha, com `random_state=42` e `n_jobs=-1`. As estratégias de construção e os demais parâmetros diferem entre os modelos, portanto a comparação avalia essas duas configurações completas.

## Organização do código

| Componente | Responsabilidade |
| --- | --- |
| `Node` | Guarda direção, limiar, filhos ou rótulo de uma folha. |
| `oDT` | Gera candidatos, escolhe cortes, constrói a árvore e faz previsões. |
| `bootstrap_sample` | Sorteia uma amostra de treino com reposição. |
| `oRF` | Treina as árvores e agrega suas previsões por votação. |
| `train_val_split` | Cria a divisão aleatória entre treino e validação. |
| `standardize` | Padroniza conjuntos usando as estatísticas do conjunto de referência. |
| `main` | Executa a comparação e gera os arquivos de saída. |

## Execução e dados

### Disponibilidade e possível origem do dataset

Os dados do EP1 foram disponibilizados pelo professor em uma competição no Kaggle. O acesso à versão original não está mais disponível, e o arquivo `data.npz` não está incluído neste repositório.

Segundo a lembrança do autor, a base utilizada era **Predict Students' Dropout and Academic Success**. Essa origem ainda não foi confirmada: o código não registra os nomes dos atributos, o significado dos rótulos nem as dimensões do arquivo original.

A [fonte original na UCI](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) contém **4.424 estudantes, 36 atributos e três classes**: `Dropout` (evasão), `Enrolled` (matriculado) e `Graduate` (concluinte). Os atributos incluem informações demográficas, socioeconômicas e de desempenho acadêmico. Há também uma [versão pública no Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention).

Essa base é compatível com a tarefa de classificação do código, mas essa compatibilidade não comprova que seja a mesma usada no EP1. Sem o arquivo original, não é possível recuperar a divisão entre treino e teste, a codificação das classes ou eventuais transformações feitas pelo professor.

### Dependências e arquivo de entrada

```bash
pip install numpy pandas scikit-learn
```

O caminho definido em `main` é:

```text
/kaggle/input/competitions/pcs-3838-2026/data.npz
```

Para executar fora do Kaggle, ajuste esse caminho para seu arquivo local. O arquivo deve conter:

| Chave | Conteúdo |
| --- | --- |
| `X_train` | Matriz numérica de atributos das amostras rotuladas, com formato `(n_amostras, n_atributos)`. |
| `y_train` | Vetor de classes, com formato `(n_amostras,)`. |
| `X_test` | Matriz de teste, com os mesmos atributos e na mesma ordem de `X_train`. |

Os relatórios e a submissão convertem os rótulos para inteiros, portanto o fluxo espera classes compatíveis com essa conversão.

### Alternativa: preparar os dados públicos

Para experimentar a implementação sem os dados da competição, é possível obter a base pública pela UCI e criar um novo `data.npz`. O exemplo abaixo usa a [interface oficial do ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo) para carregar o dataset de ID `697`.

Instale a dependência adicional:

```bash
pip install ucimlrepo
```

Execute o seguinte trecho na raiz do repositório, com acesso à internet:

```python
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=697)
X = dataset.data.features.to_numpy(dtype=float)
target = dataset.data.targets.iloc[:, 0]

# Convenção deste exemplo; a codificação original do EP1 é desconhecida.
class_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
y = target.map(class_mapping)
if y.isna().any():
    raise ValueError("O dataset contém rótulos fora do mapeamento esperado.")
y = y.to_numpy(dtype=int)

# Nova divisão para experimentação, com 20% reservados para teste.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

output_dir = Path("EP1/data")
output_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    output_dir / "data.npz",
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
)
```

Em `main`, substitua o carregamento do caminho do Kaggle por:

```python
data = np.load("EP1/data/data.npz")
```

Esse procedimento cria **um novo experimento com a base pública**. A codificação e a divisão acima são escolhas explícitas para esse exemplo e não reproduzem necessariamente o protocolo da competição. Os atributos categóricos permanecem com os códigos numéricos fornecidos pela base, sem conversão para *one-hot*.

O código ainda separa 20% de `X_train` para sua validação interna. A chave adicional `y_test` permite uma futura avaliação do teste reservado, mas não é utilizada pelo `main` atual. O CSV gerado nesse cenário contém previsões para a nova divisão local e não corresponde à submissão original do Kaggle.

### Executar o experimento

Após disponibilizar um arquivo compatível e configurar seu caminho, execute a partir da raiz:

```bash
python EP1/oDT.py
```

Os arquivos são gravados no diretório de execução:

- **`comparison_orf_vs_rf.csv`**: acurácia de validação e tempos de treino e predição dos dois modelos.
- **`submission.csv`**: colunas `ID` e `Prediction`, com IDs consecutivos a partir de 1.

## Limitações e interpretação dos resultados

A busca é aproximada: considera apenas as direções propostas e, por padrão, até 60 posições de corte por direção. Treinar SVM e LDA em muitos nós também acrescenta custo computacional, além da ordenação das projeções.

O experimento usa uma única divisão de validação e mede acurácia; não implementa validação cruzada, poda posterior ou avaliação fora do bootstrap (*out-of-bag*). Avisos de convergência são suprimidos e falhas na geração de direções supervisionadas são ignoradas, o que pode dificultar o diagnóstico.

Não há resultados numéricos de acurácia documentados aqui: é necessário executar o experimento com o dataset para avaliar o desempenho e o custo da abordagem.
