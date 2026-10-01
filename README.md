# PCS3838 — Inteligência Artificial

Repositório de exercícios, experimentos e projetos desenvolvidos para a disciplina PCS3838 — Inteligência Artificial.

## Projetos

| Pasta | Projeto | Descrição |
| --- | --- | --- |
| [EP1](EP1/README.md) | Árvores e florestas de decisão oblíquas | Combina atributos em cortes lineares e compara uma floresta oblíqua com a Random Forest tradicional. |
| [EP2](EP2/README.md) | Identificador de circuitos lógicos | Analisa diagramas para contar portas lógicas e calcular a saída do circuito a partir dos valores das entradas. |

## EP1 — Classificação com cortes oblíquos

O [oDT.py](EP1/oDT.py) constrói árvores cujos cortes combinam vários atributos. As direções vêm de SVM, LDA e vetores aleatórios; o ganho de informação escolhe o corte de cada nó. Uma floresta reúne essas árvores por bootstrap e votação majoritária.

O experimento compara acurácia e tempo de execução com uma Random Forest tradicional e gera previsões para submissão. Veja as [ideias do algoritmo e instruções de execução](EP1/README.md).

Os dados originais da competição do EP1 não estão mais acessíveis. O [README do EP1](EP1/README.md#disponibilidade-e-possível-origem-do-dataset) documenta a possível origem da base e explica como preparar os dados públicos para um novo experimento.

## EP2 — Perguntas sobre circuitos

O [identificador de circuitos](EP2/identificador_circuitos.py) combina visão computacional, extração de grafos e OCR. O modelo Qwen2.5-VL-7B-Instruct atua como alternativa quando a contagem ou a simulação determinística não pode ser utilizada.

O programa responde a três tipos de pergunta:

- **Contagem por tipo:** “Quantas portas AND existem no circuito?”
- **Contagem total:** “Quantas portas lógicas existem ao todo?”
- **Simulação:** “Qual é o valor da saída Q para os valores informados de x0, x1, …?”

As perguntas enviadas ao código usam os padrões em inglês documentados no [README do EP2](EP2/README.md#perguntas-suportadas). As respostas são números inteiros ou `True`/`False`.

### Exemplo de diagrama

![Circuito lógico com entradas x0 a x10 e saída Q](EP2/circuits/circuit_0.png)

Veja a [galeria de circuitos](EP2/README.md#imagens-de-exemplo) e as [instruções de execução](EP2/README.md#como-executar) para conhecer os dados de entrada, as dependências e o funcionamento do projeto.
