# Atividade Ponderada: MLP do zero em NumPy

Implementacao didatica de um Multi-Layer Perceptron para classificar digitos do MNIST sem PyTorch, TensorFlow ou frameworks de deep learning no calculo da rede. O PyTorch/Torchvision entra apenas como fonte de download do dataset MNIST, conforme permitido no enunciado.

## Fontes iniciais de estudo

- [Teaching a Perceptron by Hand](https://thomascountz.com/2018/03/26/perceptrons-implementing-and-part-1), Thomas Countz: usei como ponto de partida para raciocinar sobre pesos, vieses e a ideia de transformar entradas em uma decisao.
- [Parameter optimization in neural networks](https://www.deeplearning.ai/ai-notes/optimization/index.html), DeepLearning.AI: usei como base para organizar a relacao entre loss, custo medio, gradiente, learning rate e batch size.

## Estrutura

```text
.
|-- mlp/                    # Implementacao manual da rede
|-- notebooks/              # Notebook principal da ponderada
|-- results/                # Curvas, tabelas e metricas geradas
|-- scripts/                # Execucao reprodutivel dos experimentos
|-- tests/                  # Testes unitarios e gradient check
|-- README.md
`-- requirements.txt
```

## Evolucao registrada

Este README sera atualizado a cada marco relevante. A intencao e deixar claro nao so o resultado final, mas tambem as pedras no caminho encontradas durante a producao.

| Marco | Evidencia | Pedra no caminho | Decisao |
| --- | --- | --- | --- |
| Escopo inicial | Enunciado do notebook original em `Downloads` | Leitura inicial falhou porque o terminal PowerShell nao aceita heredoc no estilo Bash | Usar sintaxe nativa do PowerShell para inspecao |
| Inspecao do enunciado | Requisitos extraidos: 2 camadas ocultas, ReLU, softmax, cross-entropy, SGD, 92%+ | Console em `cp1252` quebrou ao imprimir simbolos matematicos | Forcar stdout UTF-8 nas inspecoes |
| Repositorio dedicado | Repo publico `C-Icaro/ponderada-mlp-mnist-numpy` | O repo anterior era generico e ja tinha arquivos nao relacionados | Criar repo novo, publico e isolado para a ponderada |
| Forward pass | `tests/test_forward.py` | Antes de treinar, a rede precisava provar que dimensoes e probabilidades estavam coerentes | Testar logits e soma da softmax antes de implementar backprop |
| Backpropagation | `tests/test_backprop.py` | A loss do MNIST so aponta que algo esta errado, mas nao onde | Criar gradient check numerico em uma rede pequena |
| Loader e treino | `mlp/data.py` e `scripts/train.py` | `tensorflow/keras` nao estavam instalados no ambiente local | Usar `torchvision.datasets.MNIST` apenas para baixar os dados e converter tudo para NumPy |

## Como rodar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train.py
```

## Status

Em desenvolvimento. A arquitetura, resultados, comparacao de experimentos, decisoes tecnicas e dificuldades serao preenchidos conforme os commits avancarem.
