# Analise de overfitting

O problema encontrado foi overfitting leve no modelo improved: treino quase perfeito e gap treino-validacao de 2.03 p.p. A resolucao aplicada foi early stopping por val_data_loss com restauracao do melhor checkpoint. O gap caiu para 1.20 p.p. e a acuracia de teste ficou em 97.86%.

| Modelo | Test acc | Gap treino-val | Melhor val acc | Queda val final | Epoca selecionada | Diagnostico |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| compact_relu_64_32 | 96.34% | 0.66 p.p. | 96.25% ep.8 | 0.00 p.p. | 8 | sem sinal material |
| final_relu_128_64 | 97.59% | 1.49 p.p. | 97.09% ep.10 | 0.00 p.p. | 10 | sem sinal material |
| improved_relu_256_128_momentum | 98.16% | 2.03 p.p. | 98.01% ep.13 | 0.04 p.p. | 15 | overfitting leve |
| resolved_relu_256_128_l2_earlystop | 97.86% | 1.20 p.p. | 97.86% ep.8 | 0.00 p.p. | 8 | mitigado por early stopping |

Criterio usado: gap final treino-validacao acima de 1.5 p.p. indica sinal leve; queda de validacao acima de 1 p.p. ou aumento relevante de data loss indicaria overfitting danoso.
A decisao para esta entrega e manter o modelo resolvido com early stopping como resposta ao problema encontrado.
