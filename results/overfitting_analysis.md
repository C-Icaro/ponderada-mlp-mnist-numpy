# Analise de overfitting

Ha overfitting leve no modelo melhorado: treino chega quase a 100% e o gap treino-validacao fica perto de 2.02 p.p. Ainda assim, nao ha overfitting danoso na janela treinada, porque a validacao cai apenas 0.04 p.p. do pico e a acuracia de teste sobe em relacao ao modelo anterior.

| Modelo | Test acc | Gap treino-val | Melhor val acc | Queda val final | Diagnostico |
| --- | ---: | ---: | ---: | ---: | --- |
| compact_relu_64_32 | 96.34% | 0.51 p.p. | 96.25% ep.8 | 0.00 p.p. | sem sinal material |
| final_relu_128_64 | 97.59% | 1.53 p.p. | 97.09% ep.10 | 0.00 p.p. | overfitting leve |
| improved_relu_256_128_momentum | 98.16% | 2.02 p.p. | 98.01% ep.13 | 0.04 p.p. | overfitting leve |

Criterio usado: gap final treino-validacao acima de 1.5 p.p. indica sinal leve; queda de validacao acima de 1 p.p. ou aumento relevante de data loss indicaria overfitting danoso.
A decisao para esta entrega e manter o modelo melhorado, mas registrar que early stopping na epoca de melhor validacao seria a proxima melhoria natural.
