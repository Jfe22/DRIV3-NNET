# DRIV3-NNET
Driving Recognition &amp; Inference Via Transformer Neural Networks


A DRIV3-NNET é uma rede neuronal que classifica as manobras de um condutor como agressivas ou não. Os dados usados para treinar e testar a rede foram recolhidos durante uma viagem entre Abrantes e Leiria, captados por acelerómetros e giroscópios. Após a análise, identificaram-se padrões nesses dados, refletindo manobras concretas realizadas pelo condutor. O dataset original foi ajustado para representar essas manobras.
Como a rede é treinada neste novo dataset que representa as manobras do condutor, faz sentido que ela classifique cada manobra individualmente como agressiva ou não, resolvendo assim um problema de classificação binária multi- classe, em que as classes são independentes umas das outras.
A rede principal utiliza a arquitetura multi-layer perceptron. Foram realizadas várias experiências com redes baseadas em arquitetura transformer, mas os resultados não foram tão positivos como os obtidos com a arquitetura multi-layer perceptron. No entanto, mais modelos com transformers estão a ser criados e testados, e se algum superar o desempenho do multi-layer perceptron, será adotado como o modelo principal.
