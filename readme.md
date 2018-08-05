# 作成メモ
- "Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal"(https://arxiv.org/abs/1712.02478) の実装
- chainerのpix2pixを元に一応動くところまでは作れた。
- Loss functionのパラメータはかなり適当。
- softmax cross entropyをうまく使いこなせず、sigmoid cross entropyを使用。
- batch normalizationとlamda2の調整で、影抽出まではきれいにできている。
- 影除去はうまくできているかわからない（電力不足のため、GPUが途中で落ちる）。
