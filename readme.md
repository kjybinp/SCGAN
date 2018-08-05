# 作成メモ
- "Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal"(https://arxiv.org/abs/1712.02478) の実装
- chainerのpix2pixを元に作成中。
- softmax cross entropyをうまく使いこなせず、sigmoid cross entropyを使用。
- batch normalizationとlamda2の調整で、影抽出まではできている。
