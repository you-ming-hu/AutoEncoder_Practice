

# 變分自解碼器應用練習

## 1. 變分自解碼器介紹

### 變分自編碼器(Variational autoencoder, VAE)由兩部分組成

1. Encoder:

   將樣本空間X上某點x映射至編碼空間Z上對應的常態分佈。
   $$
   \mu_{Encoder(x)},log\sigma^2_{Encoder(x)} = Encoder(x)\qquad(1)Encoder將x映射成Z上常態分佈的參數\mu,log\sigma^2\\ 
   N(\mu_{Encoder(x)},\sigma^2_{Encoder(x)})\qquad(2)換算成其常態分佈，參數為\mu,\sigma^2\\
   \\
   (以下Encoder以E代替)
   $$
   換句話說，Encoder可以把任意x映射成專屬於該x在Z上的常態分佈，如下示意圖:

   <img src="C:\Users\Atlas\Dropbox\Portfolio\AutoEncoder_Practice\images\2584918486.png" alt="2584918486" style="zoom:50%;" />

2. Decoder:

   可以將編碼空間Z中的某點z映射回樣本空間X中的某點x
   $$
   \overline{x} = Decoder(z),\quad\overline{x}\in X\\
   \\
   (以下Decoder以D代替)
   $$
   

---

### 模型目標:

- 將空間X中某點x透過Encoder映射成在Z空上的某個常態分佈，再從這個分佈中重新採樣一點z，期望可以透過Decoder將採樣點z還原回x。

$$
find \quad E^*,D^*\\
E^*,D^*=\underset{E,D}{\operatorname{argmin}} Expect_{x \sim Data}[||x-D(E(x))||^2]\qquad Autoencoder通則\\
E^*,D^*=\underset{E,D}{\operatorname{argmin}} Expect_{x \sim Data}[||x-D(\mu_{E(x)}+\sigma_{E(x)} * \epsilon)||^2],\epsilon\sim N(0,1) \qquad 加入VAE重取樣的技巧
$$

---

### 細節講解:

- **簡單的理解**

  樣本空間X中每一點x都可以透過Encoder找到在編碼空間上專屬的分布，透過重採樣的過程，可以將分布轉換為具體位置z，在通過Decoder重建x。在取樣的過程中即便是相同的輸入x，取樣的結果z也會不同，但通過Decoder又要盡量映射回輸入的x。可知在編碼空間Z上面，歐式距離相近的點應該也要對應到相似的x，因為重採樣的點出分佈必須服從專屬的常態分佈，在最小化重建誤差的條件下，距離專屬常態分佈中央越近，取樣機率越高，對應的x重建誤差必須越小，才能滿足該條件。透過足夠多次重採樣，可以在編碼空間中鄰近區域以較稠密的採樣點覆蓋，產生近似於對該區域面積完整的重建，並且應映射至類似的x。因為稠密的點分佈，不是只有獨立一點做代表，使得空間上的映射比較為平滑，減少不連續映射(如圖)的發生。簡而言之，**在編碼空間Z上歐式距離越近，重建出來的x越類似**。

  不連續分佈示意圖<img src="C:\Users\Atlas\Dropbox\Portfolio\AutoEncoder_Practice\images\不連續分佈示意圖.png" alt="不連續分佈示意圖" style="zoom:24%;" />

- **重採樣與梯度**
  $$
  從一常態分佈N_{\mu,\sigma^2}中取樣x'可以利用標準常態分佈N_{0,1}的取樣去模擬，其轉換如下:\\
  if\quad N_{\mu,\sigma^2}(\epsilon') = N_{0,1}(\epsilon),\quad then\quad \epsilon' = \mu+\epsilon*\sigma\\
  所以實際上只要從N_{0,1}取樣\epsilon，再換算成\mu+\epsilon*\sigma，即等價於從N_{\mu,\sigma^2}中取樣，示意圖如下:\\
  $$
  <img src="C:\Users\Atlas\Dropbox\Portfolio\AutoEncoder_Practice\images\重取樣.png" alt="重取樣" style="zoom:50%;" />
  $$
  (1)\qquad\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial \epsilon'}\frac{\partial \epsilon'}{\partial \mu}\\
  (2)\qquad\frac{\partial L}{\partial \sigma^2} = \frac{\partial L}{\partial \epsilon'}\frac{\partial \epsilon'}{\partial \sigma^2}\\
  \\
  即便\epsilon是隨機非固定的，但仍可視為一常數，那麼從\\
  \epsilon' = \mu+\epsilon*\sigma\\
  可看出(1),(2)都是可導的
  $$
  得知**重採樣是一個可導的操作**。

- **分佈限制**

  重建的過程當中，μ會提供確切的位置，因為ε是隨機的，σ*ε則發揮了重取樣的作用。在有限且離散的資料樣本下，對於連續編碼空間上所有鄰近區域，訓練時實際上只能對應到單一的x，而不是對應到相似的x，所以σ無可避免會增加重建誤差。由於σ持續提供誤差，在進行梯度下降時，σ肯定會被越縮越小，以減少重建誤差。σ縮小則失去重取樣的功能，這是非期望的，故在此引入對分佈的限制條件，如數學式及式意圖如下。
  $$
  N(\mu_{E(x)},\sigma^2_{E(x)}) \approx N(0,1) \qquad , x \in X \quad and \quad x \sim Data
  $$
  <img src="C:\Users\Atlas\Dropbox\Portfolio\AutoEncoder_Practice\images\看齊標準常態分佈.png" alt="看齊標準常態分佈" style="zoom:50%;" />

  簡單的想法是:
  $$
  find \quad E^*\\
  E^* = \underset{E}{\operatorname{argmin}}Expect_{x \sim Data}[L2([\mu_{E(x)},\sigma^2_{E(X)}],[0,1])]
  $$
  但**VAE有另一種處理法，既然希望兩分佈相似，可以使用KL Divergence量測分佈的差異，數值越小則代表越接近目標分佈N(0,1)**，故改成以下:
  $$
  find \quad E^*\\
  E^* = \underset{E}{\operatorname{argmin}} Expect_{x\sim Data}[KLDivergence(N(0,1),N(\mu_{E(x)},\sigma^2_{E(x)}))]\\
  \\
  KLDivergence(N(0,1),N(\mu_{E(x)},\sigma^2_{E(x)}))
  =\frac{1}{2}(-log\sigma_{E(x)}^2+\mu^2_{E(x)}+\sigma^2_{E(x)}-1)\\
  化簡過程參考 \quad 苏剑林. (Mar. 18, 2018). 《变分自编码器（一）：原来是这么一回事 》\\
  \\
  E^* = \underset{E}{\operatorname{argmin}} Expect_{x\sim Data}[\frac{1}{2}(-log\sigma_{E(x)}^2+\mu^2_{E(x)}+\sigma^2_{E(x)}-1)]\\
  $$
  故總誤差函數除了重建誤差之外，還須包含資料點映射的分佈和標準常態分佈的差異，一個數學式正確，但邏輯及上不完整的想法，就是直接把兩者相加，於是整個模型改為以下:
  $$
  E^*,D^*=\underset{E,D}{\operatorname{argmin}} Expect_{x \sim Data}[
  ||x-D(\mu_{E(x)}+\sigma_{E(x)} * \epsilon)||^2+\frac{1}{2}(-log\sigma_{E(x)}^2+\mu^2_{E(x)}+\sigma^2_{E(x)}-1)]\\
  ,\epsilon\sim N(0,1)
  $$
  完整的推導參考《变分自编码器（二）：从贝叶斯观点出发 》

  另外對於分佈的限制並不僅限於使用KL Divergence，參考《变分自编码器（三）：这样做为什么能成？ 》，裡面提及靠單一採樣點估測其專屬常態分佈與標準常態分佈N(0,1)差異的方法，且與KL Divergence是等價的，其實現程式碼在參考文獻Convolutional Variational Autoencoder中Define the loss function and the optimizer部分可以找到。

- **多變數常態分佈**

  前述所有的公式僅提及單變數常態分佈，但實際上編碼空間Z是多維的，事實上VAE中的Encoder所映射的分佈是假定多維度且各維度獨立的常態分佈。

  **在各維度獨立的條件下，把各維度的KLDivergence總合起來即可，若非獨立則不能這樣算，原來的KLDivergence修改為下式**。
  $$
  KLDivergence(N(0,1),N(\mu_{E(x)},\sigma^2_{E(x)}))
  =\frac{1}{2}\sum_{i=1}^{d}(-log\sigma_{E(x)i}^2+\mu^2_{E(x)i}+\sigma^2_{E(x)i}-1)\\
  ,d=dim(EncodingSpace)
  \\
  \\
  整體模型修改為:\\
  find \quad E^*,D^*\\
  E^*,D^*=\underset{E,D}{\operatorname{argmin}} Expect_{x \sim Data}[
  ||x-D(\mu_{E(x)}+\sigma_{E(x)} * \epsilon)||^2+\frac{1}{2}\sum_{i=1}^{d}(-log\sigma_{E(x)i}^2+\mu^2_{E(x)i}+\sigma^2_{E(x)i}-1)]\\
  ,\epsilon\sim N(0,1)\\
  ,d=dim(EncodingSpace)
  $$
  

---

### 總結:

完整的數學模型如下
$$
find \quad E^*,D^*\\
E^*,D^*=\underset{E,D}{\operatorname{argmin}} Expect_{x \sim Data}[
||x-D(\mu_{E(x)}+\sigma_{E(x)} * \epsilon)||^2+\frac{1}{2}\sum_{i=1}^{d}(-log\sigma_{E(x)i}^2+\mu^2_{E(x)i}+\sigma^2_{E(x)i}-1)]\\
,\epsilon\sim N(0,1)\\
,d=dim(EncodingSpace)
$$
VAE即為Encoder與Decoder的組合。

---

參考文獻&圖片引用: 

1. [苏剑林. (Mar. 18, 2018). 《变分自编码器（一）：原来是这么一回事 》]( https://spaces.ac.cn/archives/5253)
2. [苏剑林. (Mar. 28, 2018). 《变分自编码器（二）：从贝叶斯观点出发 》]( https://spaces.ac.cn/archives/5343)
3. [苏剑林. (Apr. 03, 2018). 《变分自编码器（三）：这样做为什么能成？ 》]( https://spaces.ac.cn/archives/5383)
4. [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae)

## 2. 應用練習

**資料來源:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

**程式碼架構:**

```
├─ Variational_AutoEncoder.ipynb : 模型實現
├─ data_description.txt
├─ test.csv
└─ train.csv
```

---

## 發想:

房子在建造的時候可能會有某些特定的配置，像是越多的房間也有較高的機會搭配越多的衛浴設備，甚至可以進行聚類，例如:二房一衛、四房兩衛等等的配置。然而這些配置也會影響房價，像是四房比二房貴的機率比較高。最終可以從潛變量空間中約略預測房價。那麼在類別變項中或許也有這樣的組合，並且對於房價有預測能力。
## 理論:
若一群資料點在高維度空間中有某種分布，且該分布可以用較低的維度表達，則可以訓練VAE將資料點投影進低維度空間，再從低維度空間重建輸入資料。
換句話說，可以透過VAE建立能夠儲存資訊的低維度空間，且給定一個位於該低維空間中的點可以反推其在高維度空間中的位置。

## 假設:
若房屋的類別變項具備某種特定分布，應該可以輸入至VAE在重建回來，同時在潛變量空間中存在有意義的分布，或許跟房價有某種程度上應對關係。
## 模型意義:
這是一個非監督式學習。因為不需要用到答案(這裡是SalePrice)，所以可以把訓練跟測試資料都拿來使用，不浪費測試資料，也比較不須擔心對SalePrice的過度擬合。若可以在潛變量空間上看出房價的分布，基本上可以確定類別變項對房價是有影響的，畢竟訓練過程模型沒有對房價進行推論，不可能從中學到關於房價的任何資訊。

## 結論:



## 心得:



---

## 備註:
