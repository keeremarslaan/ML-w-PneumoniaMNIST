# Makine Öğrenmesi ile Zatürre Tespiti Yapılması Üzerinde Görüntü Bozukluklarının Etkisi

> **Araştırma odağı:** Görüntü kalitesi değiştiğinde bir modelin başarısı, verdiği olasılıklar ve kararını açıklayan haritalar nasıl değişir?

Bu depo, PneumoniaMNIST üzerinde bu soruyu tek bir notebook akışıyla incelemek için hazırlanmıştır. Küçük bir CNN ile fine-tune edilen bir ResNet18 karşılaştırılır; ardından görüntü bozulmaları, olasılık kalibrasyonu ve dayanıklılık eğitimi birlikte değerlendirilir.

**Başlamak için:** [Çalıştırma rehberine](#calistirma) geçin. Deney tasarımını anlamak için [veri ve yöntemi](#yontem), bulguları incelemek için [sonuçları](#sonuclar) okuyun.

### Okuma rehberi

- [Kurulum ve notebook’un çalıştırılması](#calistirma)
- [Veri, modeller ve değerlendirme yöntemi](#yontem)
- [Grafikler, temel sonuçlar ve ayrıntılı ölçümler](#sonuclar)
- [Çalışmanın sınırlılıkları](#sinirliliklar)
- [Çıktılar ve çalıştırma kayıtları](#ciktilar)
- [Kaynaklar ve kullanılan araçlar](#kaynaklar)

### Deneyin mantığı

1. **Temiz veriyle karşılaştırma:** CNN ve ResNet18 eğitilir; model ve checkpoint temiz validation AUROC ile seçilir.
2. **Sabit seçimlerle değerlendirme:** Validation üzerinde belirlenen karar eşiği ve sıcaklık parametresi, temiz ve bozulmuş test görüntülerine uygulanır.
3. **Birlikte yorumlama:** Sınıflandırma başarısı, olasılık kalibrasyonu ve Grad-CAM haritaları farklı sorulara yanıt verir; sonuçlar birlikte okunur.
4. **Dayanıklılık karşılaştırması:** Bozulmalarla eğitilen model aynı seedler üzerinden baseline ile karşılaştırılır. Bu aşama keşifseldir.

### Depoyu kullanırken

Deneyin çalışma merkezi `study.ipynb`, ayarların giriş noktası notebook içindeki `cfg` sözlüğüdür. Yeni deney ayarları bu akış üzerinden yönetilir; üretilen ayar kopyaları ve sonuç kayıtları çalıştırmanın izlenebilmesini sağlar. Bağımlılıklar `requirements.txt` ile tanımlanır.

Bu depoda notebook, bağımlılık listesi ve `presentation.pptx` bulunur. Aşağıdaki raporda anılan `outputs/`, `runs/`, `config.yaml` ve sonuç arşivleri mevcut depo kopyasında yer almamaktadır. İlgili yollar, raporlanan çalıştırmanın çıktı düzenini anlatır; sonuç tabloları ve kaynaklar mevcut rapordan aynen korunmuştur.

---
=======

Bu projede gürültü, bulanıklık ve düşük kontrastın zatürre sınıflandırmasına etkisi incelenmiştir.
Sıfırdan eğitilen CNN ile ImageNet ağırlıkları kullanılarak fine-tune edilen ResNet18 karşılaştırılmıştır.
Classification metrikleri, olasılık kalibrasyonu ve Grad-CAM haritaları beraber değerlendirilmiştir.
Ayrı bir dayanıklı training ile bozulmuş görüntülerdeki başarının ve temiz görüntülerdeki değişimin ölçülmesi amaçlanmıştır.


**Buradaki sonuçlar, `outputs/` ve sunum 8 Kasım 2026 tarihinde tamamlanan yeni Colab çalıştırmasına aittir.**
Dört baseline ve üç dayanıklı training baştan yapılmıştır. Önceki Kaggle sonuçları ayrı arşivde korunur.
Yeni tekrar, baseline test verisi daha önce görüldüğü için dayanıklılık çalışmasını bağımsız doğrulama haline getirmez.

<a id="calistirma"></a>

=======

## 1. Notebook'un çalıştırılması

1. Sadece `study.ipynb` dosyasını [Google Colab](https://colab.research.google.com/) üzerine yükleyin.
2. GPU seçin. Ayarları notebook içindeki `cfg` üzerinden düzenleyin.
3. Hücreleri sırayla çalıştırın. Eksik paketler, resmi veri ve ImageNet ağırlıkları otomatik indirilir.
4. Baseline checkpoint ZIP'ini, dayanıklılık karşılaştırması yapılırsa onun checkpoint ZIP'ini ve sonuç ZIP'ini bilgisayarınıza indirin.

Bağlantı verilirse liste `runs/bootstrap/requirements.txt` içine indirilir ve kurulum öncesinde notebook içindeki listeyle karşılaştırılır.
Boş satırlar ve yorumlar dikkate alınmaz; listeler farklıysa veya indirme başarısızsa işlem durur.
GitHub sayfası yerine raw dosya bağlantısı kullanılmalıdır.

**Tek ayar girişi `cfg`'dir.** `config.yaml` sadece oluşturulmuş bir ayar kopyasıdır, notebook tarafından okunmaz.
Kaynak kimliği kontrolünden sonra `runs/results/{baseline,robustness_v1}/{mode}/config.yaml` üretilir ve ZIP'lere eklenir.
Farklı bir mevcut kayıt varsa üzerine yazılmaz. PyYAML hem gömülü listede hem `requirements.txt` içinde bulunur.

`execution.mode='full'` dört baseline training yapar. Temiz validation ResNet18'i seçerse üç seedli dayanıklılık karşılaştırması da çalışır;
CNN seçilirse bu ResNet karşılaştırması atlanır. `smoke` ise 128 training/64 validation görüntüsü ve bir epoch kullanır;
test inference ve bilimsel üç-seed özeti yapmaz. Full training CUDA/Linux, smoke CPU/MPS üzerinde de çalışır.

Colab başlangıcı için başka proje dosyası gerekmez. Yerelde Python 3.10+ ile:

```bash
python -m pip install -r requirements.txt
```

Aynı notebook isteğe bağlı olarak arayüzsüz de çalıştırılabilir:

```bash
python -m pip install nbconvert ipykernel
python -m jupyter nbconvert --execute --to notebook \
  --ExecutePreprocessor.timeout=-1 --output study.executed.ipynb study.ipynb
```

Tamamlanan trainingler sadece kod, ayarlar, ortam ve dosya kontrolleri eşleşirse tekrar kullanılır.
Uyuşmazlık varsa eski dosyaları koruyup inceleyin. Best checkpointler yarım kalan traininge devam etmek için optimizer durumu içermez.
Bu komutlar çalıştırma yönergesidir; güncel sürümün yeniden tam çalıştırıldığı anlamına gelmez.

<a id="yontem"></a>

## 2. Veri ve yöntem

| Özellik | Kullanılan ayar |
|---|---|
| Veri | Resmi PneumoniaMNIST, 128 × 128 grayscale; normal=0, zatürre=1 |
| Splitler | 4.708 train, 524 validation, 624 test; resmi splitler korunur |
| Normal/zatürre sayıları | Train 1.214/3.494; validation 135/389; test 234/390 |
| Modeller | CNN: 23.473 parametre; ImageNet ResNet18: 11.177.025 parametre, tümü fine-tune edilir |
| Normalizasyon | CNN mean/std 0.5/0.5; ResNet aynı grayscale görüntünün üç kopyası ve ImageNet normalizasyonu |
| Baseline augmentation | Sadece training: ±5° dönme, en fazla %2 kaydırma; flip ve kalite bozulması yok |
| Training | Binary cross-entropy, AdamW, batch 64, en fazla 20 epoch, patience 5 |
| Learning rate / weight decay | CNN 0.001; ResNet backbone 0.0001, head 0.001; weight decay 0.0001 |
| Seedler | Ana seed 42; seçilen model ayrıca 123/2026; corruption seed 8675309 |
| Seçim | Checkpoint ve model sadece temiz validation AUROC üzerinden seçilir |
| Threshold | Temiz validation üzerinde Youden's J; test koşullarında sabit |
| Kalibrasyon | Her checkpoint için temiz validation NLL üzerinden tek pozitif T |

Ağırlıklar train üzerinde öğrenilir. Checkpoint, model, threshold ve T seçimleri temiz validation üzerinden yapılır.
Test sadece sabit seçimleri değerlendirmek için kullanılır; test sonucuna göre seed seçilmez.
Checkpoint eşitliğinde önceki epoch; model skorları 1e-12 içinde eşitse CNN seçilir.
Threshold eşitliğinde aynı toleransla daha büyük değer alınır. Test sınıf sayıları sadece veriyi tanımlar.

Bozulmalar normalizasyondan önce grayscale [0,1] üzerinde ayrı ayrı uygulanır:
gürültü SD 0.03/0.07/0.12, bulanıklık sigma 0.5/1/2 ve kontrast 0.8/0.6/0.4.
Bulanıklık kernel boyutu 13, padding reflect'tir; gürültü [0,1] aralığına kırpılır.
Gürültü resmi görüntü ID'sine ve corruption seedine bağlıdır; modeller ve seedler aynı evaluation görüntülerini kullanır.
Kontrast katsayısı küçüldükçe bozulma artar. Bozulma şiddetleri aileler arasında eşdeğer değildir.

AUROC ve AUPRC sınıfların sıralanmasını ölçer; burada AUPRC, average precision'dır.
Sensitivity zatürreli, specificity normal görüntülerin doğru bulunma oranıdır.
Precision, zatürre tahminlerinin ne kadarının doğru olduğunu; F1, precision ve sensitivity'nin birleşimini gösterir.
Balanced accuracy, sensitivity ve specificity'nin ortalamasıdır. Youden's J = sensitivity + specificity − 1.
Üç seedin ortalaması ve örneklem standart sapması (`ddof=1`) raporlanır; bu SD hasta bazında güven aralığı değildir.

<a id="sonuclar"></a>

## 3. Grafikler ve temel sonuçlar

Grafikler PNG/PDF olarak saklanır. Aşağıdaki açıklamalar yeni Colab sonuçlarına aittir.

| Grafik | Nasıl okunur? |
|---|---|
| `training_examples` | İlk satır normal, ikinci satır zatürre; her sınıftan dört training görüntüsü. Aynı sınıfta da görünüm değişir. |
| `training_intensity` | X: görüntünün ortalama yoğunluğu; Y: sınıf içi yoğunluk. Her dağılımın alanı 1'dir. Zatürre dağılımı daha geniş olsa da sınıflar belirgin şekilde örtüşür. |
| `full_learning_curves` | Her satır bir training; X: epoch, solda loss, sağda AUROC. En iyi checkpoint son epoch olmayabilir. |
| `c_corruption_examples` | Aynı training görüntüsünün temiz ve üç bozulma seviyesindeki hali; satırlar bozulma aileleridir. |
| `full_severity` | Satırlar AUROC/F1/sensitivity/specificity, sütunlar bozulmalar. X: bozulma seviyesi, Y: metrik. İnce çizgiler seedler, siyah çizgi ortalama ± SD. |
| `d_reliability_resnet18_1/2` | On koşul, iki sayfa; solda raw, sağda scaled. X: ortalama zatürre olasılığı, Y: gerçek zatürre oranı. Köşegene yakınlık uyumu gösterir; bin sayıları azsa oran daha değişkendir. |
| `e_batch_replay_gradcam_TP/TN/FP/FN` | Satırlar görüntüler; sütunlar temiz/gürültü 0.12/bulanıklık 2/kontrast 0.4. Label ve raw olasılık gösterilir; renk ölçeği aynı görüntünün dört panelinde ortaktır. |
| `robustness_vs_baseline` | Satırlar metrikler, sütunlar bozulmalar; X: seviye, Y: metrik. Mavi kesikli daireler baseline, turuncu düz kareler dayanıklı model; ortalama ± SD gösterilir. |

### Classification ve görüntü kalitesi

Temiz testte ResNet42, CNN42'den altı metrikte de daha iyidir; AUROC 0.9858 ve 0.9540'dır.
ResNet42'de 38 FP/1 FN, CNN42'de 52 FP/16 FN vardır. Confusion tablolarında satırlar gerçek sınıf,
sütunlar tahmin; sıralama normal/zatürredir. Üç ResNet seedinin temiz AUROC değeri 0.9858 ± 0.0025'dır.

Gürültü SD 0.12'de baseline ortalama AUROC 0.7274 ± 0.1852'a düşmüştür.
Üç seed de tüm görüntüleri zatürre tahmin ettiği için sensitivity 1, specificity 0 olmuştur.
Temiz görüntülerdeki başarı, bozulmuş görüntülerde aynı başarıyı garanti etmemiştir.
Training lossun düşmesi de tek başına yeni veride başarı kanıtı değildir;
training metrikleri augmentation uygulanırken ve ağırlıklar değişirken hesaplanır.

### Kalibrasyon

NLL özellikle yüksek güvenli hataları, Brier olasılıkların karesel hatasını ölçer.
ECE, 15 eşit aralıkta ortalama olasılık ile gerçek zatürre oranını görüntü sayılarına göre karşılaştırır.
Düşük değer daha iyidir; boş aralıklar sıfır katkı verir, olasılık 1 son aralığa girer.

T, temiz validation NLL üzerinden logT optimizasyonuyla [0.05,20] arasında öğrenilir:
`p = sigmoid(logit / T)`. T tüm test koşullarında sabittir.
Threshold da dönüştürüldüğü için kararlar, AUROC ve average precision değişmez.
Baseline temiz NLL scaling öncesi 0.3296, sonrası 0.3311'dir. Dayanıklı modelde ise 0.3415 olan değer 0.2838'e düşmüştür.
Dayanıklı modelin NLL değeri tüm koşullarda azalmış, ECE ise her koşulda iyileşmemiştir.
Temiz validation üzerinde kalibrasyon yapmak testte her metriğin iyileşmesini garanti etmez.

### Grad-CAM

Hedef `layer4[-1].conv2` üzerinden raw zatürre skorudur. Pozitif katkılar gösterilir;
4 × 4 harita 128 × 128'e büyütüldüğü için yorum kabadır.
TP/TN/FP gruplarından resmi ID sırasıyla üçer görüntü seçilmiştir. FN grubunda yalnızca bir örnek bulunduğu için
bu grupta iki örnek eksik kalmış ve toplam 40 harita üretilmiştir. On haritada pozitif katkı sıfırdır.
Aynı inference batchleri tekrar kullanılarak kayıtlı skorlar korunur; model durumu değişmez.

Test ID 3 temizken doğru normal, gürültü eklendiğinde yanlış zatürre tahmin edilmiştir.
Test ID 571 temizken p=0.265 ile 0.330 validation thresholdunun altında kaldığı için FN olmuştur.
Aynı görüntü üç güçlü bozulmada da pozitif tahmin edilmiştir; bu örnek bozulmanın genel olarak faydalı olduğunu göstermez.
Vurgu tahminin doğruluğunu, sıfır harita da hastalık olmadığını kanıtlamaz.
Lezyon anotasyonu bulunmadığından haritalar lokalizasyon veya klinik geçerlilik kanıtı değildir.

### Dayanıklı training

Affine augmentation sonrasında temiz, gürültü, bulanıklık veya kontrast koşullarından biri 0.25 olasılıkla seçilir.
Aralıklar gürültü [0,0.12], bulanıklık [0.5,2], kontrast [0.4,1]'dir; aileler birleştirilmez.
Aynı üç seed ve maksimum training bütçesi kullanılır; early stopping çalıştırılan epoch sayısını değiştirebilir.
Checkpoint, threshold ve T temiz validation üzerinden seçilir. Baseline tahminleri yeniden inference yapılmadan kullanılır;
farklar aynı seed içinde hesaplanıp ortalama ve SD alınır.

Dayanıklı modelin ortalama AUROC değeri on koşulda da daha yüksektir.
Gürültü 0.12'de AUROC 0.9764 ± 0.0055, specificity 0.8362 ± 0.0215'dir.
Temiz AUROC 0.9858'den 0.9869'a yükselse de F1 0.9461'den 0.9394'e,
specificity 0.8162'den 0.7977'ye düşmüştür. Sensitivity de 0.9966'dan 0.9932'ye düşmüştür.
Bu yöntem baseline test sonuçları görüldükten sonra geliştirildiği için keşifseldir.
Sonraki Colab tekrarı bağımsız doğrulama sayılmaz.

<details>
<summary>Ayrıntılı ölçümler — Colab çalışması</summary>

Colab ortamı Tesla T4, Python 3.13.16, torch 2.14.1+cu130 ve torchvision 0.29.1+cu130'dur.
Süreler model başlangıcı, training/validation, checkpoint kaydı ve reload'u içerir; kurulum ve son analizi içermez.
Her test koşulunda aynı 624 görüntü vardır. AP, average precision'dır; ± değerler seedler arasındaki örneklem SD'dir.
Tablolar kaydedilmiş yeni sonuçlardan alınmıştır; bu rapor güncellemesinde yeni training, inference veya T fit yapılmamıştır.

| Training | En iyi epoch | Çalıştırılan epoch | Temiz validation AUROC | T | Başlangıç/reload dahil süre (saniye) |
|---|---|---|---|---|---|
| cnn_seed42 | 17 | 20 | 0.985775 | 0.880127 | 178.16 |
| resnet18_seed42 | 3 | 8 | 0.999505 | 1.169128 | 133.80 |
| resnet18_seed123 | 10 | 15 | 0.999886 | 0.817189 | 236.38 |
| resnet18_seed2026 | 7 | 12 | 0.999771 | 1.142291 | 191.61 |

| Model / seed | AUROC | AP | F1 | Sensitivity | Specificity | Balanced accuracy |
|---|---|---|---|---|---|---|
| cnn / 42 | 0.9540 | 0.9703 | 0.9167 | 0.9590 | 0.7778 | 0.8684 |
| resnet18 / 42 | 0.9858 | 0.9891 | 0.9523 | 0.9974 | 0.8376 | 0.9175 |
| resnet18 / 123 | 0.9882 | 0.9919 | 0.9511 | 0.9974 | 0.8333 | 0.9154 |
| resnet18 / 2026 | 0.9833 | 0.9889 | 0.9349 | 0.9949 | 0.7778 | 0.8863 |
| ResNet18 ortalama ± örneklem SD | 0.9858 ± 0.0025 | 0.9900 ± 0.0017 | 0.9461 ± 0.0097 | 0.9966 ± 0.0015 | 0.8162 ± 0.0334 | 0.9064 ± 0.0174 |

| Model / seed | Gerçek label | Normal tahmini | Zatürre tahmini |
|---|---|---|---|
| cnn / 42 | normal | 182 | 52 |
| cnn / 42 | pneumonia | 16 | 374 |
| resnet18 / 42 | normal | 196 | 38 |
| resnet18 / 42 | pneumonia | 1 | 389 |

| Seed | En iyi / çalıştırılan epoch | Temiz validation AUROC | Validation T | Temiz test AUROC | Test F1 | Test sensitivity | Test specificity |
|---|---|---|---|---|---|---|---|
| 42 | 3 / 8 | 0.999181 | 0.910668 | 0.9877 | 0.9417 | 0.9949 | 0.8034 |
| 123 | 10 / 15 | 0.999124 | 1.518820 | 0.9852 | 0.9281 | 0.9923 | 0.7564 |
| 2026 | 8 / 13 | 0.999295 | 1.402388 | 0.9878 | 0.9485 | 0.9923 | 0.8333 |

| Koşul | Baseline AUROC | Robust AUROC | Eşleştirilmiş AUROC farkı | Baseline specificity | Robust specificity |
|---|---|---|---|---|---|
| Temiz | 0.9858 ± 0.0025 | 0.9869 ± 0.0015 | +0.0011 ± 0.0038 | 0.8162 ± 0.0334 | 0.7977 ± 0.0388 |
| Gürültü 0.03 | 0.9477 ± 0.0319 | 0.9850 ± 0.0038 | +0.0372 ± 0.0350 | 0.3319 ± 0.2393 | 0.8134 ± 0.0510 |
| Gürültü 0.07 | 0.8604 ± 0.1397 | 0.9817 ± 0.0047 | +0.1213 ± 0.1426 | 0.0798 ± 0.1382 | 0.8405 ± 0.0215 |
| Gürültü 0.12 | 0.7274 ± 0.1852 | 0.9764 ± 0.0055 | +0.2490 ± 0.1894 | 0.0000 ± 0.0000 | 0.8362 ± 0.0215 |
| Bulanıklık 0.5 | 0.9857 ± 0.0017 | 0.9867 ± 0.0013 | +0.0010 ± 0.0030 | 0.7877 ± 0.0193 | 0.7991 ± 0.0450 |
| Bulanıklık 1 | 0.9822 ± 0.0028 | 0.9861 ± 0.0026 | +0.0038 ± 0.0016 | 0.6353 ± 0.1228 | 0.7991 ± 0.0621 |
| Bulanıklık 2 | 0.9454 ± 0.0164 | 0.9833 ± 0.0073 | +0.0378 ± 0.0191 | 0.1353 ± 0.1101 | 0.7251 ± 0.1188 |
| Kontrast 0.8 | 0.9860 ± 0.0008 | 0.9862 ± 0.0018 | +0.0002 ± 0.0022 | 0.7792 ± 0.0324 | 0.7806 ± 0.0373 |
| Kontrast 0.6 | 0.9838 ± 0.0015 | 0.9852 ± 0.0033 | +0.0014 ± 0.0019 | 0.6467 ± 0.0558 | 0.7692 ± 0.0334 |
| Kontrast 0.4 | 0.9708 ± 0.0086 | 0.9832 ± 0.0042 | +0.0123 ± 0.0049 | 0.3348 ± 0.0309 | 0.7365 ± 0.0193 |

| Koşul | Baseline raw NLL | Baseline scaled NLL | Robust raw NLL | Robust scaled NLL | Robust raw ECE | Robust scaled ECE |
|---|---|---|---|---|---|---|
| Temiz | 0.3296 | 0.3311 | 0.3415 | 0.2838 | 0.0753 | 0.0743 |
| Gürültü 0.03 | 1.4004 | 1.3083 | 0.2983 | 0.2537 | 0.0690 | 0.0687 |
| Gürültü 0.07 | 2.5109 | 2.3211 | 0.2540 | 0.2164 | 0.0579 | 0.0546 |
| Gürültü 0.12 | 2.5436 | 2.4074 | 0.2621 | 0.2281 | 0.0595 | 0.0521 |
| Bulanıklık 0.5 | 0.3771 | 0.3796 | 0.3501 | 0.2904 | 0.0732 | 0.0741 |
| Bulanıklık 1 | 0.6351 | 0.6313 | 0.3593 | 0.2987 | 0.0749 | 0.0744 |
| Bulanıklık 2 | 1.6964 | 1.6802 | 0.4541 | 0.3803 | 0.1053 | 0.1059 |
| Kontrast 0.8 | 0.3958 | 0.3975 | 0.3659 | 0.3047 | 0.0794 | 0.0797 |
| Kontrast 0.6 | 0.5806 | 0.5833 | 0.3766 | 0.3154 | 0.0839 | 0.0838 |
| Kontrast 0.4 | 1.2165 | 1.2304 | 0.4140 | 0.3542 | 0.0966 | 0.0963 |

</details>

<a id="sınırlılıklar"></a>

## 4. Sınırlılıklar

Veri, düşük çözünürlüklü ve önceden işlenmiş röntgen görüntülerinden oluşmaktadır.
Küçük validation split birden fazla seçimde kullanılmıştır. Üç seed training değişimini sınırlı şekilde gösterir;
tek sabit gürültü alanı farklı gürültü örnekleri arasındaki değişimi ölçmemektedir.
Bozulma seviyeleri classlar arasında eşdeğer değildir. Grad-CAM kaba sonuçlar verir, dayanıklılık çalışması keşifsel olarak yapılmıştır.
Bu sonuçlar klinik kullanım yeterliliği göstermemeketedir.

<a id="çıktılar"></a>

## 5. Dosyalar ve çalıştırma durumu

| Dosya / klasör | İçerik |
|---|---|
| `study.ipynb`, `requirements.txt` | Tek executable notebook ve eşleşen paket listesi |
| `config.yaml` | `cfg` üzerinden üretilmiş varsayılan ayar kopyası |
| `outputs/results.csv`, `outputs/robustness_v1/full/results.csv` | Yeni Colab baseline/dayanıklılık sonuçları: 1.560/1.300 satır |
| `outputs/figures/`, `outputs/robustness_v1/full/figures/` | Yeni Colab grafiklerinin 11/6 PNG ve PDF sayfası |
| `outputs/presentation*.pptx` | Yeni Colab verilerine göre güncellenmiş iki on slaytlık sunum |
| `outputs/{run_metadata,calibration_fits,study_provenance}.csv` | Training, validation fit ve kaynak kayıtları |
| `runs/results/{baseline,robustness_v1}/full/` | Yeni çalıştırmanın sonuçları, grafikleri, ayar ve tamamlanma kayıtları |
| `runs/cache/{baseline,robustness_v1}/full/` | Görüntü ID'leri, labellar, logitler, CAM ve kimlik kayıtları |
| `runs/checkpoints/{baseline,robustness_v1}/full/` | Best ağırlıklar |

Checkpoint ZIP'leri test öncesindeki ağırlık ve validation kayıtlarını; `full_results.zip` sonuç, grafik ve cacheleri saklar.
Kayıtlı logitler, T ve threshold ile yeni inference yapmadan olasılık ve kararlar elde edilebilir.
Veri, ağırlıklar ve cacheler Git/teslime dahil edilmez. Güncel CSV'lerdeki `runs/cache/` yolları indirilen Colab sonuç arşivine aittir;
notebook başlangıcı için gerekli değildir.

<a id="kaynaklar"></a>

## Kaynaklar ve kullanılan araçlar

- [MedMNIST](https://medmnist.com/), [resmi kod/metadata](https://github.com/MedMNIST/MedMNIST) ve [başlangıç notebook'u](https://colab.research.google.com/github/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb).
- [Yang ve arkadaşları, MedMNIST v2, Scientific Data (2023)](https://doi.org/10.1038/s41597-022-01721-8).
- [Güvenli checkpoint yükleme](https://docs.pytorch.org/docs/stable/generated/torch.load.html), [AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html) ve [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) ve [ResNet18 ağırlıkları/normalizasyonu](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- [Gaussian blur](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html), [Gaussian noise](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.GaussianNoise.html) ve [kontrast](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html).
- [SciPy sınırlı skaler optimizasyon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html);
- [Guo ve arkadaşları, On Calibration of Modern Neural Networks (2017)](https://proceedings.mlr.press/v70/guo17a.html).
- [Captum LayerGradCam](https://captum.ai/api/layer.html#gradcam), [ROC thresholdları](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html) ve [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
