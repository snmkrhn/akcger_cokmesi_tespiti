Pneumothorax Detection from Chest X-Ray Images using Deep Learning
🔍 Proje Tanımı
Bu proje, pnömotoraks (akciğer çökmesi) vakalarının X-Ray (göğüs röntgeni) görüntülerinden otomatik olarak tespit edilmesini amaçlayan bir derin öğrenme tabanlı sınıflandırma sistemidir. Projede, çeşitli akciğer rahatsızlıklarını içeren büyük ölçekli bir veri kümesi üzerinde CNN, ResNet, RNN ve Vision Transformer (ViT) gibi farklı model mimarileri karşılaştırmalı olarak değerlendirilmiştir.

🧪 Kullanılan Veri Kümesi
Toplam 26.355 adet X-Ray görüntüsü içeren veri kümesi 4 sınıfa ayrılmıştır:

Akciğer Çökmesi (Pnömotoraks): 12.047 görüntü

Verem: 3.369 görüntü

Zatürre: 4.273 görüntü

Sağlıklı: 6.666 görüntü

Veri; eğitim (%70), doğrulama (%20) ve test (%10) olmak üzere üç bölüme ayrılmıştır.

⚙️ Kullanılan Yöntemler ve Modeller
Ön İşleme:
Görüntüler gri tonlamaya çevrildi, yeniden boyutlandırıldı (224x224), normalize edildi ve histogram eşitleme uygulandı.

Modeller:

CNN (Convolutional Neural Network)

ResNet (Residual Neural Network)

RNN (Recurrent Neural Network)

Vision Transformer (ViT)

Eğitim Ayrıntıları:

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Epoch: 10

Batch Size: 32

Augmentasyon: Döndürme, yakınlaştırma, çevirme

