BLM2642 - Polinom Regresyon ve Optimizasyon Ödev Raporu

Ad - Soyad: Sema Almasri
Öğrenci No: 26011603
YouTube Sunum Videosu Linki: https://youtu.be/WbcUp6JPd54
Proje Github Linki: https://github.com/sema2817/dif_odevi/tree/main

1. Giriş ve Veri Setinin İncelenmesi
Bu çalışmanın temel amacı, bilinmeyen ve gürültülü bir fonksiyondan üretilmiş olan veri setleri üzerinde polinom regresyon modelleri kurmak, analitik ve sayısal optimizasyon (Gradyan Düşümü) yöntemlerini uygulamak, hata analizleri ve dönüşümlerle model performansını optimize etmektir.
Çalışmada kullanılan veri setleri:
egitim.csv: Modelin öğrenmesi için kullanılan 50 adet (x,y) gözlem çiftini içerir. x değerleri iki ayrı aralıktan ([-3;0.5] ve [1.5;3]) seçilmiştir.
test.csv: Modellerin (test hatasını tarafsız bir şekilde ölçmek için ayrılmış 20 adet (x,y) gözlem çiftini içerir.
tahmin_x.csv: [-4;4] aralığından seçilmiş, yalnızca x değerlerini içeren ve nihai tahminlerin üretileceği 30 adet gözlem noktasıdır.
Veri setindeki tüm y değerleri pozitiftir. Eğitim ve test verilerinin aynı grafikte görselleştirilmesi (soru1_veri.png), verinin non-linear ve karmaşık bir yapıya sahip olduğunu ve belirli aralıklarda yoğunlaştığını göstermiştir.
2. Polinom Regresyon ve Analitik Çözüm
1 ile 8 arasındaki tüm polinom dereceleri (d ∈ {1, ..., 8})   için veriye en uygun eğriyi uydurmak amacıyla Tasarım Matrisi (X) şu formda oluşturulmuştur:

Katsayı vektörü , hiçbir hazır makine öğrenmesi kütüphanesine bağımlı kalmadan kapalı form (analitik normal denklemler) çözümü ile sıfırdan hesaplanmıştır:
=(XTX)-1XTy
Her bir derece için elde edilen eğitim ve test ortalama karesel hata (MSE) değerleri Tablo 1'de listelenmiştir:
Polinom Derecesi (d)
Eğitim MSE (Normal)
Test MSE (Normal)
Eğitim MSE (ln(y) Dönüşümlü)
Test MSE (ln(y) Dönüşümlü)
1
0.21680
0.14639
1.78178
1.54245
2
0.21656
0.14045
0.02591
0.05128
3
0.04475
0.04980
0.02712
0.05990
4
0.02900
0.05129
0.02385
0.04689
5
0.02502
0.04574
0.02351
0.04598
6
0.02295
0.04668
0.02364
0.04734
7
0.02285
0.04779
0.02426
0.04529
8
0.02186
0.05387
0.02145
0.05179


MSE Değişim Grafiği (soru2_mse.png): Derece arttıkça modelin esnekliği artmış ve eğitim hatası sürekli düşmüştür. Ancak test hatası belirli bir noktadan sonra (normal modelde 5. dereceden sonra) aşırı öğrenme (overfitting) eğilimi göstererek tekrar yükselmiştir.
3. Model Seçimi ve Gerekçesi
Makine öğrenmesinde temel kural, modelin eğitim verisini ezberlemesi değil, görmediği test verisi üzerinde yüksek genelleştirme yeteneği sergilemesidir. Bu nedenle seçim yapılırken eğitim hatası yerine test MSE değerleri baz alınmıştır: Normal model için seçim, en düşük test MSE değerini (MSE=0.04574) veren 5. derece optimum normal model olarak belirlenmiştir. (y) dönüşümlü yaklaşımda ise en düşük test MSE değerini (MSE=0.04529) veren 7. derece optimum logaritmik model seçilmiştir. İki yaklaşımın test MSE değerleri birbirine oldukça yakındır. Logaritmik modelin test MSE değeri normal modelinkinden yaklaşık %1 daha düşüktür. Bu nedenle logaritmik model tercih edilmiş olsa da iki yaklaşımın genelleştirme performansları arasında büyük bir fark bulunmadığı değerlendirilmiştir. 
4. Kalıntı (Residuals) Analizi

Modelin kalıntıları, gerçek değerler ile tahmin edilen değerler arasındaki fark olarak (ei=yi-yi)  hesaplanmıştır ve x değerlerine karşılık soru4_kalinti.png grafiğinde çizdirilmiştir. Seçilen 5. derece modelin bir alt derecesi olan 4. derece ve bir üst derecesi olan 6. derece modellerin kalıntılarıyla karşılaştırma yapılmıştır. İdeal bir regresyon modelinde kalıntıların sıfır etrafında rastgele dağılması beklenir. Seçilen optimum derecelerde kalıntıların sıfır çizgisi etrafında çok daha homojen ve rastgele dağıldığı gözlemlenmiştir. Bu karşılaştırma sayesinde polinom derecesinin modelin hata davranışına etkisi incelenmiştir. 
5. Logaritmik Dönüşüm Modeli ((y)) ve Etki Analizi

Verideki y değerlerinin pozitif olması ve bazı bölgelerde gözlenen hızlı artış eğilimi nedeniyle ln(y) dönüşümü uygulanmıştır. Bu dönüşüm, bu veri setinde test MSE değerinde küçük bir iyileşme sağlamıştır. Ancak iki model arasındaki fark yaklaşık %1 olduğundan, dönüşümün belirgin bir performans üstünlüğü sağladığı söylenemez. 
z=(y)
Model z uzayında analitik olarak eğitildikten sonra, tahminler ters dönüşümle orijinal ölçeğe aktarılmıştır:
y=ez
Tüm MSE hesaplamaları adil bir karşılaştırma için doğrudan y'nin kendi orijinal ölçeğinde gerçekleştirilmiştir.
(y) dönüşümü, pozitif değerlerdeki büyük ölçekli dalgalanmaları baskılayarak modelin veriyi biraz daha kararlı öğrenmesini sağlamıştır. Özellikle 7. derece (y) modeli, normal modellere kıyasla test verisi üzerinde az miktarda daha düşük hata üretmiş ve genel performansı artırmıştır.
6. İnterpolasyon ve Ekstrapolasyon Davranışları
Normal model ile (y) dönüşümlü modelin [-4;4] aralığındaki genel tahmin eğrileri eğitim verisiyle birlikte soru6_karsilastirma.png grafiğinde karşılaştırılmıştır:
İnterpolasyon Bölgesi [0.5,1.5]: Bu aralıkta herhangi bir eğitim verisi bulunmamaktadır. Bu nedenle modeller bu bölgede gerçek anlamda interpolasyon değil, iki veri kümesi arasındaki boşluğu doldurmaya çalışmaktadır.
Ekstrapolasyon Bölgeleri [-4,-3] ve [3,4]: Eğitim verisinin bulunmadığı bu uç bölgelerde, normal yüksek dereceli polinom modellerin hızla sönümlendiği veya mantıksız salınımlar yaptığı görülmüştür. Buna karşılık, bu veri setinde ln(y) dönüşümlü modelin uç bölgelerde daha yumuşak ve daha kararlı tahminler ürettiği gözlenmiştir. Yine de logaritmik dönüşüm ekstrapolasyon bölgelerinde her zaman daha doğru tahmin sağlayacağı anlamına gelmez. Bu bölgelerde gerçek gözlemler bulunmadığından tahminlerin güvenilirliği yalnızca grafik üzerinden kesin olarak belirlenemez. 
7. Gradyan Düşümü (Gradient Descent) ile Optimizasyon
Analitik çözümle beraber, algoritmanın iteratif öğrenme becerisini test etmek amacıyla gradyan düşümü uygulanmıştır.
Özellik Ölçekleme (Feature Scaling): 
Yüksek dereceli polinom tasarım matrisindeki sütunların farklı büyüklüklerde değerlere sahip olması, gradyan düşümü algoritmasının yakınsamasını zorlaştırabilir. Bu nedenle sabit terim dışındaki polinom sütunlarına Z-score ölçeklendirmesi uygulanmıştır. Her sütunun ortalaması  (Xmean) ve standart sapması  (Xstd) hesaplanmış, standart sapması sıfır olan sütunlarda sıfıra bölünmeyi önlemek amacıyla standart sapma değeri 1 (Xstd=1) olarak ayarlanmıştır. Ardından sütunlar ortalama çıkarılıp standart sapmaya bölünerek ölçeklendirilmiştir. 
Kayıp Fonksiyonu ve Eğrisi (soru7_kayip.png):


J()=12ni=1n(yi-yi)2
formülasyonu kullanılmış, 3000 iterasyon boyunca maliyetin (Loss - MSE/2) sıfıra doğru pürüzsüz ve kararlı bir şekilde azaldığı gözlemlenmiştir.
Katsayı Kıyaslaması: Analitik yöntem ile gradyan düşümü yöntemi kullanılarak 5. derece normal modelin katsayıları hesaplanmıştır. Analitik çözümde ölçeklendirilmemiş tasarım matrisi kullanılırken, gradyan düşümü yönteminde yakınsamayı kolaylaştırmak amacıyla polinom sütunlarına Z-score ölçeklendirmesi uygulanmıştır. Bu nedenle iki yöntemden elde edilen katsayıların sayısal değerleri doğrudan karşılaştırılamaz. Gradyan düşümü sonucunda elde edilen ölçeklendirilmiş katsayılar ve analitik katsayılar ayrı ayrı incelenmiştir. Ayrıca kayıp fonksiyonunun iterasyonlar boyunca değişimi gözlemlenerek optimizasyon süreci değerlendirilmiştir.
8. Nihai Tahminler ve Teslim Dosyası
Yapılan tüm test MSE karşılaştırmaları, kalıntı analizleri ve ekstrapolasyon testleri sonucunda, genelleştirme başarısı en yüksek olan 7. derece (y) dönüşümlü model son model olarak seçilmiştir. tahmin_x.csv dosyasındaki 30 adet yeni x gözlem noktası için bu modelle üretilen orijinal ölçekteki y tahminleri, yönergeye uygun olarak iki sütunlu (id, y) şekilde 26011603_tahmin.csv dosyasına hatasız bir şekilde kaydedilmiştir.

Sonuç
Bu çalışmada polinom regresyon teknikleri kullanılarak gürültülü ve doğrusal olmayan bir veri seti modellenmiştir. Farklı polinom dereceleri için analitik çözümler oluşturulmuş, eğitim ve test MSE değerleri karşılaştırılmıştır. Ayrıca hedef değişkene logaritmik dönüşüm uygulanarak model performansı incelenmiştir. Test sonuçlarına göre normal modelde 5. derece, logaritmik modelde ise 7. derece en düşük test hatasını vermiştir. İki model arasındaki hata farkı küçük olmakla birlikte, en düşük test MSE değerine sahip 7. derece logaritmik model nihai tahminlerde kullanılmıştır. Gradyan düşümü yöntemi ise normal modelin optimizasyonunda uygulanarak kayıp fonksiyonunun iterasyonlar boyunca değişimi incelenmiştir. 
# dif_odevi
