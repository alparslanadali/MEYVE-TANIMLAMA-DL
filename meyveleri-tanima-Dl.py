from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import pickle
# ****************** UYARI ********************************
#Kodun çalışması için kod dosyalarının içinde sadece Training,Test olması gerekmektedir yoksa data için resimleri okurken hata ile karşılaşırsınız.
# ****************** UYARI ********************************



# veri dosyalarının lokasyonları
train_path = "fruits-360/Training/"
test_path = "fruits-360/Test/"

images = []


for dirname,_,filename in os.walk("fruits-360"): #dirname kök dizini _ alt dizinleri filename dosyaları alır
    for filename in filename:

        a=os.path.join(dirname,filename) #resimelerin net alanını veriyor
        img = cv2.imread(a) #resim okunuyor
        img = cv2.resize(img, (100, 100))   #resim boyutlandırılıyor.
        images.append(img) #okunan resim veri olarak listeye ekleniyor.


images = np.array(images) #listedeki veriler matrixlere dönüştürülüyor.



className = os.listdir(train_path)  #kaç adet farklı meyve olduğunu bulmak için sınıf sayısını belirliyor.
numberOfClass = len(className)
print("NumberOfClass: ",numberOfClass)

# Sinir ağı oluşturuluyor.
model = Sequential()
model.add(Conv2D(32,(3, 3),input_shape = (100,100,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(numberOfClass)) # output
model.add(Activation("softmax"))


model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])


batch_size = 32 # her eğitimde kullanılacak resim sayısını belirtiyor.

# eğitim için var olan resimler uzaklaştırıp yakınlaştırıyor ve varolan resimler çoğaltılıyor.
train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range = 0.3)


test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path, #hedef klasör belirtiliyor
        target_size = (100,100),  # her bir kullanılacak resim sayısı
        color_mode= "rgb", # resim türü
        class_mode= "categorical") #sınıflandırma metodu belirleniyor

test_generator = test_datagen.flow_from_directory(
        test_path, #hedef klasör belirtiliyor.
        target_size=(100,100), # hedef klasördeki resimlerin boyutu
        batch_size = batch_size,   # her testte kullanılacak resim sayısı
        color_mode= "rgb", # resim türü
        class_mode= "categorical")  #sınıflandırma metodu belirleniyor

hist = model.fit_generator(
        generator = train_generator,    #hangi veri kullanılacağını belirtiyor
        steps_per_epoch = 1600 // batch_size, # her bir epochda'ki eğitim sayısı
        epochs=15,  #iterasyon sayısı
        validation_data = test_generator, #Eğitilmiş veriyi doğrulamak için kullanılacak veri
        validation_steps = 800 // batch_size, #eğitilmiş veri için her bir epochda'ki doğrulama sayısı
        shuffle=1)  # ezber olmaması için verileri karıştırıyor.
#Oluşan veriler kayıt edilcek.
pickle_out = open("egitilmis-siniragı-yeni.p", "wb") #depolancak veri ismi
pickle.dump(model, pickle_out) #depolayacak.
pickle_out.close()

#verileri grafikleştiriyorum.
#print(hist.history.keys())
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["accuracy"], label = "Train acc")
plt.plot(hist.history["val_accuracy"], label = "Validation acc")
plt.legend()
plt.show()
