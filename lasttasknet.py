import os

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

negative_dir = 'frame1'
positive_dir = 'frame2'
img_size = (128, 128)



positive_images = []
for filename in os.listdir(positive_dir):
    img = cv2.imread(os.path.join(positive_dir, filename))
    img = cv2.resize(img, img_size)
    positive_images.append(img)

# Загрузка кадров с лицом, не смотрящим в камеру
negative_images = []
for filename in os.listdir(negative_dir):
    img = cv2.imread(os.path.join(negative_dir, filename))
    img = cv2.resize(img, img_size)
    negative_images.append(img)

# Создание датасета и меток классов
X = np.array


# Создание датасета и меток классов
X = np.array(positive_images + negative_images)
y = np.array([1] * len(positive_images) + [0] * len(negative_images))

# Нормализация значений пикселей изображений
X = X.astype('float32') / 255.0

# Создание модели сверточной нейронной сети
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, epochs=5, batch_size=32)

# Запуск видеопотока
cap = cv2.VideoCapture(0)

# Цикл обработки кадров видеопотока
while True:
    # Получение кадра из видеопотока
    ret, frame = cap.read()

    # Обработка кадра
    img = cv2.resize(frame, img_size)
    img = img.reshape((1,) + img.shape)
    img = img.astype('float32') / 255.0

    # Классификация кадра
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        label = f"in: {prediction*100:.2f}%"

    else:
        label = f"out: {prediction*100:.2f}%"

    # Отображение результата классификации на кадре
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Video', frame)

    # Прерывание цикла обработки кадров при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Остановка видеопотока и закрытие окна с кадрами
cap.release()
cv2.destroyAllWindows()
