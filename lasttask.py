import os
import time

import cv2

# Индекс камеры (обычно 0 для встроенной камеры)
camera_index = 0

# Открытие камеры
cap = cv2.VideoCapture(0)
frame1key = 0
frame2key = 0
while True:
    # Чтение кадра из камеры
    ret, frame = cap.read()

    # Если кадр не был прочитан, выход из цикла
    if not ret:
        break

    # Отображение кадра на экране
    cv2.imshow('frame', frame)

    # Обработка нажатий клавиш
    key = cv2.waitKey(1)


    if key == ord('s'):
        frame1key = 1
        # filepath = os.path.join('frame1', filename)
        # cv2.imwrite(filepath, frame)

    if key == ord('x'):
        frame2key = 1

    if frame1key:
        filename = f'frame_{time.time()}.jpg'
        filepath = os.path.join('frame1', filename)
        cv2.imwrite(filepath, frame)

    if frame2key:
        filename = f'frame_{time.time()}.jpg'
        filepath = os.path.join('frame2', filename)
        cv2.imwrite(filepath, frame)
        # cv2.imwrite( f'frame_{time.time()}.jpg', frame)

    if key == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
