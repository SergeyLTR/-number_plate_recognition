import matplotlib.pyplot as plt
import pytesseract
import cv2

def open_img(img_path):
    img = cv2.imread(img_path) #считать изображение
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # преобразование цветой модели в RGB
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    return img

# detectMultiScale - метод позволяет обнаруживать объекты разных размеров 
# на входном изображении и возвращает список из границ прямоугольника в которых обнаружены объекты. 
# Для каждого прямоугольника возвращается 4 значения: координаты, ширина и высота прямоугольника
# scaleFactor указывает насколько уменьшается размер изображения в каждом масштабе изображения
# minNeighbors - минимальное количество соседей, влияет на качество обнаруживаемых объектов. Более высокое значение приводит
# К меньшему количеству обнаружений, но обнаружения имеют более высокое качество и точность
def carplate_detection(img, classifier):
    carplate_rects = classifier.detectMultiScale(img, scaleFactor=1.1, minNeighbors=7)
    print(carplate_rects)
    x, y, w, h = carplate_rects[0]
    
    coord = (x, y, w, h)
    carplate_img = img[y:y+h, x:x+w]
    
    return carplate_img, coord

# Функция увеличивает изображение, чтобы улучшить распознавание. Так как получается изначально небольшим. 
# Функция принимает список координат изображения и параметр увелечения изображения 
def enlarge_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    new_size_image = (width, height)
    # plt.axis('off')
    resized_image = cv2.resize(image, new_size_image, interpolation=cv2.INTER_AREA)
    
    return resized_image

def main():
    img_rgb = open_img(img_path='src/image2.jpg') # изображение автомобилей
    
    classifier = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    carplate_img, coords = carplate_detection(img_rgb, classifier=classifier)
    x, y, w, h = coords
    # print(carplate_img)
    carplate_img = enlarge_image(carplate_img, 1000)
    plt.imshow(carplate_img)
    plt.show()

    carplate_img_gray = cv2.cvtColor(carplate_img, cv2.COLOR_RGB2GRAY) # Перевести изображение в оттенки серого
    plt.imshow(carplate_img_gray, cmap='gray')
 

    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' # для работы программы необходимо установить Tesseract и тут указать путь до tesseract.exe; 
    car_number = pytesseract.image_to_string(
        carplate_img_gray,
        config='--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    print(' Номер автомобиля ', car_number)


    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(img_rgb, (x, y - 40), (x + w + 1, y), (0, 255, 0), -1)
    cv2.putText(img_rgb, car_number, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
    plt.imshow(img_rgb)
    plt.show()

if __name__ == '__main__':
    main()