import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 데이터 가져오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 정규화
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 이미지 증강 설정(각도, 기울기, 범위, 뒤집기, 확대축소 범위) 데이터 변형하여 다양한 데이터로 학습시키기 위해 
gen = ImageDataGenerator(
    rotation_range=20, 
    shear_range=0.2, 
    width_shift_range=0.2, 
    height_shift_range=0.2, 
    horizontal_flip=True,
    zoom_range=0.2
)

# 데이터 증강
augment_ratio = 1.5
augment_size = int(augment_ratio * x_train.shape[0])

# 랜덤배치
randidx = np.random.randint(x_train.shape[0], size=augment_size)
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

# 이미지 증강 및 데이터 가져오기
iterator = gen.flow(x_augmented, y_augmented, batch_size=augment_size, shuffle=False)
x_augmented, y_augmented = next(iterator)

# 데이터 결합
x_train = np.concatenate((x_train, x_augmented))
y_train = np.concatenate((y_train, y_augmented))

# 데이터 섞기
s = np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]

# 모델가져오기
cnn = Sequential()

# 학습 padding 데이터 축소 방지 kernel_regularizer:L2 가중치의 크기를 제한하고 과적합을 방지
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001), input_shape=(32, 32, 3)))
# 배치 정규화로 학습을 안정화 하고 학습속도를 높임
cnn.add(BatchNormalization())
cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn.add(BatchNormalization())
# 중요한 특징을 추출.
cnn.add(MaxPooling2D(pool_size=(2, 2)))
# 무작위로 25%의 뉴런을 비활성화.
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn.add(BatchNormalization())
cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001)))
cnn.add(BatchNormalization())
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Dropout(0.25))

# 1차원 배열로 변환
cnn.add(Flatten())
cnn.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

# 모델 학습준비
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# 모델의 개선이 없을 경우, 학습율을 조절해 모델의 개선을 유도하는 콜백함수
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# 학습
hist = cnn.fit(x_train, y_train, batch_size=256, epochs=250, validation_data=(x_test, y_test), callbacks=[reduce_lr])

# 그래프화
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.grid()
plt.show()
