import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import os

class AdvancedLipReader:
    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.phoneme_classes = ['а', 'и', 'о', 'у', 'э', 'м', 'п', 'б', 'neutral']
        self.model = None
        self.is_trained = False
        
        # Инициализация MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

        # точки губ из MediaPipe Face Mesh
        self.lip_indices = [
            61, 84, 314, 17,     # Внешние углы
            78, 308, 13, 14,     # Основные точки
            80, 81, 82, 87, 88, 89, 95, 96, 97,     # Верхняя губа
            317, 318, 319, 324, 325, 326, 375, 376, 377  # Нижняя губа
        ]
        
        self.sequence_buffer = deque(maxlen=sequence_length)
        
    def build_model(self):
        """Создание улучшенной модели для распознавания речи по губам"""
        model = models.Sequential([
            # простая архитектура для демо
            layers.LSTM(128, return_sequences=True, 
                       input_shape=(self.sequence_length, len(self.lip_indices) * 2),
                       dropout=0.2),
            
            layers.LSTM(64, return_sequences=False, dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Выходной слой
            layers.Dense(len(self.phoneme_classes), activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def extract_lip_landmarks(self, image):
        """Извлечение landmarks губ из изображения"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = []
                
                for idx in self.lip_indices:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        landmarks.extend([landmark.x, landmark.y])
                    else:
                        landmarks.extend([0.0, 0.0])
                
                return np.array(landmarks)
            return None
        except Exception as e:
            print(f"Ошибка при извлечении landmarks: {e}")
            return None
    
    def add_frame_to_sequence(self, image):
        """Добавление кадра в последовательность"""
        landmarks = self.extract_lip_landmarks(image)
        if landmarks is not None:
            if len(self.sequence_buffer) == 0:
                for _ in range(self.sequence_length):
                    self.sequence_buffer.append(landmarks)
            else:
                self.sequence_buffer.append(landmarks)
            return True
        else:
            # Если landmarks не найдены, добавляем нулевой вектор
            zero_landmarks = np.zeros(len(self.lip_indices) * 2)
            if len(self.sequence_buffer) == 0:
                for _ in range(self.sequence_length):
                    self.sequence_buffer.append(zero_landmarks)
            else:
                self.sequence_buffer.append(zero_landmarks)
            return False
    
    def get_current_sequence(self):
        """Получение текущей последовательности для предсказания"""
        if len(self.sequence_buffer) == self.sequence_length:
            return np.array([list(self.sequence_buffer)])
        return None
    
    def predict_phoneme(self, image):
        """Предсказание фонемы для текущего кадра"""
        if not self.is_trained or self.model is None:
            return "Модель не обучена", 0.0
        
        # Добавляем кадр в последовательность
        self.add_frame_to_sequence(image)
        
        # Получаем последовательность для предсказания
        sequence = self.get_current_sequence()
        if sequence is not None:
            try:
                predictions = self.model.predict(sequence, verbose=0)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                if confidence > 0.3:  # Пониженный порог для демо
                    return self.phoneme_classes[predicted_class], confidence
                else:
                    return "uncertain", confidence
            except Exception as e:
                return f"Ошибка: {e}", 0.0
        
        return "collecting_data", 0.0
    
    def create_synthetic_data(self, num_samples=500):
        X_train = []
        y_train = []
        
        for i in range(num_samples):
            sequence = []
            base_point = np.random.random(len(self.lip_indices) * 2) * 0.5 + 0.25
            
            for j in range(self.sequence_length):
                # Добавляем небольшой шум к базовой точке
                noise = np.random.normal(0, 0.05, len(self.lip_indices) * 2)
                frame_data = base_point + noise * (j / self.sequence_length)
                sequence.append(frame_data)
            
            X_train.append(sequence)
            y_train.append(np.random.randint(0, len(self.phoneme_classes)))
        
        X_train = np.array(X_train)
        y_train = tf.keras.utils.to_categorical(y_train, len(self.phoneme_classes))
        
        return X_train, y_train
    
    def train(self, X_train=None, y_train=None, epochs=20):
        """Обучение модели"""
        if self.model is None:
            self.build_model()
        
        # Если данные не предоставлены, создаем синтетические
        if X_train is None or y_train is None:
            print("Создание синтетических данных для демонстрации...")
            X_train, y_train = self.create_synthetic_data(300)  # Меньше данных для быстрого обучения
        
        print(f"Форма данных: {X_train.shape}")
        
        # Простое обучение без валидации для демо
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=16,
            verbose=1,
            validation_split=0.2
        )
        
        self.is_trained = True
        print("Модель обучена!")
        return history
    
    def save_model(self, filepath):
        """Сохранение модели"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath):
        """Загрузка модели"""
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            self.is_trained = True
            print(f"Модель загружена из {filepath}")
        else:
            print(f"Файл {filepath} не найден")

# Упрощенное приложение
class LipReadingApp:
    def __init__(self):
        self.lip_reader = AdvancedLipReader(sequence_length=10)  # Укороченная последовательность
        self.current_text = ""
        self.prediction_history = deque(maxlen=5)  # Укороченная история
        
    def run(self):
        """Запуск приложения с веб-камерой"""
        print("Инициализация камеры...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Ошибка: не удалось открыть камеру!")
            return
        
        print("Обучение модели на синтетических данных...")
        self.lip_reader.train(epochs=15)  # Быстрое обучение для демо
        
        print("Запуск распознавания...")
        
        while True:
            success, image = cap.read()
            if not success:
                print("Ошибка чтения кадра")
                break
            
            # зеркальное отображение
            image = cv2.flip(image, 1)
            
            # Предсказание фонемы
            phoneme, confidence = self.lip_reader.predict_phoneme(image)
            
            # Обновление истории и текста
            if phoneme not in ["collecting_data", "uncertain", "Модель не обучена"] and not phoneme.startswith("Ошибка"):
                self.prediction_history.append(phoneme)
                
                # Если несколько одинаковых предсказаний подряд, добавляем в текст
                if len(self.prediction_history) >= 3:
                    last_three = list(self.prediction_history)[-3:]
                    if all(p == last_three[0] for p in last_three):
                        if not self.current_text or self.current_text[-1] != last_three[0]:
                            self.current_text += last_three[0]
                            print(f"Добавлена фонема: {last_three[0]}")
            
            # Отображение информации
            cv2.putText(image, f"Phoneme: {phoneme}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Text: {self.current_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(image, "Press 'c' to clear, 'q' to quit", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Lip Reading Demo', image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
                self.prediction_history.clear()
                print("Текст очищен")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Приложение завершено")

# Альтернативный простой запуск
def simple_demo():
    """Упрощенная демонстрация"""
    print("=== Lip Reading Demo ===")
    print("Убедитесь, что веб-камера подключена и есть хорошее освещение")
    print("Нажмите 'q' для выхода, 'c' для очистки текста")
    print("=" * 30)
    
    app = LipReadingApp()
    app.run()

if __name__ == "__main__":
    simple_demo()
