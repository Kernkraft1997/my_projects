import speech_recognition as sr
import pyttsx3

# Голосовой движок
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # скорость речи


def speak(text):
    engine.say(text)
    engine.runAndWait()


# Слушаем микрофон
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("🎤 Скажи что-нибудь...")
    recognizer.adjust_for_ambient_noise(source, duration=1)
    audio = recognizer.listen(source)

print("🧠 Распознаю...")
try:
    text = recognizer.recognize_google(audio, language="ru-RU")
    print(f"✅ Ты сказал: {text}")

    # Отвечаем
    speak(f"Ты сказал: {text}")

except sr.UnknownValueError:
    print("❌ Я тебя не расслышал бро")
    speak("Я тебя не расслышал бро")
except sr.RequestError:
    print("❌ Ошибка соединения")
