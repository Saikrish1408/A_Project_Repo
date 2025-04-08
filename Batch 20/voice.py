from googletrans import Translator
from gtts import gTTS
import pygame
from datetime import datetime

def english_to_tamil_voice(english_text, language):
    # Translate the text to Tamil
    translator = Translator()
    translated = translator.translate(english_text, src='en', dest=language)
    tamil_text = translated.text
    
    # Print the Tamil text
    print("Tamil Text:", tamil_text)
    
    # Convert the Tamil text to speech
    tts = gTTS(text=tamil_text, lang=language, slow=False)
    # Save the audio file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_file = f"audio_{timestamp}.mp3"  # Change the extension as needed
    tts.save(audio_file)
    # Initialize pygame mixer
    pygame.mixer.init()

    # Load and play the sound
    pygame.mixer.music.load(audio_file)  # Use the full path if needed
    pygame.mixer.music.play()

    # Keep the program running while the sound plays
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    return 'Completed'
