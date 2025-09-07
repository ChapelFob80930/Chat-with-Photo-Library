import speech_recognition as sr
import subprocess
import webbrowser
import os

def listen_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        command = r.recognize_google(audio).lower()
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def execute_command(command):
    if "open terminal" in command:
        subprocess.Popen("wt" if os.name == "nt" else "gnome-terminal")
    elif "open vs code" in command:
        subprocess.Popen(["code"])
    elif "new project" in command:
        os.makedirs("MyProject", exist_ok=True)
        subprocess.Popen(["code", "MyProject"])
    elif "news" in command:
        webbrowser.open("https://news.google.com")
    else:
        print("Command not recognized.")

while True:
    cmd = listen_command()
    execute_command(cmd)
