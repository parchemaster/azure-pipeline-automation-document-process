import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
from configparser import ConfigParser
from datetime import datetime

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        print(event.src_path)
        if event.is_directory:
            return None
        elif event.src_path.endswith('.pdf'):
            subprocess.run(["python", "app.py"])

config_object = ConfigParser()
config_object.read("config.ini")


observer = Observer()
event_handler = NewFileHandler()
observer.schedule(event_handler, path=config_object["DIRECTION"]["InPath"], recursive=True)
observer.start()

def run_program():
    currentDateAndTime = datetime.now()
    day_of_week = datetime.today().weekday()
    subprocess.run(["python", "app.py"])
    try:
        while True:
            current_time = datetime.now().time()
            if ((current_time.hour >= int(config_object["OTHER"]["FinishTime"].split(":")[0])) and int(current_time.minute >= config_object["OTHER"]["FinishTime"].split(":")[1])):
                time.sleep(3600 * 18)
            if (is_holidays):
                time.sleep(3600 * 24)
            if(day_of_week <= 5):
                time.sleep(3600 * 24)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

def is_holidays():
    now = datetime.now()
    current_date = f"{now.day} {now.strftime('%B')}"
    formated_current_date = str(current_date).replace(" ", "")
    with open("holidays.txt", "r") as file:
        contents = file.read()
        rows = contents.split("\n")
        for row in rows:
            formated_holiday = "".join(row.split("\t")[0].split()).strip()
            if (formated_holiday == formated_current_date):
                return 

run_program()