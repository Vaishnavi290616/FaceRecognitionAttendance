from datetime import datetime

now = datetime.now()
date = now.strftime("%d-%m-%y")
time = now.strftime("%H:%M:%S")
print("Date:",date)
print("Time:",time)
