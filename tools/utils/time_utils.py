from datetime import datetime

def current_time():
  now = str(datetime.now()).split(" ")
  now[0] = now[0].replace("-", "")
  now[1] = now[1].split(".")[0].replace(":","")
  now = now[0]+"_"+now[1]
  return now
  