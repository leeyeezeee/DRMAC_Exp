import os

folder = r"e:\my_paper\ExpoComm\src\config\exp"
for fname in os.listdir(folder):
    if fname.startswith("ExpoComm") and fname.endswith(".yaml"):
        with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
            content = f.read()
        new_fname = fname.replace("ExpoComm", "DRMAC")
        new_content = content.replace("ExpoComm", "DRMAC")
        with open(os.path.join(folder, new_fname), "w", encoding="utf-8") as f:
            f.write(new_content)