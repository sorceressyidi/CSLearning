name = input("File name: ").strip().lower().split(".")[-1]
extentions = {
    "gif": "image/gif",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "pdf": "application/pdf",
    "txt": "text/plain",
    "zip": "application/zip",
}
print(extentions.get(name, "application/octet-stream"))