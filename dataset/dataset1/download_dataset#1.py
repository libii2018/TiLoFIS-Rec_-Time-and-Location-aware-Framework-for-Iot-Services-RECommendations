import urllib.request, urllib.error, urllib.parse # Mis à jour pour Python 3
import re, os
import zipfile
import shutil


url = 'http://wsdream.github.io/dataset/wsdream_dataset1'
page = urllib.request.urlopen(url) # Mis à jour pour Python 3
html = page.read().decode('utf-8') # Ajout de .decode('utf-8') pour Python 3 (lire les octets en chaîne)
pattern = r'<a id="downloadlink" href="(.*?)"'
downloadlink = re.findall(pattern, html)[0]
file_name = downloadlink.split('/')[-1]
page = urllib.request.urlopen(downloadlink) # Mis à jour pour Python 3
meta = page.info()
file_size = int(meta.get("Content-Length")[0]) # Mis à jour : getheaders est déprécié, utilisez get
print("Downloading: %s (%s bytes)" % (file_name, file_size)) # Syntaxe print de Python 3
urllib.request.urlretrieve(downloadlink, file_name) # Mis à jour pour Python 3

print('Unzip data files...') # Syntaxe print de Python 3
with zipfile.ZipFile(file_name, 'r') as z:
   for name in z.namelist():
      filename = os.path.basename(name)
      # ignore les répertoires
      if not filename:
         continue
      # copie le fichier (tiré de l'extraction de zipfile)
      print(filename) # Syntaxe print de Python 3
      source = z.open(name)
      target = open(filename, "wb") # Remplacé file() par open() pour Python 3
      with source, target:
         shutil.copyfileobj(source, target)

os.remove(file_name)

print('==============================================') # Syntaxe print de Python 3
print('Downloading data done!\n') # Syntaxe print de Python 3