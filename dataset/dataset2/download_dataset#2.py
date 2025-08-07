import urllib.request, urllib.error, urllib.parse # Mis à jour pour Python 3
import re, os
import zipfile
import shutil


url = 'http://wsdream.github.io/dataset/wsdream_dataset2'
page = urllib.request.urlopen(url) # Utilisez urllib.request.urlopen pour Python 3
html = page.read().decode('utf-8') # Décodez les octets en chaîne pour les expressions régulières
pattern = r'<a id="downloadlink" href="(.*?)"'
downloadlink = re.findall(pattern, html)[0]
file_name = downloadlink.split('/')[-1]
page = urllib.request.urlopen(downloadlink) # Utilisez urllib.request.urlopen pour Python 3
meta = page.info()
file_size = int(meta.get("Content-Length")[0]) # Utilisez .get() au lieu de .getheaders() et accédez au premier élément
print("Downloading: %s (%s bytes)" % (file_name, file_size))
urllib.request.urlretrieve(downloadlink, file_name) # Utilisez urllib.request.urlretrieve pour Python 3

print('Unzip data files...')
with zipfile.ZipFile(file_name, 'r') as z:
   for name in z.namelist():
      filename = os.path.basename(name)
      # ignore les répertoires
      if not filename:
         continue
      # copie le fichier (tiré de l'extraction de zipfile)
      print(filename)
      source = z.open(name)
      target = open(filename, "wb") # Utilisez open() au lieu de file() pour Python 3
      with source, target:
         shutil.copyfileobj(source, target)

os.remove(file_name)

print('==============================================')
print('Downloading data done!\n')