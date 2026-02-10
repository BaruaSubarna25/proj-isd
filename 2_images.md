---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ba540a64a5c4b5b4f5bf9318ad3609d7", "grade": false, "grade_id": "cell-19c6eb4994e2384a", "locked": true, "schema_version": 3, "solution": false}}

# Manipuler des images

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "367b2aaa29a7a087acf944d330ea8b11", "grade": false, "grade_id": "cell-19c6eb4994e2384b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans cette feuille, vous allez apprendre à effectuer quelques
manipulations et traitements simples sur les images.  Nous allons
commencer par nous entraîner sur une image riche en couleurs (source:
[wikimedia](https://commons.wikimedia.org/wiki/File:Apple_icon_2.png)):

:::{figure} media/apple.png
:alt: media/apple.png
:width: 40px
:align: center
:::

Pour cela, nous la chargeons avec la bibliothèque `PIL` (Python
Imaging Library) en précisant le nom du fichier la contenant, puis
l'affectons à une variable `img` pour pouvoir la manipuler par la
suite:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: fc4316407949214dd25659ce94a2c8ff
  grade: false
  grade_id: cell-38a01921463de697
  locked: true
  schema_version: 3
  solution: false
---
from PIL import Image
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 7cd24c6d30411319b4b3ad9ee00354fc
  grade: false
  grade_id: cell-b5659c2e482c3848
  locked: true
  schema_version: 3
  solution: false
---
img = Image.open("media/apple.png")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2d659b6c78fbcb6dc4b70deb2c31d7bc", "grade": false, "grade_id": "cell-e75aecf3bd8946db", "locked": true, "schema_version": 3, "solution": false}}

Voici cette image:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b50b98470eb5f237c85aeee0a899fd14
  grade: false
  grade_id: cell-5f412c59d2396365
  locked: true
  schema_version: 3
  solution: false
---
img
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e40db6e9befd8bd904ea5c08897869cf", "grade": false, "grade_id": "cell-450a9499627740e6", "locked": true, "schema_version": 3, "solution": false}}

Pour l'afficher avec des axes et -- lorsque l'image a une basse
résolution -- mieux repérer les pixels individuels, on peut utiliser
`matplotlib`:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 10493e3d9ca1f599aa7c62052f5abf3b
  grade: false
  grade_id: cell-1b1461380b6fef35
  locked: true
  schema_version: 3
  solution: false
---
import matplotlib.pyplot as plt
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 1c892a66a9b93dd8f844691500a9461d
  grade: false
  grade_id: cell-1b1461380b6fef36
  locked: true
  schema_version: 3
  solution: false
---
plt.imshow(img);
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "254806d17d1bb93bf6d739397403bee8", "grade": false, "grade_id": "cell-b74d7516557d88dd", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{hint} Interfaces traditionnelle et objet de matplotlib

Pourquoi un `;` à la fin de la commande précédente?  Parce que
`plt.imshow` ne renvoie pas une image, mais l'affiche par effet de
bord. Le `;` évite l'affichage de ce que renvoie vraiment `plt.imshow`
(un objet de type figure).

Cette approche quelque peu datée est traditionnelle dans des systèmes
comme `Matlab`. La bibliothèque `matplotlib.pyplot` l'a reproduit pour
faciliter la migration d'utilisateurs de ces systèmes. Par habitude
beaucoup d'exemples sur internet utilisent encore cette approche; cela
peut rester pratique comme raccourci dans des exemples en une ligne
comme ci-dessus.

Mais on sait depuis -- et c'est ce que nous vous enseignons depuis le
début de l'année -- que l'on obtient du code beaucoup plus modulaire
si l'on sépare proprement les traitements et calculs (par exemple
construire une figure) des entrées et sorties (par exemple afficher la
figure).

De ce fait, pour tout usage non trivial, il est préférable d'utiliser
l'interface objet de `matplotlib`, comme dans l'exemple suivant:
    
:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 4296a87605d5600bd42c2ec885c8b47b
  grade: false
  grade_id: cell-811d54735c3ff4e5
  locked: true
  schema_version: 3
  solution: false
  task: false
---
from matplotlib.figure import Figure
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b4f1aba47e03a2738c7446c1894a2ddf
  grade: false
  grade_id: cell-0c1a66846231a7bb
  locked: true
  schema_version: 3
  solution: false
  task: false
---
fig = Figure()              # Construction d'une nouvelle figure
subplot = fig.add_subplot() # Ajout d'une zone de dessin (appelée «axes» dans matplotlib) à la figure
subplot.imshow(img)         # Ajout d'une image à la zone de dessin
fig                         # Affichage de la figure
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a1395327180c929e710c10338ec18529", "grade": false, "grade_id": "cell-67d310808fac6650", "locked": true, "schema_version": 3, "solution": false}}

Consultez la documentation de **PIL Image** sur internet, pour trouver
comment obtenir la largeur et la hauteur de cette image. Stockez le
résultat dans des variables `width` et `height` et vérifiez la
cohérence avec la figure ci-dessus.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 0591dd4e4263702807ca56712faf4fea
  grade: false
  grade_id: cell-46598b84e0c79fc6
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# VOTRE CODE ICI
width = img.width
height = img.height
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 727a5f8f2c889479e7681bff4e7b22f4
  grade: true
  grade_id: cell-c6bfc2a73d6866ce
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert width == 256
assert height == 256
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "5aa69c14fc4e4590042c8e5692162775", "grade": false, "grade_id": "cell-5aa2ff2d91d5b0b3", "locked": true, "schema_version": 3, "solution": false}}

## Images comme tableaux

On souhaite maintenant pouvoir accéder au contenu de l'image pour
pouvoir calculer avec. Pour cela, nous allons convertir l'image en un
tableau de nombres `NumPy`, tels ceux que nous avons manipulés dans la
[fiche précédente](1_tableaux.md).

Voici le tableau associé à l'image:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: bbb2d1e391bf0019b32da5b93f05a5b1
  grade: false
  grade_id: cell-8f62152c1e513665
  locked: true
  schema_version: 3
  solution: false
  task: false
---
import numpy as np
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 14c2ac7ae3388ba529818944cb00339a
  grade: false
  grade_id: cell-f1cf391e96f32f52
  locked: true
  schema_version: 3
  solution: false
---
M = np.array(img)
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fb79bd105e1614d14f1c213fe77ee593", "grade": true, "grade_id": "cell-1a34fbbc93618b40", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

En vous référant éventuellement au cours, combien de lignes, de
colonnes et de couches devrait avoir ce tableau?

VOTRE RÉPONSE ICI
4 couches (rgba) de 256 lignes et 256 colonnes
Vérifier avec l'attribut `shape`:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: fbad0bebc7fe362ecae7f9d867a41349
  grade: false
  grade_id: cell-f3258fc6004a71d6
  locked: false
  schema_version: 3
  solution: true
---
# VOTRE CODE ICI
M.shape
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "28ce720e19aa96998b118b7f225a0a8d", "grade": false, "grade_id": "cell-397f146412f6fb75", "locked": true, "schema_version": 3, "solution": false}}

Pourquoi quatre couches? Rouge, Vert, Bleu, ... et transparence!

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9ee110b88a5ebad3e627effe4d6182fc", "grade": false, "grade_id": "cell-397f146412f6fb76", "locked": true, "schema_version": 3, "solution": false}}

### Comprendre les couches de couleurs

Comme toujours, pour mieux comprendre des données, il faut les
visualiser !  Voici une figure représentant notre image et ses trois
couches rouge, vert, bleu.  Observez comment les couleurs de l'image
de départ (blanc, vert, noir, rouge) se décomposent dans les
différentes couches.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: bcf0dc465bbe1270b29f119b907ad266
  grade: false
  grade_id: cell-49b556a7e4717b09
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Échelles de couleur (colormap) allant du noir à la couleur primaire correspondante
from matplotlib.colors import LinearSegmentedColormap
black_red_cmap   = LinearSegmentedColormap.from_list('black_red_cmap',   ["black", "red"])
black_green_cmap = LinearSegmentedColormap.from_list('black_green_cmap', ["black", "green"])
black_blue_cmap  = LinearSegmentedColormap.from_list('black_blue_cmap',  ["black", "blue"])

fig = Figure(figsize=(30, 5));
(subplot, subplotr, subplotg, subplotb) = fig.subplots(1, 4)  # Quatre zones de dessin
# Dessin de l'image et de ses trois couches
subplot.imshow(M)
imgr = subplotr.imshow(M[:,:,0], cmap=black_red_cmap,   vmin=0, vmax=255)
imgg = subplotg.imshow(M[:,:,1], cmap=black_green_cmap, vmin=0, vmax=255)
imgb = subplotb.imshow(M[:,:,2], cmap=black_blue_cmap,  vmin=0, vmax=255)
# Ajout des barres d'échelle de couleur aux images
fig.colorbar(imgr, ax=subplotr);
fig.colorbar(imgg, ax=subplotg);
fig.colorbar(imgb, ax=subplotb);
fig
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b11a9ebb0dfb591f0e9eb7b1cda60cc1", "grade": false, "grade_id": "cell-953569768ed43403", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Par la suite, nous visualiserons de même de nombreuses images.  Il est
donc temps d'automatiser la construction de la figure ci-dessus.

:::{admonition} Exercice

Ouvrez le fichier [utilities.py](utilities.py) et complétez-y la
fonction `show_color_channels` à partir du code ci-dessus.

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 0e0c41454956ed3735d0ea836320cb54
  grade: false
  grade_id: cell-40f2b1ca26aac943
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
from intro_science_donnees import *
from utilities import *
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9f2425888eb560cc5960d99333f9763e
  grade: false
  grade_id: cell-f01f30acda19b78b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(show_color_channels)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9ad91a641c3bf70ea432483f5d39ff4b
  grade: true
  grade_id: cell-66040947f316edfb
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
show_color_channels(img)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b3ae860daf9ea17e4e03590118be55dc", "grade": false, "grade_id": "cell-fda75a63d59e2209", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vérification: `show_color_channels` renvoie bien une figure

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: cb8ea7ebed14f226e1daef4773e4ea17
  grade: true
  grade_id: cell-80328561c59cf158
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert isinstance(show_color_channels(img), Figure)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f7bd8a7b56a3821945eb64ca61b1cebd", "grade": false, "grade_id": "cell-0cf719f423e37ef5", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Étudions maintenant les images du jeu de données de la semaine
dernière:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 2459c104673389925f906a48ed6fe013
  grade: false
  grade_id: cell-1c62320d37fe2e8f
  locked: true
  schema_version: 3
  solution: false
  task: false
---
import os.path
dataset_dir = os.path.join(data.dir, 'ApplesAndBananasSimple')
images = load_images(dataset_dir, "*.png")
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b1f6c574776c5e7928f57c3f9b898b58
  grade: false
  grade_id: cell-a7f1576c7b9d8917
  locked: true
  schema_version: 3
  solution: false
  task: false
---
image_grid(images, titles=images.index)
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "641aa9d1643283fdc0edfe561349d3b0", "grade": true, "grade_id": "cell-9f449b989ea74be6", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

:::{admonition} Exercice

Observez l'image suivante et ses couches. Expliquez ce que vous
voyez. Essayez d'autres exemples.

VOTRE RÉPONSE ICI

:::

```{code-cell} ipython3
img = images[10]
show_color_channels(img)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "12b7f4e53942fba1052f2d8c5ace6a4c", "grade": false, "grade_id": "cell-291f681a5aab593d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous allons maintenant observer l'**histogramme des couleurs**
apparaissant dans une image, en utilisant l'utilitaire
`color_histogram` (vous pouvez comme d'habitude en consulter la
documentation et le code par introspection avec `color_histogram?` et
`color_histogram??`):

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 150948f151ad9f3824cbb0a43b071a9e
  grade: false
  grade_id: cell-ed77018895a029a9
  locked: true
  schema_version: 3
  solution: false
  task: false
---
color_histogram(img)
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8a1d0feb47db8a2737140feed95b45a5", "grade": true, "grade_id": "cell-a887ca1b2eac11df", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}}

:::{admonition} Exercice

Observez les histogrammes ci-dessous de la dixième et la troisième
image, et interprétez-les.

VOTRE RÉPONSE ICI

:::

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: fe1db908c96b5fabf540deecbd4de345
  grade: false
  grade_id: cell-3828ddf071670eff
  locked: true
  schema_version: 3
  solution: false
  task: false
---
img = images[9]
show_color_channels(img)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: bb06a95dbdef21b812d75e4c3598e7d7
  grade: false
  grade_id: cell-3d6d2c2781569087
  locked: true
  schema_version: 3
  solution: false
  task: false
---
color_histogram(img)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9eee30bb80b5461f77c8bab61aec3fa8
  grade: false
  grade_id: cell-4f5ef22b7b70c0c8
  locked: true
  schema_version: 3
  solution: false
  task: false
---
img = images[2]
show_color_channels(img)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 86bdf73b6de46bb6e03d27e2049141b2
  grade: false
  grade_id: cell-7ef43bfaf750914f
  locked: true
  schema_version: 3
  solution: false
  task: false
---
color_histogram(img)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6d2ac3d0585daa5448d3557fb35bfd64", "grade": false, "grade_id": "cell-5d6255f7be07beda", "locked": true, "schema_version": 3, "solution": false}}

## Séparation des couleurs

Nous allons maintenant extraire les trois canaux, rouge, vert,
bleu. Pour le canal des rouges, on extrait le sous-tableau à deux
dimensions de toutes les cases d'indice $(i,j,k)$ avec $k=0$. Le 
`* 1.0` sert à convertir les valeurs en nombres à virgule.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: ebd7f3ac0452d4af2805ee65c072992a
  grade: false
  grade_id: cell-dfe51efcf6df748b
  locked: true
  schema_version: 3
  solution: false
---
R = M[:,:,0] * 1.0
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1956de28640b45b1d33bca95a1ef2388", "grade": false, "grade_id": "cell-59920f8c83495b86", "locked": true, "schema_version": 3, "solution": false}}

Regarder le résultat directement n'est pas très informatif :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 818ecd7d2413dbfd33d86b9a881f88b8
  grade: false
  grade_id: cell-79a4159977138616
  locked: true
  schema_version: 3
  solution: false
---
R
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f78969aede85e0e0845474e62b274a2e", "grade": false, "grade_id": "cell-58b15c024481ac3d", "locked": true, "schema_version": 3, "solution": false}}

Comme d'habitude, il vaut mieux le *visualiser* :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: d3f8a542750fb667a0006e2731499d76
  grade: false
  grade_id: cell-403d27ca87e158c3
  locked: true
  schema_version: 3
  solution: false
---
fig = Figure(figsize=(5,5))
ax, axr = fig.subplots(1,2)
ax.imshow(M)
axr.imshow(R, cmap='Greys_r', vmin=0, vmax=255)
fig
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0ba3504692210914a0772b0a5b258bcc", "grade": false, "grade_id": "cell-f5748504b4d52000", "locked": true, "schema_version": 3, "solution": false, "task": false}}

:::{admonition} Exercice

Extrayez de même le canal des verts et des bleus de la première image
dans les variables `G` et `B`. N'hésitez pas à les visualiser !

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 84ef9c8b6022f20aaaf3339aa5a393dc
  grade: false
  grade_id: cell-2793bd4b1f83558d
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# VOTRE CODE ICI
G = M[:,:,1] * 1.0
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 554525ffc81338eba8930f0cf73ca91e
  grade: true
  grade_id: cell-841034979ffe6094
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert G.shape == (256, 256)
assert abs(G.mean() - 158.27) < 0.1
```

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 00dac5227fae9117048c559d1793e90d
  grade: false
  grade_id: cell-8baf6161cb011920
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# VOTRE CODE ICI
B = M[:,:,2] * 1.0
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 0026e93a099e056f8b2f5bb578ebf29b
  grade: true
  grade_id: cell-841034979ffe6095
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert B.shape == (256, 256)
assert abs(B.mean() - 148.39) < 0.1
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a795bb7e0891ac3388498b32016ec6f1", "grade": false, "grade_id": "cell-233c14413c6e3e6f", "locked": true, "schema_version": 3, "solution": false}}

Il est maintenant facile de faire de l'arithmétique sur tous les
pixels. Par exemple la somme des intensités en vert et rouge s'écrit:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 8c5959d639d7951b07c72c42d2327905
  grade: false
  grade_id: cell-0bd25c9e1e6c6abf
  locked: true
  schema_version: 3
  solution: false
---
G + R
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c4b27d04842cd440b2dc42486b8dc3f8", "grade": false, "grade_id": "cell-5d168111627796e1", "locked": true, "schema_version": 3, "solution": false}}

:::{admonition} Exercice

Calculez et visualisez la luminosité de tous les pixels de l'image, la
*luminosité* d'un pixel $(r,g,b)$ étant définie comme la moyenne
$v=\frac{r+g+b}{3}$:

:::

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 93109febd0552684cd6e4de30f5ab725
  grade: false
  grade_id: cell-4956fa101f9c567c
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# VOTRE CODE ICI
V = (R + G + B) / 3
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 7c5e41e9533f15b0c0cca359420e2595
  grade: true
  grade_id: cell-841034979ffe6096
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert V.shape == (256, 256)
assert abs(V.mean() - 172.44) < 0.1
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f52b82486f95773b2b9c2e42805e921c", "grade": false, "grade_id": "cell-5d168111627796e2", "locked": true, "schema_version": 3, "solution": false}}

Vous venez de transformer l'image en niveaux de gris! Pour que cela
colle au mieux avec notre perception visuelle, il faudrait en fait
utiliser une moyenne légèrement pondérée; voir par exemple la
[Wikipedia](https://fr.wikipedia.org/wiki/Niveau_de_gris#Convertir_une_image_couleur_en_niveau_de_gris).

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7f4af73eb1742c465180687ba662f362", "grade": false, "grade_id": "cell-d0fb04b5c7bf1744", "locked": true, "schema_version": 3, "solution": false}}

## Conclusion

Vous avons vu dans cette feuille comment charger une image dans Python
et effectuer quelques manipulations, visualisations et calculs simples
dessus. Cela a été l'occasion de mieux comprendre la décomposition
d'une image en couches de couleur.

:::{admonition} Exercice

Mettez à jour votre rapport, et notamment la section « revue de code »
pour vérifier vos utilitaires dans [utilities.py](utilities.py).

:::

Vous pouvez maintenant passer à
l'[extraction d'attributs](3_extraction_d_attributs.md)
