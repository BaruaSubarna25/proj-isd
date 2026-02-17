##############################################################################
# Imports

from numbers import Number
from typing import List, Iterable, Union, Any
import warnings

from PIL import Image  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore

from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.axes import Axes  # type: ignore
from matplotlib.colors import LinearSegmentedColormap  # type: ignore


warnings.simplefilter(action="ignore", category=FutureWarning)

##############################################################################
# Traitement d'images

black_red_cmap = LinearSegmentedColormap.from_list("black_red_cmap", ["black", "red"])
black_green_cmap = LinearSegmentedColormap.from_list(
    "black_green_cmap", ["black", "green"]
)
black_blue_cmap = LinearSegmentedColormap.from_list(
    "black_blue_cmap", ["black", "blue"]
)


def show_color_channels(img: Image.Image) -> Figure:
    """
    Return a figure displaying the image together with its red, green, and blue layers
    """
    M = np.asarray(img)

    fig = Figure(figsize=(30, 5))
    (subplot, subplotr, subplotg, subplotb) = fig.subplots(1, 4)

    # Dessin de l'image et de ses trois couches
    subplot.imshow(M)
    imgr = subplotr.imshow(M[:, :, 0], cmap=black_red_cmap, vmin=0, vmax=255)
    imgg = subplotg.imshow(M[:, :, 1], cmap=black_green_cmap, vmin=0, vmax=255)
    imgb = subplotb.imshow(M[:, :, 2], cmap=black_blue_cmap, vmin=0, vmax=255)

    # Ajout des barres d'échelle de couleur aux images
    fig.colorbar(imgr, ax=subplotr)
    fig.colorbar(imgg, ax=subplotg)
    fig.colorbar(imgb, ax=subplotb)
    return fig


def color_histogram(img: Image.Image) -> Figure:
    """
    Return a histogram of the color channels of the image
    """
    M = np.array(img)
    # Si je mets un commentaire dans mon code c'est bien.
    # Mais s'il respecte les conventions de codage c'est mieux.
   # Décalez ce commentaire pour respecter les conventions !
    n, p, m = M.shape
    MM = np.reshape(M, (n * p, m))
    if m == 4:  # Discard transparency channel if present
        MM = MM[:, 0:3]
    colors = ["red", "green", "blue"]
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot()
    ax.hist(MM, bins=10, density=True, histtype="bar", color=colors, label=colors)
    ax.set_xlabel("Pixel amplitude in each color channel")
    ax.set_ylabel("Pixel density")
    return fig


def foreground_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    M = np.asarray(img)
    
    G = np.min(M[:, :, 0:3], axis=2)
    
    F = G < theta
    return F

def transparent_background_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 10
) -> Image.Image:
    """Create a cropped image with transparent background."""
    F = foreground_filter(img, theta=theta)
    M = np.array(img)
    N = np.zeros([M.shape[0], M.shape[1], 4], dtype=M.dtype)
    N[:, :, :3] = M[:, :, :3]
    N[:, :, 3] = F * 255
    return Image.fromarray(N)


def transparent_background(img: Image.Image) -> Image.Image:
    """Sets the white background of an image to transparent"""
    data = img.getdata()  # Get a list of tuples
    newData = []
    for a in data:
        a = a[:3]  # Shorten to RGB
        if np.mean(np.array(a)) == 255:  # the background is white
            a = a + (0,)  # Put a transparent value in A channel (the fourth one)
        else:
            a = a + (255,)  # Put a non- transparent value in A channel
        newData.append(a)
    img.putdata(newData)  # Get new img ready
    return img


##############################################################################
# Attributs


def redness(img: Image.Image) -> float:
    """Return the redness of a PIL image."""
    M = np.asarray(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    
    F = foreground_filter(img)
    
    mean_R = np.mean(R[F])
    mean_G = np.mean(G[F])

    return mean_R - mean_G


def elongation(img: Image.Image) -> float: 
    """Extract the scalar value elongation from a PIL image."""
    F = foreground_filter(img)
    # Build the cloud of points given by the foreground image pixels
    xy = np.argwhere(F)
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    return s[0] / s[1]


def elongation_plot( img: Image.Image, subplot: Axes ) -> None:
    """Plot the principal axes of the SVD when computing the elongation"""
    # Build the cloud of points defined by the foreground image pixels
    F = foreground_filter(img)
    xy = np.argwhere(F)
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)

    N = len(xy)
    a0 = s[0] / np.sqrt(N)
    a1 = s[1] / np.sqrt(N)

    # Plot the center
    subplot.plot(
        C[1], C[0], "ro", linewidth=50, markersize=10
    )  # x and y are j and i in matrix coord.
    # Plot the principal axes
    subplot.plot(
        [C[1], C[1] + a0 * V[0, 1]], [C[0], C[0] + a0 * V[0, 0]], "r-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] - a0 * V[0, 1]], [C[0], C[0] - a0 * V[0, 0]], "r-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] + a1 * V[1, 1]], [C[0], C[0] + a1 * V[1, 0]], "g-", linewidth=3
    )
    subplot.plot(
        [C[1], C[1] - a1 * V[1, 1]], [C[0], C[0] - a1 * V[1, 0]], "g-", linewidth=3
    )


##############################################################################
# Nouveaux attributs

def brightness(img: Image.Image) -> Number:
    """Luminosité moyenne de l'image (en niveaux de gris)."""
    M = np.array(img.convert("RGB"), dtype=float)
    gray = M.mean(axis=2)          
    return float(gray.mean())

def vertical_line(img):
    M = np.array(img.convert("RGB"), dtype=float)
    gray = M.mean(axis=2)

    mask = gray < 245
    if not mask.any():
        return 0.0

    haut = mask[:5, :].sum()
    bas = mask[-5:, :].sum()

    return float((haut - bas) / (haut + bas + 1e-6))

def redness_tete(img: Image.Image) -> Number:
   
    M = np.array(img.convert("RGB"), dtype=float)

    # Extraire les 12 premières lignes
    part_haut = M[:12, :, :]

    R = part_haut[:, :, 0]
    G = part_haut[:, :, 1]
    
    redness_val = np.mean(R - G)

    return float(redness_val)

##############################################################################
# Analyse de performance


def error_rate(solutions: pd.Series, predictions: pd.Series) -> Any:
    """
    Return the error rate between two vectors.
    """
    solutions, predictions = solutions.align(predictions, join="inner")
    return float((solutions != predictions).mean())

def split_data(X, Y, verbose=True, seed=0):
    """Make a 50/50 training/test data split (stratified).
    Return the indices of the split train_idx and test_idx."""
    SSS = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    ((train_index, test_index),) = SSS.split(X, Y)
    if verbose:
        print("TRAIN:", train_index, "TEST:", test_index)
    return (train_index, test_index)


def make_scatter_plot(
    df,
    images,
    train_index=[],
    test_index=[],
    filter=None,
    predicted_labels=[],
    show_diag=False,
    axis="normal",
    feat=None,
    theta=None,
) -> Figure:
    """This scatter plot function allows us to show the images.

    predicted_labels can either be:
                    - None (queries shown as question marks)
                    - a vector of +-1 predicted values
                    - the string "GroundTruth" (to display the test images).
    Other optional arguments:
            show_diag: add diagonal dashed line if True.
            feat and theta: add horizontal or vertical line at position theta
            axis: make axes identical if 'square'."""
    fruit = np.array(["B", "A"])

    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot()

    nsample, nfeat = df.shape
    if len(train_index) == 0:
        train_index = range(nsample)
    # Plot training examples
    x = df.iloc[train_index, 0]
    y = df.iloc[train_index, 1]
    f = images.iloc[train_index]
    ax.scatter(x, y, s=750, marker="o", c="w")

    for x0, y0, img in zip(x, y, f):
        ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
        ax.add_artist(ab)

    # Plot test examples
    x = df.iloc[test_index, 0]
    y = df.iloc[test_index, 1]

    if isinstance(predicted_labels, str) and predicted_labels == "GroundTruth":
        f = images.iloc[test_index]
        ax.scatter(x, y, s=500, marker="s", color="c")
        for x0, y0, img in zip(x, y, f):
            ab = AnnotationBbox(OffsetImage(img), (x0, y0), frameon=False)
            ax.add_artist(ab)

    elif (len(predicted_labels) > 0 and not
            (predicted_labels == "GroundTruth").all()):
        label = (predicted_labels + 1) / 2
        ax.scatter(x, y, s=250, marker="s", color="c")
        for x0, y0, lbl in zip(x, y, label):
            ax.text(
                x0 - 0.03,
                y0 - 0.03,
                fruit[int(lbl)],
                color="w",
                fontsize=12,
                weight="bold",
            )

    else:  # Plot UNLABELED test examples
        f = images[test_index]
        ax.scatter(x, y, s=250, marker="s", c="c")
        ax.scatter(x, y, s=100, marker="$?$", c="w")

    if axis == "square":
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(f"$x_1$ = {df.columns[0]}")
    ax.set_ylabel(f"$x_2$ = {df.columns[1]}")

    # Add line on the diagonal
    if show_diag:
        ax.plot([-3, 3], [-3, 3], "k--")

    # Add separating line along one of the axes
    if theta is not None:
        if feat == 0:  # vertical line
            ax.plot([theta, theta], [-3, 3], "k--")
        else:  # horizontal line
            ax.plot([-3, 3], [theta, theta], "k--")

    return fig


##############################################################################
# Classifier


class OneRule:
    def __init__(self) -> None:
        """
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.
        """
        self.is_trained = False
        self.ig = 0  # Index of the good feature G
        self.w = 1  # Feature polarity
        self.theta = 0  # Threshold on the good feature

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        """
        This function should train the model parameters.

        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            Y: Training label matrix of dim num_train_samples * 1.
        Both inputs are panda dataframes.
        """
        # Compute the correlation between the class and the attributes
        # Hint:
        # - Use pd.concat to reconstruct the original table with both attributes and
        #   class
        # - Compute the correlation matrix
        # - Extract the "class" column, and remove its "class" row
        # Remplacer la ligne suivante par le code adéquat
        raise NotImplementedError("code non implanté ligne 307");
        # Store in self.attribute the attribute which maximizes the correlation
        # Hint: use the idxmax method
        # Remplacer la ligne suivante par le code adéquat
        raise NotImplementedError("code non implanté ligne 311");
        # Store in self.sign the sign of the correlation for this attribute (1 or -1)
        # Hint: use np.sign
        # Remplacer la ligne suivante par le code adéquat
        raise NotImplementedError("code non implanté ligne 315");
        # Choose a threshold and store it in self.theta
        # Hint:
        # - Compute the mean of the value of the attribute for the elements
        #   in the first class
        # - Compute the mean of the value of the attribute for the elements
        #   in the second class
        # - Take as threshold the average of these
        # Remplacer la ligne suivante par le code adéquat
        raise NotImplementedError("code non implanté ligne 324");
        self.is_trained = True
        print(
            f"FIT: Training Successful. Feature selected: {self.attribute}; "
            f"Polarity: {self.sign}; Threshold: {self.theta:5.2f}."
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Return predictions for the elements described by X

        Args:
            X: Test data matrix of dim num_test_samples * num_feat.
        Return:
            Y: Predicted label matrix of dim num_test_samples * 1.
        """
        # Fetch the feature of interest and multiply by polarity
        G = X[self.attribute]
        # Make decisions according to the threshold theta and sign
        Y = G.copy()
        Y[G < self.theta] = -self.sign
        Y[G >= self.theta] = self.sign
        print("PREDICT: Prediction done")
        return Y
