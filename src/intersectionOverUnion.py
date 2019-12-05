import numpy as np

def intersectionOverUnion(imPred, imLabel):
    imPred = np.asarray(imPred).copy()
    imLabel = np.asarray(imLabel).copy()

    #go from 0/1 to 1/2 (depending on what are known as unlabeled)
    imPred += 1
    imLabel += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLabel > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLabel)
    (area_intersection, _) = np.histogram(
        intersection, bins=2, range=(1, 2))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=2, range=(1, 2))
    (area_lab, _) = np.histogram(imLabel, bins=2, range=(1, 2))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection/area_union