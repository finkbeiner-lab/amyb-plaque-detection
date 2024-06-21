import qupath.lib.roi.ROIs
import qupath.lib.regions.ImagePlane
import qupath.lib.objects.PathObjects
import qupath.lib.roi.RectangleROI
import qupath.lib.objects.PathAnnotationObject

int z = 0
int t = 0
def plane = ImagePlane.getPlane(z, t)
def roi = ROIs.createRectangleROI(96064-22526,7323-189,  1024, 1024 , plane)
print roi
def rgb = getColorRGB(50, 50, 200)
def pathClass = getPathClass('Other', rgb)
def annotation = new PathAnnotationObject(roi, pathClass)
//def annotation = PathObjects.createAnnotationObject(roi)
addObject(annotation)