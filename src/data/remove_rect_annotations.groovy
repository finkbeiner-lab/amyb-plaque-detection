import static qupath.lib.gui.scripting.QPEx.*
import qupath.lib.roi.*
import qupath.lib.objects.*


def viewer = qupath.lib.gui.scripting.QPEx.getCurrentViewer()
def hierarchy = viewer.hierarchy


def annotations = hierarchy.annotationObjects
def firstAnnotation = getAnnotationObjects().findAll{it.ROI.getRoiName() == "Rectangle"}
removeObjects(firstAnnotation, true)


